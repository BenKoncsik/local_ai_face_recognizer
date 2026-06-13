"""Background worker for the image-browser "re-recognize faces" workflow.

A short-lived :class:`QThread` that, for a set of images, snapshots their
unresolved Unknown faces, scores each against the named-person profiles, applies
the confident (AUTO) merges, and hands the uncertain (SUGGEST) matches back to
the UI for review.  All DB access happens on this thread via ``session_scope``,
keeping the Qt UI thread responsive.

Scoring is sequential here (a handful of vectorised matmuls per face), which is
plenty fast and avoids cross-thread ORM/session hazards; progress is emitted —
throttled — so the modal progress dialog can update live and offer Cancel.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import List, Optional, Sequence

from PySide6.QtCore import QThread, Signal

from app.config import AppConfig
from app.db.database import session_scope
from app.jobs.cancellation import CancellationToken, OperationCancelled
from app.services.rerecognition_service import (
    KIND_AUTO,
    KIND_SUGGEST,
    AutoItem,
    ReRecognitionResult,
    ReRecognitionService,
)

log = logging.getLogger(__name__)

_PROGRESS_THROTTLE_MS = 80.0


class ReRecognitionWorker(QThread):
    """Re-recognize Unknown faces on the given images in the background.

    Signals:
        progress:        ``(processed, total, auto_count, suggest_count)`` — throttled.
        finished_result: ``ReRecognitionResult`` — emitted once on completion.
        failed:          ``str`` — emitted with an error message on failure.
    """

    progress = Signal(int, int, int, int)
    finished_result = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        image_ids: Sequence[int],
        config: AppConfig,
        *,
        auto_threshold: Optional[float] = None,
        suggest_threshold: Optional[float] = None,
        engine: str = "classic",   # "classic" | "deep"
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._image_ids = [int(i) for i in image_ids]
        self._config = config
        self._auto_threshold = auto_threshold
        self._suggest_threshold = suggest_threshold
        self._engine = engine
        self._token = CancellationToken()
        self._last_emit = 0.0
        # Set when running under the Task Manager (cancel/pause + progress).
        self._ctx = None

    def cancel(self) -> None:
        self._token.cancel()

    # ------------------------------------------------------------------

    def run_in_task(self, ctx) -> ReRecognitionResult:  # noqa: ANN001
        """Run under the Task Manager; return the ReRecognitionResult.

        Reuses the task's cancellation token (so Stop/Pause in the Task Manager
        control this run) and routes progress through ``ctx``.  Raises
        :class:`OperationCancelled` on cancel; raises on hard failure.
        """
        self._ctx = ctx
        self._token = ctx.token
        result = self._execute_deep() if self._engine == "deep" else self._execute()
        self._run_ai_face_detection(result)
        return result

    def run(self) -> None:  # noqa: D401 — QThread entry point (legacy)
        try:
            result = self._execute_deep() if self._engine == "deep" else self._execute()
        except OperationCancelled:
            log.info("Re-recognition cancelled by user")
            self.finished_result.emit(ReRecognitionResult(cancelled=True))
            return
        except Exception as exc:  # noqa: BLE001
            log.exception("Re-recognition worker failed")
            self.failed.emit(str(exc))
            return
        self._run_ai_face_detection(result)
        self.finished_result.emit(result)

    def _run_ai_face_detection(self, result: ReRecognitionResult) -> None:
        """Best-effort AI face-detection pass on the re-recognized images.

        Analysis only: stores where the pretrained deep-learning detector sees
        faces (``ai_face_detections`` table) without touching the classic
        re-recognition outcome.  Never raises — if the AI model is missing or
        anything fails, the error is logged and re-recognition completes
        exactly as before.
        """
        try:
            cfg = getattr(self._config, "ai_face_detection", None)
            if cfg is None or not cfg.enabled or not cfg.run_on_rerecognition:
                return

            from app.services.ai_face_detection_service import (
                SOURCE_RERECOGNITION,
                AiFaceDetectionService,
            )

            with session_scope() as session:
                svc = AiFaceDetectionService(session, cfg)
                stats = svc.detect_images(
                    self._image_ids,
                    source=SOURCE_RERECOGNITION,
                    cancel_check=lambda: self._token.cancelled,
                )
            if not stats.available:
                log.warning("AI face detection skipped: %s", stats.error)
                return
            result.ai_images_scanned = stats.images_processed
            result.ai_faces_found = stats.faces_found
        except Exception as exc:  # noqa: BLE001
            log.warning("AI face detection after re-recognition failed: %s", exc)

    def _execute(self) -> ReRecognitionResult:
        with session_scope() as session:
            svc = ReRecognitionService(
                session,
                self._config.recognition,
                auto_threshold=self._auto_threshold,
                suggest_threshold=self._suggest_threshold,
            )
            profiles = svc.load_profiles()
            candidates = svc.extract_candidates(self._image_ids)
            rejected = svc.load_rejected_targets(candidates)

            result = ReRecognitionResult(n_examined=len(candidates))
            total = len(candidates)
            if not profiles or total == 0:
                log.info(
                    "Re-recognition: nothing to do (%d profile(s), %d candidate(s))",
                    len(profiles), total,
                )
                self._emit_progress(total, total, 0, 0, force=True)
                return result

            auto_items: List[AutoItem] = []
            for i, face in enumerate(candidates, 1):
                self._token.raise_if_cancelled()
                kind, item = svc.classify(face, profiles, rejected)
                if kind == KIND_AUTO:
                    auto_items.append(item)  # type: ignore[arg-type]
                    result.n_auto += 1
                elif kind == KIND_SUGGEST:
                    result.suggest_items.append(item)  # type: ignore[arg-type]
                else:
                    result.n_none += 1
                self._emit_progress(i, total, result.n_auto, result.n_suggest)

            # Final progress tick (always emitted, not throttled).
            self._emit_progress(total, total, result.n_auto, result.n_suggest, force=True)

            # One batch id per run, so any later review-dialog merges group with
            # this run's automatic merges (and undo together).
            batch_id = uuid.uuid4().hex
            result.batch_id = batch_id
            if auto_items:
                svc.apply_auto_merges(auto_items, batch_id)
            log.info(
                "Re-recognition done: %d examined, %d auto, %d to review, %d none",
                result.n_examined, result.n_auto, result.n_suggest, result.n_none,
            )
            return result

    def _execute_deep(self) -> ReRecognitionResult:
        """Re-recognition using the deep learning (MLP ensemble) classifier."""
        from pathlib import Path

        from app.deep.classifier import DeepFaceClassifier
        from app.services.rerecognition_service import FaceData, SuggestItem

        with session_scope() as session:
            svc = ReRecognitionService(
                session,
                self._config.recognition,
                auto_threshold=self._auto_threshold,
                suggest_threshold=self._suggest_threshold,
            )
            candidates = svc.extract_candidates(self._image_ids)
            result = ReRecognitionResult(n_examined=len(candidates))
            total = len(candidates)

            if total == 0:
                self._emit_progress(0, 0, 0, 0, force=True)
                return result

            # Load the saved deep model (no retraining — use whatever is on disk)
            model_dir = Path(
                self._config.resolve(self._config.deep_recognition.model_dir)
            )
            classifier = DeepFaceClassifier(self._config.deep_recognition)
            if not classifier.load(model_dir / "deep_face_model.pkl"):
                log.warning("Deep re-recognition: no trained model found")
                self._emit_progress(total, total, 0, 0, force=True)
                return result

            auto_thr = self._auto_threshold or self._config.recognition.rerecognition_auto_threshold
            suggest_thr = self._suggest_threshold or self._config.recognition.rerecognition_suggest_threshold

            auto_items: List[AutoItem] = []
            for i, face in enumerate(candidates, 1):
                self._token.raise_if_cancelled()
                pred = classifier.predict(face.embedding)

                if pred.person_id is None:
                    result.n_none += 1
                elif pred.score >= auto_thr:
                    auto_items.append(
                        AutoItem(
                            face=face,
                            target_person_id=pred.person_id,
                            target_name=pred.person_name or str(pred.person_id),
                            score=pred.score,
                        )
                    )
                    result.n_auto += 1
                elif pred.score >= suggest_thr:
                    from app.services.rerecognition_service import Candidate
                    result.suggest_items.append(
                        SuggestItem(
                            face=face,
                            candidates=[
                                Candidate(
                                    person_id=pred.person_id,
                                    name=pred.person_name or str(pred.person_id),
                                    score=pred.score,
                                )
                            ],
                        )
                    )
                else:
                    result.n_none += 1

                self._emit_progress(i, total, result.n_auto, result.n_suggest)

            self._emit_progress(total, total, result.n_auto, result.n_suggest, force=True)

            batch_id = uuid.uuid4().hex
            result.batch_id = batch_id
            if auto_items:
                svc.apply_auto_merges(auto_items, batch_id)

            log.info(
                "Deep re-recognition done: %d examined, %d auto, %d to review, %d none",
                result.n_examined, result.n_auto, result.n_suggest, result.n_none,
            )
            return result

    def _emit_progress(
        self, processed: int, total: int, auto: int, suggest: int,
        *, force: bool = False,
    ) -> None:
        if not force:
            now = time.monotonic() * 1000.0
            if now - self._last_emit < _PROGRESS_THROTTLE_MS and processed < total:
                return
            self._last_emit = now
        if self._ctx is not None:
            pct = int(processed / total * 100) if total else 0
            self._ctx.report(
                min(max(pct, 0), 100),
                f"{processed}/{total} (auto {auto}, ? {suggest})",
            )
            # Block while paused; raises OperationCancelled on cancel.
            self._ctx.token.wait_if_paused()
        else:
            self.progress.emit(processed, total, auto, suggest)
