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
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._image_ids = [int(i) for i in image_ids]
        self._config = config
        self._auto_threshold = auto_threshold
        self._suggest_threshold = suggest_threshold
        self._token = CancellationToken()
        self._last_emit = 0.0

    def cancel(self) -> None:
        self._token.cancel()

    # ------------------------------------------------------------------

    def run(self) -> None:  # noqa: D401 — QThread entry point
        try:
            result = self._execute()
        except OperationCancelled:
            log.info("Re-recognition cancelled by user")
            self.finished_result.emit(ReRecognitionResult(cancelled=True))
            return
        except Exception as exc:  # noqa: BLE001
            log.exception("Re-recognition worker failed")
            self.failed.emit(str(exc))
            return
        self.finished_result.emit(result)

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
                self.progress.emit(total, total, 0, 0)
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
            self.progress.emit(total, total, result.n_auto, result.n_suggest)

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

    def _emit_progress(
        self, processed: int, total: int, auto: int, suggest: int
    ) -> None:
        now = time.monotonic() * 1000.0
        if now - self._last_emit < _PROGRESS_THROTTLE_MS and processed < total:
            return
        self._last_emit = now
        self.progress.emit(processed, total, auto, suggest)
