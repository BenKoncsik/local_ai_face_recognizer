"""Deep-learning pipeline worker (the "new" recognition path).

Runs the full AI pipeline in a QThread so the GUI stays responsive.  Two
modes exist, matching the two cards of the AI tab in Scan & Maintenance:

``rescan``
    Scans for new images, detects + embeds the new faces, removes duplicate
    overlapping boxes (a box assigned to a person always survives), retrains
    the neural network from every labeled face, then tries to place the
    still-unknown faces with the trained model.  Existing recognitions are
    never displaced.

``rebuild``
    Rebuilds the face database from scratch: every automatically detected
    box is deleted (manually drawn boxes and human-confirmed person
    assignments survive — they are the training data), every image is
    re-detected and re-embedded, then the same train + recognize flow runs.

``train``
    Embeds the missing faces, then retrains the model — nothing else.

``detect_faces``
    Analysis-only AI face detection over every image: stores where the
    pretrained deep-learning detector sees faces (bounding box + confidence)
    in the ``ai_face_detections`` table.  No Face row is created or changed.

Accuracy is preferred over speed throughout: training may take minutes and
is allowed to saturate every CPU core.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal

from app.config import AppConfig
from app.db.database import init_db, session_scope
from app.db.models import Face, Image, Person
from app.deep.dataset import TRUSTED_MANUAL_SOURCES
from app.detectors.factory import create_detector
from app.embeddings.tflite_embedder import TFLiteEmbedder
from app.jobs.cancellation import OperationCancelled
from app.workers.pipeline_worker import PipelineResult
from app.services.clustering_service import ClusteringService, ClusteringStats
from app.services.deep_recognition_service import (
    DeepRecognitionService,
    TrainAndRecognizeResult,
)
from app.services.detection_service import DetectionService
from app.services.embedding_service import EmbeddingService
from app.services.intra_image_consistency_service import (
    IntraImageConsistencyService,
    IntraImageConsistencyStats,
)
from app.services.overlap_resolution_service import (
    OverlapResolutionService,
    OverlapResolutionStats,
)
from app.services.scan_service import ScanService
from app.services.suggestion_service import SuggestionService

log = logging.getLogger(__name__)

MODE_RESCAN = "rescan"
MODE_REBUILD = "rebuild"
MODE_TRAIN = "train"
# Analysis-only AI face detection (where are the faces + confidence); never
# creates Face rows and never touches the classic recognition results.
MODE_DETECT_FACES = "detect_faces"


class DeepPipelineWorker(QThread):
    """QThread running the deep-learning processing pipeline.

    Signals:
        progress:                ``(current, total, stage, detail)``
        log_message:             ``(message)``
        suggestions_ready:       ``(count)`` — open merge/name suggestions
        auto_assignments_ready:  ``(count)`` — reviewable automatic groupings
        finished:                ``(success, summary)``
        error:                   ``(message)``
        face_debug:              ``(DeepDebugInfo)`` — emitted per-face when
                                 ``ai_visualization`` or ``ai_debug_log`` is on
    """

    progress = Signal(int, int, str, str)
    log_message = Signal(str)
    suggestions_ready = Signal(int)
    auto_assignments_ready = Signal(int)
    finished = Signal(bool, str)
    error = Signal(str)
    face_debug = Signal(object)

    def __init__(
        self,
        root_folder: str,
        config: AppConfig,
        mode: str = MODE_RESCAN,
        parent=None,
        db_path_override: Optional[str] = None,
        drive_client=None,
        drive_root_folder_id: Optional[str] = None,
        drive_mirror_dir: Optional[Path] = None,
        ai_visualization: bool = False,
        ai_debug_log: bool = False,
    ) -> None:
        super().__init__(parent)
        if mode not in (MODE_RESCAN, MODE_REBUILD, MODE_TRAIN, MODE_DETECT_FACES):
            raise ValueError(f"Unknown deep pipeline mode: {mode!r}")
        self._root_folder = root_folder
        self._config = config
        self._mode = mode
        self._abort = False
        self._db_path_override = db_path_override
        self._drive_client = drive_client
        self._drive_root_folder_id = drive_root_folder_id
        self._drive_mirror_dir = drive_mirror_dir
        self._ai_visualization = ai_visualization
        self._ai_debug_log = ai_debug_log
        # Set when running under the Task Manager (cancel/pause + progress).
        self._ctx = None

    @property
    def _drive_mode(self) -> bool:
        return self._drive_client is not None

    @property
    def mode(self) -> str:
        return self._mode

    def abort(self) -> None:
        """Request a graceful stop (legacy QThread mode)."""
        self._abort = True
        log.info("Deep pipeline abort requested")

    # ------------------------------------------------------------------
    # Progress / log / cancel routing (mode-aware)
    # ------------------------------------------------------------------

    def _emit_progress(self, current: int, total: int, stage: str, detail: str) -> None:
        if self._ctx is not None:
            pct = int(current / total * 100) if total else 0
            self._ctx.report(min(max(pct, 0), 100), f"{stage}: {detail}")
            self._ctx.checkpoint()  # responsive pause/cancel mid-stage
        else:
            self.progress.emit(current, total, stage, detail)

    def _emit_log(self, message: str) -> None:
        # Queued cross-thread signal in both modes (worker lives on UI thread).
        self.log_message.emit(message)

    def _checkpoint(self) -> None:
        """Raise OperationCancelled on cancel; block while paused."""
        if self._ctx is not None:
            self._ctx.checkpoint()
        elif self._abort:
            raise OperationCancelled()

    def _is_cancel_requested(self) -> bool:
        if self._ctx is not None:
            return self._ctx.token.cancelled
        return self._abort

    def _write_debug_log(self, info: object) -> None:
        """Append one JSON line to data/deep_debug.jsonl."""
        import json
        from app.deep.debug_info import DeepDebugInfo
        if not isinstance(info, DeepDebugInfo):
            return
        try:
            log_path = self._config.resolve("data/deep_debug.jsonl")
            entry = {
                "face_id": info.face_id,
                "crop_path": info.crop_path,
                "mode": info.mode,
                "embedding_norm": round(info.embedding_norm, 4),
                "embedding_top_dims": [(int(i), round(float(v), 4)) for i, v in info.embedding_top_dims],
                "all_similarities": {k: round(v, 4) for k, v in info.all_similarities.items()},
                "gates": [
                    {
                        "name": g.name,
                        "passed": g.passed,
                        "value": round(g.value, 4),
                        "threshold": round(g.threshold, 4),
                    }
                    for g in info.gates
                ],
                "output_probs": {k: round(v, 4) for k, v in list(info.output_probs.items())[:10]},
                "decision": {
                    "person_id": info.prediction.person_id,
                    "person_name": info.prediction.person_name,
                    "score": round(info.prediction.score, 4),
                    "probability": round(info.prediction.probability, 4),
                    "similarity": round(info.prediction.similarity, 4),
                    "margin": round(info.prediction.margin, 4),
                    "reason": info.prediction.reason,
                },
            }
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001
            log.warning("Debug log write failed: %s", exc)

    def run_in_task(self, ctx) -> PipelineResult:  # noqa: ANN001
        """Run on the Task Manager's worker thread; return a PipelineResult.

        Raises :class:`OperationCancelled` on cancel and blocks while paused
        (both via ``ctx.checkpoint()``); raises on hard failure.
        """
        self._ctx = ctx
        return self._run_pipeline()

    def run(self) -> None:
        """Execute the deep pipeline as a standalone QThread (legacy)."""
        try:
            result = self._run_pipeline()
        except OperationCancelled:
            self.finished.emit(False, "Aborted")
        except Exception as exc:  # noqa: BLE001
            msg = f"Deep pipeline error: {exc}\n{traceback.format_exc()}"
            log.error(msg)
            self.error.emit(str(exc))
            self.finished.emit(False, str(exc))
        else:
            self.suggestions_ready.emit(result.n_suggestions)
            self.auto_assignments_ready.emit(result.n_auto_assignments)
            self.finished.emit(result.success, result.summary)

    # ------------------------------------------------------------------

    def _run_pipeline(self) -> PipelineResult:
        db_path = self._db_path_override or str(self._config.db_path_resolved)
        init_db(db_path)

        if self._mode == MODE_TRAIN:
            return self._run_train_only_pipeline()

        if self._mode == MODE_DETECT_FACES:
            return self._run_detect_only_pipeline()

        n_stages = 10 if self._mode == MODE_REBUILD else 9
        stage = [0]

        def announce(message: str) -> None:
            stage[0] += 1
            self._emit_log(f"Stage {stage[0]}/{n_stages}: {message}")

        # --- Rebuild only: wipe automatic data, keep human decisions ---
        if self._mode == MODE_REBUILD:
            announce("Rebuilding from scratch — clearing automatic data …")
            n_deleted, n_kept = self._reset_for_rebuild()
            self._emit_log(
                f"  Removed {n_deleted} automatic face box(es); "
                f"kept {n_kept} human-confirmed face(s) as training data."
            )
            self._checkpoint()

        # --- Scan ---
        announce(
            "Scanning Google Drive project folder …"
            if self._drive_mode
            else "Scanning image folder …"
        )
        new_ids = self._run_scan()
        self._checkpoint()

        # --- Detection ---
        pending = self._get_pending_detection_ids()
        announce(f"Detecting faces in {len(pending)} image(s) …")
        total_faces = self._run_detection(pending)
        self._checkpoint()

        # --- AI face detection (analysis only, best-effort) ---
        announce(f"AI face detection on {len(pending)} image(s) …")
        ai_stats = self._run_ai_face_detection(pending)
        self._checkpoint()

        # --- Embedding ---
        announce("Generating face embeddings …")
        try:
            embedded = self._run_embedding()
        except ImportError as exc:
            log.error("TFLite backend missing: %s", exc)
            raise RuntimeError(
                "Hiányzik a TFLite futtatókörnyezet. "
                "Telepítsd/javítsd a függőségeket:\n"
                "  pip install ai-edge-litert\n"
                f"Részletek: {exc}"
            ) from exc
        self._checkpoint()

        # --- Overlapping-box resolution ---
        announce("Resolving overlapping face boxes …")
        overlap_stats = self._run_overlap_resolution()
        if overlap_stats.faces_removed:
            self._emit_log(
                f"  Removed {overlap_stats.faces_removed} duplicate box(es); "
                f"assigned faces always kept."
            )
        self._checkpoint()

        # --- Permanently-ignored face filter ---
        self._run_ignored_filter()

        # --- Train + recognize (the deep learning core) ---
        announce(
            "Training the neural network from your categorized faces "
            "(this may take a while — accuracy over speed) …"
        )
        result = self._run_deep_train_and_recognize()
        self._checkpoint()

        # --- Cluster the remaining unknown faces into groups ---
        announce("Grouping remaining unknown faces …")
        cluster_stats = self._run_clustering()

        # --- Same-image identity consistency ---
        announce("Unifying same-person faces within each image …")
        consistency_stats = self._run_intra_image_consistency()

        # --- Suggestions / review counters ---
        announce("Collecting review items …")
        n_suggestions = self._run_suggestions()

        train = result.train
        rec = result.recognition
        acc = (
            f"{train.validation_accuracy * 100:.1f}%"
            if train.validation_accuracy is not None
            else "n/a"
        )
        ai_part = (
            f"AI detect: {ai_stats.faces_found} face(s) on "
            f"{ai_stats.images_processed} image(s)"
            if ai_stats.available
            else "AI detect: unavailable"
        )
        summary = (
            f"Done ({self._mode}) — {len(new_ids)} new image(s), "
            f"{total_faces} face(s) detected, {embedded} embedded | "
            f"{ai_part} | "
            f"model: {train.n_persons} person(s), {train.n_examples} example(s), "
            f"accuracy {acc}"
            f"{' (model reused — data unchanged)' if train.reused_existing_model else ''} | "
            f"AI assigned {rec.n_assigned} of {rec.n_candidates} unknown face(s) "
            f"(outlier={rec.n_rejected_outlier}, "
            f"unsure={rec.n_rejected_threshold + rec.n_rejected_margin + rec.n_rejected_prototype}) | "
            f"overlaps removed: {overlap_stats.faces_removed} | "
            f"unknown groups: +{cluster_stats.n_new_persons} | "
            f"intra-image fixes: {consistency_stats.n_faces_reassigned} | "
            f"{n_suggestions} suggestion(s)"
        )
        self._emit_log(summary)
        return PipelineResult(
            True, summary,
            n_suggestions=n_suggestions,
            n_auto_assignments=rec.n_assigned,
        )

    # ------------------------------------------------------------------
    # Train-only pipeline
    # ------------------------------------------------------------------

    def _run_train_only_pipeline(self) -> PipelineResult:
        """Retrain the model from every person-assigned face; touch nothing.

        Walks the already-recognized faces only: no scan, no detection, no
        assignment — the user's data stays exactly as it is.  Always retrains
        (the explicit action must not be skipped as "data unchanged").
        """
        self._emit_log("Stage 1/2: Generating missing face embeddings …")
        try:
            embedded = self._run_embedding()
        except ImportError as exc:
            log.error("TFLite backend missing: %s", exc)
            raise RuntimeError(
                "Hiányzik a TFLite futtatókörnyezet. "
                "Telepítsd/javítsd a függőségeket:\n"
                "  pip install ai-edge-litert\n"
                f"Részletek: {exc}"
            ) from exc
        self._checkpoint()

        self._emit_log(
            "Stage 2/2: Training the neural network from your categorized "
            "faces (this may take a while — accuracy over speed) …"
        )

        def train_cb(current, total, detail):
            self._emit_progress(current, total or 0, "Training", detail)

        with session_scope() as session:
            svc = DeepRecognitionService(
                session=session,
                config=self._config.deep_recognition,
                model_dir=self._config.resolve(
                    self._config.deep_recognition.model_dir
                ),
            )
            _, train, _ = svc.train(
                mode=MODE_TRAIN, progress_cb=train_cb, force=True
            )

        acc = (
            f"{train.validation_accuracy * 100:.1f}%"
            if train.validation_accuracy is not None
            else "n/a"
        )
        summary = (
            f"Done (train) — {embedded} face(s) newly embedded | "
            f"model trained on {train.n_examples} face(s) of "
            f"{train.n_persons} person(s) (+{train.n_augmented} synthetic), "
            f"accuracy {acc} | no faces were modified"
        )
        self._emit_log(summary)
        return PipelineResult(True, summary)

    # ------------------------------------------------------------------
    # AI face detection (analysis only)
    # ------------------------------------------------------------------

    def _run_detect_only_pipeline(self) -> PipelineResult:
        """Run only the AI face-detection analysis, over every image.

        Stores where the AI sees faces (bounding box + confidence) in the
        ``ai_face_detections`` table; no Face row is created or modified, no
        identity is assigned — the classic results stay untouched.
        """
        from app.services.ai_face_detection_service import (
            SOURCE_MANUAL,
            AiFaceDetectionService,
        )

        def cb(current, total, path):
            self._emit_progress(current, total or 0, "AI Detect", Path(path).name)

        with session_scope() as session:
            svc = AiFaceDetectionService(session, self._config.ai_face_detection)
            image_ids = svc.all_image_ids()
            self._emit_log(
                f"Stage 1/1: AI face detection on {len(image_ids)} image(s) …"
            )
            stats = svc.detect_images(
                image_ids,
                source=SOURCE_MANUAL,
                progress_cb=cb,
                cancel_check=self._is_cancel_requested,
            )

        if not stats.available:
            raise RuntimeError(f"AI face detection unavailable: {stats.error}")
        self._checkpoint()

        summary = (
            f"Done (AI face detection) — {stats.faces_found} face(s) found on "
            f"{stats.images_processed} image(s) "
            f"({stats.images_failed} unreadable) | detector: {stats.detector_name} | "
            f"analysis only — no face was assigned, moved or deleted"
        )
        self._emit_log(summary)
        return PipelineResult(True, summary)

    def _run_ai_face_detection(self, image_ids: list):
        """Best-effort AI face-detection stage inside rescan/rebuild.

        Never raises: a missing model or any other failure is logged and an
        empty/errored stats object is returned, so the rest of the pipeline
        (the existing recognition flow) continues unaffected.
        """
        from app.services.ai_face_detection_service import (
            SOURCE_PIPELINE,
            AiFaceDetectionService,
            AiFaceDetectionStats,
        )

        def cb(current, total, path):
            self._emit_progress(current, total or 0, "AI Detect", Path(path).name)

        try:
            with session_scope() as session:
                svc = AiFaceDetectionService(
                    session, self._config.ai_face_detection
                )
                stats = svc.detect_images(
                    image_ids,
                    source=SOURCE_PIPELINE,
                    progress_cb=cb,
                    cancel_check=self._is_cancel_requested,
                )
            if stats.available:
                self._emit_log(
                    f"  AI detected {stats.faces_found} face(s) on "
                    f"{stats.images_processed} image(s)."
                )
            else:
                self._emit_log(
                    f"  AI face detection skipped: {stats.error}"
                )
            return stats
        except Exception as exc:  # noqa: BLE001
            log.warning("AI face detection stage failed: %s", exc)
            return AiFaceDetectionStats(error=str(exc))

    # ------------------------------------------------------------------
    # Rebuild reset
    # ------------------------------------------------------------------

    def _reset_for_rebuild(self) -> tuple[int, int]:
        """Delete automatic faces, keep human decisions; reset all images.

        Kept (these are the training data the network learns from):
        * manually drawn boxes;
        * faces assigned to a real named person by a human decision
          (manual / merge / approved suggestion / confirmed AI grouping,
          including legacy rows with no recorded source).

        Everything else — unknown boxes, auto-clustered and auto-recognized
        faces — is deleted and will be re-created from scratch.
        """
        with session_scope() as session:
            named_ids = {
                pid
                for (pid,) in session.query(Person.id)
                .filter(Person.is_auto_named == False)  # noqa: E712
                .filter(Person.is_protected == False)  # noqa: E712
                .all()
            }

            kept = 0
            to_delete: list[int] = []
            for face in session.query(Face).all():
                if face.detector_backend == "manual":
                    kept += 1
                    continue
                if (
                    face.person_id in named_ids
                    and face.assignment_source in TRUSTED_MANUAL_SOURCES
                ):
                    kept += 1
                    continue
                to_delete.append(face.id)

            deleted = 0
            if to_delete:
                deleted = (
                    session.query(Face)
                    .filter(Face.id.in_(to_delete))
                    .delete(synchronize_session="fetch")
                )

            session.query(Image).update(
                {"detection_done": False, "embedding_done": False}
            )

            orphans = (
                session.query(Person)
                .filter(Person.is_auto_named == True)  # noqa: E712
                .filter(~Person.faces.any())
                .all()
            )
            for person in orphans:
                session.delete(person)

        log.info(
            "Rebuild reset: deleted %d automatic face(s), kept %d human-confirmed.",
            deleted, kept,
        )
        return deleted, kept

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _run_scan(self) -> list:
        def cb(current, total, path):
            detail = Path(path).name
            label = "Drive Scan" if self._drive_mode else "Scanning"
            self._emit_progress(current, total or 0, label, detail)
            if current % 50 == 0:
                self._emit_log(f"  Scanned {current}/{total or '?'} files …")

        if self._drive_mode:
            from app.gdrive.drive_scan_service import DriveScanService
            with session_scope() as session:
                svc = DriveScanService(
                    session=session,
                    client=self._drive_client,
                    root_folder_id=self._drive_root_folder_id,
                    local_mirror_dir=self._drive_mirror_dir,
                    config=self._config.scan,
                    progress_cb=cb,
                )
                return svc.scan()
        else:
            from app.services.image_library_service import get_image_library_optional
            with session_scope() as session:
                svc = ScanService(
                    session=session,
                    config=self._config.scan,
                    progress_cb=cb,
                    image_library_svc=get_image_library_optional(),
                )
                return svc.scan(self._root_folder)

    def _get_pending_detection_ids(self) -> list:
        with session_scope() as session:
            return [
                r[0]
                for r in session.query(Image.id)
                .filter(Image.detection_done == False)  # noqa: E712
                .all()
            ]

    def _run_detection(self, image_ids: list) -> int:
        if not image_ids:
            return 0

        detector = create_detector(self._config.detection)
        self._emit_log(f"  Using detector: {detector.backend_name}")

        def cb(current, total, path):
            self._emit_progress(current, total or 0, "Detecting", Path(path).name)

        with session_scope() as session:
            svc = DetectionService(
                session=session,
                detector=detector,
                config=self._config,
                progress_cb=cb,
                high_accuracy=self._config.deep_recognition.high_accuracy_detection,
            )
            return svc.process(image_ids)

    def _run_embedding(self) -> int:
        embedder = TFLiteEmbedder(
            model_path=self._config.embedding.model_path,
            embedding_dim=self._config.embedding.embedding_dim,
            input_size=self._config.embedding.input_size,
        )
        self._emit_log(
            f"  Embedder backend: {getattr(embedder, '_backend', '?')}"
        )

        def cb(current, total, face_id):
            self._emit_progress(current, total or 0, "Embedding", f"face #{face_id}")

        with session_scope() as session:
            svc = EmbeddingService(
                session=session,
                embedder=embedder,
                config=self._config,
                progress_cb=cb,
            )
            return svc.process_pending(
                exclude_low_quality=self._config.deep_recognition.strict_quality_filter
            )

    def _run_overlap_resolution(self) -> OverlapResolutionStats:
        try:
            with session_scope() as session:
                svc = OverlapResolutionService(
                    session=session,
                    config=self._config.overlap_resolution,
                )
                stats = svc.resolve()
                self._emit_progress(
                    1, 1, "Overlaps", f"{stats.faces_removed} duplicate(s) removed"
                )
                return stats
        except Exception as exc:  # noqa: BLE001
            log.warning("Overlap resolution failed: %s", exc)
            return OverlapResolutionStats()

    def _run_ignored_filter(self) -> None:
        from app.services.ignored_face_service import IgnoredFaceService
        try:
            with session_scope() as session:
                svc = IgnoredFaceService(
                    session=session,
                    config=getattr(self._config, "ignored_faces", None),
                )
                stats = svc.suppress_matching_unassigned()
                if stats.n_suppressed:
                    self._emit_log(
                        f"  Suppressed {stats.n_suppressed} permanently-ignored face(s)."
                    )
        except Exception as exc:  # noqa: BLE001
            log.warning("Ignored-face filter failed: %s", exc)

    def _run_deep_train_and_recognize(self) -> TrainAndRecognizeResult:
        def train_cb(current, total, detail):
            self._emit_progress(current, total or 0, "Training", detail)

        def recognize_cb(current, total, detail):
            self._emit_progress(current, total or 0, "Recognizing", detail)

        debug_cb = None
        if self._ai_visualization or self._ai_debug_log:
            def debug_cb(info):  # type: ignore[misc]
                if self._ai_debug_log:
                    self._write_debug_log(info)
                if self._ai_visualization:
                    self.face_debug.emit(info)

        with session_scope() as session:
            svc = DeepRecognitionService(
                session=session,
                config=self._config.deep_recognition,
                model_dir=self._config.resolve(
                    self._config.deep_recognition.model_dir
                ),
            )
            result = svc.train_and_recognize(
                mode=self._mode,
                train_progress_cb=train_cb,
                recognize_progress_cb=recognize_cb,
                debug_cb=debug_cb,
            )
        train = result.train
        acc = (
            f"{train.validation_accuracy * 100:.1f}%"
            if train.validation_accuracy is not None
            else "n/a"
        )
        if train.reused_existing_model:
            self._emit_log(
                "  Labeled data unchanged — reusing the previously trained model."
            )
        else:
            self._emit_log(
                f"  Model trained on {train.n_examples} face(s) of "
                f"{train.n_persons} person(s) "
                f"(+{train.n_augmented} synthetic), accuracy {acc}."
            )
        self._emit_log(
            f"  AI placed {result.recognition.n_assigned} unknown face(s) "
            f"with known people."
        )
        return result

    def _run_clustering(self) -> ClusteringStats:
        try:
            with session_scope() as session:
                svc = ClusteringService(
                    session=session,
                    config=self._config.clustering,
                    exclude_low_quality=self._config.deep_recognition.strict_quality_filter,
                )
                stats = svc.cluster_unassigned()
                self._emit_progress(
                    1, 1, "Clustering", f"{stats.n_new_persons} new unknown group(s)"
                )
                return stats
        except Exception as exc:  # noqa: BLE001
            log.warning("Unknown clustering stage failed: %s", exc)
            return ClusteringStats()

    def _run_intra_image_consistency(self) -> IntraImageConsistencyStats:
        try:
            with session_scope() as session:
                svc = IntraImageConsistencyService(
                    session=session,
                    config=self._config.intra_image,
                    exclude_low_quality=self._config.deep_recognition.strict_quality_filter,
                )
                return svc.run()
        except Exception as exc:  # noqa: BLE001
            log.warning("Intra-image consistency stage failed: %s", exc)
            return IntraImageConsistencyStats()

    def _run_suggestions(self) -> int:
        try:
            with session_scope() as session:
                svc = SuggestionService(
                    session=session,
                    config=self._config.suggestions,
                    exclude_low_quality=self._config.deep_recognition.strict_quality_filter,
                )
                return svc.count_suggestions()
        except Exception as exc:  # noqa: BLE001
            log.warning("Suggestion stage failed: %s", exc)
            return 0
