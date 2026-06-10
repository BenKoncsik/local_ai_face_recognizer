"""Background pipeline worker.

Runs the full scan → detect → embed → recognize → suggest pipeline in a
QThread so the GUI remains responsive.  Progress is communicated via Qt
signals.

Usage::

    worker = PipelineWorker(root_folder="/home/user/photos", config=cfg)
    worker.progress.connect(on_progress)
    worker.log_message.connect(on_log)
    worker.finished.connect(on_finished)
    worker.start()
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from app.config import AppConfig
from app.db.database import init_db, session_scope
from app.detectors.factory import create_detector
from app.embeddings.tflite_embedder import TFLiteEmbedder
from app.services.clustering_service import ClusteringService, ClusteringStats
from app.services.detection_service import DetectionService
from app.services.embedding_service import EmbeddingService
from app.services.intra_image_consistency_service import (
    IntraImageConsistencyService,
    IntraImageConsistencyStats,
)
from app.services.intra_image_duplicate_service import (
    IntraImageDuplicateService,
    IntraImageDuplicateStats,
)
from app.services.recognition_service import RecognitionService, RecognitionStats
from app.services.scan_service import ScanService
from app.services.suggestion_service import SuggestionService
from typing import Optional

log = logging.getLogger(__name__)


class PipelineWorker(QThread):
    """QThread that runs the complete processing pipeline.

    Signals:
        progress:          ``(current: int, total: int, stage: str, detail: str)``
        log_message:       ``(message: str)``
        suggestions_ready: ``(count: int)`` — number of name suggestions found
        finished:          ``(success: bool, summary: str)``
        error:             ``(message: str)``
    """

    progress = Signal(int, int, str, str)
    log_message = Signal(str)
    suggestions_ready = Signal(int)
    finished = Signal(bool, str)
    error = Signal(str)

    def __init__(
        self,
        root_folder: str,
        config: AppConfig,
        parent=None,
        high_accuracy: bool = False,
        db_path_override: Optional[str] = None,
        drive_client=None,          # GDriveClient when Drive mode is active
        drive_root_folder_id: Optional[str] = None,
        drive_mirror_dir: Optional[Path] = None,
    ) -> None:
        super().__init__(parent)
        self._root_folder = root_folder
        self._config = config
        self._abort = False
        self._high_accuracy = high_accuracy
        # Drive mode — all three must be set together or none.
        self._drive_client = drive_client
        self._drive_root_folder_id = drive_root_folder_id
        self._drive_mirror_dir = drive_mirror_dir
        # Override DB path (needed so Drive mode uses the downloaded DB, not
        # the default config path which may point at the wrong file).
        self._db_path_override = db_path_override

    @property
    def _drive_mode(self) -> bool:
        return self._drive_client is not None

    def abort(self) -> None:
        """Request a graceful stop (checked between pipeline stages)."""
        self._abort = True
        log.info("Pipeline abort requested")

    def run(self) -> None:
        """Execute the pipeline.  Called by QThread.start()."""
        try:
            self._run_pipeline()
        except Exception as exc:  # noqa: BLE001
            msg = f"Pipeline error: {exc}\n{traceback.format_exc()}"
            log.error(msg)
            self.error.emit(str(exc))
            self.finished.emit(False, str(exc))

    # ------------------------------------------------------------------

    def _run_pipeline(self) -> None:
        # Read user preference for face quality filtering.
        from app.app_settings import app_qsettings
        _qs = app_qsettings()
        exclude_low_quality: bool = _qs.value(
            "face_quality/exclude_low_quality", True, type=bool
        )

        # Initialise the DB — use the override path when Drive mode is active
        # so we operate on the downloaded Drive DB rather than the config default.
        db_path = self._db_path_override or str(self._config.db_path_resolved)
        init_db(db_path)

        # --- Stage 1: Scan ---
        if self._drive_mode:
            self.log_message.emit("Stage 1/7: Scanning Google Drive project folder …")
        else:
            self.log_message.emit("Stage 1/7: Scanning image folder …")
        new_ids = self._run_scan()
        if self._abort:
            self.finished.emit(False, "Aborted after scan")
            return

        # --- Stage 2: Detection ---
        all_pending = self._get_pending_detection_ids()
        mode_label = "high-accuracy" if self._high_accuracy else "fast"
        self.log_message.emit(
            f"Stage 2/7: Detecting faces in {len(all_pending)} image(s) [{mode_label} mode] …"
        )
        total_faces = self._run_detection(all_pending)
        if self._abort:
            self.finished.emit(False, "Aborted after detection")
            return

        # --- Stage 3: Embedding ---
        self.log_message.emit("Stage 3/7: Generating face embeddings …")
        try:
            embedded = self._run_embedding(exclude_low_quality)
        except ImportError as exc:
            log.error("TFLite backend missing: %s", exc)
            user_msg = (
                "Hiányzik a TFLite futtatókörnyezet. "
                "Telepítsd/javítsd a Windows AI runtime függőségeket:\n"
                "  pip install ai-edge-litert\n"
                f"Részletek: {exc}"
            )
            self.error.emit(user_msg)
            self.finished.emit(False, "Missing TFLite runtime — embedding skipped")
            return
        if self._abort:
            self.finished.emit(False, "Aborted after embedding")
            return

        # --- Stage 3b: Same-image duplicate-detection cleanup ---
        # Drop freshly detected boxes that are really the same physical face as
        # one already marked on the same photo, before they get clustered into
        # spurious Unknown identities.
        dedup_stats = self._run_intra_image_dedup(exclude_low_quality)
        if dedup_stats.faces_removed:
            self.log_message.emit(
                f"  Removed {dedup_stats.faces_removed} duplicate detection(s) "
                f"of already-known faces."
            )

        # --- Stage 4: Recognition ---
        self.log_message.emit(
            "Stage 4/7: Recognizing faces from learned people …"
        )
        rec_stats = self._run_recognition(exclude_low_quality)

        # --- Stage 4b: Permanently-ignored face filter ---
        # After recognition so known people keep their faces; before unknown
        # clustering so suppressed faces never spawn new "Unknown N" persons.
        ignored_stats = self._run_ignored_filter()
        if ignored_stats.n_suppressed:
            self.log_message.emit(
                f"  Suppressed {ignored_stats.n_suppressed} face(s) matching "
                f"the permanent ignore list."
            )

        # --- Stage 5: Unknown-face clustering ---
        self.log_message.emit(
            "Stage 5/7: Clustering unassigned faces into Unknown persons …"
        )
        cluster_stats = self._run_clustering(exclude_low_quality)

        # --- Stage 6: Same-image identity consistency ---
        self.log_message.emit(
            "Stage 6/7: Unifying same-person faces within each image …"
        )
        consistency_stats = self._run_intra_image_consistency(exclude_low_quality)

        # --- Stage 7: Name suggestions ---
        self.log_message.emit(
            "Stage 7/7: Matching unknown faces against named people …"
        )
        n_suggestions = self._run_suggestions(exclude_low_quality)
        self.suggestions_ready.emit(n_suggestions)

        summary = (
            f"Done — {len(new_ids)} new image(s), "
            f"{total_faces} detected, "
            f"{embedded} embedded | "
            f"known: {rec_stats.n_assigned}"
            f" (of {rec_stats.n_candidates} candidates, "
            f"{rec_stats.n_profiles} profiles, "
            f"{rec_stats.n_below_threshold} below-thr, "
            f"{rec_stats.n_margin_rejected} margin-rej) | "
            f"unknown: +{cluster_stats.n_new_persons} new persons "
            f"({cluster_stats.n_assigned_to_new} faces), "
            f"+{cluster_stats.n_assigned_to_existing} → existing, "
            f"{cluster_stats.n_singletons} unassigned | "
            f"intra-image: {consistency_stats.n_faces_reassigned} face(s) reunified, "
            f"{consistency_stats.n_persons_removed} fragment(s) removed | "
            f"{n_suggestions} suggestion(s)"
        )
        self.log_message.emit(summary)
        self.finished.emit(True, summary)

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _run_scan(self) -> list:
        def cb(current, total, path):
            detail = Path(path).name
            label = "Drive Scan" if self._drive_mode else "Scanning"
            self.progress.emit(current, total or 0, label, detail)
            if current % 50 == 0:
                self.log_message.emit(f"  Scanned {current}/{total or '?'} files …")

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
        from app.db.database import get_session
        from app.db.models import Image

        session = get_session()
        try:
            ids = [
                r[0]
                for r in session.query(Image.id)
                .filter(Image.detection_done == False)  # noqa: E712
                .all()
            ]
            return ids
        finally:
            session.close()

    def _run_detection(self, image_ids: list) -> int:
        if not image_ids:
            return 0

        detector = create_detector(self._config.detection)
        self.log_message.emit(f"  Using detector: {detector.backend_name}")

        def cb(current, total, path):
            detail = Path(path).name
            self.progress.emit(current, total or 0, "Detecting", detail)

        with session_scope() as session:
            svc = DetectionService(
                session=session,
                detector=detector,
                config=self._config,
                progress_cb=cb,
                high_accuracy=self._high_accuracy,
            )
            return svc.process(image_ids)

    def _run_embedding(self, exclude_low_quality: bool = False) -> int:
        embedder = TFLiteEmbedder(
            model_path=self._config.embedding.model_path,
            embedding_dim=self._config.embedding.embedding_dim,
            input_size=self._config.embedding.input_size,
        )
        self.log_message.emit(f"  Embedder backend: {getattr(embedder, '_backend', '?')}")

        counter = [0]

        def cb(current, total, face_id):
            counter[0] = current
            self.progress.emit(current, total or 0, "Embedding", f"face #{face_id}")

        with session_scope() as session:
            svc = EmbeddingService(
                session=session,
                embedder=embedder,
                config=self._config,
                progress_cb=cb,
            )
            return svc.process_pending(exclude_low_quality=exclude_low_quality)

    def _run_intra_image_dedup(
        self, exclude_low_quality: bool = False
    ) -> IntraImageDuplicateStats:
        """Remove same-image duplicate detections of already-known faces."""
        try:
            with session_scope() as session:
                svc = IntraImageDuplicateService(
                    session=session,
                    config=self._config.intra_image_duplicate,
                    exclude_low_quality=exclude_low_quality,
                )
                stats = svc.remove_duplicates()
                self.progress.emit(
                    1, 1, "Deduplicating",
                    f"{stats.faces_removed} duplicate(s) removed",
                )
                return stats
        except Exception as exc:  # noqa: BLE001
            log.warning("Same-image duplicate cleanup failed: %s", exc)
            return IntraImageDuplicateStats()

    def _run_recognition(self, exclude_low_quality: bool = False) -> RecognitionStats:
        with session_scope() as session:
            svc = RecognitionService(
                session=session,
                config=self._config.recognition,
                exclude_low_quality=exclude_low_quality,
            )
            assignments, stats = svc.recognize_pending()
            self.progress.emit(1, 1, "Recognition", f"{stats.n_assigned} face(s) assigned")
            return stats

    def _run_ignored_filter(self) -> "IgnoredFilterStats":
        """Suppress unassigned faces that match the permanent ignore list."""
        from app.services.ignored_face_service import (
            IgnoredFaceService,
            IgnoredFilterStats,
        )
        try:
            with session_scope() as session:
                svc = IgnoredFaceService(
                    session=session,
                    config=getattr(self._config, "ignored_faces", None),
                )
                stats = svc.suppress_matching_unassigned()
                self.progress.emit(
                    1, 1, "Ignore filter",
                    f"{stats.n_suppressed} face(s) suppressed",
                )
                return stats
        except Exception as exc:  # noqa: BLE001
            log.warning("Ignored-face filter stage failed: %s", exc)
            return IgnoredFilterStats()

    def _run_clustering(self, exclude_low_quality: bool = False) -> ClusteringStats:
        """Cluster unassigned embedded faces into Unknown N persons."""
        try:
            with session_scope() as session:
                svc = ClusteringService(
                    session=session,
                    config=self._config.clustering,
                    exclude_low_quality=exclude_low_quality,
                )
                stats = svc.cluster_unassigned()
                self.progress.emit(
                    1, 1, "Clustering",
                    f"{stats.n_new_persons} new Unknown persons",
                )
                return stats
        except Exception as exc:  # noqa: BLE001
            log.warning("Unknown clustering stage failed: %s", exc)
            return ClusteringStats()

    def _run_intra_image_consistency(
        self, exclude_low_quality: bool = False
    ) -> IntraImageConsistencyStats:
        """Reunify same-person faces split across identities within one image."""
        try:
            with session_scope() as session:
                svc = IntraImageConsistencyService(
                    session=session,
                    config=self._config.intra_image,
                    exclude_low_quality=exclude_low_quality,
                )
                stats = svc.run()
                self.progress.emit(
                    1, 1, "Consistency",
                    f"{stats.n_faces_reassigned} face(s) reunified",
                )
                return stats
        except Exception as exc:  # noqa: BLE001
            log.warning("Intra-image consistency stage failed: %s", exc)
            return IntraImageConsistencyStats()

    def _run_suggestions(self, exclude_low_quality: bool = False) -> int:
        """Compute name suggestions; never aborts the pipeline on failure."""
        try:
            with session_scope() as session:
                svc = SuggestionService(
                    session=session,
                    config=self._config.suggestions,
                    exclude_low_quality=exclude_low_quality,
                )
                n = svc.count_suggestions()
                self.progress.emit(1, 1, "Suggestions", f"{n} match(es)")
                return n
        except Exception as exc:  # noqa: BLE001
            log.warning("Suggestion stage failed: %s", exc)
            return 0
