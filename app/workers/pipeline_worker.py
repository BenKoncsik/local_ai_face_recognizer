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

from PySide6.QtCore import QSettings, QThread, Signal

from app.config import AppConfig
from app.db.database import init_db, session_scope
from app.detectors.factory import create_detector
from app.embeddings.tflite_embedder import TFLiteEmbedder
from app.services.detection_service import DetectionService
from app.services.embedding_service import EmbeddingService
from app.services.recognition_service import RecognitionService
from app.services.scan_service import ScanService
from app.services.suggestion_service import SuggestionService

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
    ) -> None:
        super().__init__(parent)
        self._root_folder = root_folder
        self._config = config
        self._abort = False
        self._high_accuracy = high_accuracy

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
        _qs = QSettings("FaceLocal", "FaceLocal")
        exclude_low_quality: bool = _qs.value(
            "face_quality/exclude_low_quality", True, type=bool
        )

        # Each stage gets its own session to avoid long-held transactions.
        init_db(self._config.db_path_resolved)

        # --- Stage 1: Scan ---
        self.log_message.emit("Stage 1/4: Scanning image folder …")
        new_ids = self._run_scan()
        if self._abort:
            self.finished.emit(False, "Aborted after scan")
            return

        # --- Stage 2: Detection ---
        all_pending = self._get_pending_detection_ids()
        mode_label = "high-accuracy" if self._high_accuracy else "fast"
        self.log_message.emit(
            f"Stage 2/5: Detecting faces in {len(all_pending)} image(s) [{mode_label} mode] …"
        )
        total_faces = self._run_detection(all_pending)
        if self._abort:
            self.finished.emit(False, "Aborted after detection")
            return

        # --- Stage 3: Embedding ---
        self.log_message.emit("Stage 3/5: Generating face embeddings …")
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

        # --- Stage 4: Recognition ---
        self.log_message.emit(
            "Stage 4/5: Recognizing faces from learned people …"
        )
        n_assigned = self._run_recognition(exclude_low_quality)

        # --- Stage 5: Name suggestions ---
        self.log_message.emit(
            "Stage 5/5: Matching unknown faces against named people …"
        )
        n_suggestions = self._run_suggestions(exclude_low_quality)
        self.suggestions_ready.emit(n_suggestions)

        summary = (
            f"Done — {len(new_ids)} new image(s), "
            f"{total_faces} face(s) detected, "
            f"{embedded} embedded, "
            f"{n_assigned} face(s) auto-assigned, "
            f"{n_suggestions} name suggestion(s)"
        )
        self.log_message.emit(summary)
        self.finished.emit(True, summary)

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _run_scan(self) -> list:
        def cb(current, total, path):
            detail = Path(path).name
            self.progress.emit(current, total or 0, "Scanning", detail)
            if current % 50 == 0:
                self.log_message.emit(f"  Scanned {current}/{total or '?'} files …")

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

    def _run_recognition(self, exclude_low_quality: bool = False) -> int:
        with session_scope() as session:
            svc = RecognitionService(
                session=session,
                config=self._config.recognition,
                exclude_low_quality=exclude_low_quality,
            )
            assignments = svc.recognize_pending()
            n = len(assignments)
            self.progress.emit(1, 1, "Recognition", f"{n} face(s)")
            return n

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
