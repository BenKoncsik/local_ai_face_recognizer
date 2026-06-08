"""Background worker for the "embed persons into image files" export.

Writing the recognised persons into every image's metadata touches the
filesystem once per image and must not run on the UI thread — otherwise the
window freezes and the user cannot react (including pressing *Cancel*).  This
:class:`QThread` owns its own database session (cross-thread session sharing is
unsafe with SQLite/SQLAlchemy), reports progress back to the UI via Qt signals,
and stops cooperatively when its :class:`CancellationToken` is cancelled.
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QThread, Signal

from app.db.database import session_scope
from app.jobs.cancellation import CancellationToken
from app.services.face_metadata_export_service import (
    FaceMetadataExportOptions,
    FaceMetadataExportService,
)

log = logging.getLogger(__name__)


class FaceMetadataExportWorker(QThread):
    """Runs :meth:`FaceMetadataExportService.export_all` off the UI thread.

    Signals:
        progress: ``(done: int, total: int, current_name: str)`` — ``done``
                  images of ``total`` finished; ``current_name`` is the file
                  currently being processed (may be empty).
        finished_ok: ``(summary: object)`` — the :class:`FaceMetadataExportSummary`.
        failed:   ``(message: str)`` — an unexpected crash (cancellation is
                  *not* an error and arrives via ``finished_ok``).
    """

    progress = Signal(int, int, str)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        options: FaceMetadataExportOptions,
        cancel_token: CancellationToken,
    ) -> None:
        super().__init__()
        self._options = options
        self._cancel_token = cancel_token

    def cancel(self) -> None:
        """Request a cooperative stop; safe to call from the UI thread."""
        self._cancel_token.cancel()

    def run(self) -> None:  # noqa: D401 — QThread entry point
        def cb(done: int, total: int, name: Optional[str]) -> None:
            self.progress.emit(int(done), int(total), str(name or ""))

        try:
            with session_scope() as session:
                summary = FaceMetadataExportService(session).export_all(
                    self._options,
                    progress_cb=cb,
                    cancel_token=self._cancel_token,
                )
            self.finished_ok.emit(summary)
        except Exception as exc:  # noqa: BLE001
            log.exception("Face metadata export crashed")
            self.failed.emit(str(exc))
