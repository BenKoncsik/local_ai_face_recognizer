"""Background worker for the Astro static-site export.

The export resizes every image into three variants and then runs ``npm run
build``, which is far too slow to run on the UI thread. This :class:`QThread`
owns its own database session (cross-thread session sharing is unsafe with
SQLite/SQLAlchemy) and reports progress back to the UI via Qt signals.
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QThread, Signal

from app.db.database import session_scope
from app.services.export_service import ExportService

log = logging.getLogger(__name__)


class AstroExportWorker(QThread):
    """Runs :meth:`ExportService.export_astro` off the UI thread.

    Signals:
        progress: ``(percent: int, message: str)`` — ``percent`` is 0–100, or
                  ``-1`` for an indeterminate phase (the opaque npm build).
        finished_ok: ``(out_path: str)`` — the finished site directory.
        failed:   ``(error_kind: str, message: str)`` — ``error_kind`` is
                  ``"node"`` when Node/npm is missing, else ``"error"``.
    """

    progress = Signal(int, str)
    finished_ok = Signal(str)
    failed = Signal(str, str)

    def __init__(self, target_dir: str, person_id: Optional[int] = None) -> None:
        super().__init__()
        self._target_dir = target_dir
        self._person_id = person_id

    def run(self) -> None:  # noqa: D401 — QThread entry point
        try:
            with session_scope() as session:
                out = ExportService(session).export_astro(
                    target_dir=self._target_dir,
                    person_id=self._person_id,
                    progress_callback=lambda pct, msg: self.progress.emit(int(pct), str(msg)),
                )
            self.finished_ok.emit(str(out))
        except RuntimeError as exc:
            # _run_astro_build raises RuntimeError when Node/npm is missing.
            kind = "node" if "npm" in str(exc).lower() or "node" in str(exc).lower() else "error"
            log.warning("Astro export failed: %s", exc)
            self.failed.emit(kind, str(exc))
        except Exception as exc:  # noqa: BLE001
            log.exception("Astro export crashed")
            self.failed.emit("error", str(exc))
