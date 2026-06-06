"""Background workers for lazy Drive image loading in the Image Browser.

Two worker classes:

* :class:`DriveThumbRunnable` — download a Drive file to the local mirror
  (if it isn't there yet) and then generate a small thumbnail QImage.
  Used to populate tree-item icons for images whose mirror file is absent.

* :class:`DriveFetchRunnable` — download a Drive file to the local mirror
  and emit ``ready`` when done.  Used for:
  - full-image display when the user clicks an item
  - silent prefetch of adjacent images before the user navigates to them

Both workers resolve the Drive ``file_id`` automatically by looking up the
:class:`~app.db.models.RemoteImage` record for the given ``image_id``.  All
network I/O happens off the GUI thread.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Signal
from PySide6.QtGui import QImage

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thumbnail worker
# ---------------------------------------------------------------------------

class _DriveThumbSignals(QObject):
    ready  = Signal(str, QImage)  # cache_key, thumbnail QImage
    failed = Signal(str)          # cache_key


class DriveThumbRunnable(QRunnable):
    """Download a Drive file (if missing) then emit a small thumbnail QImage.

    Args:
        drive_client: Authenticated :class:`~app.gdrive.protocols.GDriveClient`.
        image_id:     :class:`~app.db.models.Image` primary key.
        local_path:   Expected local mirror path (may or may not exist yet).
        cache_key:    Passed through to ``signals.ready`` for tree-item lookup.
        size:         Thumbnail max-side in pixels (default 56 for the tree).
    """

    def __init__(
        self,
        drive_client,
        image_id: int,
        local_path: str,
        cache_key: str,
        size: int = 56,
    ) -> None:
        super().__init__()
        self.signals = _DriveThumbSignals()
        self._client = drive_client
        self._image_id = image_id
        self._local = local_path
        self._key = cache_key
        self._size = size
        self.setAutoDelete(True)

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:
            log.warning("DriveThumb failed for image %d: %s", self._image_id, exc)
            self.signals.failed.emit(self._key)

    def _run(self) -> None:
        local = Path(self._local)
        if not local.exists():
            file_id = _resolve_drive_file_id(self._image_id)
            if file_id is None:
                log.warning("No RemoteImage record for image_id=%d", self._image_id)
                self.signals.failed.emit(self._key)
                return
            _download(self._client, file_id, local)

        # Generate thumbnail from the local mirror file.
        import cv2
        from app.utils.image_utils import load_image_bgr_normalized

        img_bgr = load_image_bgr_normalized(str(local))
        if img_bgr is None:
            self.signals.failed.emit(self._key)
            return

        h, w = img_bgr.shape[:2]
        scale = self._size / max(w, h)
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        qimg = QImage(bytes(rgb.tobytes()), nw, nh, 3 * nw, QImage.Format_RGB888)
        qimg = qimg.copy()
        self.signals.ready.emit(self._key, qimg)


# ---------------------------------------------------------------------------
# Full-image fetch worker
# ---------------------------------------------------------------------------

class _DriveFetchSignals(QObject):
    ready  = Signal(int, str)  # image_id, local_path
    failed = Signal(int, str)  # image_id, error_message


class DriveFetchRunnable(QRunnable):
    """Download a Drive file to *local_path* if it isn't already there.

    Emits ``ready(image_id, local_path)`` on success so the panel can
    display the image (or silently complete a prefetch).

    Args:
        drive_client: Authenticated Drive client.
        image_id:     :class:`~app.db.models.Image` primary key.
        local_path:   Target local mirror path.
        silent:       When ``True`` suppress UI notification on completion
                      (used for prefetch).
    """

    def __init__(
        self,
        drive_client,
        image_id: int,
        local_path: str,
        silent: bool = False,
    ) -> None:
        super().__init__()
        self.signals = _DriveFetchSignals()
        self._client = drive_client
        self._image_id = image_id
        self._local = local_path
        self._silent = silent
        self.setAutoDelete(True)

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:
            msg = str(exc)
            log.warning("DriveFetch failed for image %d: %s", self._image_id, msg)
            if not self._silent:
                self.signals.failed.emit(self._image_id, msg)

    def _run(self) -> None:
        local = Path(self._local)
        if local.exists():
            # Already in mirror — nothing to do, just notify.
            self.signals.ready.emit(self._image_id, str(local))
            return

        file_id = _resolve_drive_file_id(self._image_id)
        if file_id is None:
            raise RuntimeError(
                f"No RemoteImage record for image_id={self._image_id}"
            )
        _download(self._client, file_id, local)
        self.signals.ready.emit(self._image_id, str(local))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _resolve_drive_file_id(image_id: int) -> "str | None":
    """Look up the Drive file_id for *image_id* via RemoteImage."""
    from app.db.database import session_scope
    from app.db.models import RemoteImage
    try:
        with session_scope() as session:
            rec = (
                session.query(RemoteImage)
                .filter(RemoteImage.image_id == image_id)
                .first()
            )
            return rec.drive_file_id if rec is not None else None
    except Exception as exc:  # noqa: BLE001
        log.warning("_resolve_drive_file_id(%d) failed: %s", image_id, exc)
        return None


def _download(client, file_id: str, local: Path) -> None:
    """Download *file_id* from Drive to *local*, creating parent dirs."""
    local.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(file_id, str(local))
    log.debug("Drive downloaded: %s → %s", file_id, local.name)
