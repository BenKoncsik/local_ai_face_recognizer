"""Background thumbnail generator using QRunnable + QThreadPool.

Usage
-----
    worker = ThumbnailRunnable(file_path, cache_key, size=160)
    worker.signals.ready.connect(my_slot)   # (cache_key, QImage)
    worker.signals.failed.connect(my_err)   # (cache_key,)
    QThreadPool.globalInstance().start(worker)

The worker emits a *QImage* (not QPixmap) because QImage is thread-safe
while QPixmap must only be created/used on the GUI thread.  Convert with
QPixmap.fromImage() in the receiving slot.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, QRunnable, Signal
from PySide6.QtGui import QImage

log = logging.getLogger(__name__)


class _ThumbnailSignals(QObject):
    """Signal carrier — must be a QObject, created before the runnable starts."""

    ready = Signal(str, QImage)   # cache_key, image
    failed = Signal(str)          # cache_key


class ThumbnailRunnable(QRunnable):
    """Resize an image to *size*×*size* (keeping aspect ratio) in a thread-pool worker."""

    def __init__(self, file_path: str, cache_key: str, size: int = 160) -> None:
        super().__init__()
        self.signals = _ThumbnailSignals()
        self._path = file_path
        self._key = cache_key
        self._size = size
        self.setAutoDelete(True)

    # ------------------------------------------------------------------

    def run(self) -> None:
        try:
            self._generate()
        except Exception as exc:
            log.warning("Thumbnail generation failed for %s: %s", self._path, exc)
            self.signals.failed.emit(self._key)

    def _generate(self) -> None:
        import cv2

        from app.utils.image_utils import load_image_bgr_normalized

        img_bgr = load_image_bgr_normalized(self._path)
        if img_bgr is None:
            log.warning("Thumbnail: cannot load %s", self._path)
            self.signals.failed.emit(self._key)
            return

        h, w = img_bgr.shape[:2]
        if w == 0 or h == 0:
            self.signals.failed.emit(self._key)
            return

        scale = self._size / max(w, h)
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))

        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # QImage is thread-safe; QPixmap conversion happens in the main thread.
        qimg = QImage(bytes(rgb.tobytes()), nw, nh, 3 * nw, QImage.Format_RGB888)
        qimg = qimg.copy()  # detach from the numpy buffer
        self.signals.ready.emit(self._key, qimg)
