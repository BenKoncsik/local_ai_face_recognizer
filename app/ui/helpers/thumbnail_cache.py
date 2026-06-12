"""Shared, content-aware thumbnail cache for face crops.

Face crops are regenerated in place (the same path can hold new pixels after a
re-crop), so a plain path-keyed pixmap cache would serve stale images.  This
cache keys on ``path + mtime + size`` so a changed file misses the cache and is
reloaded, while unchanged crops are served from memory across panels (grid,
timeline, hover previews).

The cache is a process-wide LRU bounded by entry count.  Pixmaps are cheap to
hold (already downscaled to thumbnail size), so a few thousand entries stay
small.  Loading is failure-tolerant: a missing/corrupt file yields ``None``.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QImageReader, QPixmap

log = logging.getLogger(__name__)

_MAX_ENTRIES = 4096
_cache: "OrderedDict[str, QPixmap]" = OrderedDict()


def _key(path: str, mtime: float, size: int) -> str:
    return f"{path}|{mtime:.3f}|{size}"


def get_thumbnail(
    crop_path: Optional[str],
    size: int,
    face_id: Optional[int] = None,
) -> Optional[QPixmap]:
    """Return a cached/loaded ``size``-square thumbnail for *crop_path*.

    Returns ``None`` when the path is empty, missing, or unreadable.  The result
    is cached keyed by the file's modification time so a regenerated crop is
    picked up automatically.
    """
    if not crop_path:
        return None
    try:
        mtime = os.path.getmtime(crop_path)
    except OSError:
        return None

    key = _key(crop_path, mtime, size)
    hit = _cache.get(key)
    if hit is not None:
        _cache.move_to_end(key)         # LRU touch
        return hit

    pixmap = _load_scaled(crop_path, size, face_id)
    if pixmap is None:
        return None
    _cache[key] = pixmap
    _cache.move_to_end(key)
    # Drop any older entries for the same path+size (superseded by new mtime).
    stale_prefix = f"{crop_path}|"
    suffix = f"|{size}"
    for k in [k for k in _cache if k != key and k.startswith(stale_prefix) and k.endswith(suffix)]:
        del _cache[k]
    while len(_cache) > _MAX_ENTRIES:
        _cache.popitem(last=False)      # evict least-recently-used
    return pixmap


def _load_scaled(crop_path: str, size: int, face_id: Optional[int]) -> Optional[QPixmap]:
    if not Path(crop_path).exists():
        return None
    reader = QImageReader(crop_path)
    reader.setAutoTransform(True)
    image = reader.read()
    if image.isNull():
        log.warning(
            "Crop thumbnail load failed: FaceId=%s preview_source=%s error=%s",
            face_id, crop_path, reader.errorString(),
        )
        return None
    return QPixmap.fromImage(image).scaled(
        size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation
    )


def clear_cache() -> None:
    """Drop all cached thumbnails (e.g. after a bulk re-crop)."""
    _cache.clear()


def cache_size() -> int:
    """Number of thumbnails currently held (for tests/diagnostics)."""
    return len(_cache)
