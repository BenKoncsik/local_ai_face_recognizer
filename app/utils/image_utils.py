"""Image processing utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image as PilImage

from app.detectors.base import Detection

log = logging.getLogger(__name__)


def save_face_crop(
    img_bgr: np.ndarray,
    detection: Detection,
    crops_dir: Path,
    image_id: int,
    thumbnail_size: Tuple[int, int] = (128, 128),
    face_index: int = 0,
) -> Optional[Path]:
    """Extract a face crop and save it as a JPEG thumbnail.

    Args:
        img_bgr:        Full image in BGR format.
        detection:      Bounding box detection result.
        crops_dir:      Directory to write the crop file.
        image_id:       Parent image DB ID (used in filename).
        thumbnail_size: Target size ``(width, height)`` for the saved crop.
        face_index:     Index of this face within the image (for naming).

    Returns:
        :class:`Path` to the saved crop file, or ``None`` on failure.
    """
    x, y, w, h = detection.as_tuple()

    if w <= 0 or h <= 0:
        log.debug("Skipping zero-area crop for image_id=%d", image_id)
        return None

    crop = img_bgr[y : y + h, x : x + w]
    if crop.size == 0:
        return None

    # Add a small margin (10% each side) to include face context
    img_h, img_w = img_bgr.shape[:2]
    margin_x = int(w * 0.10)
    margin_y = int(h * 0.10)
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(img_w, x + w + margin_x)
    y2 = min(img_h, y + h + margin_y)
    crop = img_bgr[y1:y2, x1:x2]

    # Resize to thumbnail
    thumb = cv2.resize(crop, thumbnail_size, interpolation=cv2.INTER_AREA)

    filename = f"img{image_id:06d}_face{face_index:03d}.jpg"
    dest = crops_dir / filename
    success = save_image_bgr(dest, thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])

    if not success:
        log.warning("Failed to write crop: %s", dest)
        return None

    return dest


def load_image_bgr(path: str) -> Optional[np.ndarray]:
    """Load an image file to a BGR numpy array, with Pillow fallback.

    Decoding is done via ``np.fromfile`` + ``cv2.imdecode`` rather than
    ``cv2.imread``: on Windows ``cv2.imread`` cannot open paths containing
    non-ASCII characters (e.g. accented Hungarian folder names) and silently
    returns ``None``.  For WEBP and other edge cases we fall back to Pillow.

    Args:
        path: Absolute or relative path to the image file.

    Returns:
        BGR uint8 numpy array, or ``None`` if loading fails.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size > 0:
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                return img
    except Exception as exc:  # noqa: BLE001
        log.debug("cv2.imdecode failed for %s: %s", path, exc)

    # Pillow fallback (handles WEBP, some TIFF variants, etc.)
    try:
        pil_img = PilImage.open(path).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not load image %s: %s", path, exc)
        return None


def save_image_bgr(path, img: np.ndarray, params: Optional[list] = None) -> bool:
    """Write a BGR image to *path*, supporting non-ASCII paths on Windows.

    Mirrors ``cv2.imwrite`` but encodes in memory and writes the bytes with
    ``ndarray.tofile`` so Unicode paths work everywhere.

    Args:
        path:   Destination file path; the extension selects the format.
        img:    BGR uint8 numpy array.
        params: Optional OpenCV encode params (e.g. JPEG quality).

    Returns:
        ``True`` on success, ``False`` otherwise.
    """
    path = Path(path)
    ext = path.suffix or ".jpg"
    try:
        ok, buf = cv2.imencode(ext, img, params or [])
        if not ok:
            log.warning("Failed to encode image for %s", path)
            return False
        buf.tofile(str(path))
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not write image %s: %s", path, exc)
        return False


def qt_pixmap_from_path(path: str, max_size: Tuple[int, int] = (400, 400)):
    """Load an image and return a scaled QPixmap.

    Keeps aspect ratio; fits within *max_size*.

    Args:
        path:     Image file path.
        max_size: Maximum ``(width, height)`` for the returned pixmap.

    Returns:
        :class:`PySide6.QtGui.QPixmap` or ``None`` if loading fails.
    """
    try:
        from PySide6.QtGui import QPixmap
        from PySide6.QtCore import Qt

        pixmap = QPixmap(path)
        if pixmap.isNull():
            return None
        return pixmap.scaled(
            max_size[0],
            max_size[1],
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("QPixmap load failed for %s: %s", path, exc)
        return None
