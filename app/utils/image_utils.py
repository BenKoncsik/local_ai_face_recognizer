"""Image processing utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image as PilImage, ImageOps

from app.detectors.base import Detection

log = logging.getLogger(__name__)

# Standard EXIF tag id for Orientation
_EXIF_ORIENTATION_TAG = 274


def save_face_crop(
    img_bgr: np.ndarray,
    detection: Detection,
    crops_dir: Path,
    image_id: int,
    thumbnail_size: Tuple[int, int] = (128, 128),
    face_index: int = 0,
    dest_path: Optional[Path] = None,
    crop_mode: str = "legacy",
    landmarks: Optional[object] = None,
) -> Optional[Path]:
    """Extract a face crop and save it as a JPEG thumbnail.

    Args:
        img_bgr:        Full image in BGR format.
        detection:      Bounding box detection result.
        crops_dir:      Directory to write the crop file.
        image_id:       Parent image DB ID (used in filename).
        thumbnail_size: Target size ``(width, height)`` for the saved crop.
        face_index:     Index of this face within the image (for naming).
                        Ignored when *dest_path* is provided.
        dest_path:      Explicit output path; overrides the auto-generated name.
                        Use this to overwrite an existing crop in-place without
                        introducing filename collisions.
        crop_mode:      ``"legacy"`` (stretch), ``"square"`` (aspect-preserving)
                        or ``"aligned"`` (5-point landmark alignment).  See
                        :mod:`app.embeddings.alignment`.
        landmarks:      Optional 5×2 facial landmarks, used by ``"aligned"``.

    Returns:
        :class:`Path` to the saved crop file, or ``None`` on failure.
    """
    from app.embeddings.alignment import make_face_crop

    x, y, w, h = detection.as_tuple()

    if w <= 0 or h <= 0:
        log.debug("Skipping zero-area crop for image_id=%d", image_id)
        return None

    thumb = make_face_crop(
        img_bgr,
        bbox=(x, y, w, h),
        output_size=thumbnail_size,
        mode=crop_mode,
        landmarks=landmarks,
        legacy_margin_frac=0.10,
    )
    if thumb is None:
        return None

    if dest_path is not None:
        dest = dest_path
    else:
        filename = f"img{image_id:06d}_face{face_index:06d}.jpg"
        dest = crops_dir / filename
    success = save_image_bgr(dest, thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])

    if not success:
        log.warning("Failed to write crop: %s", dest)
        return None

    return dest


def load_image_bgr_normalized(path: str) -> Optional[np.ndarray]:
    """Load an image and apply EXIF orientation correction.

    Uses Pillow's exif_transpose for correct orientation on all platforms.
    Falls back to load_image_bgr when EXIF reading fails.

    Args:
        path: Absolute or relative path to the image file.

    Returns:
        BGR uint8 numpy array with correct orientation, or ``None`` on failure.
    """
    try:
        with PilImage.open(path) as pil:
            exif = pil.getexif()
            orientation = exif.get(_EXIF_ORIENTATION_TAG, 1) if exif else 1
            if orientation in (None, 1):
                # Normal orientation — fast OpenCV path
                pass
            else:
                pil_corrected = ImageOps.exif_transpose(pil)
                log.debug(
                    "EXIF orientation=%d applied for: %s", orientation, Path(path).name
                )
                return cv2.cvtColor(
                    np.array(pil_corrected.convert("RGB")), cv2.COLOR_RGB2BGR
                )
    except Exception as exc:  # noqa: BLE001
        log.debug("EXIF normalisation failed for %s: %s", path, exc)

    return load_image_bgr(path)


def get_exif_orientation(path: str) -> int:
    """Return the EXIF Orientation tag value for *path*, or 1 if absent/unreadable."""
    try:
        with PilImage.open(path) as pil:
            exif = pil.getexif()
            return exif.get(_EXIF_ORIENTATION_TAG, 1) if exif else 1
    except Exception:  # noqa: BLE001
        return 1


def apply_clahe(img_bgr: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Apply CLAHE contrast enhancement in the L channel of LAB colour space.

    Args:
        img_bgr:    BGR uint8 image.
        clip_limit: CLAHE clip limit (higher = stronger contrast).

    Returns:
        Contrast-enhanced BGR uint8 image.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_histeq(img_bgr: np.ndarray) -> np.ndarray:
    """Apply global histogram equalization to each BGR channel independently.

    Helps old / underexposed photos where the face-detector was trained on
    well-exposed modern images.

    Args:
        img_bgr: BGR uint8 image.

    Returns:
        Histogram-equalised BGR uint8 image.
    """
    b, g, r = cv2.split(img_bgr)
    return cv2.merge([cv2.equalizeHist(b), cv2.equalizeHist(g), cv2.equalizeHist(r)])


def apply_bilateral(img_bgr: np.ndarray, d: int = 9, sigma: float = 75.0) -> np.ndarray:
    """Apply bilateral filter to reduce noise while preserving edges.

    Useful for grainy scanned or old photographs where high-frequency noise
    can lower detector confidence.

    Args:
        img_bgr: BGR uint8 image.
        d:       Pixel neighbourhood diameter.
        sigma:   Colour and space sigma for the filter.

    Returns:
        Filtered BGR uint8 image.
    """
    return cv2.bilateralFilter(img_bgr, d, sigma, sigma)


def apply_gamma(img_bgr: np.ndarray, gamma: float = 0.7) -> np.ndarray:
    """Apply gamma correction to brighten (gamma<1) or darken (gamma>1) an image.

    Args:
        img_bgr: BGR uint8 image.
        gamma:   Gamma value; 0.7 brightens, 1.3 darkens.

    Returns:
        Gamma-corrected BGR uint8 image.
    """
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8
    )
    return cv2.LUT(img_bgr, table)


def compute_dhash(img_bgr: np.ndarray, hash_size: int = 8) -> int:
    """Compute a difference hash (dHash) of *img_bgr*.

    Two images with the same or very similar content produce close hash values.
    Use ``bin(a ^ b).count('1')`` to count differing bits between hashes.

    Args:
        img_bgr:   BGR uint8 image.
        hash_size: Side length of the internal hash grid.

    Returns:
        Integer hash (hash_size ** 2 bits).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = diff.flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


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
