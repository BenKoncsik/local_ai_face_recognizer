"""Image loading and preprocessing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def load_image_rgb(path: str | Path) -> Optional[np.ndarray]:
    """Load an image from *path* and return it as RGB uint8.

    Handles EXIF orientation via cv2.IMREAD_UNCHANGED workaround.
    Returns None if the file cannot be read.
    """
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def load_image_bgr(path: str | Path) -> Optional[np.ndarray]:
    """Load an image as BGR (OpenCV native format)."""
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def resize_keep_aspect(
    img: np.ndarray, max_side: int
) -> Tuple[np.ndarray, float]:
    """Resize *img* so its longest side is at most *max_side*.

    Returns ``(resized, scale)`` where scale < 1.0 means the image was shrunk.
    """
    h, w = img.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale == 1.0:
        return img, 1.0
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def augment_brightness_contrast(
    img: np.ndarray,
    brightness_delta: float = 0.15,
    contrast_delta: float = 0.15,
    seed: int = 0,
) -> np.ndarray:
    """Apply a small, deterministic brightness/contrast jitter to *img*.

    *img* must be RGB or BGR uint8.  Returns a new array.

    Args:
        brightness_delta: Maximum absolute brightness change [0, 1].
        contrast_delta:   Maximum absolute contrast change [0, 1].
        seed:             Which variant to generate (0 or 1, etc.).
    """
    rng = np.random.default_rng(seed)
    alpha = 1.0 + rng.uniform(-contrast_delta, contrast_delta)   # contrast
    beta = rng.uniform(-brightness_delta, brightness_delta) * 255  # brightness
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return out


def crop_face(img: np.ndarray, x: int, y: int, w: int, h: int, margin: float = 0.1) -> np.ndarray:
    """Crop a face region with an optional proportional margin.

    Clamps to image boundaries.  Returns the cropped BGR/RGB array.
    """
    ih, iw = img.shape[:2]
    mx = int(w * margin)
    my = int(h * margin)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(iw, x + w + mx)
    y2 = min(ih, y + h + my)
    return img[y1:y2, x1:x2].copy()
