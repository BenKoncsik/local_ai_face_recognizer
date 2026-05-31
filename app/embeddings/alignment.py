"""Face crop geometry: aspect-preserving square crops and 5-point alignment.

Why this exists
---------------
MobileFaceNet / ArcFace embedding models are trained on faces that have been
*aligned* — warped by a similarity transform so the eyes, nose and mouth land
on a fixed canonical template at 112×112.  The original pipeline instead fed
the raw bounding box, stretched to a square, straight into the model
(:func:`app.utils.image_utils.save_face_crop`).  Two problems follow:

1. **No alignment** — a tilted or profile face produces a very different
   embedding from a frontal one of the same person.
2. **Aspect distortion** — a non-square bounding box squashed to a square warps
   facial geometry differently depending on the box shape, so two crops of the
   same person drift apart in embedding space.

Both inflate intra-person embedding distance and feed the identity
fragmentation this module set out to fix.  Two crop strategies are offered:

``square_crop``
    No landmarks required.  Expands the box to a square about its centre and
    resizes without changing aspect ratio.  Removes problem (2).

``align_face_5pt``
    Requires 5 facial landmarks.  Warps the face to the standard ArcFace
    template, removing problems (1) and (2).

Switching an existing library to a new crop mode changes every embedding, so
callers must treat a mode change as requiring a full re-embed.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Canonical 5-point template (right eye, left eye, nose, right mouth, left
# mouth) for a 112×112 aligned face — the de-facto ArcFace / InsightFace
# reference points.
_ARCFACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def square_crop(
    image_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    output_size: Tuple[int, int] = (112, 112),
    margin_frac: float = 0.10,
) -> Optional[np.ndarray]:
    """Return an aspect-preserving square crop centred on *bbox*.

    The bounding box is expanded to a square about its centre (plus a margin)
    so the face is never horizontally or vertically squashed.  When the square
    would run off the image edge it is clamped and the result is padded back to
    square with edge replication, keeping facial proportions intact.

    Args:
        image_bgr:   Full image, BGR uint8.
        bbox:        ``(x, y, w, h)`` face box in image pixels.
        output_size: ``(width, height)`` of the returned crop.
        margin_frac: Extra context added on each side as a fraction of the
                     square side length.

    Returns:
        The resized BGR crop, or ``None`` when the box is degenerate.
    """
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return None

    img_h, img_w = image_bgr.shape[:2]
    cx = x + w / 2.0
    cy = y + h / 2.0
    side = max(w, h) * (1.0 + 2.0 * margin_frac)
    half = side / 2.0

    x1 = int(round(cx - half))
    y1 = int(round(cy - half))
    x2 = int(round(cx + half))
    y2 = int(round(cy + half))

    # Clamp to image, recording how much padding each side needs to stay square.
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - img_w)
    pad_bottom = max(0, y2 - img_h)

    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(img_w, x2), min(img_h, y2)
    if cx2 <= cx1 or cy2 <= cy1:
        return None

    crop = image_bgr[cy1:cy2, cx1:cx2]
    if pad_left or pad_top or pad_right or pad_bottom:
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_REPLICATE,
        )
    if crop.size == 0:
        return None
    return cv2.resize(crop, output_size, interpolation=cv2.INTER_AREA)


def align_face_5pt(
    image_bgr: np.ndarray,
    landmarks: Sequence[Sequence[float]],
    output_size: Tuple[int, int] = (112, 112),
) -> Optional[np.ndarray]:
    """Warp a face to the canonical ArcFace template using 5 landmarks.

    Args:
        image_bgr:   Full image, BGR uint8.
        landmarks:   5 ``(x, y)`` points in image pixels, ordered
                     right-eye, left-eye, nose, right-mouth, left-mouth.
        output_size: ``(width, height)`` of the aligned output (must match the
                     template scale; defaults to 112×112).

    Returns:
        The aligned BGR crop, or ``None`` if the transform cannot be estimated.
    """
    pts = np.asarray(landmarks, dtype=np.float32)
    if pts.shape != (5, 2):
        log.debug("align_face_5pt: expected 5x2 landmarks, got %s", pts.shape)
        return None

    template = _ARCFACE_TEMPLATE_112.copy()
    out_w, out_h = output_size
    if (out_w, out_h) != (112, 112):
        template[:, 0] *= out_w / 112.0
        template[:, 1] *= out_h / 112.0

    # Partial affine = similarity transform (rotation + uniform scale + shift),
    # which is what ArcFace alignment uses (no shear).
    matrix, _ = cv2.estimateAffinePartial2D(pts, template, method=cv2.LMEDS)
    if matrix is None:
        return None
    return cv2.warpAffine(
        image_bgr, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR
    )


def make_face_crop(
    image_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    output_size: Tuple[int, int],
    *,
    mode: str = "legacy",
    landmarks: Optional[Sequence[Sequence[float]]] = None,
    legacy_margin_frac: float = 0.10,
) -> Optional[np.ndarray]:
    """Produce a face crop using the requested *mode*.

    Modes:
        ``"legacy"`` — original behaviour: bbox + margin stretched to fill the
            output (aspect ratio NOT preserved).
        ``"square"`` — aspect-preserving square crop (:func:`square_crop`).
        ``"aligned"`` — 5-point alignment (:func:`align_face_5pt`); falls back
            to ``"square"`` when landmarks are missing.

    Returns the BGR crop, or ``None`` on failure.
    """
    if mode == "aligned" and landmarks is not None:
        aligned = align_face_5pt(image_bgr, landmarks, output_size)
        if aligned is not None:
            return aligned
        log.debug("Alignment failed — falling back to square crop")
        mode = "square"

    if mode in ("aligned", "square"):
        return square_crop(image_bgr, bbox, output_size, margin_frac=legacy_margin_frac)

    # Legacy: stretch the (margined) box to the output size.
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return None
    img_h, img_w = image_bgr.shape[:2]
    mx = int(w * legacy_margin_frac)
    my = int(h * legacy_margin_frac)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(img_w, x + w + mx)
    y2 = min(img_h, y + h + my)
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, output_size, interpolation=cv2.INTER_AREA)
