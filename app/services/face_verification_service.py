"""Detection verification — a second-opinion gate against false positives.

The low-threshold detection passes (high-accuracy multi-variant, adaptive
escalation) and the landmark-less backends (Coral SSD, Caffe SSD) regularly
produce boxes over ears, hair, walls, clothing and objects.  This service
re-examines each uncertain detection by cropping the candidate region with a
generous margin, upscaling it, and re-running the YuNet detector on the crop:

* a **real face** is re-found in its own centered crop with a high score,
  even when the full-image pass only scored it 0.25–0.45;
* a **non-face** (ear, wall, fabric, toy) is not re-found and is dropped.

As a side benefit, when the verifier's box is much tighter than the original
(oversized SSD boxes), the detection is refined to the verified box, and
landmarks found by the verifier are adopted by landmark-less detections so
the "aligned" crop mode works for them too.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from app.config import DetectionConfig
from app.detectors.base import Detection, FaceDetector
from app.utils.image_utils import apply_clahe

log = logging.getLogger(__name__)

# A verified box must overlap the original candidate at least this much
# (IoU in crop coordinates) — or have its center inside the candidate box —
# so a *different* face elsewhere in the margin never confirms a non-face.
_MIN_OVERLAP_IOU = 0.25

# Adopt the verifier's (tighter) box only when it is clearly smaller than the
# original — fixes oversized boxes without jittering well-fitted ones.
_REFINE_AREA_RATIO = 0.5


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """IoU of two (x, y, w, h) boxes."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


class FaceVerifier:
    """Confirms or rejects face detections via a crop-level re-detection.

    Args:
        config:   The detection configuration block (verification_* fields).
        detector: Optional detector to verify with.  ``None`` → a YuNet
                  instance is created from ``config.yunet_model_path``; when
                  that fails (model missing) the verifier reports itself as
                  unavailable and :meth:`verify` passes everything through.
    """

    def __init__(
        self,
        config: DetectionConfig,
        detector: Optional[FaceDetector] = None,
    ) -> None:
        self._config = config
        if detector is not None:
            self._detector: Optional[FaceDetector] = detector
        else:
            try:
                from app.detectors.yunet_detector import YuNetDetector

                self._detector = YuNetDetector(model_path=config.yunet_model_path)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "Face verification unavailable (%s) — the false-positive "
                    "gate is disabled for this run.", exc,
                )
                self._detector = None

    @property
    def available(self) -> bool:
        return self._detector is not None

    # ------------------------------------------------------------------

    def verify(self, image_bgr: np.ndarray, det: Detection) -> Optional[Detection]:
        """Re-detect *det* inside its own enlarged crop.

        Returns:
            The confirmed detection — possibly refined (tighter bbox and/or
            adopted landmarks) — or ``None`` when no face could be confirmed
            at the candidate location.  When the verifier is unavailable the
            detection is passed through unchanged.
        """
        if self._detector is None:
            return det
        if det.w <= 1 or det.h <= 1:
            return None

        crop, offset, scale = self._extract_crop(image_bgr, det)
        if crop is None:
            return det  # degenerate geometry — do not reject on our failure

        # Candidate box in crop coordinates.
        local = (
            (det.x - offset[0]) * scale,
            (det.y - offset[1]) * scale,
            det.w * scale,
            det.h * scale,
        )
        # The face inside the crop is roughly the candidate box size; allow it
        # to be re-found notably smaller (loose original boxes, turned heads —
        # 0.3 measured as the best junk/real-face separation on real crops).
        min_face = max(12, int(min(det.w, det.h) * scale * 0.3))

        variants = (crop, apply_clahe(crop))
        for variant in variants:
            try:
                found: List[Detection] = self._detector.detect(
                    variant,
                    confidence_threshold=self._config.verification_threshold,
                    min_face_size=min_face,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("Verifier detect failed: %s — keeping detection", exc)
                return det
            for f in found:
                if self._confirms(local, f):
                    return self._refine(det, f, offset, scale)
        return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_crop(
        self, image_bgr: np.ndarray, det: Detection
    ) -> Tuple[Optional[np.ndarray], Tuple[int, int], float]:
        """Crop *det* with margin and upscale; return (crop, offset, scale)."""
        import cv2

        img_h, img_w = image_bgr.shape[:2]
        margin = self._config.verification_margin
        x1 = max(0, int(det.x - det.w * margin))
        y1 = max(0, int(det.y - det.h * margin))
        x2 = min(img_w, int(det.x + det.w * (1 + margin)))
        y2 = min(img_h, int(det.y + det.h * (1 + margin)))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None, (0, 0), 1.0

        crop = image_bgr[y1:y2, x1:x2]
        short_side = min(crop.shape[:2])
        scale = 1.0
        min_crop = self._config.verification_min_crop
        if short_side < min_crop:
            scale = min_crop / short_side
            crop = cv2.resize(
                crop,
                (int(round(crop.shape[1] * scale)), int(round(crop.shape[0] * scale))),
                interpolation=cv2.INTER_CUBIC,
            )
        return crop, (x1, y1), scale

    @staticmethod
    def _confirms(
        local: Tuple[float, float, float, float], found: Detection
    ) -> bool:
        """True when *found* sits on the candidate box, not elsewhere in the margin."""
        fbox = (float(found.x), float(found.y), float(found.w), float(found.h))
        if _iou(local, fbox) >= _MIN_OVERLAP_IOU:
            return True
        fcx = found.x + found.w / 2.0
        fcy = found.y + found.h / 2.0
        return (
            local[0] <= fcx <= local[0] + local[2]
            and local[1] <= fcy <= local[1] + local[3]
        )

    @staticmethod
    def _refine(
        det: Detection,
        found: Detection,
        offset: Tuple[int, int],
        scale: float,
    ) -> Detection:
        """Map the verified box back to image space and merge with *det*."""
        fx = found.x / scale + offset[0]
        fy = found.y / scale + offset[1]
        fw = found.w / scale
        fh = found.h / scale

        landmarks = None
        if found.landmarks is not None:
            landmarks = [
                [px / scale + offset[0], py / scale + offset[1]]
                for px, py in found.landmarks
            ]

        # A verified face is a real face: lift the confidence to the verifier's
        # crop-level score so the downstream quality filter does not exclude a
        # genuinely rescued (low full-image score) face from clustering.
        confidence = max(det.confidence, found.confidence)

        if fw * fh <= _REFINE_AREA_RATIO * det.w * det.h:
            # Original box is clearly oversized — adopt the tight verified box.
            return Detection(
                x=int(round(fx)), y=int(round(fy)),
                w=int(round(fw)), h=int(round(fh)),
                confidence=confidence,
                landmarks=landmarks,
            )
        return Detection(
            x=det.x, y=det.y, w=det.w, h=det.h,
            confidence=confidence,
            landmarks=det.landmarks if det.landmarks is not None else landmarks,
        )
