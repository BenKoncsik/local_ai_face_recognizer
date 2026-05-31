"""CPU face detector using OpenCV's YuNet model (``cv2.FaceDetectorYN``).

Unlike :class:`~app.detectors.cpu_detector.CpuDetector` (which only returns
bounding boxes), YuNet additionally returns **5 facial landmarks** per face —
right eye, left eye, nose tip, right mouth corner, left mouth corner.  Those
landmarks are exactly what :func:`app.embeddings.alignment.align_face_5pt`
needs to warp a face onto the ArcFace template, so this detector is what makes
``embedding.crop_mode = "aligned"`` actually align instead of silently falling
back to a square crop.

Model file
----------
``models/face_detection_yunet_2023mar.onnx`` (~230 KB).  Download from the
OpenCV Model Zoo::

    https://github.com/opencv/opencv_zoo/raw/main/models/\\
    face_detection_yunet/face_detection_yunet_2023mar.onnx

Place it in ``models/`` (the default search location) or point
``detection.yunet_model_path`` at it in ``config.yaml``.

IMPORTANT: This is a real CPU implementation — no stubs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from app.detectors.base import Detection, FaceDetector
from app.paths import resource_path

log = logging.getLogger(__name__)

# Default model location relative to the project root / bundle.
_DEFAULT_MODEL = "models/face_detection_yunet_2023mar.onnx"

# YuNet's raw detection row layout (one row per face):
#   [0:4]  bbox  x, y, w, h
#   [4:14] 5 landmarks (x, y) pairs in this order:
#            right eye, left eye, nose, right mouth, left mouth
#   [14]   confidence score
_LANDMARK_SLICE = slice(4, 14)
_SCORE_INDEX = 14


class YuNetDetector(FaceDetector):
    """CPU face detector using OpenCV's YuNet ONNX model.

    Args:
        model_path: Optional ``.onnx`` path or directory override.  ``None`` →
                    look for :data:`_DEFAULT_MODEL` relative to the project root.

    Raises:
        FileNotFoundError: when no YuNet model file can be located.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        resolved = self._resolve_model_path(model_path)
        if resolved is None:
            raise FileNotFoundError(
                "YuNet model not found. Download "
                "face_detection_yunet_2023mar.onnx into models/ "
                "(see app/detectors/yunet_detector.py for the URL) or set "
                "detection.yunet_model_path in config.yaml."
            )
        self._model_path = resolved
        # Input size is updated per-image in detect(); the constructor value is
        # just a placeholder.  Thresholds are likewise overridden per call.
        self._net = cv2.FaceDetectorYN.create(
            str(resolved),
            "",
            (320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000,
        )
        log.info("CPU detector: OpenCV YuNet (%s)", resolved.name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_model_path(model_path: Optional[str]) -> Optional[Path]:
        """Find the YuNet ``.onnx`` model file."""
        candidates: List[Path] = []
        if model_path:
            p = Path(model_path)
            if p.is_dir():
                candidates.append(p / "face_detection_yunet_2023mar.onnx")
            else:
                candidates.append(p)
        candidates.append(resource_path(_DEFAULT_MODEL))
        return next((c for c in candidates if c.exists()), None)

    @staticmethod
    def _is_valid_face(det: Detection, min_face_size: int) -> bool:
        """Return True if the detection looks like a plausible face crop."""
        if det.w < min_face_size or det.h < min_face_size:
            return False
        aspect = det.w / det.h if det.h > 0 else 0
        return 0.4 <= aspect <= 2.5

    # ------------------------------------------------------------------

    @property
    def backend_name(self) -> str:
        return "yunet"

    def detect(
        self,
        image_bgr: np.ndarray,
        confidence_threshold: float = 0.65,
        min_face_size: int = 50,
    ) -> List[Detection]:
        """Detect faces (with 5 landmarks) in *image_bgr*.

        Args:
            image_bgr:            OpenCV BGR uint8 image.
            confidence_threshold: Minimum confidence to keep a detection.
            min_face_size:        Minimum face width AND height in pixels.

        Returns:
            List of :class:`Detection` objects, each carrying ``landmarks``.
            YuNet applies its own NMS internally, so no extra dedup is needed.
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        img_h, img_w = image_bgr.shape[:2]
        # YuNet requires the input size to match the image before each detect().
        self._net.setInputSize((img_w, img_h))
        self._net.setScoreThreshold(float(confidence_threshold))

        _, faces = self._net.detect(image_bgr)
        if faces is None:
            return []

        results: List[Detection] = []
        for row in faces:
            x, y, w, h = (int(round(v)) for v in row[0:4])
            conf = float(row[_SCORE_INDEX])
            lm = row[_LANDMARK_SLICE].reshape(5, 2).astype(float)
            landmarks = [[float(px), float(py)] for px, py in lm]

            det = Detection(
                x=x, y=y, w=w, h=h, confidence=conf, landmarks=landmarks
            ).clamp(img_w, img_h)
            if self._is_valid_face(det, min_face_size):
                results.append(det)

        log.debug("YuNet detected %d face(s)", len(results))
        return results
