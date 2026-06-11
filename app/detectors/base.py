"""Abstract base class for face detectors.

All concrete detectors (Coral, CPU) implement :class:`FaceDetector`.
The rest of the application only depends on this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Detection:
    """A single face detection result.

    Attributes:
        x:          Left edge of bounding box in image pixels.
        y:          Top edge of bounding box in image pixels.
        w:          Width of bounding box in pixels.
        h:          Height of bounding box in pixels.
        confidence: Detector confidence score [0.0 – 1.0].
        landmarks:  Optional 5 facial landmarks in image pixels, ordered
                    right-eye, left-eye, nose, right-mouth, left-mouth — the
                    canonical ArcFace order consumed by
                    :func:`app.embeddings.alignment.align_face_5pt`.  ``None``
                    for detectors that do not produce landmarks (Coral, the
                    Caffe SSD CPU path, Haar).
    """

    x: int
    y: int
    w: int
    h: int
    confidence: float
    landmarks: Optional[List[List[float]]] = None

    @property
    def x2(self) -> int:
        """Right edge."""
        return self.x + self.w

    @property
    def y2(self) -> int:
        """Bottom edge."""
        return self.y + self.h

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Return ``(x, y, w, h)``."""
        return (self.x, self.y, self.w, self.h)

    def clamp(self, img_w: int, img_h: int) -> "Detection":
        """Return a new Detection with coordinates clamped to image bounds."""
        x = max(0, min(self.x, img_w - 1))
        y = max(0, min(self.y, img_h - 1))
        x2 = max(0, min(self.x2, img_w))
        y2 = max(0, min(self.y2, img_h))
        return Detection(
            x=x, y=y, w=x2 - x, h=y2 - y,
            confidence=self.confidence,
            landmarks=self.landmarks,
        )


def detection_iou(a: Detection, b: Detection) -> float:
    """Intersection-over-Union of two detections' bounding boxes."""
    iw = max(0, min(a.x2, b.x2) - max(a.x, b.x))
    ih = max(0, min(a.y2, b.y2) - max(a.y, b.y))
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def greedy_nms(detections: List[Detection], iou_threshold: float = 0.4) -> List[Detection]:
    """Greedy non-maximum suppression: keep the highest-confidence box of each
    overlapping group, suppress the rest."""
    if len(detections) <= 1:
        return list(detections)
    kept: List[Detection] = []
    for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if all(detection_iou(det, k) < iou_threshold for k in kept):
            kept.append(det)
    return kept


class FaceDetector(ABC):
    """Interface that all face detectors must implement."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Short identifier, e.g. ``"coral"`` or ``"cpu"``."""

    @abstractmethod
    def detect(
        self,
        image_bgr: np.ndarray,
        confidence_threshold: float = 0.5,
    ) -> List[Detection]:
        """Detect faces in a BGR image array.

        Args:
            image_bgr:            OpenCV BGR uint8 image.
            confidence_threshold: Discard detections below this score.

        Returns:
            List of :class:`Detection` objects, possibly empty.
        """

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} backend={self.backend_name!r}>"
