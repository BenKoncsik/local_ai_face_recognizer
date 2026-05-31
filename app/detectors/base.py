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
