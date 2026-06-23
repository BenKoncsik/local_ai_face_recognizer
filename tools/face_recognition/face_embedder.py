"""ArcFace embedding extraction using InsightFace buffalo_l.

Produces L2-normalized 512-dimensional embeddings after face alignment.
Supports double-embedding via slight image augmentation.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from face_detector import DetectedFace, FaceDetector
from utils.image import augment_brightness_contrast

log = logging.getLogger(__name__)

EMBEDDING_DIM = 512


class FaceEmbedder:
    """Extract ArcFace embeddings from detected faces.

    Relies on the same InsightFace ``FaceAnalysis`` app created by
    :class:`FaceDetector` — pass *detector* to share the loaded models.

    Args:
        detector: A pre-initialised :class:`FaceDetector`.  Its internal
                  ``_app`` is reused so the ArcFace model loads only once.
    """

    def __init__(self, detector: FaceDetector) -> None:
        self._app = detector._app

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, img_rgb: np.ndarray, face: DetectedFace) -> Optional[np.ndarray]:
        """Compute an L2-normalised 512-D ArcFace embedding for *face*.

        InsightFace performs 5-point alignment internally when landmarks
        (kps) are available.  If no embedding can be produced, returns None.

        Args:
            img_rgb: Full image (RGB uint8) the face was detected in.
            face:    Detection from :class:`FaceDetector`.
        """
        raw = self._get_raw_embedding(img_rgb, face)
        if raw is None:
            return None
        return self._l2_normalize(raw)

    def embed_double(
        self,
        img_rgb: np.ndarray,
        face: DetectedFace,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Compute two augmented embeddings for the double-check gate.

        Applies two distinct brightness/contrast jitters and returns both
        L2-normalised embeddings.  Both must match a reference for a
        verification to count as valid.

        Returns ``(emb_a, emb_b)`` — either may be None if extraction fails.
        """
        aug0 = augment_brightness_contrast(img_rgb, seed=0)
        aug1 = augment_brightness_contrast(img_rgb, seed=1)
        emb0 = self.embed(aug0, face)
        emb1 = self.embed(aug1, face)
        return emb0, emb1

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_raw_embedding(
        self, img_rgb: np.ndarray, face: DetectedFace
    ) -> Optional[np.ndarray]:
        """Ask InsightFace to produce a raw (unnormalised) ArcFace embedding."""
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        try:
            # Re-run get() to obtain InsightFace face objects with embeddings.
            raw_faces = self._app.get(img_bgr, max_num=0)
        except Exception as exc:  # noqa: BLE001
            log.error("InsightFace embedding failed: %s", exc)
            return None

        if not raw_faces:
            return None

        # Match by bounding-box proximity to the original detection.
        best = self._closest_face(raw_faces, face)
        if best is None or best.embedding is None:
            return None

        return best.embedding.copy()

    @staticmethod
    def _closest_face(raw_faces, face: DetectedFace):
        """Return the InsightFace face object whose bbox centre is nearest to *face*."""
        fx, fy, fw, fh = face.bbox
        cx = fx + fw / 2.0
        cy = fy + fh / 2.0
        best = None
        best_dist = float("inf")
        for rf in raw_faces:
            bx1, by1, bx2, by2 = rf.bbox.astype(float)
            rcx = (bx1 + bx2) / 2.0
            rcy = (by1 + by2) / 2.0
            dist = (rcx - cx) ** 2 + (rcy - cy) ** 2
            if dist < best_dist:
                best_dist = dist
                best = rf
        return best

    @staticmethod
    def _l2_normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            return v
        return v / norm
