"""Dual-threshold face matching with optional double-embedding verification.

A match is only valid when BOTH conditions hold:
  cosine_similarity > COSINE_THRESHOLD    (default 0.45)
  euclidean_distance < EUCLIDEAN_THRESHOLD  (default 0.85)

The double-embedding gate generates two augmented embeddings and requires
BOTH to individually satisfy both thresholds — significantly reducing false
positives at the cost of slightly higher computation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ── Thresholds ───────────────────────────────────────────────────────────────
COSINE_THRESHOLD: float = 0.45
EUCLIDEAN_THRESHOLD: float = 0.85


@dataclass
class MatchResult:
    """Result of a single face verification attempt."""

    is_match: bool
    cosine: float
    euclidean: float
    cosine_ok: bool
    euclidean_ok: bool
    # Double-embedding results (None when not requested)
    cosine_aug0: Optional[float] = None
    euclidean_aug0: Optional[float] = None
    cosine_aug1: Optional[float] = None
    euclidean_aug1: Optional[float] = None


class FaceMatcher:
    """Compare L2-normalised ArcFace embeddings.

    Args:
        cosine_threshold:    Minimum cosine similarity to accept a match.
        euclidean_threshold: Maximum L2 distance to accept a match.
    """

    def __init__(
        self,
        cosine_threshold: float = COSINE_THRESHOLD,
        euclidean_threshold: float = EUCLIDEAN_THRESHOLD,
    ) -> None:
        self._cos_thresh = cosine_threshold
        self._l2_thresh = euclidean_threshold

    # ------------------------------------------------------------------
    # Core comparators
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity in [-1, 1].  Assumes L2-normalised inputs."""
        return float(np.clip(np.dot(emb1, emb2), -1.0, 1.0))

    @staticmethod
    def euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """L2 (Euclidean) distance.  For normalised vectors, range is [0, 2]."""
        return float(np.linalg.norm(emb1 - emb2))

    def is_same_person(self, emb1: np.ndarray, emb2: np.ndarray) -> bool:
        """True iff BOTH cosine similarity AND euclidean distance pass.

        Both embeddings must be L2-normalised 512-D ArcFace vectors.
        """
        cos = self.cosine_similarity(emb1, emb2)
        l2 = self.euclidean_distance(emb1, emb2)
        return cos > self._cos_thresh and l2 < self._l2_thresh

    # ------------------------------------------------------------------
    # Full verification with optional double-embedding
    # ------------------------------------------------------------------

    def verify_embeddings(
        self,
        probe_emb: np.ndarray,
        reference_emb: np.ndarray,
        probe_emb_aug0: Optional[np.ndarray] = None,
        probe_emb_aug1: Optional[np.ndarray] = None,
    ) -> MatchResult:
        """Compare a probe embedding against a reference.

        When *probe_emb_aug0* and *probe_emb_aug1* are provided the
        double-embedding gate is active: the match is only accepted if BOTH
        augmented embeddings also individually satisfy both thresholds.

        Args:
            probe_emb:      Main L2-normalised embedding of the probe face.
            reference_emb:  L2-normalised reference embedding.
            probe_emb_aug0: First augmented embedding (optional).
            probe_emb_aug1: Second augmented embedding (optional).
        """
        cos = self.cosine_similarity(probe_emb, reference_emb)
        l2 = self.euclidean_distance(probe_emb, reference_emb)
        cos_ok = cos > self._cos_thresh
        l2_ok = l2 < self._l2_thresh
        base_match = cos_ok and l2_ok

        # Double-embedding gate
        cos0 = cos1 = l2_0 = l2_1 = None
        double_ok = True
        if probe_emb_aug0 is not None and probe_emb_aug1 is not None:
            cos0 = self.cosine_similarity(probe_emb_aug0, reference_emb)
            l2_0 = self.euclidean_distance(probe_emb_aug0, reference_emb)
            cos1 = self.cosine_similarity(probe_emb_aug1, reference_emb)
            l2_1 = self.euclidean_distance(probe_emb_aug1, reference_emb)
            double_ok = (
                cos0 > self._cos_thresh and l2_0 < self._l2_thresh
                and cos1 > self._cos_thresh and l2_1 < self._l2_thresh
            )

        return MatchResult(
            is_match=base_match and double_ok,
            cosine=cos,
            euclidean=l2,
            cosine_ok=cos_ok,
            euclidean_ok=l2_ok,
            cosine_aug0=cos0,
            euclidean_aug0=l2_0,
            cosine_aug1=cos1,
            euclidean_aug1=l2_1,
        )


# ------------------------------------------------------------------
# High-level verify_face convenience function
# ------------------------------------------------------------------

def verify_face(
    img_rgb: np.ndarray,
    reference_embedding: np.ndarray,
    detector,
    embedder,
    quality_filter,
    matcher: Optional[FaceMatcher] = None,
    double_check: bool = True,
    run_liveness: bool = True,
) -> dict:
    """Full pipeline: detect → quality → embed → match.

    Args:
        img_rgb:             Probe image (RGB uint8).
        reference_embedding: L2-normalised ArcFace 512-D reference vector.
        detector:            :class:`FaceDetector` instance.
        embedder:            :class:`FaceEmbedder` instance.
        quality_filter:      :class:`FaceQualityFilter` instance.
        matcher:             :class:`FaceMatcher` (default thresholds if None).
        double_check:        Use double-embedding augmentation gate.
        run_liveness:        Run liveness heuristic filter.

    Returns:
        A dict with keys:
          ``is_match``   bool
          ``cosine``     float or None
          ``euclidean``  float or None
          ``error``      str or None  (populated when the pipeline failed early)
          ``quality``    QualityResult or None
          ``n_faces``    int — number of faces detected in probe image
    """
    if matcher is None:
        matcher = FaceMatcher()

    faces = detector.detect(img_rgb)
    if not faces:
        return {"is_match": False, "cosine": None, "euclidean": None,
                "error": "no_face_detected", "quality": None, "n_faces": 0}

    # Take the highest-confidence face (typically the primary subject).
    face = faces[0]

    from face_quality import FaceQualityFilter
    qresult = quality_filter.check(face, img_rgb, run_liveness=run_liveness)
    if not qresult.accepted:
        return {"is_match": False, "cosine": None, "euclidean": None,
                "error": f"quality_rejected: {', '.join(qresult.reasons)}",
                "quality": qresult, "n_faces": len(faces)}

    emb = embedder.embed(img_rgb, face)
    if emb is None:
        return {"is_match": False, "cosine": None, "euclidean": None,
                "error": "embedding_failed", "quality": qresult, "n_faces": len(faces)}

    aug0 = aug1 = None
    if double_check:
        aug0, aug1 = embedder.embed_double(img_rgb, face)

    result = matcher.verify_embeddings(emb, reference_embedding, aug0, aug1)
    return {
        "is_match": result.is_match,
        "cosine": result.cosine,
        "euclidean": result.euclidean,
        "cosine_aug0": result.cosine_aug0,
        "euclidean_aug0": result.euclidean_aug0,
        "cosine_aug1": result.cosine_aug1,
        "euclidean_aug1": result.euclidean_aug1,
        "error": None,
        "quality": qresult,
        "n_faces": len(faces),
    }
