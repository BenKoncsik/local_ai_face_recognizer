"""Face quality evaluation service.

Scores each detected face for usability in embedding / training pipelines.
Poor-quality faces (back-facing, blurry, too small, low confidence) are
flagged so downstream services can skip them when the user's quality filter
is enabled.

Evaluation criteria (all without extra ML models):
  - Detection confidence       → low_confidence
  - Bounding-box area          → too_small
  - Crop blur (Laplacian var.) → blurry
  - Bbox aspect ratio          → bad_aspect_ratio

The overall quality_score is 1.0 minus deductions for each reason found.
is_low_quality is True when quality_score < QUALITY_THRESHOLD.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
from sqlalchemy.orm import Session

from app.db.models import Face

log = logging.getLogger(__name__)

# ── Tuneable thresholds ────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.55       # below → low_confidence
MIN_FACE_AREA = 40 * 40           # pixels² below → too_small  (40×40 px)
BLUR_THRESHOLD = 35.0             # Laplacian variance below → blurry
ASPECT_RATIO_MIN = 0.45           # w/h below → bad_aspect_ratio
ASPECT_RATIO_MAX = 2.20           # w/h above → bad_aspect_ratio
QUALITY_THRESHOLD = 0.50          # score below → is_low_quality = True

# Penalty subtracted from score per issue found (total never below 0.0)
_PENALTY = {
    "low_confidence":   0.35,
    "too_small":        0.30,
    "blurry":           0.25,
    "bad_aspect_ratio": 0.20,
}


class FaceQualityEvaluator:
    """Stateless quality evaluator: call :meth:`evaluate` on any Face."""

    def evaluate(self, face: Face) -> tuple[float, list[str]]:
        """Compute a quality score and list of reason codes for *face*.

        Returns:
            ``(score, reasons)`` where *score* is in ``[0.0, 1.0]`` and
            *reasons* is a (possibly empty) list of string codes.
        """
        reasons: list[str] = []

        if face.confidence < CONFIDENCE_THRESHOLD:
            reasons.append("low_confidence")

        area = face.bbox_w * face.bbox_h
        if area < MIN_FACE_AREA:
            reasons.append("too_small")

        if face.bbox_w > 0 and face.bbox_h > 0:
            ratio = face.bbox_w / face.bbox_h
            if ratio < ASPECT_RATIO_MIN or ratio > ASPECT_RATIO_MAX:
                reasons.append("bad_aspect_ratio")

        if face.crop_path:
            blur = self._laplacian_variance(face.crop_path)
            if blur is not None and blur < BLUR_THRESHOLD:
                reasons.append("blurry")

        penalty = sum(_PENALTY.get(r, 0.0) for r in reasons)
        score = max(0.0, 1.0 - penalty)
        return score, reasons

    def evaluate_and_update(self, face: Face) -> None:
        """Evaluate *face* and write results into its ORM attributes.

        The caller is responsible for committing the session.
        """
        score, reasons = self.evaluate(face)
        face.quality_score = score
        face.quality_reasons = ",".join(reasons) if reasons else None
        face.is_low_quality = score < QUALITY_THRESHOLD
        if face.is_low_quality:
            log.debug(
                "Face id=%d flagged as low quality (score=%.2f reasons=%s)",
                face.id, score, reasons,
            )

    @staticmethod
    def _laplacian_variance(crop_path: str) -> Optional[float]:
        """Return Laplacian variance of *crop_path* as a blur metric.

        Higher = sharper.  Returns None if the image cannot be read.
        """
        path = Path(crop_path)
        if not path.exists():
            return None
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return None
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class FaceQualityBatchService:
    """Re-evaluates all faces in the database (e.g. after threshold changes)."""

    def __init__(self, session: Session) -> None:
        self._session = session
        self._evaluator = FaceQualityEvaluator()

    def evaluate_all(self, progress_cb=None) -> int:
        """Evaluate every face and persist quality fields.

        Args:
            progress_cb: Optional ``(current, total)`` callable.

        Returns:
            Number of faces evaluated.
        """
        from sqlalchemy.orm import lazyload

        # Quality scoring reads crops, never embeddings — skip the default
        # selectin blob load for this full-table query.
        faces: list[Face] = (
            self._session.query(Face).options(lazyload(Face.blob)).all()
        )
        total = len(faces)
        log.info("Quality batch evaluation: %d face(s)", total)

        for idx, face in enumerate(faces, start=1):
            if progress_cb:
                progress_cb(idx, total)
            self._evaluator.evaluate_and_update(face)
            if idx % 200 == 0:
                self._session.commit()

        self._session.commit()
        low = sum(1 for f in faces if f.is_low_quality)
        log.info(
            "Quality batch done: %d evaluated, %d flagged as low quality",
            total, low,
        )
        return total
