"""Per-face recognition diagnostics — "why does this face have this identity?".

Surfaces, for a single face, exactly what the matching pipeline saw:

* the embedding similarity to every known (named) person profile,
* the similarity to every Unknown-person centroid,
* the adaptive threshold that face would face in recognition,
* its current assignment and how it was made,
* a plain-language verdict explaining the outcome.

This is the developer/debug counterpart to the automatic pipeline: it answers
the fragmentation question "why did this face become Unknown 155 instead of
joining Unknown 98 / the named person it clearly matches?" by laying the scores
and thresholds side by side.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from sqlalchemy.orm import Session

from app.config import RecognitionConfig
from app.db.models import Face, Person
from app.services.vector_scoring import FaceVectorScorer

log = logging.getLogger(__name__)


@dataclass
class CandidateScore:
    """One ranked identity candidate for a face."""

    person_id: int
    name: str
    is_auto_named: bool
    similarity: float
    face_count: int


@dataclass
class FaceDiagnostics:
    """Full diagnostic record for one face."""

    face_id: int
    has_embedding: bool
    current_person_id: Optional[int]
    current_person_name: Optional[str]
    assignment_source: Optional[str]
    assignment_confidence: Optional[float]
    quality_score: Optional[float]
    is_low_quality: Optional[bool]
    bbox: tuple
    adaptive_threshold: float
    base_threshold: float
    margin_required: float
    top_named: List[CandidateScore] = field(default_factory=list)
    top_unknown: List[CandidateScore] = field(default_factory=list)
    verdict: str = ""


class FaceDiagnosticsService:
    """Explains the recognition outcome for individual faces."""

    def __init__(
        self,
        session: Session,
        config: Optional[RecognitionConfig] = None,
        exclude_low_quality: bool = False,
        top_k: int = 5,
    ) -> None:
        self._session = session
        self._config = config or RecognitionConfig()
        self._top_k = top_k
        # Reuse the shared vector scorer so thresholds/scoring match production.
        self._recognizer = FaceVectorScorer(
            session, config=self._config, exclude_low_quality=exclude_low_quality
        )

    def explain(self, face_id: int) -> Optional[FaceDiagnostics]:
        """Return diagnostics for *face_id*, or ``None`` if the face is gone."""
        face = self._session.get(Face, face_id)
        if face is None:
            return None

        person = face.person
        diag = FaceDiagnostics(
            face_id=face.id,
            has_embedding=face.get_embedding() is not None,
            current_person_id=face.person_id,
            current_person_name=person.name if person else None,
            assignment_source=face.assignment_source,
            assignment_confidence=face.assignment_confidence,
            quality_score=face.quality_score,
            is_low_quality=face.is_low_quality,
            bbox=(face.bbox_x, face.bbox_y, face.bbox_w, face.bbox_h),
            adaptive_threshold=self._recognizer._compute_adaptive_threshold(face),
            base_threshold=self._config.auto_assign_threshold,
            margin_required=self._config.min_margin,
        )

        embedding = self._recognizer._normalise(face.get_embedding())
        if embedding is None:
            diag.verdict = (
                "No embedding stored — the face was never embedded, so no "
                "matching could run."
            )
            return diag

        diag.top_named = self._score_named(embedding)
        diag.top_unknown = self._score_unknown(embedding, exclude_person=face.person_id)
        diag.verdict = self._build_verdict(diag)
        return diag

    # ------------------------------------------------------------------

    def _score_named(self, embedding: np.ndarray) -> List[CandidateScore]:
        profiles = self._recognizer.build_profiles()
        scored = [
            CandidateScore(
                person_id=p.person_id,
                name=p.name,
                is_auto_named=False,
                similarity=self._recognizer._score(embedding, p),
                face_count=p.face_count,
            )
            for p in profiles.values()
        ]
        scored.sort(key=lambda c: c.similarity, reverse=True)
        return scored[: self._top_k]

    def _score_unknown(
        self, embedding: np.ndarray, exclude_person: Optional[int]
    ) -> List[CandidateScore]:
        auto_persons: List[Person] = (
            self._session.query(Person)
            .filter(Person.is_auto_named == True)  # noqa: E712
            .all()
        )
        scored: List[CandidateScore] = []
        for person in auto_persons:
            vecs = []
            for f in person.faces:
                v = self._recognizer._normalise(f.get_embedding())
                if v is not None:
                    vecs.append(v)
            if not vecs:
                continue
            centroid = self._recognizer._normalise(np.mean(vecs, axis=0))
            if centroid is None:
                continue
            scored.append(
                CandidateScore(
                    person_id=person.id,
                    name=person.name,
                    is_auto_named=True,
                    similarity=float(np.dot(embedding, centroid)),
                    face_count=len(vecs),
                )
            )
        scored.sort(key=lambda c: c.similarity, reverse=True)
        return scored[: self._top_k]

    def _build_verdict(self, diag: FaceDiagnostics) -> str:
        best_named = diag.top_named[0] if diag.top_named else None
        thr = diag.adaptive_threshold

        if best_named is None:
            return (
                "No named-person profiles exist yet, so this face can only be "
                "clustered into an Unknown identity."
            )

        if best_named.similarity >= thr:
            second = diag.top_named[1].similarity if len(diag.top_named) > 1 else 0.0
            margin = best_named.similarity - second
            if margin < diag.margin_required:
                return (
                    f"Best match {best_named.name!r} scores "
                    f"{best_named.similarity:.3f} (≥ threshold {thr:.3f}), but the "
                    f"margin to the runner-up is only {margin:.3f} "
                    f"(< {diag.margin_required:.3f}) — too ambiguous to auto-assign."
                )
            return (
                f"Best match {best_named.name!r} scores "
                f"{best_named.similarity:.3f} ≥ threshold {thr:.3f} with margin "
                f"{margin:.3f} — eligible for automatic recognition."
            )

        return (
            f"Best named match {best_named.name!r} scores "
            f"{best_named.similarity:.3f}, below this face's adaptive threshold "
            f"{thr:.3f} — left for Unknown clustering. The threshold is lowered "
            f"for small/blurry/profile faces (quality="
            f"{diag.quality_score if diag.quality_score is not None else 'n/a'})."
        )
