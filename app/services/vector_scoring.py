"""Embedding (vector) scoring helpers shared across the app.

This module holds the cosine-similarity scoring primitives that several
features need *without* being the (now removed) classic recognition pipeline:

* the person-selector popups order candidates by match score
  (:mod:`app.services.match_scoring`),
* the reviewable Unknown auto-merge ranks faces against known people
  (:mod:`app.services.unknown_merge_service`),
* the face diagnostics view explains why a face does/doesn't match
  (:mod:`app.services.face_diagnostics_service`).

It builds an embedding *profile* per known person from their trusted training
faces and scores a probe embedding against those profiles.  It never assigns
faces or mutates the database — it is pure read-only scoring.  Actual identity
assignment is the job of the deep-learning recognition engine
(:mod:`app.services.deep_recognition_service`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.config import RecognitionConfig
from app.db.models import Face, FaceCorrection, Person

log = logging.getLogger(__name__)

# Assignment sources that mean a human (or trusted legacy data) made the call;
# such faces are always eligible as training examples for a person profile.
_TRUSTED_MANUAL_SOURCES = {None, "manual", "manual_merge", "suggestion_approved"}


@dataclass
class PersonRecognitionProfile:
    """Embedding profile learned for one known person."""

    person_id: int
    name: str
    centroid: np.ndarray
    examples: np.ndarray
    face_count: int


class FaceVectorScorer:
    """Score embeddings against known-person profiles (read-only).

    The profiles mirror what automatic recognition would compare against
    (non-protected, non-auto-named people with enough trusted training faces),
    so scores produced here are consistent with how the app groups identities.
    """

    def __init__(
        self,
        session: Session,
        config: Optional[RecognitionConfig] = None,
        exclude_low_quality: bool = False,
    ) -> None:
        self._session = session
        self._config = config or RecognitionConfig()
        self._exclude_low_quality = exclude_low_quality

    # ------------------------------------------------------------------
    # Profile building
    # ------------------------------------------------------------------

    def build_profiles(self) -> Dict[int, PersonRecognitionProfile]:
        """Build scoring profiles for all known non-protected people."""
        persons: List[Person] = (
            self._session.query(Person)
            .filter(Person.is_auto_named == False)  # noqa: E712
            .filter(Person.is_protected == False)  # noqa: E712
            .all()
        )

        profiles: Dict[int, PersonRecognitionProfile] = {}
        for person in persons:
            vectors = [
                self._normalise(face.get_embedding())
                for face in person.faces
                if self._is_training_face(face)
            ]
            vectors = [v for v in vectors if v is not None]
            if len(vectors) < max(1, self._config.min_examples_per_person):
                continue

            examples = np.vstack(vectors).astype(np.float32)
            centroid = self._normalise(examples.mean(axis=0))
            if centroid is None:
                continue

            profiles[person.id] = PersonRecognitionProfile(
                person_id=person.id,
                name=person.name,
                centroid=centroid,
                examples=examples,
                face_count=len(vectors),
            )

        log.debug("Built %d scoring profile(s)", len(profiles))
        return profiles

    def _is_training_face(self, face: Face) -> bool:
        if face.is_excluded or face.get_embedding() is None:
            return False
        if face.person is None or face.person.is_auto_named or face.person.is_protected:
            return False

        if face.assignment_source in _TRUSTED_MANUAL_SOURCES:
            # Manually assigned faces bypass quality filtering — the user
            # explicitly chose this face as a training example.
            return True

        # Quality filter applies to auto-recognised faces only.
        if self._exclude_low_quality and face.is_low_quality:
            return False

        if face.assignment_source != "recognition":
            return False

        if not self._config.use_recognized_faces_for_training:
            return False

        confidence = face.assignment_confidence
        return (
            confidence is not None
            and confidence >= self._config.profile_auto_min_confidence
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_persons(self, embedding: Optional[np.ndarray]) -> Dict[int, float]:
        """Score *embedding* against every known person's profile.

        Returns a mapping of ``person_id`` → similarity score (higher is a
        closer match), restricted to the same non-protected, non-auto-named
        people that automatic recognition considers.  Persons without enough
        trusted training examples to build a profile are simply absent from the
        result.  An empty mapping is returned when *embedding* is missing or
        cannot be normalised, so callers can safely fall back to the default
        ordering without special-casing ``None``.
        """
        normalised = self._normalise(embedding)
        if normalised is None:
            return {}
        return {
            person_id: self._score(normalised, profile)
            for person_id, profile in self.build_profiles().items()
        }

    def rank_persons(
        self,
        embedding: Optional[np.ndarray],
        profiles: Optional[Dict[int, PersonRecognitionProfile]] = None,
    ) -> "List[tuple[int, str, float]]":
        """Rank known people for *embedding*, best match first.

        Unlike :meth:`score_persons`, the caller may pass a pre-built *profiles*
        mapping so a batch of faces can be scored without rebuilding profiles for
        every face (re-recognition scores many faces against the same people).
        When *profiles* is omitted it is built once via :meth:`build_profiles`.

        Returns a list of ``(person_id, name, score)`` sorted by descending
        score.  An empty list is returned when *embedding* is missing/unusable or
        no profiles exist, so callers can treat "no match" uniformly.
        """
        normalised = self._normalise(embedding)
        if normalised is None:
            return []
        if profiles is None:
            profiles = self.build_profiles()
        ranked = [
            (profile.person_id, profile.name, self._score(normalised, profile))
            for profile in profiles.values()
        ]
        ranked.sort(key=lambda item: item[2], reverse=True)
        return ranked

    def _score(self, embedding: np.ndarray, profile: PersonRecognitionProfile) -> float:
        centroid_similarity = float(np.dot(embedding, profile.centroid))
        example_similarities = profile.examples @ embedding
        best_example = float(np.max(example_similarities))
        centroid_weight = min(1.0, max(0.0, self._config.centroid_weight))
        return (
            centroid_weight * centroid_similarity
            + (1.0 - centroid_weight) * best_example
        )

    # ------------------------------------------------------------------
    # Quality-adaptive threshold (used by the diagnostics view)
    # ------------------------------------------------------------------

    def _compute_adaptive_threshold(self, face: Face) -> float:
        """Compute a face-specific score threshold.

        Smaller / blurrier / more profile-angled faces receive a lower
        threshold so they are not unfairly excluded.  The threshold is always
        clamped to ``[adaptive_min_threshold, auto_assign_threshold]``.
        """
        base = self._config.auto_assign_threshold
        if not self._config.adaptive_threshold_enabled:
            return base

        minimum = self._config.adaptive_min_threshold
        if minimum >= base:
            return base

        # Quality score [0, 1]: 0 = terrible, 1 = perfect
        quality = float(face.quality_score) if face.quality_score is not None else 0.70

        # Face area: larger → embedding model has more pixels to work with
        area = face.bbox_w * face.bbox_h
        size_factor = min(1.0, area / (80 * 80))

        # Aspect ratio: a narrow face is more likely to be a profile view
        if face.bbox_h > 0 and face.bbox_w > 0:
            ratio = face.bbox_w / face.bbox_h
            frontality = min(1.0, ratio / 0.75)
        else:
            frontality = 1.0

        combined = quality * 0.5 + size_factor * 0.3 + frontality * 0.2
        combined = max(0.0, min(1.0, combined))

        threshold = minimum + (base - minimum) * combined
        log.debug(
            "Face id=%d: adaptive threshold=%.3f "
            "(quality=%.2f size_factor=%.2f frontality=%.2f)",
            face.id, threshold, quality, size_factor, frontality,
        )
        return threshold

    # ------------------------------------------------------------------
    # Correction-aware helpers (used by re-recognition)
    # ------------------------------------------------------------------

    def _load_rejected_face_targets(
        self, candidates: List[Face]
    ) -> Dict[int, "set[int]"]:
        """Map candidate face IDs to known person IDs they must not match.

        Built from negative ("not the same person") face corrections so a probe
        face is never re-matched to a person the user explicitly rejected.
        """
        candidate_ids = {face.id for face in candidates}
        if not candidate_ids:
            return {}

        corrections: List[FaceCorrection] = (
            self._session.query(FaceCorrection)
            .filter(FaceCorrection.same_person == False)  # noqa: E712
            .filter(
                or_(
                    FaceCorrection.face_id_a.in_(candidate_ids),
                    FaceCorrection.face_id_b.in_(candidate_ids),
                )
            )
            .all()
        )
        if not corrections:
            return {}

        other_face_ids = set()
        for correction in corrections:
            if correction.face_id_a in candidate_ids:
                other_face_ids.add(correction.face_id_b)
            if correction.face_id_b in candidate_ids:
                other_face_ids.add(correction.face_id_a)

        person_of: Dict[int, Optional[int]] = {
            face.id: face.person_id
            for face in self._session.query(Face)
            .filter(Face.id.in_(other_face_ids))
            .all()
        }

        rejected: Dict[int, "set[int]"] = {}
        for correction in corrections:
            if correction.face_id_a in candidate_ids:
                target_person_id = person_of.get(correction.face_id_b)
                if target_person_id is not None:
                    rejected.setdefault(correction.face_id_a, set()).add(target_person_id)
            if correction.face_id_b in candidate_ids:
                target_person_id = person_of.get(correction.face_id_a)
                if target_person_id is not None:
                    rejected.setdefault(correction.face_id_b, set()).add(target_person_id)
        return rejected

    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(vector: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if vector is None:
            return None
        vec = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-8:
            return None
        return vec / norm
