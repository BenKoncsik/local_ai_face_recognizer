"""Learned person recognition service.

Builds person profiles from already labeled faces and assigns new or currently
unresolved faces to known people when the embedding match is strong enough.

This replaces the old pipeline's DBSCAN-first identity assignment path for the
main workflow. Existing named people are preserved: faces already assigned to a
real, non-protected person are never moved by this service.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Dict, List, Optional, Set

import numpy as np
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.config import RecognitionConfig
from app.db.models import Face, FaceCorrection, Person

log = logging.getLogger(__name__)

_TRUSTED_MANUAL_SOURCES = {None, "manual", "manual_merge", "suggestion_approved"}


def _utcnow_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


@dataclass
class PersonRecognitionProfile:
    """Embedding profile learned for one known person."""

    person_id: int
    name: str
    centroid: np.ndarray
    examples: np.ndarray
    face_count: int


@dataclass
class RecognitionAssignment:
    """One automatic face-to-person assignment."""

    face_id: int
    target_person_id: int
    target_name: str
    score: float
    margin: float


class RecognitionService:
    """Recognizes unresolved faces from labeled person profiles."""

    def __init__(
        self,
        session: Session,
        config: Optional[RecognitionConfig] = None,
        exclude_low_quality: bool = False,
    ) -> None:
        self._session = session
        self._config = config or RecognitionConfig()
        self._exclude_low_quality = exclude_low_quality

    def recognize_pending(self) -> List[RecognitionAssignment]:
        """Assign unresolved faces to known people when confidence is high.

        Returns:
            Automatic assignments made during this run.
        """
        profiles = self.build_profiles()
        assignments: List[RecognitionAssignment] = []

        if not profiles:
            log.info("Recognition skipped: no known people with trusted examples")
        else:
            candidates = self._load_candidate_faces()
            if not candidates:
                log.info("Recognition skipped: no unresolved embedded faces")
            else:
                rejected_face_targets = self._load_rejected_face_targets(candidates)
                rejected_person_pairs = self._load_rejected_person_pairs()

                now = _utcnow_naive()
                for face in candidates:
                    match = self._match_face(
                        face=face,
                        profiles=profiles,
                        rejected_face_targets=rejected_face_targets,
                        rejected_person_pairs=rejected_person_pairs,
                    )
                    if match is None:
                        continue

                    face.person_id = match.target_person_id
                    face.assignment_source = "recognition"
                    face.assignment_confidence = match.score
                    face.assigned_at = now
                    assignments.append(match)

        self._delete_orphan_auto_persons()
        self._session.commit()
        log.info(
            "Recognition assigned %d unresolved face(s)",
            len(assignments),
        )
        return assignments

    def build_profiles(self) -> Dict[int, PersonRecognitionProfile]:
        """Build recognition profiles for all known non-protected people."""
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

        log.debug("Built %d recognition profile(s)", len(profiles))
        return profiles

    def _match_face(
        self,
        *,
        face: Face,
        profiles: Dict[int, PersonRecognitionProfile],
        rejected_face_targets: Dict[int, Set[int]],
        rejected_person_pairs: Set[frozenset],
    ) -> Optional[RecognitionAssignment]:
        embedding = self._normalise(face.get_embedding())
        if embedding is None:
            return None

        scored: List[tuple[float, PersonRecognitionProfile]] = []
        for profile in profiles.values():
            if profile.person_id in rejected_face_targets.get(face.id, set()):
                continue
            if (
                face.person_id is not None
                and frozenset((face.person_id, profile.person_id))
                in rejected_person_pairs
            ):
                continue
            score = self._score(embedding, profile)
            scored.append((score, profile))

        if not scored:
            return None

        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_profile = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        margin = best_score - second_score

        if best_score < self._config.auto_assign_threshold:
            return None
        if margin < self._config.min_margin:
            return None

        return RecognitionAssignment(
            face_id=face.id,
            target_person_id=best_profile.person_id,
            target_name=best_profile.name,
            score=best_score,
            margin=margin,
        )

    def _score(self, embedding: np.ndarray, profile: PersonRecognitionProfile) -> float:
        centroid_similarity = float(np.dot(embedding, profile.centroid))
        example_similarities = profile.examples @ embedding
        best_example = float(np.max(example_similarities))
        centroid_weight = min(1.0, max(0.0, self._config.centroid_weight))
        return (
            centroid_weight * centroid_similarity
            + (1.0 - centroid_weight) * best_example
        )

    def _load_candidate_faces(self) -> List[Face]:
        """Return embedded faces that are safe for automatic recognition."""
        query = (
            self._session.query(Face)
            .outerjoin(Person, Face.person_id == Person.id)
            .filter(Face._embedding.isnot(None))
            .filter(Face.is_excluded == False)  # noqa: E712
            .filter(
                or_(
                    Face.person_id.is_(None),
                    Person.is_auto_named == True,  # noqa: E712
                    Person.is_protected == True,  # noqa: E712
                )
            )
        )
        if self._exclude_low_quality:
            query = query.filter(
                or_(Face.is_low_quality.is_(None), Face.is_low_quality == False)  # noqa: E712
            )
        return query.all()

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

    def _load_rejected_face_targets(
        self, candidates: List[Face]
    ) -> Dict[int, Set[int]]:
        """Map candidate face IDs to known person IDs they must not match."""
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

        rejected: Dict[int, Set[int]] = {}
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

    def _load_rejected_person_pairs(self) -> Set[frozenset]:
        """Return current person pairs that have a negative correction."""
        corrections: List[FaceCorrection] = (
            self._session.query(FaceCorrection)
            .filter(FaceCorrection.same_person == False)  # noqa: E712
            .all()
        )
        if not corrections:
            return set()

        face_ids = set()
        for correction in corrections:
            face_ids.add(correction.face_id_a)
            face_ids.add(correction.face_id_b)

        person_of: Dict[int, Optional[int]] = {
            face.id: face.person_id
            for face in self._session.query(Face).filter(Face.id.in_(face_ids)).all()
        }

        rejected: Set[frozenset] = set()
        for correction in corrections:
            person_a = person_of.get(correction.face_id_a)
            person_b = person_of.get(correction.face_id_b)
            if person_a is not None and person_b is not None and person_a != person_b:
                rejected.add(frozenset((person_a, person_b)))
        return rejected

    def _delete_orphan_auto_persons(self) -> int:
        orphans: List[Person] = (
            self._session.query(Person)
            .filter(Person.is_auto_named == True)  # noqa: E712
            .filter(~Person.faces.any())
            .all()
        )
        for person in orphans:
            self._session.delete(person)
        return len(orphans)

    @staticmethod
    def _normalise(vector: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if vector is None:
            return None
        vec = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-8:
            return None
        return vec / norm
