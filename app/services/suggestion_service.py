"""Name-suggestion service.

Compares automatically-named ("Unknown N") persons against the embedding
profiles of manually-named persons and proposes likely identity matches.

Design
------
* For every manually-named person a representative embedding *profile* is
  built: the L2-normalised centroid of all that person's non-excluded,
  embedded face vectors.
* The same profile is built for every auto-named candidate person.
* Cosine similarity between candidate and named profiles drives the
  suggestions; only matches above a configurable threshold are returned.
* Nothing is merged automatically — :meth:`approve` and :meth:`reject`
  must be called explicitly by the UI in response to a user decision.
* A rejection records a ``different`` :class:`~app.db.models.FaceCorrection`
  so the same pair is not suggested again.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np
from sqlalchemy.orm import Session

from app.config import SuggestionConfig
from app.db.models import Face, FaceCorrection, Person
from app.services.identity_service import IdentityService

log = logging.getLogger(__name__)


@dataclass
class PersonProfile:
    """A representative embedding profile for one person."""

    person_id: int
    name: str
    centroid: np.ndarray
    face_count: int
    rep_face_id: int
    rep_crop_path: Optional[str]


@dataclass
class Suggestion:
    """A proposed match: an unknown person likely equals a named person.

    All fields are plain values so the suggestion survives after the
    SQLAlchemy session that produced it has closed.
    """

    candidate_person_id: int
    candidate_name: str
    candidate_face_id: int
    candidate_crop_path: Optional[str]
    candidate_face_count: int
    target_person_id: int
    target_name: str
    target_face_id: int
    target_crop_path: Optional[str]
    target_face_count: int
    similarity: float


class SuggestionService:
    """Builds person profiles and proposes Unknown → Known matches.

    Args:
        session: SQLAlchemy session.
        config:  Suggestion configuration block (defaults used if omitted).
    """

    def __init__(
        self, session: Session, config: Optional[SuggestionConfig] = None
    ) -> None:
        self._session = session
        self._config = config or SuggestionConfig()

    # ------------------------------------------------------------------
    # Centroid
    # ------------------------------------------------------------------

    @staticmethod
    def centroid(embeddings: List[np.ndarray]) -> np.ndarray:
        """Return the L2-normalised mean of L2-normalised embedding vectors.

        Each input vector is normalised first so a single face with an
        unusually large magnitude cannot dominate the average.

        Raises:
            ValueError: if *embeddings* is empty.
        """
        if not embeddings:
            raise ValueError("Cannot compute a centroid from zero embeddings")

        matrix = np.vstack(
            [np.asarray(e, dtype=np.float32) for e in embeddings]
        )
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        unit = matrix / norms

        mean = unit.mean(axis=0)
        mean_norm = float(np.linalg.norm(mean))
        return mean / mean_norm if mean_norm > 1e-8 else mean

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_named_profiles(self) -> Dict[int, PersonProfile]:
        """Return embedding profiles for every manually-named person."""
        return self._build_profiles(named=True)

    def generate_suggestions(
        self, threshold: Optional[float] = None
    ) -> List[Suggestion]:
        """Propose likely Unknown → Known matches, ranked by similarity.

        Args:
            threshold: Optional override for the configured similarity
                threshold (used by the UI for live re-filtering).

        Returns:
            Suggestions sorted by descending similarity.  Each unknown
            person yields at most ``max_suggestions_per_person`` entries.
        """
        named = self._build_profiles(named=True)
        if not named:
            log.info("No manually-named persons with embeddings — no suggestions")
            return []

        candidates = self._build_profiles(named=False)
        if not candidates:
            log.info("No auto-named candidate persons with embeddings")
            return []

        if threshold is None:
            threshold = self._config.similarity_threshold
        max_per = max(1, self._config.max_suggestions_per_person)
        rejected = self._load_rejected_pairs()

        suggestions: List[Suggestion] = []
        for cand in candidates.values():
            scored: List[tuple[float, PersonProfile]] = []
            for prof in named.values():
                if frozenset((cand.person_id, prof.person_id)) in rejected:
                    continue
                similarity = float(np.dot(cand.centroid, prof.centroid))
                if similarity >= threshold:
                    scored.append((similarity, prof))

            scored.sort(key=lambda item: item[0], reverse=True)
            for similarity, prof in scored[:max_per]:
                suggestions.append(
                    Suggestion(
                        candidate_person_id=cand.person_id,
                        candidate_name=cand.name,
                        candidate_face_id=cand.rep_face_id,
                        candidate_crop_path=cand.rep_crop_path,
                        candidate_face_count=cand.face_count,
                        target_person_id=prof.person_id,
                        target_name=prof.name,
                        target_face_id=prof.rep_face_id,
                        target_crop_path=prof.rep_crop_path,
                        target_face_count=prof.face_count,
                        similarity=similarity,
                    )
                )

        suggestions.sort(key=lambda s: s.similarity, reverse=True)
        log.info("Generated %d name suggestion(s)", len(suggestions))
        return suggestions

    def count_suggestions(self) -> int:
        """Return how many suggestions the current data would produce."""
        return len(self.generate_suggestions())

    def approve(self, candidate_person_id: int, target_person_id: int) -> Person:
        """Confirm a suggestion: merge the unknown person into the named one.

        All faces of the unknown person are reassigned to the named person
        and the unknown person row is deleted.
        """
        log.info(
            "Approving suggestion: merging person %d into %d",
            candidate_person_id, target_person_id,
        )
        return IdentityService(self._session).merge_persons(
            source_id=candidate_person_id, target_id=target_person_id
        )

    def reject(self, candidate_person_id: int, target_person_id: int) -> None:
        """Reject a suggestion and record it as a 'different person' decision.

        A small sample of face pairs between the two persons is recorded as
        :class:`~app.db.models.FaceCorrection` rows with ``same_person=False``
        so the same pair is not proposed again.
        """
        candidate_faces = self._sample_face_ids(candidate_person_id)
        target_faces = self._sample_face_ids(target_person_id)
        if not candidate_faces or not target_faces:
            log.warning(
                "Cannot record rejection — missing faces for person %d / %d",
                candidate_person_id, target_person_id,
            )
            return

        identity = IdentityService(self._session)
        for face_a in candidate_faces:
            for face_b in target_faces:
                identity.record_different(face_a, face_b)

        log.info(
            "Rejected suggestion: person %d marked different from %d",
            candidate_person_id, target_person_id,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_profiles(self, *, named: bool) -> Dict[int, PersonProfile]:
        """Build embedding profiles for named or auto-named persons.

        Persons with no non-excluded, embedded face are skipped — a profile
        cannot be computed for them.
        """
        want_auto_named = not named
        persons: List[Person] = (
            self._session.query(Person)
            .filter(Person.is_auto_named == want_auto_named)
            .all()
        )
        if named:
            persons = [p for p in persons if not p.is_protected]

        profiles: Dict[int, PersonProfile] = {}
        for person in persons:
            usable = [
                f
                for f in person.faces
                if not f.is_excluded and f.get_embedding() is not None
            ]
            if not usable:
                continue

            embeddings = [f.get_embedding() for f in usable]
            representative = self._pick_representative(person, usable)
            profiles[person.id] = PersonProfile(
                person_id=person.id,
                name=person.name,
                centroid=self.centroid(embeddings),
                face_count=len(usable),
                rep_face_id=representative.id,
                rep_crop_path=representative.crop_path,
            )

        return profiles

    @staticmethod
    def _pick_representative(person: Person, faces: List[Face]) -> Face:
        """Choose a face to visually stand for *person*."""
        if person.thumbnail_path:
            for face in faces:
                if face.crop_path == person.thumbnail_path:
                    return face
        for face in faces:
            if face.crop_path:
                return face
        return faces[0]

    def _sample_face_ids(self, person_id: int, limit: int = 5) -> List[int]:
        """Return up to *limit* non-excluded face IDs for a person."""
        faces = (
            self._session.query(Face)
            .filter(Face.person_id == person_id)
            .filter(Face.is_excluded == False)  # noqa: E712
            .limit(limit)
            .all()
        )
        return [f.id for f in faces]

    def _load_rejected_pairs(self) -> Set[frozenset]:
        """Return person-id pairs already marked as 'different person'.

        Each ``same_person=False`` correction links two faces; both faces
        are mapped to their current person so a rejection keeps suppressing
        the suggestion even if individual faces are reassigned later.
        """
        corrections: List[FaceCorrection] = (
            self._session.query(FaceCorrection)
            .filter(FaceCorrection.same_person == False)  # noqa: E712
            .all()
        )
        if not corrections:
            return set()

        face_ids: Set[int] = set()
        for correction in corrections:
            face_ids.add(correction.face_id_a)
            face_ids.add(correction.face_id_b)

        person_of: Dict[int, Optional[int]] = {
            f.id: f.person_id
            for f in self._session.query(Face).filter(Face.id.in_(face_ids)).all()
        }

        rejected: Set[frozenset] = set()
        for correction in corrections:
            person_a = person_of.get(correction.face_id_a)
            person_b = person_of.get(correction.face_id_b)
            if person_a is not None and person_b is not None and person_a != person_b:
                rejected.add(frozenset((person_a, person_b)))
        return rejected
