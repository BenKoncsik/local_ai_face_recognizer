"""Learned person recognition service.

Builds person profiles from already labeled faces and assigns new or currently
unresolved faces to known people when the embedding match is strong enough.

This replaces the old pipeline's DBSCAN-first identity assignment path for the
main workflow. Existing named people are preserved: faces already assigned to a
real, non-protected person are never moved by this service.

Recognition runs in two passes:

Pass 1 — Adaptive threshold
    Each candidate face gets a quality-aware threshold derived from its
    detection confidence, bounding-box area, and aspect ratio (a proxy for
    frontal vs. profile).  Small / blurry / profile faces receive a lower
    threshold so they are not unfairly excluded.

Pass 2 — Same-image context assist
    Faces that survive pass 1 unresolved are retried with a lower threshold
    when the same image already contains at least one manually confirmed
    person.  Matching is restricted to those confirmed persons only, which
    constrains the search and lowers the false-positive risk.
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


@dataclass
class RecognitionStats:
    """Diagnostic counters from a single :meth:`RecognitionService.recognize_pending` run."""

    n_profiles: int = 0           # known persons with usable training examples
    n_candidates: int = 0         # unresolved embedded faces tried
    n_assigned: int = 0           # successfully assigned to known person
    n_below_threshold: int = 0    # rejected: best score < threshold
    n_margin_rejected: int = 0    # rejected: margin between 1st and 2nd too small
    n_no_embedding: int = 0       # skipped: face had no embedding vector
    n_pass2_assigned: int = 0     # extra assignments from same-image context pass


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

    def recognize_pending(self) -> "tuple[List[RecognitionAssignment], RecognitionStats]":
        """Assign unresolved faces to known people when confidence is high.

        Returns:
            Tuple of (assignments, stats) where *assignments* is the list of
            automatic assignments made this run and *stats* holds diagnostic
            counters useful for the pipeline summary.
        """
        profiles = self.build_profiles()
        assignments: List[RecognitionAssignment] = []
        stats = RecognitionStats(n_profiles=len(profiles))

        if not profiles:
            log.info(
                "Recognition skipped: no known people with trusted training examples"
            )
        else:
            candidates = self._load_candidate_faces()
            stats.n_candidates = len(candidates)
            if not candidates:
                log.info("Recognition skipped: no unresolved embedded faces")
            else:
                log.info(
                    "Recognition: %d profile(s), %d candidate face(s)",
                    len(profiles), len(candidates),
                )
                rejected_face_targets = self._load_rejected_face_targets(candidates)
                rejected_person_pairs = self._load_rejected_person_pairs()
                now = _utcnow_naive()

                # --- Pass 1: quality-adaptive threshold per face ---
                unresolved: List[Face] = []
                for face in candidates:
                    threshold = self._compute_adaptive_threshold(face)
                    match, reject_reason = self._match_face(
                        face=face,
                        profiles=profiles,
                        rejected_face_targets=rejected_face_targets,
                        rejected_person_pairs=rejected_person_pairs,
                        threshold=threshold,
                        margin=self._config.min_margin,
                    )
                    if match is not None:
                        self._apply_match(face, match, now)
                        assignments.append(match)
                        stats.n_assigned += 1
                    else:
                        if reject_reason == "no_embedding":
                            stats.n_no_embedding += 1
                        elif reject_reason == "below_threshold":
                            stats.n_below_threshold += 1
                        elif reject_reason == "margin":
                            stats.n_margin_rejected += 1
                        unresolved.append(face)

                # --- Pass 2: same-image context-assisted recognition ---
                if self._config.same_image_assist_enabled and unresolved:
                    n_extra = self._run_assisted_pass(
                        unresolved=unresolved,
                        profiles=profiles,
                        rejected_face_targets=rejected_face_targets,
                        rejected_person_pairs=rejected_person_pairs,
                        now=now,
                        assignments=assignments,
                    )
                    stats.n_pass2_assigned = n_extra
                    stats.n_assigned += n_extra
                    if n_extra:
                        log.info(
                            "Same-image assist: %d additional face(s) assigned", n_extra
                        )

        self._delete_orphan_auto_persons()
        self._session.commit()
        log.info(
            "Recognition summary: %d profile(s), %d candidate(s), "
            "%d assigned (%d pass-2), %d below threshold, %d margin rejected, "
            "%d no embedding",
            stats.n_profiles, stats.n_candidates,
            stats.n_assigned, stats.n_pass2_assigned,
            stats.n_below_threshold, stats.n_margin_rejected,
            stats.n_no_embedding,
        )
        return assignments, stats

    # ------------------------------------------------------------------
    # Profile building
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Pass 1: adaptive threshold
    # ------------------------------------------------------------------

    def _compute_adaptive_threshold(self, face: Face) -> float:
        """Compute a face-specific recognition threshold.

        Smaller / blurrier / more profile-angled faces receive a lower
        threshold so they are not unfairly excluded from recognition.
        The threshold is always clamped to [adaptive_min_threshold, base].
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
    # Pass 2: same-image context assist
    # ------------------------------------------------------------------

    def _run_assisted_pass(
        self,
        *,
        unresolved: List[Face],
        profiles: Dict[int, PersonRecognitionProfile],
        rejected_face_targets: Dict[int, Set[int]],
        rejected_person_pairs: Set[frozenset],
        now: datetime,
        assignments: List[RecognitionAssignment],
    ) -> int:
        """Retry unresolved faces on images that have confirmed labeled faces.

        Matching is restricted to the confirmed persons on each image, which
        lowers the chance of false positives while allowing a lower threshold.

        Returns:
            Number of additional faces assigned.
        """
        image_ids = {f.image_id for f in unresolved if f.image_id is not None}
        if not image_ids:
            return 0

        image_confirmed = self._load_image_confirmed_persons(image_ids)
        if not image_confirmed:
            return 0

        threshold = self._config.same_image_assist_threshold
        margin = self._config.same_image_assist_margin
        n_assigned = 0

        for face in unresolved:
            confirmed_persons = image_confirmed.get(face.image_id, set())
            if len(confirmed_persons) < self._config.same_image_assist_min_confirmed:
                continue

            restricted = {
                pid: p for pid, p in profiles.items() if pid in confirmed_persons
            }
            if not restricted:
                continue

            match, _ = self._match_face(
                face=face,
                profiles=restricted,
                rejected_face_targets=rejected_face_targets,
                rejected_person_pairs=rejected_person_pairs,
                threshold=threshold,
                margin=margin,
            )
            if match is None:
                continue

            log.debug(
                "Face id=%d: assisted pass → %r (score=%.3f, image_id=%d)",
                face.id, match.target_name, match.score, face.image_id,
            )
            self._apply_match(face, match, now)
            assignments.append(match)
            n_assigned += 1

        return n_assigned

    def _load_image_confirmed_persons(
        self, image_ids: Set[int]
    ) -> Dict[int, Set[int]]:
        """Return {image_id: set(person_id)} for manually confirmed faces.

        Only non-auto, non-protected persons assigned via trusted manual
        sources (including legacy NULL-source assignments) are included.
        """
        if not image_ids:
            return {}

        confirmed: List[Face] = (
            self._session.query(Face)
            .join(Person, Face.person_id == Person.id)
            .filter(Face.image_id.in_(image_ids))
            .filter(Face.is_excluded == False)  # noqa: E712
            .filter(Face.person_id.isnot(None))
            .filter(Person.is_auto_named == False)  # noqa: E712
            .filter(Person.is_protected == False)  # noqa: E712
            .filter(
                or_(
                    Face.assignment_source.is_(None),
                    Face.assignment_source.in_(
                        ["manual", "manual_merge", "suggestion_approved"]
                    ),
                )
            )
            .all()
        )

        result: Dict[int, Set[int]] = {}
        for face in confirmed:
            if face.person_id is not None:
                result.setdefault(face.image_id, set()).add(face.person_id)
        return result

    # ------------------------------------------------------------------
    # Core matching
    # ------------------------------------------------------------------

    def _apply_match(
        self, face: Face, match: RecognitionAssignment, now: datetime
    ) -> None:
        face.person_id = match.target_person_id
        face.assignment_source = "recognition"
        face.assignment_confidence = match.score
        face.assigned_at = now

    def _match_face(
        self,
        *,
        face: Face,
        profiles: Dict[int, PersonRecognitionProfile],
        rejected_face_targets: Dict[int, Set[int]],
        rejected_person_pairs: Set[frozenset],
        threshold: float,
        margin: float,
    ) -> "tuple[Optional[RecognitionAssignment], Optional[str]]":
        """Try to match *face* against *profiles*.

        Returns
        -------
        ``(assignment, None)`` on success, or ``(None, reason)`` on failure
        where *reason* is one of ``"no_embedding"``, ``"below_threshold"``,
        ``"margin"``, or ``"no_profiles"``.
        """
        embedding = self._normalise(face.get_embedding())
        if embedding is None:
            log.debug("Face id=%d: no embedding — skipped", face.id)
            return None, "no_embedding"

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
            log.debug("Face id=%d: no eligible profiles after corrections", face.id)
            return None, "no_profiles"

        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_profile = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        actual_margin = best_score - second_score

        if best_score < threshold:
            log.debug(
                "Face id=%d: rejected — score=%.3f < threshold=%.3f "
                "(best=%r, quality=%.2f, bbox=%dx%d)",
                face.id, best_score, threshold, best_profile.name,
                face.quality_score if face.quality_score is not None else -1.0,
                face.bbox_w, face.bbox_h,
            )
            return None, "below_threshold"

        if actual_margin < margin:
            log.debug(
                "Face id=%d: rejected — margin=%.3f < %.3f "
                "(best=%r score=%.3f vs second=%r score=%.3f)",
                face.id, actual_margin, margin,
                best_profile.name, best_score,
                scored[1][1].name if len(scored) > 1 else "none",
                second_score,
            )
            return None, "margin"

        log.debug(
            "Face id=%d: assigned → %r (score=%.3f, margin=%.3f, threshold=%.3f)",
            face.id, best_profile.name, best_score, actual_margin, threshold,
        )
        return RecognitionAssignment(
            face_id=face.id,
            target_person_id=best_profile.person_id,
            target_name=best_profile.name,
            score=best_score,
            margin=actual_margin,
        ), None

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
    # Data loading helpers
    # ------------------------------------------------------------------

    def _load_candidate_faces(self) -> List[Face]:
        """Return embedded faces that are safe for automatic recognition."""
        query = (
            self._session.query(Face)
            .outerjoin(Person, Face.person_id == Person.id)
            .filter(Face.embedding_exists())
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
