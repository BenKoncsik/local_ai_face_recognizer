"""Background person merge-suggestion engine.

Responsibilities
----------------
* Build embedding *profiles* (L2-normalised centroids) for candidate
  (auto-named) and target (named) persons.
* Score candidates against targets combining **face similarity** (dominant)
  with **fuzzy name similarity** into a single confidence — see
  :func:`compute_confidence`.  Scoring is pure / read-only so it can run on a
  thread pool without touching the database.
* Persist suggestions incrementally into :class:`MergeSuggestion` with
  staleness/conflict guards so a long background job never overwrites a fresh
  manual user edit and concurrent chunks never duplicate a pair.
* Provide the accept / reject / defer / dismiss decision API.  Only *accept*
  performs the real :class:`Person` merge (via :class:`IdentityService`).

The class is intentionally split into "pure compute" methods (safe to call
from worker threads on read-only sessions) and "persist/decide" methods
(serialised, called from the owning job thread).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from sqlalchemy.orm import Session, selectinload

from app.config import MatchingConfig
from app.db.models import (
    MERGE_OPEN_STATUSES,
    MERGE_STATUS_DEFERRED,
    MERGE_STATUS_DISMISSED,
    MERGE_STATUS_PENDING,
    MERGE_STATUS_REJECTED,
    MERGE_SUPPRESSED_STATUSES,
    Face,
    FaceCorrection,
    MergeSuggestion,
    Person,
)
from app.services.identity_service import IdentityService
from app.services.name_matching import name_key, name_similarity, phonetic_key

log = logging.getLogger(__name__)

# Confidence blending weights — face similarity always dominates.
_FACE_WEIGHT = 0.75
_NAME_WEIGHT = 0.25
# A name-only signal is capped well below auto-merge grade: a matching name
# with no face evidence must never look like a confident match.
_NAME_ONLY_CAP = 0.5


def _utcnow_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def compute_confidence(
    face_similarity: Optional[float],
    name_similarity: Optional[float],
) -> float:
    """Combine face + name similarity into a single confidence in ``[0, 1]``.

    Rules:
    * Both present  → weighted blend, face dominant.
    * Face only     → the face similarity itself.
    * Name only     → name similarity scaled down by :data:`_NAME_ONLY_CAP`
                      so it can never reach auto-merge-grade confidence.
    * Neither       → 0.
    """
    has_face = face_similarity is not None
    has_name = name_similarity is not None and name_similarity > 0.0
    f = float(face_similarity) if has_face else 0.0
    n = float(name_similarity) if has_name else 0.0

    if has_face and has_name:
        conf = _FACE_WEIGHT * f + _NAME_WEIGHT * n
    elif has_face:
        conf = f
    elif has_name:
        conf = n * _NAME_ONLY_CAP
    else:
        conf = 0.0
    return max(0.0, min(1.0, conf))


@dataclass
class MergeProfile:
    """Read-only embedding/name profile for one person (no live ORM state)."""

    person_id: int
    name: str
    nickname: Optional[str]
    centroid: np.ndarray
    face_count: int
    is_auto_named: bool
    updated_at: Optional[datetime]
    rep_face_id: Optional[int] = None
    rep_crop_path: Optional[str] = None
    rep_image_path: Optional[str] = None
    rep_bbox: Optional[Tuple[int, int, int, int]] = None


@dataclass
class MergeCandidateResult:
    """A scored candidate→target pair produced by pure scoring."""

    candidate_person_id: int
    target_person_id: int
    confidence: float
    face_similarity: Optional[float]
    name_similarity: Optional[float]


@dataclass
class MergeSuggestionDTO:
    """Detached, render-ready view of a persisted suggestion."""

    suggestion_id: int
    status: str
    confidence: float
    face_similarity: Optional[float]
    name_similarity: Optional[float]
    candidate_person_id: int
    candidate_name: str
    candidate_face_count: int
    candidate_crop_path: Optional[str]
    candidate_image_path: Optional[str]
    candidate_bbox: Optional[Tuple[int, int, int, int]]
    target_person_id: int
    target_name: str
    target_face_count: int
    target_crop_path: Optional[str]
    target_image_path: Optional[str]
    target_bbox: Optional[Tuple[int, int, int, int]]


class MergeSuggestionService:
    """Engine + persistence + decision API for background merge suggestions."""

    def __init__(
        self,
        session: Session,
        config: Optional[MatchingConfig] = None,
        exclude_low_quality: bool = False,
    ) -> None:
        self._session = session
        self._config = config or MatchingConfig()
        self._exclude_low_quality = exclude_low_quality

    # ------------------------------------------------------------------
    # Profile loading (read-only)
    # ------------------------------------------------------------------

    # Eager-load faces (and each face's image) in a couple of batched queries
    # instead of lazy-loading them one person/face at a time — profile building
    # is otherwise O(persons) round-trips (a classic N+1) that dominates the
    # job time on large libraries.
    _PROFILE_LOAD_OPTS = (
        selectinload(Person.faces).selectinload(Face.image),
    )

    def load_targets(self) -> List[MergeProfile]:
        """Profiles for every manually-named, non-protected person."""
        persons = (
            self._session.query(Person)
            .filter(Person.is_auto_named == False)  # noqa: E712
            .filter(Person.is_protected == False)  # noqa: E712
            .options(*self._PROFILE_LOAD_OPTS)
            .all()
        )
        return self._profiles_for(persons)

    def load_candidates(
        self, person_ids: Optional[Sequence[int]] = None
    ) -> List[MergeProfile]:
        """Profiles for auto-named candidate persons (optionally restricted)."""
        query = self._session.query(Person).filter(
            Person.is_auto_named == True  # noqa: E712
        )
        if person_ids is not None:
            ids = list(person_ids)
            if not ids:
                return []
            query = query.filter(Person.id.in_(ids))
        return self._profiles_for(query.options(*self._PROFILE_LOAD_OPTS).all())

    def _profiles_for(self, persons: List[Person]) -> List[MergeProfile]:
        profiles: List[MergeProfile] = []
        for person in persons:
            # Deserialise each embedding exactly once (it is a blob decode +
            # copy), keeping it alongside its face so the centroid reuses it.
            usable: List[Face] = []
            embeddings: List[np.ndarray] = []
            for f in person.faces:
                if f.is_excluded:
                    continue
                if self._exclude_low_quality and f.is_low_quality:
                    continue
                emb = f.get_embedding()
                if emb is None:
                    continue
                usable.append(f)
                embeddings.append(emb)
            if not usable:
                continue
            centroid = self._centroid(embeddings)
            rep = self._pick_representative(person, usable)
            rep_bbox = (rep.bbox_x, rep.bbox_y, rep.bbox_w, rep.bbox_h)
            profiles.append(
                MergeProfile(
                    person_id=person.id,
                    name=person.name,
                    nickname=person.nickname,
                    centroid=centroid,
                    face_count=len(usable),
                    is_auto_named=person.is_auto_named,
                    updated_at=person.updated_at,
                    rep_face_id=rep.id,
                    rep_crop_path=rep.crop_path,
                    rep_image_path=rep.image.file_path if rep.image else None,
                    rep_bbox=rep_bbox,
                )
            )
        return profiles

    @staticmethod
    def _centroid(embeddings: List[np.ndarray]) -> np.ndarray:
        matrix = np.vstack([np.asarray(e, dtype=np.float32) for e in embeddings])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        unit = matrix / norms
        mean = unit.mean(axis=0)
        mean_norm = float(np.linalg.norm(mean))
        return mean / mean_norm if mean_norm > 1e-8 else mean

    @staticmethod
    def _pick_representative(person: Person, faces: List[Face]) -> Face:
        if person.thumbnail_path:
            for face in faces:
                if face.crop_path == person.thumbnail_path:
                    return face
        for face in faces:
            if face.crop_path:
                return face
        return faces[0]

    # ------------------------------------------------------------------
    # Suppressed pairs (read-only)
    # ------------------------------------------------------------------

    def load_suppressed_pairs(self) -> Set[frozenset]:
        """Person-id pairs the engine must not re-suggest.

        Combines explicit reject/dismiss decisions on
        :class:`MergeSuggestion` with ``different``
        :class:`FaceCorrection` judgements mapped to their current persons.
        """
        suppressed: Set[frozenset] = set()

        rows = (
            self._session.query(
                MergeSuggestion.source_person_id,
                MergeSuggestion.target_person_id,
            )
            .filter(MergeSuggestion.status.in_(MERGE_SUPPRESSED_STATUSES))
            .all()
        )
        for src, tgt in rows:
            suppressed.add(frozenset((src, tgt)))

        corrections = (
            self._session.query(FaceCorrection)
            .filter(FaceCorrection.same_person == False)  # noqa: E712
            .all()
        )
        if corrections:
            face_ids: Set[int] = set()
            for c in corrections:
                face_ids.add(c.face_id_a)
                face_ids.add(c.face_id_b)
            person_of = {
                f.id: f.person_id
                for f in self._session.query(Face)
                .filter(Face.id.in_(face_ids))
                .all()
            }
            for c in corrections:
                pa = person_of.get(c.face_id_a)
                pb = person_of.get(c.face_id_b)
                if pa is not None and pb is not None and pa != pb:
                    suppressed.add(frozenset((pa, pb)))
        return suppressed

    # ------------------------------------------------------------------
    # Pure scoring (thread-pool safe — no DB access)
    # ------------------------------------------------------------------

    def score_candidates(
        self,
        candidates: Sequence[MergeProfile],
        targets: Sequence[MergeProfile],
        suppressed: Set[frozenset],
    ) -> List[MergeCandidateResult]:
        """Score *candidates* against *targets* — pure, no DB side effects.

        Pre-filtering keeps this well below O(candidates × targets) work:
        face similarity is one vectorised matmul per candidate, and the
        expensive fuzzy name comparison is only computed for a shortlist
        (targets that already clear the face floor, plus name/phonetic-block
        neighbours).
        """
        if not candidates or not targets:
            return []

        target_matrix = np.vstack([t.centroid for t in targets]).astype(np.float32)

        # Name/phonetic blocking index over targets (cheap to build once).
        block: Dict[str, List[int]] = {}
        for idx, t in enumerate(targets):
            for key in {name_key(t.name), phonetic_key(t.name)}:
                if key:
                    block.setdefault(key, []).append(idx)

        cfg = self._config
        face_floor = cfg.name_supported_face_floor
        results: List[MergeCandidateResult] = []

        for cand in candidates:
            face_sims = target_matrix @ cand.centroid  # cosine (unit vectors)

            # Shortlist: targets clearing the face floor …
            shortlist: Set[int] = {
                int(i) for i in np.nonzero(face_sims >= face_floor)[0]
            }
            # … plus name/phonetic-block neighbours (catch strong name matches
            # even when the face signal is weaker).
            for key in {name_key(cand.name), phonetic_key(cand.name)}:
                shortlist.update(block.get(key, ()))
            if not shortlist:
                continue

            scored: List[MergeCandidateResult] = []
            for idx in shortlist:
                target = targets[idx]
                if frozenset((cand.person_id, target.person_id)) in suppressed:
                    continue
                face_sim = float(face_sims[idx])
                nsim = name_similarity(
                    cand.name,
                    target.name,
                    nicknames_a=(cand.nickname,),
                    nicknames_b=(target.nickname,),
                )
                name_sim = nsim if nsim >= cfg.name_threshold else None
                # A name match only counts when there is at least minimal face
                # evidence — never suggest on name alone.
                if name_sim is not None and face_sim < face_floor:
                    name_sim = None

                face_component = face_sim if face_sim >= cfg.face_threshold else None
                if face_component is None and name_sim is None:
                    continue

                confidence = compute_confidence(face_component, name_sim)
                if confidence < cfg.min_confidence:
                    continue
                scored.append(
                    MergeCandidateResult(
                        candidate_person_id=cand.person_id,
                        target_person_id=target.person_id,
                        confidence=confidence,
                        face_similarity=face_component,
                        name_similarity=name_sim,
                    )
                )

            scored.sort(key=lambda r: r.confidence, reverse=True)
            results.extend(scored[: max(1, cfg.max_suggestions_per_person)])

        return results

    # ------------------------------------------------------------------
    # Persistence (serialised — owning job thread only)
    # ------------------------------------------------------------------

    def persist_results(
        self,
        results: Sequence[MergeCandidateResult],
        job_id: str,
        job_started_at: datetime,
    ) -> int:
        """Upsert scored results into :class:`MergeSuggestion`.

        Conflict / staleness guards (requirement: a background job must never
        clobber a fresh manual edit):

        * The candidate must still exist and still be auto-named — if the user
          named or merged it while the job ran, the suggestion is dropped.
        * Either person being modified after ``job_started_at`` means the user
          touched it more recently than this job's snapshot → skip.
        * A pair already decided (accepted/rejected/dismissed) is left intact.

        Writes happen in a single transaction (batched) to avoid
        ``database is locked`` churn under WAL.

        Returns:
            Number of pending suggestions inserted or refreshed.
        """
        if not results:
            return 0

        # Pre-load the persons referenced this batch for cheap validation.
        person_ids: Set[int] = set()
        for r in results:
            person_ids.add(r.candidate_person_id)
            person_ids.add(r.target_person_id)
        persons = {
            p.id: p
            for p in self._session.query(Person)
            .filter(Person.id.in_(person_ids))
            .all()
        }

        # Pre-load existing suggestions for every pair in this batch in one
        # query (avoids a per-result SELECT — an N+1 over the result set).
        existing_by_pair: Dict[Tuple[int, int], MergeSuggestion] = {
            (s.source_person_id, s.target_person_id): s
            for s in self._session.query(MergeSuggestion)
            .filter(
                MergeSuggestion.source_person_id.in_(person_ids),
                MergeSuggestion.target_person_id.in_(person_ids),
            )
            .all()
        }

        n_written = 0
        try:
            for r in results:
                cand = persons.get(r.candidate_person_id)
                tgt = persons.get(r.target_person_id)
                if cand is None or tgt is None:
                    continue
                # User renamed/merged the candidate since the snapshot.
                if not cand.is_auto_named:
                    continue
                if self._is_stale(cand, job_started_at) or self._is_stale(
                    tgt, job_started_at
                ):
                    log.debug(
                        "Skipping stale suggestion %d→%d (person edited after job start)",
                        r.candidate_person_id, r.target_person_id,
                    )
                    continue

                src_id, dst_id = self._normalise_pair(
                    r.candidate_person_id, r.target_person_id
                )
                existing = existing_by_pair.get((src_id, dst_id))
                if existing is not None:
                    # Never resurrect a decided pair; just refresh open ones.
                    if existing.status not in MERGE_OPEN_STATUSES:
                        continue
                    existing.confidence = r.confidence
                    existing.face_similarity = r.face_similarity
                    existing.name_similarity = r.name_similarity
                    existing.status = MERGE_STATUS_PENDING
                    existing.job_id = job_id
                else:
                    new_row = MergeSuggestion(
                        source_person_id=src_id,
                        target_person_id=dst_id,
                        confidence=r.confidence,
                        face_similarity=r.face_similarity,
                        name_similarity=r.name_similarity,
                        status=MERGE_STATUS_PENDING,
                        job_id=job_id,
                    )
                    self._session.add(new_row)
                    # Guard against a repeated pair within this same batch
                    # inserting a duplicate row.
                    existing_by_pair[(src_id, dst_id)] = new_row
                n_written += 1
            self._session.commit()
        except Exception:
            self._session.rollback()
            raise
        return n_written

    @staticmethod
    def _is_stale(person: Person, job_started_at: datetime) -> bool:
        return person.updated_at is not None and person.updated_at > job_started_at

    @staticmethod
    def _normalise_pair(a: int, b: int) -> Tuple[int, int]:
        return (a, b) if a < b else (b, a)

    # ------------------------------------------------------------------
    # Read API for the UI
    # ------------------------------------------------------------------

    def count_open(self) -> int:
        return (
            self._session.query(MergeSuggestion)
            .filter(MergeSuggestion.status == MERGE_STATUS_PENDING)
            .count()
        )

    def list_open(
        self, include_deferred: bool = False, limit: Optional[int] = None
    ) -> List[MergeSuggestionDTO]:
        """Return render-ready pending (and optionally deferred) suggestions."""
        statuses = (
            list(MERGE_OPEN_STATUSES) if include_deferred else [MERGE_STATUS_PENDING]
        )
        query = (
            self._session.query(MergeSuggestion)
            .filter(MergeSuggestion.status.in_(statuses))
            .order_by(MergeSuggestion.confidence.desc())
        )
        if limit:
            query = query.limit(limit)
        rows = query.all()

        # Resolve representative crops for the persons involved.
        person_ids: Set[int] = set()
        for row in rows:
            person_ids.add(row.source_person_id)
            person_ids.add(row.target_person_id)
        reps = self._representatives(person_ids)

        dtos: List[MergeSuggestionDTO] = []
        for row in rows:
            cand_id, tgt_id = self._orient(row)
            cand = reps.get(cand_id)
            tgt = reps.get(tgt_id)
            if cand is None or tgt is None:
                continue
            dtos.append(
                MergeSuggestionDTO(
                    suggestion_id=row.id,
                    status=row.status,
                    confidence=row.confidence,
                    face_similarity=row.face_similarity,
                    name_similarity=row.name_similarity,
                    candidate_person_id=cand_id,
                    candidate_name=cand.name,
                    candidate_face_count=cand.face_count,
                    candidate_crop_path=cand.rep_crop_path,
                    candidate_image_path=cand.rep_image_path,
                    candidate_bbox=cand.rep_bbox,
                    target_person_id=tgt_id,
                    target_name=tgt.name,
                    target_face_count=tgt.face_count,
                    target_crop_path=tgt.rep_crop_path,
                    target_image_path=tgt.rep_image_path,
                    target_bbox=tgt.rep_bbox,
                )
            )
        return dtos

    def _orient(self, row: MergeSuggestion) -> Tuple[int, int]:
        """Return ``(candidate_id, target_id)`` — auto-named side as candidate."""
        src = self._session.get(Person, row.source_person_id)
        if src is not None and src.is_auto_named:
            return row.source_person_id, row.target_person_id
        return row.target_person_id, row.source_person_id

    def _representatives(self, person_ids: Set[int]) -> Dict[int, MergeProfile]:
        if not person_ids:
            return {}
        persons = (
            self._session.query(Person).filter(Person.id.in_(person_ids)).all()
        )
        # Reuse profile building but only for representative metadata.
        return {p.person_id: p for p in self._profiles_for(persons)}

    # ------------------------------------------------------------------
    # Decision API (serialised)
    # ------------------------------------------------------------------

    def accept(self, suggestion_id: int) -> Person:
        """Approve a suggestion → merge candidate into target.

        Merging deletes the candidate person, which CASCADE-deletes this (and
        any other) suggestion referencing it.  The row is expunged first so the
        cascade can't trigger a stale UPDATE on flush.
        """
        row = self._require(suggestion_id)
        cand_id, tgt_id = self._orient(row)
        log.info("Accepting merge suggestion %d: %d → %d", suggestion_id, cand_id, tgt_id)
        self._session.expunge(row)
        return IdentityService(self._session).merge_persons(
            source_id=cand_id, target_id=tgt_id
        )

    def reject(self, suggestion_id: int) -> None:
        """Reject this specific suggestion (records a 'different' correction)."""
        self._decide(suggestion_id, MERGE_STATUS_REJECTED)
        row = self._session.get(MergeSuggestion, suggestion_id)
        if row is not None:
            self._record_different(row.source_person_id, row.target_person_id)

    def defer(self, suggestion_id: int) -> None:
        """Mark 'decide later' — hidden from the default pending list."""
        self._decide(suggestion_id, MERGE_STATUS_DEFERRED)

    def dismiss(self, suggestion_id: int) -> None:
        """'Never suggest this pair again' — permanently suppressed."""
        self._decide(suggestion_id, MERGE_STATUS_DISMISSED)
        row = self._session.get(MergeSuggestion, suggestion_id)
        if row is not None:
            self._record_different(row.source_person_id, row.target_person_id)

    def auto_merge_suggestions(
        self,
        min_confidence: Optional[float] = None,
        max_unknown_faces: Optional[int] = None,
    ) -> int:
        """Automatically merge high-confidence suggestions for small unknown persons.

        Args:
            min_confidence: Minimum confidence threshold (defaults to config).
            max_unknown_faces: Maximum faces in unknown person (defaults to config).

        Returns:
            Number of suggestions that were automatically merged.
        """
        if min_confidence is None:
            min_confidence = self._config.auto_merge_min_confidence
        if max_unknown_faces is None:
            max_unknown_faces = self._config.auto_merge_max_unknown_faces

        # Query all pending suggestions sorted by confidence (highest first)
        rows = (
            self._session.query(MergeSuggestion)
            .filter(MergeSuggestion.status == MERGE_STATUS_PENDING)
            .order_by(MergeSuggestion.confidence.desc())
            .all()
        )

        merged_count = 0
        for row in rows:
            # Only auto-merge if confidence meets threshold
            if row.confidence < min_confidence:
                continue

            # Determine which side is the auto-named (unknown) person
            src = self._session.get(Person, row.source_person_id)
            tgt = self._session.get(Person, row.target_person_id)

            if src is None or tgt is None:
                continue

            # The auto-named person is the candidate; check face count
            if src.is_auto_named and len(src.faces) <= max_unknown_faces:
                candidate_id = row.source_person_id
                target_id = row.target_person_id
            elif tgt.is_auto_named and len(tgt.faces) <= max_unknown_faces:
                candidate_id = row.target_person_id
                target_id = row.source_person_id
            else:
                # Neither side qualifies for auto-merge
                continue

            # Accept this suggestion (merges the persons)
            try:
                log.info(
                    "Auto-merging suggestion %d: %d → %d (confidence=%.2f)",
                    row.id,
                    candidate_id,
                    target_id,
                    row.confidence,
                )
                # The row will be cascade-deleted when the candidate person is deleted
                self._session.expunge(row)
                IdentityService(self._session).merge_persons(
                    source_id=candidate_id, target_id=target_id
                )
                merged_count += 1
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Failed to auto-merge suggestion %d: %s", row.id, exc
                )
                # Continue with the next suggestion
                self._session.rollback()

        return merged_count

    def _decide(self, suggestion_id: int, status: str) -> None:
        row = self._require(suggestion_id)
        row.status = status
        row.decided_at = _utcnow_naive()
        self._session.commit()
        log.info("Merge suggestion %d → %s", suggestion_id, status)

    def _record_different(self, person_a: int, person_b: int) -> None:
        """Persist a few 'different person' face corrections to back a rejection."""
        identity = IdentityService(self._session)
        faces_a = self._sample_face_ids(person_a)
        faces_b = self._sample_face_ids(person_b)
        for fa in faces_a:
            for fb in faces_b:
                identity.record_different(fa, fb)

    def _sample_face_ids(self, person_id: int, limit: int = 3) -> List[int]:
        rows = (
            self._session.query(Face.id)
            .filter(Face.person_id == person_id)
            .filter(Face.is_excluded == False)  # noqa: E712
            .limit(limit)
            .all()
        )
        return [r[0] for r in rows]

    def _require(self, suggestion_id: int) -> MergeSuggestion:
        row = self._session.get(MergeSuggestion, suggestion_id)
        if row is None:
            raise ValueError(f"MergeSuggestion id={suggestion_id} not found")
        return row
