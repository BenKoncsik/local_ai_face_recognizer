"""Identity Repair Scan.

Consolidates *fragmented* identities: the situation where one real person has
been split across several auto-named ("Unknown N") persons because incremental
clustering kept spawning fresh clusters at the threshold boundary over many
pipeline runs.

Where the intra-image consistency pass
(:mod:`app.services.intra_image_consistency_service`) heals splits *within one
photo*, this scan works *globally* across the whole database, comparing every
Unknown person to every other and proposing merges for those whose embeddings
say they are the same individual.

Output is a ranked list of :class:`RepairCandidate` proposals
(``"Unknown 98 ≈ Unknown 155 (92%)"``).  Nothing is merged until
:meth:`IdentityRepairService.apply` is called with the user-approved pairs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sqlalchemy.orm import Session

from app.config import IdentityRepairConfig
from app.db.models import Face, FaceCorrection, Person
from app.db.query_utils import in_chunks
from app.services.identity_service import IdentityService

log = logging.getLogger(__name__)


@dataclass
class _RepairProfile:
    """Embedding profile of one Unknown person used during the scan."""

    person_id: int
    name: str
    centroid: np.ndarray          # L2-normalised centroid
    faces: np.ndarray             # (n, dim) L2-normalised face matrix
    face_count: int
    rep_crop_path: Optional[str]


@dataclass
class RepairCandidate:
    """One proposed merge between two fragmented Unknown identities."""

    person_a_id: int
    person_b_id: int
    name_a: str
    name_b: str
    centroid_similarity: float
    best_pair_similarity: float
    face_count_a: int
    face_count_b: int
    rep_crop_a: Optional[str]
    rep_crop_b: Optional[str]

    @property
    def confidence(self) -> float:
        """Blended 0–1 score (centroid dominant, individual pair supporting)."""
        return 0.7 * self.centroid_similarity + 0.3 * self.best_pair_similarity


@dataclass
class RepairApplyResult:
    """Outcome of applying approved merges."""

    requested_pairs: int = 0
    persons_merged_away: int = 0
    groups_consolidated: int = 0
    surviving_person_ids: Tuple[int, ...] = ()


class IdentityRepairService:
    """Finds and consolidates fragmented Unknown identities across the DB."""

    def __init__(
        self,
        session: Session,
        config: Optional[IdentityRepairConfig] = None,
        exclude_low_quality: bool = False,
    ) -> None:
        self._session = session
        self._config = config or IdentityRepairConfig()
        self._exclude_low_quality = exclude_low_quality

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def scan(self) -> List[RepairCandidate]:
        """Return ranked merge proposals between Unknown persons.

        Each unordered pair is reported at most once; the highest-confidence
        proposals come first.
        """
        profiles = self._load_profiles()
        if len(profiles) < 2:
            log.info("Identity repair: %d Unknown profile(s) — nothing to scan",
                     len(profiles))
            return []

        suppressed = self._load_suppressed_pairs()

        # Pre-stack centroids for a single matrix multiply.
        ids = [p.person_id for p in profiles]
        centroids = np.vstack([p.centroid for p in profiles]).astype(np.float32)
        sim_matrix = centroids @ centroids.T  # (n, n) cosine similarities

        cfg = self._config
        by_id = {p.person_id: p for p in profiles}
        # Per person, keep only its top-N partners so big DBs stay manageable.
        per_person: Dict[int, List[RepairCandidate]] = {pid: [] for pid in ids}
        seen: Set[frozenset] = set()

        for i in range(len(profiles)):
            for j in range(i + 1, len(profiles)):
                centroid_sim = float(sim_matrix[i, j])
                if centroid_sim < cfg.merge_similarity:
                    continue
                pa, pb = profiles[i], profiles[j]
                if frozenset((pa.person_id, pb.person_id)) in suppressed:
                    continue

                pair_sim = self._best_pair_similarity(pa, pb)
                if pair_sim < cfg.min_pair_similarity:
                    continue

                cand = RepairCandidate(
                    person_a_id=pa.person_id,
                    person_b_id=pb.person_id,
                    name_a=pa.name,
                    name_b=pb.name,
                    centroid_similarity=centroid_sim,
                    best_pair_similarity=pair_sim,
                    face_count_a=pa.face_count,
                    face_count_b=pb.face_count,
                    rep_crop_a=pa.rep_crop_path,
                    rep_crop_b=pb.rep_crop_path,
                )
                key = frozenset((pa.person_id, pb.person_id))
                if key in seen:
                    continue
                seen.add(key)
                per_person[pa.person_id].append(cand)
                per_person[pb.person_id].append(cand)

        # Cap per person, then flatten + de-duplicate + rank globally.
        kept: Set[frozenset] = set()
        results: List[RepairCandidate] = []
        for pid in ids:
            cands = sorted(
                per_person[pid], key=lambda c: c.confidence, reverse=True
            )[: cfg.max_candidates_per_person]
            for c in cands:
                key = frozenset((c.person_a_id, c.person_b_id))
                if key in kept:
                    continue
                kept.add(key)
                results.append(c)

        results.sort(key=lambda c: c.confidence, reverse=True)
        log.info(
            "Identity repair scan: %d Unknown profile(s) → %d merge candidate(s)",
            len(profiles), len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Applying
    # ------------------------------------------------------------------

    def apply(self, pairs: List[Tuple[int, int]]) -> RepairApplyResult:
        """Merge the approved person pairs, consolidating transitive groups.

        Pairs that share a member are unioned so ``(98,155)`` and ``(155,184)``
        collapse into a single surviving identity.  Within each group the
        person with the most faces survives (ties broken by lowest id).
        """
        result = RepairApplyResult(requested_pairs=len(pairs))
        if not pairs:
            return result

        # Union-find over the approved pairs.
        parent: Dict[int, int] = {}

        def find(x: int) -> int:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for a, b in pairs:
            union(a, b)

        groups: Dict[int, List[int]] = {}
        for node in parent:
            groups.setdefault(find(node), []).append(node)

        identity_svc = IdentityService(self._session)
        survivors: List[int] = []

        for members in groups.values():
            persons = [self._session.get(Person, pid) for pid in members]
            persons = [p for p in persons if p is not None and p.is_auto_named]
            if len(persons) < 2:
                continue
            # Survivor = most faces, tie → lowest id.
            winner = max(persons, key=lambda p: (len(p.faces), -p.id))
            for loser in persons:
                if loser.id == winner.id:
                    continue
                identity_svc.merge_persons(source_id=loser.id, target_id=winner.id)
                result.persons_merged_away += 1
            result.groups_consolidated += 1
            survivors.append(winner.id)

        result.surviving_person_ids = tuple(survivors)
        log.info(
            "Identity repair applied: %d group(s) consolidated, %d person(s) merged away",
            result.groups_consolidated, result.persons_merged_away,
        )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_profiles(self) -> List[_RepairProfile]:
        persons: List[Person] = (
            self._session.query(Person)
            .filter(Person.is_auto_named == True)  # noqa: E712
            .filter(Person.is_protected == False)  # noqa: E712
            .all()
        )

        profiles: List[_RepairProfile] = []
        for person in persons:
            vectors: List[np.ndarray] = []
            rep_crop: Optional[str] = None
            for face in person.faces:
                if face.is_excluded:
                    continue
                if self._exclude_low_quality and face.is_low_quality:
                    continue
                emb = face.get_embedding()
                if emb is None:
                    continue
                vec = self._normalise(emb)
                if vec is None:
                    continue
                vectors.append(vec)
                if rep_crop is None and face.crop_path:
                    rep_crop = face.crop_path
            if not vectors:
                continue
            faces = np.vstack(vectors).astype(np.float32)
            centroid = self._normalise(faces.mean(axis=0))
            if centroid is None:
                continue
            profiles.append(
                _RepairProfile(
                    person_id=person.id,
                    name=person.name,
                    centroid=centroid,
                    faces=faces,
                    face_count=len(vectors),
                    rep_crop_path=rep_crop,
                )
            )
        return profiles

    @staticmethod
    def _best_pair_similarity(a: _RepairProfile, b: _RepairProfile) -> float:
        """Highest cosine similarity between any face of *a* and any of *b*."""
        sims = a.faces @ b.faces.T
        return float(np.max(sims))

    def _load_suppressed_pairs(self) -> Set[frozenset]:
        """Person pairs the user marked as different (via face corrections)."""
        corrections: List[FaceCorrection] = (
            self._session.query(FaceCorrection)
            .filter(FaceCorrection.same_person == False)  # noqa: E712
            .all()
        )
        if not corrections:
            return set()

        face_ids: Set[int] = set()
        for c in corrections:
            face_ids.add(c.face_id_a)
            face_ids.add(c.face_id_b)
        person_of: Dict[int, Optional[int]] = {}
        for chunk in in_chunks(list(face_ids)):
            for f in self._session.query(Face).filter(Face.id.in_(chunk)).all():
                person_of[f.id] = f.person_id
        suppressed: Set[frozenset] = set()
        for c in corrections:
            pa = person_of.get(c.face_id_a)
            pb = person_of.get(c.face_id_b)
            if pa is not None and pb is not None and pa != pb:
                suppressed.add(frozenset((pa, pb)))
        return suppressed

    @staticmethod
    def _normalise(vector: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if vector is None:
            return None
        vec = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-8:
            return None
        return vec / norm
