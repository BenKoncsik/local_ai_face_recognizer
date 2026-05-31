"""Same-image identity consistency pass.

Heals intra-image identity fragmentation: the bug where two detections of the
*same* person on the *same* photo end up under different identities because one
face landed just inside a clustering threshold and a sibling face landed just
outside it.

Why a dedicated pass?
---------------------
The incremental clusterer assigns each unassigned face independently against a
hard cosine-distance threshold (see
:meth:`app.services.clustering_service.ClusteringService.cluster_unassigned`).
That makes the outcome order-dependent and brittle at the threshold boundary:

    face A → Unknown 98   (distance 0.44, accepted)
    face B → Unknown 155  (distance 0.46, rejected → new cluster)

Both faces are the same person on the same image.  This pass runs *after*
clustering and re-unifies such faces using a single, symmetric rule: faces on
one image whose embeddings are mutually near-identical must share one identity.

Safety
------
* Two genuinely different people who merely co-occur are never merged — the
  similarity bar (:attr:`IntraImageConsistencyConfig.merge_similarity`) is
  deliberately stricter than the clustering boundary.
* A user "not the same person" correction (:class:`FaceCorrection` with
  ``same_person=False``) always blocks a merge.
* If a connected component spans two *different manually-named* persons we treat
  it as a real signal (two distinct identities in one frame) and skip it — we
  never silently fold one named person into another.
* Only faces that are unassigned or attached to an auto-named ("Unknown")
  person are ever moved.  Manually named and protected faces are anchors, never
  victims.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.config import IntraImageConsistencyConfig
from app.db.models import Face, FaceCorrection, Person

log = logging.getLogger(__name__)


def _utcnow_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


@dataclass
class IntraImageConsistencyStats:
    """Diagnostic counters from one :meth:`IntraImageConsistencyService.run`."""

    n_images_examined: int = 0      # images with >=2 embedded faces inspected
    n_components_merged: int = 0    # same-image groups that triggered a merge
    n_faces_reassigned: int = 0     # individual faces moved to a new identity
    n_persons_removed: int = 0      # Unknown persons emptied + deleted by merges
    n_conflicts_skipped: int = 0    # groups spanning 2+ named persons (left alone)


@dataclass
class _UnionFind:
    """Tiny union-find over integer face ids."""

    parent: Dict[int, int] = field(default_factory=dict)

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression.
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def components(self) -> Dict[int, List[int]]:
        groups: Dict[int, List[int]] = defaultdict(list)
        for node in self.parent:
            groups[self.find(node)].append(node)
        return groups


class IntraImageConsistencyService:
    """Unifies same-person faces within a single image after clustering."""

    def __init__(
        self,
        session: Session,
        config: Optional[IntraImageConsistencyConfig] = None,
        exclude_low_quality: bool = False,
    ) -> None:
        self._session = session
        self._config = config or IntraImageConsistencyConfig()
        self._exclude_low_quality = exclude_low_quality

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> IntraImageConsistencyStats:
        """Scan every multi-face image and heal split identities.

        Returns:
            :class:`IntraImageConsistencyStats` with what changed.
        """
        stats = IntraImageConsistencyStats()
        if not self._config.enabled:
            log.info("Intra-image consistency: disabled by config")
            return stats

        faces_by_image = self._load_faces_by_image()
        if not faces_by_image:
            log.info("Intra-image consistency: no multi-face images to inspect")
            return stats

        forbidden = self._load_forbidden_pairs()

        for image_id, faces in faces_by_image.items():
            stats.n_images_examined += 1
            self._process_image(image_id, faces, forbidden, stats)

        self._delete_orphan_auto_persons(stats)
        self._session.commit()
        log.info(
            "Intra-image consistency summary: %d image(s) examined, "
            "%d group(s) merged, %d face(s) reassigned, "
            "%d Unknown person(s) removed, %d conflict(s) skipped",
            stats.n_images_examined, stats.n_components_merged,
            stats.n_faces_reassigned, stats.n_persons_removed,
            stats.n_conflicts_skipped,
        )
        return stats

    # ------------------------------------------------------------------
    # Per-image processing
    # ------------------------------------------------------------------

    def _process_image(
        self,
        image_id: int,
        faces: List[Face],
        forbidden: Set[frozenset],
        stats: IntraImageConsistencyStats,
    ) -> None:
        if len(faces) > self._config.max_faces_per_image:
            log.debug(
                "Image id=%d: %d faces exceeds max_faces_per_image=%d — skipped",
                image_id, len(faces), self._config.max_faces_per_image,
            )
            return

        # Normalise embeddings once.
        normed: Dict[int, np.ndarray] = {}
        for face in faces:
            vec = self._normalise(face.get_embedding())
            if vec is not None:
                normed[face.id] = vec
        usable = [f for f in faces if f.id in normed]
        if len(usable) < 2:
            return

        # Build the same-identity graph: an edge when two faces on this image
        # are mutually near-identical and not forbidden by a user correction.
        uf = _UnionFind()
        threshold = self._config.merge_similarity
        for i in range(len(usable)):
            fa = usable[i]
            uf.find(fa.id)  # ensure singletons appear in components()
            for j in range(i + 1, len(usable)):
                fb = usable[j]
                if frozenset((fa.id, fb.id)) in forbidden:
                    continue
                sim = float(np.dot(normed[fa.id], normed[fb.id]))
                if sim >= threshold:
                    uf.union(fa.id, fb.id)
                    log.debug(
                        "Image id=%d: faces %d ~ %d (sim=%.3f) → same identity",
                        image_id, fa.id, fb.id, sim,
                    )

        face_by_id = {f.id: f for f in usable}
        for _root, member_ids in uf.components().items():
            if len(member_ids) < 2:
                continue
            members = [face_by_id[fid] for fid in member_ids]
            self._merge_component(image_id, members, stats)

    def _merge_component(
        self,
        image_id: int,
        members: List[Face],
        stats: IntraImageConsistencyStats,
    ) -> None:
        """Collapse one same-identity group of faces onto a single person."""
        named: Dict[int, Person] = {}
        autos: Dict[int, Person] = {}
        for face in members:
            person = face.person
            if person is None:
                continue
            if person.is_protected:
                # Protected placeholders are immovable anchors; their presence
                # alone should not force a merge of the rest, so just note it.
                named.setdefault(person.id, person)
            elif person.is_auto_named:
                autos.setdefault(person.id, person)
            else:
                named.setdefault(person.id, person)

        if len(named) > 1:
            # Two distinct manually-named/protected identities co-occur — a real
            # signal, never merge across them.
            stats.n_conflicts_skipped += 1
            log.info(
                "Image id=%d: same-image group spans %d named identities "
                "(%s) — skipped to avoid cross-identity merge",
                image_id, len(named),
                ", ".join(sorted(p.name for p in named.values())),
            )
            return

        target = self._choose_target(named, autos, members)
        if target is None:
            # All faces unassigned: leave them for the clusterer, which owns the
            # creation of brand-new Unknown identities.
            return

        now = _utcnow_naive()
        merged_any = False
        reassigned_face_ids: List[int] = []

        for face in members:
            if face.person_id == target.id:
                continue
            person = face.person
            # Never move a manually-named or protected face.
            if person is not None and not person.is_auto_named:
                continue
            face.person_id = target.id
            face.assignment_source = "intra_image"
            face.assignment_confidence = None
            face.assigned_at = now
            reassigned_face_ids.append(face.id)
            stats.n_faces_reassigned += 1
            merged_any = True

        # When the target is an Unknown person, fold the *whole* losing Unknown
        # identities into it (their faces on other images belong to the same
        # person too).  Manually-named targets only absorb the co-occurring
        # faces above — promoting an entire Unknown into a named person is the
        # job of the (user-gated) suggestion flow, not this automatic pass.
        if target.is_auto_named:
            for pid, person in autos.items():
                if pid == target.id:
                    continue
                self._absorb_person(person, target, now, stats)
                merged_any = True

        if merged_any:
            stats.n_components_merged += 1
            self._record_same_corrections(target.id, members)

    def _choose_target(
        self,
        named: Dict[int, Person],
        autos: Dict[int, Person],
        members: List[Face],
    ) -> Optional[Person]:
        """Pick the surviving identity for a same-image group."""
        if len(named) == 1:
            return next(iter(named.values()))
        if autos:
            # Winner = the Unknown person with the most faces overall, breaking
            # ties on the lowest id for deterministic output.
            return max(
                autos.values(),
                key=lambda p: (len(p.faces), -p.id),
            )
        return None

    def _absorb_person(
        self,
        loser: Person,
        winner: Person,
        now: datetime,
        stats: IntraImageConsistencyStats,
    ) -> None:
        """Move every remaining face of *loser* onto *winner*."""
        # Flush pending per-face reassignments first so the query below sees the
        # current owners, then move whatever still belongs to the loser via the
        # ORM (keeps loaded objects in sync before the loser row is deleted).
        self._session.flush()
        remaining: List[Face] = (
            self._session.query(Face)
            .filter(Face.person_id == loser.id)
            .all()
        )
        for face in remaining:
            face.person_id = winner.id
            face.assignment_source = "intra_image"
            face.assignment_confidence = None
            face.assigned_at = now
        self._session.flush()
        moved = len(remaining)
        stats.n_faces_reassigned += moved
        log.info(
            "Intra-image: absorbed %r (id=%d, %d face(s)) → %r (id=%d)",
            loser.name, loser.id, moved, winner.name, winner.id,
        )

    # ------------------------------------------------------------------
    # Corrections / cleanup
    # ------------------------------------------------------------------

    def _record_same_corrections(self, target_id: int, members: List[Face]) -> None:
        """Persist same-person links so future re-clustering stays consistent.

        Records pairs among the (now co-identified) faces so the constrained
        clusterer keeps them together.  Sampled to avoid an O(n²) explosion.
        """
        ids = sorted({f.id for f in members})
        anchor = ids[0]
        for other in ids[1:6]:
            self._upsert_same_correction(anchor, other)

    def _upsert_same_correction(self, face_a: int, face_b: int) -> None:
        a, b = sorted((face_a, face_b))
        if a == b:
            return
        existing: Optional[FaceCorrection] = (
            self._session.query(FaceCorrection)
            .filter(FaceCorrection.face_id_a == a, FaceCorrection.face_id_b == b)
            .first()
        )
        if existing is not None:
            if not existing.same_person:
                # A user previously said "different" — respect that, don't flip.
                return
            return
        self._session.add(
            FaceCorrection(face_id_a=a, face_id_b=b, same_person=True)
        )

    def _delete_orphan_auto_persons(
        self, stats: IntraImageConsistencyStats
    ) -> None:
        self._session.flush()
        orphans: List[Person] = (
            self._session.query(Person)
            .filter(Person.is_auto_named == True)  # noqa: E712
            .filter(~Person.faces.any())
            .all()
        )
        for person in orphans:
            # The face reassignments above leave each loser's loaded ``faces``
            # collection stale.  Without expiring it, deleting the person would
            # nullify the FK of faces that now belong to the winner.
            self._session.expire(person, ["faces"])
            self._session.delete(person)
        stats.n_persons_removed += len(orphans)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_faces_by_image(self) -> Dict[int, List[Face]]:
        """Return ``{image_id: [faces]}`` for images with >=2 embedded faces."""
        query = (
            self._session.query(Face)
            .filter(Face._embedding.isnot(None))
            .filter(Face.is_excluded == False)  # noqa: E712
        )
        if self._exclude_low_quality:
            query = query.filter(
                or_(Face.is_low_quality.is_(None), Face.is_low_quality == False)  # noqa: E712
            )

        by_image: Dict[int, List[Face]] = defaultdict(list)
        for face in query.all():
            if face.image_id is not None:
                by_image[face.image_id].append(face)

        return {iid: faces for iid, faces in by_image.items() if len(faces) >= 2}

    def _load_forbidden_pairs(self) -> Set[frozenset]:
        """Return face-id pairs the user marked as different people."""
        corrections: List[FaceCorrection] = (
            self._session.query(FaceCorrection)
            .filter(FaceCorrection.same_person == False)  # noqa: E712
            .all()
        )
        return {
            frozenset((c.face_id_a, c.face_id_b))
            for c in corrections
        }

    @staticmethod
    def _normalise(vector: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if vector is None:
            return None
        vec = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-8:
            return None
        return vec / norm
