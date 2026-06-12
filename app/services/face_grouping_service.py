"""Group a person's faces into visually near-identical clusters.

The "merged faces" view collapses faces that are almost the same shot (the same
moment captured twice, near-duplicate scans, or burst frames) into a single
stacked tile.  This module performs that grouping purely from face embeddings,
reusing the cosine-similarity convention of
:mod:`app.services.clustering_service` (L2-normalised vectors, similarity =
dot product).

The algorithm is a single greedy pass: each face joins the first existing group
whose running centroid is similar enough, otherwise it starts a new group.
Faces without an embedding always form their own singleton group.  Grouping is
order-stable: groups appear in first-seen order and members keep their incoming
order, so the result is deterministic for a given input and threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from app.db.models import Face
from app.services.face_date_service import FaceDateResolver
from app.utils.fuzzy_date import FuzzyDate

# Default cosine-similarity threshold for collapsing two faces into one tile.
# 0.92 ≈ "the same person in a near-identical shot", deliberately stricter than
# the recognition/clustering thresholds which only need same-identity.
DEFAULT_MERGE_THRESHOLD = 0.92


@dataclass
class FaceGroup:
    """A cluster of near-identical faces shown as one stacked tile."""

    members: List[Face] = field(default_factory=list)
    representative: Optional[Face] = None
    # Running L2-normalised centroid of member embeddings (None if none had one).
    _centroid: Optional[np.ndarray] = None

    @property
    def count(self) -> int:
        return len(self.members)

    @property
    def is_singleton(self) -> bool:
        return len(self.members) <= 1

    @property
    def member_ids(self) -> List[int]:
        return [f.id for f in self.members]

    def date_span(self, resolver: FaceDateResolver) -> FuzzyDate:
        """The combined date span of the group's members.

        Returns the earliest-starting / latest-ending known member date, or the
        representative's date if none are known.  Used to position the group on
        the timeline.
        """
        known = [resolver.for_face(f) for f in self.members]
        known = [d for d in known if d.is_known]
        if not known:
            return resolver.for_face(self.representative or self.members[0])
        earliest = min((d.earliest for d in known if d.earliest), default=None)
        latest = max((d.latest for d in known if d.latest), default=None)
        # Reuse a representative's precision/estimation flags for display.
        rep = min(known, key=lambda d: d.sort_key)
        return FuzzyDate(
            earliest or rep.earliest,
            latest or rep.latest,
            rep.precision,
            any(d.is_estimated for d in known),
            rep.raw,
        )


def _representative(members: List[Face]) -> Face:
    """Pick the clearest face as the group's cover.

    Prefers higher quality score (treating "not evaluated" as usable), then
    detection confidence, then the lowest id for determinism.
    """
    def key(f: Face):
        q = f.quality_score if f.quality_score is not None else 0.75
        c = f.confidence if f.confidence is not None else 0.0
        return (q, c, -f.id)

    return max(members, key=key)


def group_faces(
    faces: List[Face],
    *,
    threshold: float = DEFAULT_MERGE_THRESHOLD,
) -> List[FaceGroup]:
    """Group *faces* into :class:`FaceGroup`s by embedding similarity."""
    groups: List[FaceGroup] = []

    for face in faces:
        emb = face.get_embedding()
        vec: Optional[np.ndarray] = None
        if emb is not None and emb.size:
            norm = float(np.linalg.norm(emb))
            if norm > 0.0:
                vec = (emb / norm).astype(np.float32)

        if vec is None:
            # No usable embedding → always its own group.
            groups.append(FaceGroup(members=[face], representative=face))
            continue

        best_group: Optional[FaceGroup] = None
        best_sim = threshold
        for g in groups:
            if g._centroid is None:
                continue
            sim = float(np.dot(vec, g._centroid))
            if sim >= best_sim:
                best_sim = sim
                best_group = g

        if best_group is None:
            groups.append(FaceGroup(members=[face], representative=face, _centroid=vec))
        else:
            n = sum(1 for m in best_group.members if m.get_embedding() is not None)
            # Incremental mean of normalised vectors, then renormalise.
            updated = (best_group._centroid * n + vec) / (n + 1)
            unorm = float(np.linalg.norm(updated))
            best_group._centroid = (
                (updated / unorm).astype(np.float32) if unorm > 0 else best_group._centroid
            )
            best_group.members.append(face)

    # Choose a representative per group now that membership is final.
    for g in groups:
        g.representative = _representative(g.members)
    return groups
