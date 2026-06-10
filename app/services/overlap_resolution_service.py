"""Same-image overlapping face-box resolution.

When the detector marked the same physical face more than once on one photo
(shifted boxes, nested boxes from different detector runs, re-detection after
a rebuild), exactly one box must survive:

* a box assigned to a *named person* (or drawn manually) always wins over an
  unknown one — an existing recognition is never displaced by a duplicate;
* between two unknown boxes the better one is kept (quality, detector
  confidence, size);
* two boxes assigned to two *different named* persons are never touched —
  that is a user-level conflict, not detector noise.

The geometric overlap is combined with an embedding check so that two real,
tightly cropped faces (cheek-to-cheek) are not mistaken for duplicates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy.orm import Session

from app.config import OverlapResolutionConfig
from app.db.models import Face, Person

log = logging.getLogger(__name__)

# Sources marking a human decision (legacy rows have NULL source).
_MANUAL_SOURCES = {"manual", "manual_merge", "suggestion_approved", "deep_confirmed"}


@dataclass
class OverlapResolutionStats:
    """Summary of one overlap-resolution pass."""

    images_examined: int = 0
    pairs_resolved: int = 0
    faces_removed: int = 0
    conflicts_skipped: int = 0  # two different named persons — left alone


def _bbox_metrics(a: Face, b: Face) -> Tuple[float, float]:
    """Return ``(iou, containment)`` of two faces' bounding boxes."""
    ax2, ay2 = a.bbox_x + a.bbox_w, a.bbox_y + a.bbox_h
    bx2, by2 = b.bbox_x + b.bbox_w, b.bbox_y + b.bbox_h
    ix1, iy1 = max(a.bbox_x, b.bbox_x), max(a.bbox_y, b.bbox_y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0, 0.0
    area_a = a.bbox_w * a.bbox_h
    area_b = b.bbox_w * b.bbox_h
    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0.0
    smaller = min(area_a, area_b)
    containment = inter / smaller if smaller > 0 else 0.0
    return iou, containment


class OverlapResolutionService:
    """Collapses multiple detections of the same physical face into one."""

    def __init__(
        self,
        session: Session,
        config: Optional[OverlapResolutionConfig] = None,
    ) -> None:
        self._session = session
        self._config = config or OverlapResolutionConfig()
        self._person_cache: Dict[int, Person] = {}

    # ------------------------------------------------------------------

    def resolve(self) -> OverlapResolutionStats:
        """Find and resolve overlapping boxes across all images."""
        stats = OverlapResolutionStats()
        if not self._config.enabled:
            return stats

        faces_by_image = self._load_faces_by_image()
        to_delete: List[int] = []

        for image_id, faces in faces_by_image.items():
            if len(faces) < 2:
                continue
            if len(faces) > self._config.max_faces_per_image:
                log.debug(
                    "Overlap resolution: skipping image %d (%d faces)",
                    image_id, len(faces),
                )
                continue
            stats.images_examined += 1
            removed = self._resolve_image(faces, stats)
            to_delete.extend(removed)

        if to_delete:
            self._session.query(Face).filter(Face.id.in_(to_delete)).delete(
                synchronize_session="fetch"
            )
            stats.faces_removed = len(to_delete)
        self._session.commit()

        if stats.faces_removed:
            log.info(
                "Overlap resolution: removed %d duplicate box(es) across %d image(s) "
                "(%d conflict(s) left alone)",
                stats.faces_removed, stats.images_examined, stats.conflicts_skipped,
            )
        return stats

    # ------------------------------------------------------------------
    # Per-image resolution
    # ------------------------------------------------------------------

    def _resolve_image(
        self, faces: List[Face], stats: OverlapResolutionStats
    ) -> List[int]:
        """Return ids of faces on one image that must be deleted."""
        removed: set[int] = set()
        units = {face.id: self._unit(face) for face in faces}

        for i in range(len(faces)):
            a = faces[i]
            if a.id in removed:
                continue
            for j in range(i + 1, len(faces)):
                b = faces[j]
                if b.id in removed or a.id in removed:
                    continue
                if not self._is_same_physical_face(a, b, units):
                    continue

                winner, loser = self._pick_winner(a, b)
                if winner is None or loser is None:
                    stats.conflicts_skipped += 1
                    continue
                stats.pairs_resolved += 1
                removed.add(loser.id)
                log.debug(
                    "Image %d: box %d duplicates box %d — keeping %d",
                    a.image_id, loser.id, winner.id, winner.id,
                )
        return list(removed)

    def _is_same_physical_face(
        self, a: Face, b: Face, units: Dict[int, Optional[np.ndarray]]
    ) -> bool:
        iou, containment = _bbox_metrics(a, b)
        overlapping = (
            iou >= self._config.iou_threshold
            or containment >= self._config.containment_threshold
        )
        if not overlapping:
            return False

        # Embedding guard: two genuinely different faces photographed close
        # together can overlap a little — require embedding agreement unless
        # the boxes sit on essentially the same spot.
        vec_a, vec_b = units.get(a.id), units.get(b.id)
        if vec_a is not None and vec_b is not None:
            similarity = float(np.dot(vec_a, vec_b))
            if (
                similarity < self._config.embedding_guard_similarity
                and iou < self._config.hard_iou_threshold
            ):
                return False
        return True

    def _pick_winner(
        self, a: Face, b: Face
    ) -> Tuple[Optional[Face], Optional[Face]]:
        """Decide which face survives; ``(None, None)`` ⇒ untouchable conflict."""
        named_a, named_b = self._named_person_id(a), self._named_person_id(b)
        if named_a is not None and named_b is not None and named_a != named_b:
            # Two different named people on heavily overlapping boxes — a user
            # decision is required, automation must not destroy either.
            return None, None

        rank_a, rank_b = self._rank(a), self._rank(b)
        if rank_a > rank_b:
            return a, b
        if rank_b > rank_a:
            return b, a

        # Equal rank (typically two unknown boxes): keep the better detection.
        for key in (self._quality, self._confidence, self._area):
            ka, kb = key(a), key(b)
            if ka > kb:
                return a, b
            if kb > ka:
                return b, a
        # Fully tied — keep the older row (the earlier recognition stays).
        return (a, b) if a.id <= b.id else (b, a)

    # ------------------------------------------------------------------
    # Ranking helpers
    # ------------------------------------------------------------------

    def _rank(self, face: Face) -> int:
        """Priority class of a face box; higher survives."""
        if face.detector_backend == "manual":
            return 5
        named = self._named_person_id(face)
        if named is not None:
            if face.assignment_source in _MANUAL_SOURCES or face.assignment_source is None:
                return 4
            return 3
        if face.person_id is not None:
            return 2  # auto-named "Unknown N" group
        return 1      # unassigned

    def _named_person_id(self, face: Face) -> Optional[int]:
        """Person id when the face belongs to a real named person."""
        if face.person_id is None:
            return None
        person = self._person_cache.get(face.person_id)
        if person is None:
            person = self._session.get(Person, face.person_id)
            if person is None:
                return None
            self._person_cache[face.person_id] = person
        if person.is_auto_named or person.is_protected:
            return None
        return person.id

    @staticmethod
    def _quality(face: Face) -> float:
        return face.quality_score if face.quality_score is not None else 0.5

    @staticmethod
    def _confidence(face: Face) -> float:
        return face.confidence if face.confidence is not None else 0.0

    @staticmethod
    def _area(face: Face) -> int:
        return face.bbox_w * face.bbox_h

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_faces_by_image(self) -> Dict[int, List[Face]]:
        grouped: Dict[int, List[Face]] = {}
        query = (
            self._session.query(Face)
            .filter(Face.is_excluded == False)  # noqa: E712
        )
        for face in query.all():
            grouped.setdefault(face.image_id, []).append(face)
        return grouped

    @staticmethod
    def _unit(face: Face) -> Optional[np.ndarray]:
        emb = face.get_embedding()
        if emb is None:
            return None
        vec = np.asarray(emb, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-8:
            return None
        return vec / norm
