"""Find and remove unknown face boxes that overlap named faces."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable

from sqlalchemy.orm import Session, selectinload

from app.db.models import Face, Image

log = logging.getLogger(__name__)

DEFAULT_OVERLAP_IOU_THRESHOLD = 0.35

# A tight "?" box nested inside a generous named-face box has a low IoU (the
# large box dominates the union) but a high containment ratio. Flag the pair
# when this fraction of the *smaller* box lies inside the other box.
DEFAULT_CONTAINMENT_THRESHOLD = 0.80

# Matches common placeholder/unknown name patterns (case-insensitive, trimmed):
#   "?", "??", "Unknown", "Unknown 96", "Unknown_96", "unknown",
#   "Ismeretlen", "Ismeretlen 5", "ismeretlen_3", etc.
_PLACEHOLDER_RE = re.compile(
    r"^(\?+|unknown[\s_]*\d*|ismeretlen[\s_]*\d*)$",
    re.IGNORECASE,
)


def is_placeholder_name(name: str | None) -> bool:
    """Return True if *name* is a placeholder / unknown stand-in."""
    if not name or not name.strip():
        return True
    return bool(_PLACEHOLDER_RE.match(name.strip()))


@dataclass(frozen=True)
class OverlappingUnknownFaceMatch:
    """One suspicious unknown face overlapping one known face."""

    image_id: int
    image_path: str
    image_relative_path: str | None
    unknown_face_id: int
    known_face_id: int
    known_person_name: str
    overlap: float
    unknown_bbox: tuple[int, int, int, int]
    known_bbox: tuple[int, int, int, int]

    @property
    def display_path(self) -> str:
        return self.image_relative_path or self.image_path


@dataclass(frozen=True)
class DeleteUnknownFacesResult:
    requested: int
    deleted: int
    image_ids: tuple[int, ...]
    missing_or_changed: tuple[int, ...]


class DuplicateUnknownFaceFinder:
    """Find unassigned or placeholder faces overlapping named faces.

    "Unknown" faces are:
      - faces with no person assignment (person_id is None)
      - faces assigned to an auto-named placeholder person (is_auto_named=True)
      - faces whose person has a placeholder name (?, Unknown XX, Ismeretlen, …)

    Additionally, same-person duplicates (two face boxes with the same person_id
    that significantly overlap) are flagged as potential spurious detections.
    """

    def __init__(
        self,
        session: Session,
        iou_threshold: float = DEFAULT_OVERLAP_IOU_THRESHOLD,
        containment_threshold: float = DEFAULT_CONTAINMENT_THRESHOLD,
    ) -> None:
        self._session = session
        self._iou_threshold = iou_threshold
        self._containment_threshold = containment_threshold
        self.images_examined: int = 0
        # Face IDs that were flagged as same-person duplicates; may be deleted
        # even if _is_unknown() returns False for them.
        self._same_person_duplicate_ids: set[int] = set()

    def find(self) -> list[OverlappingUnknownFaceMatch]:
        """Return suspicious face boxes, one best match per unknown/duplicate."""
        self._same_person_duplicate_ids = set()
        self.images_examined = self._session.query(Image).count()
        images = (
            self._session.query(Image)
            .options(selectinload(Image.faces).selectinload(Face.person))
            .filter(Image.faces.any())
            .order_by(Image.id)
            .all()
        )

        matches: list[OverlappingUnknownFaceMatch] = []
        reported_unknown_ids: set[int] = set()

        for image in images:
            visible_faces = [face for face in image.faces if not face.is_excluded]
            unknown_faces = [face for face in visible_faces if self._is_unknown(face)]
            known_faces = [face for face in visible_faces if self._is_known(face)]

            # ── Pass 1: unknown/placeholder overlapping a named face ──────────
            for unknown in unknown_faces:
                best_known: Face | None = None
                best_key = (0.0, 0.0)
                for known in known_faces:
                    key = self._overlap_score(unknown, known)
                    if key is not None and key > best_key:
                        best_known = known
                        best_key = key

                if best_known is None or best_known.person is None:
                    continue
                best_score = best_key[0]

                matches.append(
                    OverlappingUnknownFaceMatch(
                        image_id=image.id,
                        image_path=image.file_path,
                        image_relative_path=image.relative_path,
                        unknown_face_id=unknown.id,
                        known_face_id=best_known.id,
                        known_person_name=best_known.person.name,
                        overlap=best_score,
                        unknown_bbox=_bbox_tuple(unknown),
                        known_bbox=_bbox_tuple(best_known),
                    )
                )
                reported_unknown_ids.add(unknown.id)

            # ── Pass 2: same-person duplicate detections ──────────────────────
            # Group faces by person_id and flag overlapping pairs. This covers
            # not only manually named people but also auto-named / placeholder
            # clusters (e.g. two boxes both landing in "Unknown 96") — the
            # clustering already decided they are one identity, so an overlap
            # within a single image is a duplicate detection.
            from collections import defaultdict
            by_person: dict[int, list[Face]] = defaultdict(list)
            for face in visible_faces:
                if self._is_dedup_candidate(face):
                    by_person[face.person_id].append(face)

            for pid, pfaces in by_person.items():
                if len(pfaces) < 2:
                    continue
                for i, fa in enumerate(pfaces):
                    for fb in pfaces[i + 1:]:
                        key = self._overlap_score(fa, fb)
                        if key is None:
                            continue
                        score = key[0]
                        # The lower-confidence (or higher-id) face is the duplicate.
                        conf_a = fa.confidence or 0.0
                        conf_b = fb.confidence or 0.0
                        if conf_a >= conf_b:
                            unk, kno = fb, fa
                        else:
                            unk, kno = fa, fb
                        if unk.id in reported_unknown_ids:
                            continue
                        person_name = kno.person.name if kno.person else ""
                        matches.append(
                            OverlappingUnknownFaceMatch(
                                image_id=image.id,
                                image_path=image.file_path,
                                image_relative_path=image.relative_path,
                                unknown_face_id=unk.id,
                                known_face_id=kno.id,
                                known_person_name=person_name,
                                overlap=score,
                                unknown_bbox=_bbox_tuple(unk),
                                known_bbox=_bbox_tuple(kno),
                            )
                        )
                        reported_unknown_ids.add(unk.id)
                        self._same_person_duplicate_ids.add(unk.id)

        log.info(
            "Overlapping face search: examined %d image(s), found %d candidate(s) "
            "(%d same-person duplicates)",
            self.images_examined,
            len(matches),
            len(self._same_person_duplicate_ids),
        )
        return matches

    def delete_unknown_faces(
        self,
        face_ids: Iterable[int],
    ) -> DeleteUnknownFacesResult:
        """Delete selected faces that are still unknown or same-person duplicates.

        Named faces that are not same-person duplicates are never deleted.
        If a face was assigned to a real person after the list was shown,
        it is skipped and reported.
        """
        requested_ids = sorted(set(face_ids))
        if not requested_ids:
            return DeleteUnknownFacesResult(0, 0, (), ())

        deleted_image_ids: set[int] = set()
        missing_or_changed: list[int] = []
        deleted = 0

        for face_id in requested_ids:
            face = self._session.get(Face, face_id)
            deletable = (
                face is not None
                and (
                    self._is_unknown(face)
                    or face_id in self._same_person_duplicate_ids
                )
            )
            if not deletable:
                missing_or_changed.append(face_id)
                continue
            deleted_image_ids.add(face.image_id)  # type: ignore[union-attr]
            self._session.delete(face)
            deleted += 1

        log.info(
            "Overlapping face cleanup: requested=%d deleted=%d skipped=%d",
            len(requested_ids),
            deleted,
            len(missing_or_changed),
        )
        return DeleteUnknownFacesResult(
            requested=len(requested_ids),
            deleted=deleted,
            image_ids=tuple(sorted(deleted_image_ids)),
            missing_or_changed=tuple(missing_or_changed),
        )

    def _overlap_score(self, a: Face, b: Face) -> tuple[float, float] | None:
        """Return a ``(score, iou)`` ranking key if *a* and *b* overlap, else None.

        Two boxes count as overlapping when their IoU clears ``iou_threshold``
        *or* one box is largely contained in the other (``containment`` clears
        ``containment_threshold``). The latter catches a small face box nested
        inside a much larger one, where IoU alone is misleadingly low.

        ``score`` (the larger of the two metrics) is what gets reported; the raw
        ``iou`` is carried alongside so that, among equally-scoring candidates,
        the most precisely matching box (highest IoU) wins the tie.
        """
        iou, containment = face_overlap_metrics(a, b)
        if iou >= self._iou_threshold or containment >= self._containment_threshold:
            return max(iou, containment), iou
        return None

    @staticmethod
    def _is_unknown(face: Face) -> bool:
        """Return True for unassigned, auto-named, or placeholder-named faces."""
        if face.person_id is None:
            return True
        person = face.person
        if person is None:
            return True
        if person.is_auto_named:
            return True
        if is_placeholder_name(person.name):
            return True
        return False

    @staticmethod
    def _is_dedup_candidate(face: Face) -> bool:
        """Return True for faces that belong to exactly one identity cluster.

        Such faces (any non-protected person — named, auto-named, or
        placeholder) can be safely deduplicated against their siblings in the
        same image. The protected catch-all bucket (e.g. "Ismeretlen") is
        excluded because its single ``person_id`` aggregates many distinct
        identities, so two of its boxes overlapping need not be the same face.
        """
        person = face.person
        return (
            face.person_id is not None
            and person is not None
            and not person.is_protected
        )

    @staticmethod
    def _is_known(face: Face) -> bool:
        person = face.person
        return (
            person is not None
            and not person.is_auto_named
            and not person.is_protected
            and not is_placeholder_name(person.name)
        )


def face_overlap_metrics(a: Face, b: Face) -> tuple[float, float]:
    """Return ``(iou, containment)`` for two stored face bounding boxes.

    ``iou`` is the classic intersection-over-union. ``containment`` is the
    intersection divided by the area of the *smaller* box; it stays close to
    1.0 when a small box is nested inside a much larger one — exactly the case
    (a tight "?" box sitting inside a generous named-face box) where IoU is
    misleadingly low because the large box dominates the union.
    """
    ax1, ay1, ax2, ay2 = a.bbox_x, a.bbox_y, a.bbox_x + a.bbox_w, a.bbox_y + a.bbox_h
    bx1, by1, bx2, by2 = b.bbox_x, b.bbox_y, b.bbox_x + b.bbox_w, b.bbox_y + b.bbox_h
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0, 0.0
    area_a = max(0, a.bbox_w) * max(0, a.bbox_h)
    area_b = max(0, b.bbox_w) * max(0, b.bbox_h)
    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0.0
    smaller = min(area_a, area_b)
    containment = inter / smaller if smaller > 0 else 0.0
    return iou, containment


def face_iou(a: Face, b: Face) -> float:
    """Compute Intersection-over-Union for two stored face bounding boxes."""
    return face_overlap_metrics(a, b)[0]


def _bbox_tuple(face: Face) -> tuple[int, int, int, int]:
    return face.bbox_x, face.bbox_y, face.bbox_w, face.bbox_h
