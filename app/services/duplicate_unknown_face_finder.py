"""Find and remove unknown face boxes that overlap named faces."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from sqlalchemy.orm import Session, selectinload

from app.db.models import Face, Image

log = logging.getLogger(__name__)

DEFAULT_OVERLAP_IOU_THRESHOLD = 0.35


@dataclass(frozen=True)
class OverlappingUnknownFaceMatch:
    """One suspicious unknown face overlapping one known face."""

    image_id: int
    image_path: str
    image_relative_path: str | None
    unknown_face_id: int
    known_face_id: int
    known_person_name: str
    iou: float
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
    """Find unassigned question-mark faces overlapping manually named faces.

    A question-mark face is represented by ``Face.person_id is None`` in the
    current UI. A known face must belong to a real user-named person, not an
    auto-generated or protected placeholder identity.
    """

    def __init__(
        self,
        session: Session,
        iou_threshold: float = DEFAULT_OVERLAP_IOU_THRESHOLD,
    ) -> None:
        self._session = session
        self._iou_threshold = iou_threshold
        self.images_examined: int = 0

    def find(self) -> list[OverlappingUnknownFaceMatch]:
        """Return suspicious unknown face boxes, one best match per unknown."""
        self.images_examined = self._session.query(Image).count()
        images = (
            self._session.query(Image)
            .options(selectinload(Image.faces).selectinload(Face.person))
            .filter(Image.faces.any())
            .order_by(Image.id)
            .all()
        )

        matches: list[OverlappingUnknownFaceMatch] = []
        for image in images:
            visible_faces = [face for face in image.faces if not face.is_excluded]
            unknown_faces = [face for face in visible_faces if self._is_unknown(face)]
            known_faces = [face for face in visible_faces if self._is_known(face)]
            if not unknown_faces or not known_faces:
                continue

            for unknown in unknown_faces:
                best_known: Face | None = None
                best_iou = 0.0
                for known in known_faces:
                    iou = face_iou(unknown, known)
                    if iou >= self._iou_threshold and iou > best_iou:
                        best_known = known
                        best_iou = iou

                if best_known is None or best_known.person is None:
                    continue

                matches.append(
                    OverlappingUnknownFaceMatch(
                        image_id=image.id,
                        image_path=image.file_path,
                        image_relative_path=image.relative_path,
                        unknown_face_id=unknown.id,
                        known_face_id=best_known.id,
                        known_person_name=best_known.person.name,
                        iou=best_iou,
                        unknown_bbox=_bbox_tuple(unknown),
                        known_bbox=_bbox_tuple(best_known),
                    )
                )

        log.info(
            "Overlapping unknown face search: examined %d image(s), found %d candidate(s)",
            self.images_examined,
            len(matches),
        )
        return matches

    def delete_unknown_faces(
        self,
        face_ids: Iterable[int],
    ) -> DeleteUnknownFacesResult:
        """Delete selected faces only if they are still unknown.

        Known faces are never deleted by this method. If a face was assigned to
        a person after the list was shown, it is skipped and reported.
        """
        requested_ids = sorted(set(face_ids))
        if not requested_ids:
            return DeleteUnknownFacesResult(0, 0, (), ())

        deleted_image_ids: set[int] = set()
        missing_or_changed: list[int] = []
        deleted = 0

        for face_id in requested_ids:
            face = self._session.get(Face, face_id)
            if face is None or not self._is_unknown(face):
                missing_or_changed.append(face_id)
                continue
            deleted_image_ids.add(face.image_id)
            self._session.delete(face)
            deleted += 1

        log.info(
            "Overlapping unknown face cleanup: requested=%d deleted=%d skipped=%d",
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

    @staticmethod
    def _is_unknown(face: Face) -> bool:
        return face.person_id is None

    @staticmethod
    def _is_known(face: Face) -> bool:
        person = face.person
        return (
            person is not None
            and not person.is_auto_named
            and not person.is_protected
        )


def face_iou(a: Face, b: Face) -> float:
    """Compute Intersection-over-Union for two stored face bounding boxes."""
    ax1, ay1, ax2, ay2 = a.bbox_x, a.bbox_y, a.bbox_x + a.bbox_w, a.bbox_y + a.bbox_h
    bx1, by1, bx2, by2 = b.bbox_x, b.bbox_y, b.bbox_x + b.bbox_w, b.bbox_y + b.bbox_h
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = max(0, a.bbox_w) * max(0, a.bbox_h)
    area_b = max(0, b.bbox_w) * max(0, b.bbox_h)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _bbox_tuple(face: Face) -> tuple[int, int, int, int]:
    return face.bbox_x, face.bbox_y, face.bbox_w, face.bbox_h
