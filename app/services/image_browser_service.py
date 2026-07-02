"""Service layer for the 3-column Image Browser panel.

Provides lightweight data-only queries — no UI, no file I/O beyond path
comparison.  All methods accept an already-open SQLAlchemy session so the
caller controls transaction lifetime.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sqlalchemy import case, func
from sqlalchemy.orm import Session

from app.db.models import Face, Image
from app.db.query_utils import in_chunks

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data-transfer objects
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FolderSummary:
    folder_path: str
    display_name: str
    total_images: int
    processed_images: int
    unprocessed_images: int
    detected_faces_count: int
    assigned_faces_count: int
    unknown_faces_count: int


@dataclass
class ImageSummary:
    image_id: int
    file_path: str
    filename: str
    width: Optional[int]
    height: Optional[int]
    detection_done: bool
    embedding_done: bool
    face_count: int
    assigned_face_count: int
    unknown_face_count: int
    thumbnail_cache_key: str
    place_id: Optional[int] = None
    place_name: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Service
# ──────────────────────────────────────────────────────────────────────────────

class ImageBrowserService:
    """Data queries for the image browser — folder grouping and image summaries."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_folders(self) -> List[FolderSummary]:
        """Return one FolderSummary per distinct parent folder in the DB.

        Two SQL queries total: one for images, one for face counts.
        Grouping is done in Python.
        """
        # Column-only query: full Image entities are ~10× heavier to build and
        # this runs on every panel refresh, over the whole table.
        images = self._session.query(
            Image.id, Image.file_path, Image.detection_done
        ).all()
        if not images:
            return []

        face_by_image = self._face_counts_by_image(
            [img_id for img_id, _, _ in images]
        )

        groups: Dict[str, List[tuple]] = defaultdict(list)
        for row in images:
            folder = str(Path(row[1]).parent)
            groups[folder].append(row)

        result: List[FolderSummary] = []
        for folder_path in sorted(groups.keys()):
            imgs = groups[folder_path]
            total = len(imgs)
            processed = sum(1 for _, _, done in imgs if done)

            detected = assigned = 0
            for img_id, _, _ in imgs:
                t, a = face_by_image.get(img_id, (0, 0))
                detected += t
                assigned += a

            result.append(FolderSummary(
                folder_path=folder_path,
                display_name=Path(folder_path).name or folder_path,
                total_images=total,
                processed_images=processed,
                unprocessed_images=total - processed,
                detected_faces_count=detected,
                assigned_faces_count=assigned,
                unknown_faces_count=detected - assigned,
            ))
            log.debug(
                "Folder loaded: %s  images=%d  processed=%d  faces=%d",
                folder_path, total, processed, detected,
            )
        return result

    def list_images(
        self,
        folder_path: str,
        place_id: Optional[int] = None,
        person_id: Optional[int] = None,
        second_person_id: Optional[int] = None,
        person_search_mode: str = "both",
        relationship_filter: str = "any",
    ) -> List[ImageSummary]:
        """Return ImageSummary list for every image whose parent is *folder_path*.

        Two SQL queries: one for images filtered by folder, one for face counts.
        """
        # Use exact parent comparison (avoids sub-folder collisions).
        # A LIKE filter on the prefix speeds up SQLite on large tables; the
        # in-Python check provides exactness.
        safe_prefix = folder_path.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        q = self._session.query(Image).filter(
            Image.file_path.like(f"{safe_prefix}%", escape="\\")
        )
        if place_id is not None:
            q = q.filter(Image.place_id == place_id)
        allowed_ids: Optional[set] = None
        if person_id is not None and second_person_id is not None:
            from app.services.family_service import FamilyService
            if person_id == second_person_id:
                return []
            family_results, _ = FamilyService(self._session).search_images_for_people(
                person_id,
                second_person_id,
                mode="exact" if person_search_mode == "exact" else "both",
                relationship_filter=relationship_filter,  # type: ignore[arg-type]
                limit=100000,
                offset=0,
                folder_path=folder_path,
                place_id=place_id,
            )
            allowed_ids = {r.image_id for r in family_results}
            if not allowed_ids:
                return []
            # Membership is enforced in Python below (see post-filter) rather
            # than via Image.id.in_(allowed_ids): the set can be large and an
            # IN (?, …) over it would blow SQLite's parameter limit on Windows.
        elif person_id is not None:
            person_images = (
                self._session.query(Face.image_id)
                .filter(Face.is_excluded == False)  # noqa: E712
                .filter(Face.person_id == person_id)
                .distinct()
                .subquery()
            )
            q = q.join(person_images, person_images.c.image_id == Image.id)
        raw = q.order_by(Image.file_path).all()
        images = [
            img for img in raw
            if str(Path(img.file_path).parent) == folder_path
            and (allowed_ids is None or img.id in allowed_ids)
        ]

        if not images:
            return []

        image_ids = [img.id for img in images]
        face_by_image = self._face_counts_by_image(image_ids)

        result: List[ImageSummary] = []
        for img in images:
            total_f, assigned_f = face_by_image.get(img.id, (0, 0))
            result.append(ImageSummary(
                image_id=img.id,
                file_path=img.file_path,
                filename=Path(img.file_path).name,
                width=img.width,
                height=img.height,
                detection_done=img.detection_done,
                embedding_done=img.embedding_done,
                face_count=total_f,
                assigned_face_count=assigned_f,
                unknown_face_count=total_f - assigned_f,
                thumbnail_cache_key=f"thumb_{img.id}",
                place_id=img.place_id,
                place_name=img.place.name if img.place else None,
            ))
        log.debug(
            "Images loaded for folder %s: %d images", folder_path, len(result)
        )
        return result

    def get_image_faces(
        self, image_id: int
    ) -> List[Tuple[int, int, int, int, int, Optional[str], Optional[int], bool]]:
        """Return face data tuples for *image_id*.

        Each tuple: (face_id, x, y, w, h, person_name_or_None, person_id_or_None, is_uncertain)
        Excluded faces are omitted.
        """
        # Column-only query: loading Face entities via ``img.faces`` also
        # batch-loaded every face's embedding blob (lazy="selectin") and
        # lazy-loaded each face's person/image rows, which made every image
        # selection scale with face count on a large database.  This runs on
        # each image open AND after every assignment, so it must stay cheap.
        from app.db.models import Person

        rows = (
            self._session.query(
                Face.id,
                Face.bbox_x, Face.bbox_y, Face.bbox_w, Face.bbox_h,
                Person.name,
                Face.person_id,
                Face.is_uncertain_identification,
            )
            .outerjoin(Person, Person.id == Face.person_id)
            .filter(Face.image_id == image_id)
            .filter(Face.is_excluded == False)  # noqa: E712
            .all()
        )
        return [
            (fid, x, y, w, h, person_name, person_id, bool(uncertain))
            for fid, x, y, w, h, person_name, person_id, uncertain in rows
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _face_counts_by_image(
        self, image_ids: List[int]
    ) -> Dict[int, Tuple[int, int]]:
        """Return {image_id: (total_faces, assigned_faces)} for *image_ids*.

        Only non-excluded faces are counted.
        """
        if not image_ids:
            return {}

        # Chunk the IN list so we never exceed SQLite's per-statement parameter
        # limit (999 on older/Windows builds) — each image_id is one parameter.
        result: Dict[int, Tuple[int, int]] = {}
        for chunk in in_chunks(image_ids):
            rows = (
                self._session.query(
                    Face.image_id,
                    func.count(Face.id).label("total"),
                    func.sum(
                        case((Face.person_id.isnot(None), 1), else_=0)
                    ).label("assigned"),
                )
                .filter(Face.is_excluded == False)  # noqa: E712
                .filter(Face.image_id.in_(chunk))
                .group_by(Face.image_id)
                .all()
            )
            for row in rows:
                result[row.image_id] = (row.total, row.assigned or 0)
        return result
