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
from app.services.face_crop_service import face_debug_state

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
        images = self._session.query(Image).all()
        if not images:
            return []

        face_by_image = self._face_counts_by_image(
            [img.id for img in images]
        )

        groups: Dict[str, List[Image]] = defaultdict(list)
        for img in images:
            folder = str(Path(img.file_path).parent)
            groups[folder].append(img)

        result: List[FolderSummary] = []
        for folder_path in sorted(groups.keys()):
            imgs = groups[folder_path]
            total = len(imgs)
            processed = sum(1 for i in imgs if i.detection_done)

            detected = assigned = 0
            for img in imgs:
                t, a = face_by_image.get(img.id, (0, 0))
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

    def list_images(self, folder_path: str) -> List[ImageSummary]:
        """Return ImageSummary list for every image whose parent is *folder_path*.

        Two SQL queries: one for images filtered by folder, one for face counts.
        """
        # Use exact parent comparison (avoids sub-folder collisions).
        # A LIKE filter on the prefix speeds up SQLite on large tables; the
        # in-Python check provides exactness.
        safe_prefix = folder_path.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        raw = (
            self._session.query(Image)
            .filter(Image.file_path.like(f"{safe_prefix}%", escape="\\"))
            .order_by(Image.file_path)
            .all()
        )
        images = [img for img in raw if str(Path(img.file_path).parent) == folder_path]

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
            ))
        log.debug(
            "Images loaded for folder %s: %d images", folder_path, len(result)
        )
        return result

    def get_image_faces(self, image_id: int) -> List[Tuple[int, int, int, int, int, Optional[str], Optional[int]]]:
        """Return face data tuples for *image_id*.

        Each tuple: (face_id, x, y, w, h, person_name_or_None, person_id_or_None)
        Excluded faces are omitted.
        """
        img = self._session.get(Image, image_id)
        if img is None:
            return []
        for f in img.faces:
            _ = f.person
            log.debug("Image-browser face entity: %s", face_debug_state(f, f.crop_path))
        return [
            (
                f.id,
                f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h,
                f.person.name if f.person else None,
                f.person_id,
            )
            for f in img.faces
            if not f.is_excluded
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

        rows = (
            self._session.query(
                Face.image_id,
                func.count(Face.id).label("total"),
                func.sum(
                    case((Face.person_id.isnot(None), 1), else_=0)
                ).label("assigned"),
            )
            .filter(Face.is_excluded == False)  # noqa: E712
            .filter(Face.image_id.in_(image_ids))
            .group_by(Face.image_id)
            .all()
        )
        return {row.image_id: (row.total, row.assigned or 0) for row in rows}
