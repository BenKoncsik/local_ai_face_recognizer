"""Helpers for keeping face crop thumbnails keyed by Face.id."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
from sqlalchemy.orm import Session

from app.db.models import Face
from app.detectors.base import Detection
from app.utils.image_utils import load_image_bgr, save_face_crop

log = logging.getLogger(__name__)


def canonical_face_crop_path(crops_dir: Path, image_id: int, face_id: int) -> Path:
    """Return the canonical crop path for one face row."""
    return crops_dir / f"img{image_id:06d}_face{face_id:06d}.jpg"


def face_debug_state(face: Face, preview_source: Optional[str] = None) -> str:
    """Compact diagnostic string for a face and its preview source."""
    image_path = face.image.file_path if face.image else None
    person_id = face.person_id
    return (
        f"FaceId={face.id} PersonId={person_id} crop_path={face.crop_path!r} "
        f"image_path={image_path!r} bbox=({face.bbox_x},{face.bbox_y},"
        f"{face.bbox_w},{face.bbox_h}) preview_source={preview_source!r}"
    )


def save_crop_for_face(
    face: Face,
    crops_dir: Path,
    thumbnail_size: Tuple[int, int],
    img_bgr: Optional[np.ndarray] = None,
) -> Optional[Path]:
    """Regenerate one face crop into its Face.id-keyed canonical file."""
    if face.id is None:
        raise ValueError("Face must be flushed before saving a crop")

    if img_bgr is None:
        if face.image is None:
            log.warning("Cannot save crop for face_id=%s: image relationship missing", face.id)
            return None
        img_bgr = load_image_bgr(face.image.file_path)
        if img_bgr is None:
            log.warning(
                "Cannot save crop for face_id=%s: image cannot be loaded: %s",
                face.id,
                face.image.file_path,
            )
            return None

    detection = Detection(
        x=face.bbox_x,
        y=face.bbox_y,
        w=face.bbox_w,
        h=face.bbox_h,
        confidence=face.confidence,
    ).clamp(img_bgr.shape[1], img_bgr.shape[0])
    dest = canonical_face_crop_path(crops_dir, face.image_id, face.id)
    crop_path = save_face_crop(
        img_bgr=img_bgr,
        detection=detection,
        crops_dir=crops_dir,
        image_id=face.image_id,
        thumbnail_size=thumbnail_size,
        face_index=face.id,
        dest_path=dest,
    )
    if crop_path is None:
        return None

    old_crop_path = face.crop_path
    face.crop_path = str(crop_path)
    _update_matching_person_thumbnail(face, old_crop_path, str(crop_path))
    log.debug("Crop saved for %s", face_debug_state(face, str(crop_path)))
    return crop_path


def ensure_unique_face_crops(
    session: Session,
    faces: Iterable[Face],
    crops_dir: Path,
    thumbnail_size: Tuple[int, int],
) -> int:
    """Repair missing or duplicated crop paths among loaded face rows.

    A missing, non-canonical, or duplicated ``crop_path`` means a FaceId can
    render a stale or shared physical preview file.  Regenerating those rows
    into Face.id-keyed files keeps assignment state and preview source aligned.
    """
    face_list = [f for f in faces if f.id is not None and not f.is_excluded]
    if not face_list:
        return 0

    by_path: dict[str, list[Face]] = {}
    to_repair: dict[int, Face] = {}
    for face in face_list:
        if not face.crop_path:
            to_repair[face.id] = face
            continue
        canonical = str(canonical_face_crop_path(crops_dir, face.image_id, face.id))
        if face.crop_path != canonical:
            to_repair[face.id] = face
        by_path.setdefault(face.crop_path, []).append(face)

    for crop_path, owners in by_path.items():
        if len(owners) > 1:
            log.warning(
                "Duplicate crop_path detected: path=%s owners=%s",
                crop_path,
                [f.id for f in owners],
            )
            for face in owners:
                to_repair[face.id] = face

    repaired = 0
    image_cache: dict[int, Optional[np.ndarray]] = {}
    for face in to_repair.values():
        if face.image_id not in image_cache:
            image_cache[face.image_id] = (
                load_image_bgr(face.image.file_path) if face.image else None
            )
        crop_path = save_crop_for_face(
            face,
            crops_dir=crops_dir,
            thumbnail_size=thumbnail_size,
            img_bgr=image_cache[face.image_id],
        )
        if crop_path is not None:
            repaired += 1

    if repaired:
        session.flush()
        log.info("Repaired %d face crop preview mapping(s)", repaired)
    return repaired


def crop_path_is_shared(session: Session, face: Face) -> bool:
    """Return True when another face row points at this face's crop path."""
    if not face.crop_path:
        return False
    return (
        session.query(Face.id)
        .filter(Face.id != face.id, Face.crop_path == face.crop_path)
        .first()
        is not None
    )


def _update_matching_person_thumbnail(
    face: Face,
    old_crop_path: Optional[str],
    new_crop_path: str,
) -> None:
    if not old_crop_path or face.person_id is None:
        return
    person = face.person
    if person is None:
        return
    if person.thumbnail_path == old_crop_path:
        person.thumbnail_path = new_crop_path
