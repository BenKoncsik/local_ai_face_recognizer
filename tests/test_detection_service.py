"""Unit tests for face detection service."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.config import AppConfig
from app.db.database import init_db, session_scope
from app.db.models import Face, Image
from app.detectors.base import Detection, FaceDetector
from app.services.detection_service import DetectionService


class _DummyDetector(FaceDetector):
    @property
    def backend_name(self) -> str:
        return "dummy"

    def detect(
        self,
        image_bgr: np.ndarray,
        confidence_threshold: float = 0.5,
        min_face_size: int = 50,
    ):
        return [
            Detection(x=10, y=10, w=70, h=70, confidence=0.95),
            Detection(x=120, y=15, w=65, h=65, confidence=0.92),
        ]


def test_detection_service_uses_face_id_for_crop_naming(monkeypatch, tmp_path):
    """Each crop filename must be keyed by the face's DB primary key.

    Using face.id (rather than a per-image sequential counter) guarantees that
    later bbox-update operations can overwrite the exact same file without
    accidentally colliding with another face's crop stored in the same image.
    """
    db_path = tmp_path / "faces.db"
    init_db(db_path)

    image_path = tmp_path / "family.jpg"
    img = np.full((240, 320, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), img)

    with session_scope() as session:
        image = Image(
            file_path=str(image_path),
            file_hash="hash",
            file_mtime=image_path.stat().st_mtime,
        )
        session.add(image)
        session.flush()
        image_id = image.id

    captured_indexes: list[int] = []

    def fake_save_face_crop(
        img_bgr,
        detection,
        crops_dir,
        image_id,
        thumbnail_size,
        face_index=0,
        dest_path=None,
    ):
        captured_indexes.append(face_index)
        dest = dest_path or (Path(crops_dir) / f"img{image_id:06d}_face{face_index:06d}.jpg")
        return dest

    monkeypatch.setattr(
        "app.services.detection_service.save_face_crop",
        fake_save_face_crop,
    )

    cfg = AppConfig(base_dir=str(tmp_path))
    cfg.storage.db_path = str(db_path)
    cfg.storage.crops_dir = "crops"

    with session_scope() as session:
        service = DetectionService(
            session=session,
            detector=_DummyDetector(),
            config=cfg,
        )
        detected = service.process([image_id])

    assert detected == 2

    with session_scope() as session:
        faces = session.query(Face).order_by(Face.id).all()

    assert len(faces) == 2
    face_ids = [f.id for f in faces]

    # face_index passed to save_face_crop must be the face's actual DB id,
    # not a per-image sequential counter (0, 1, 2, ...).
    assert captured_indexes == face_ids, (
        "crop filename index must equal face.id to prevent collisions "
        "with future bbox-update operations"
    )

    # Every face must have a distinct, non-None crop path.
    assert faces[0].crop_path is not None
    assert faces[1].crop_path is not None
    assert faces[0].crop_path != faces[1].crop_path
