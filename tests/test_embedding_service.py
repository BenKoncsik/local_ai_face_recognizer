"""Tests for EmbeddingService — in particular the reembed_all() trigger."""

from __future__ import annotations

import cv2
import numpy as np

from app.config import AppConfig
from app.db.database import init_db, session_scope
from app.db.models import Face, Image
from app.embeddings.base import FaceEmbedder
from app.services.embedding_service import EmbeddingService


class _CountingEmbedder(FaceEmbedder):
    """Returns a deterministic vector and counts how many times it ran."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def embedding_dim(self) -> int:
        return 4

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.full(4, float(self.calls), dtype=np.float32)


def _make_face_with_crop(tmp_path) -> int:
    crop = tmp_path / "crop.jpg"
    assert cv2.imwrite(str(crop), np.full((112, 112, 3), 128, np.uint8))
    with session_scope() as session:
        image = Image(file_path=str(tmp_path / "i.jpg"), file_hash="h", file_mtime=0.0)
        session.add(image)
        session.flush()
        face = Face(
            image_id=image.id,
            bbox_x=0, bbox_y=0, bbox_w=50, bbox_h=50,
            confidence=0.9,
            crop_path=str(crop),
        )
        session.add(face)
        session.flush()
        return face.id


def test_reembed_all_clears_and_recomputes(tmp_path):
    db_path = tmp_path / "faces.db"
    init_db(db_path)
    face_id = _make_face_with_crop(tmp_path)

    cfg = AppConfig(base_dir=str(tmp_path))
    embedder = _CountingEmbedder()

    # First pass: process_pending embeds the one pending face.
    with session_scope() as session:
        svc = EmbeddingService(session=session, embedder=embedder, config=cfg)
        assert svc.process_pending() == 1
    with session_scope() as session:
        first = session.get(Face, face_id).get_embedding()
    assert first is not None

    # A second process_pending does nothing (embedding already present).
    with session_scope() as session:
        svc = EmbeddingService(session=session, embedder=embedder, config=cfg)
        assert svc.process_pending() == 0

    # reembed_all clears and recomputes — the vector changes (new call count).
    with session_scope() as session:
        svc = EmbeddingService(session=session, embedder=embedder, config=cfg)
        assert svc.reembed_all() == 1
    with session_scope() as session:
        second = session.get(Face, face_id).get_embedding()
    assert second is not None
    assert not np.array_equal(first, second)
