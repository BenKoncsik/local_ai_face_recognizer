"""Tests for EmbeddingService — in particular the reembed_all() trigger."""

from __future__ import annotations

import cv2
import numpy as np

from app.config import AppConfig
from app.db.database import init_db, session_scope
from app.db.models import Face, Image
from app.embeddings.base import FaceEmbedder
from app.services.embedding_service import EmbeddingService, embed_manual_face


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


def _make_manual_face_with_crop(tmp_path) -> int:
    """A manually-marked face: detector_backend='manual', crop but no vector."""
    crop = tmp_path / "manual_crop.jpg"
    assert cv2.imwrite(str(crop), np.full((112, 112, 3), 90, np.uint8))
    with session_scope() as session:
        image = Image(file_path=str(tmp_path / "m.jpg"), file_hash="mh", file_mtime=0.0)
        session.add(image)
        session.flush()
        face = Face(
            image_id=image.id,
            bbox_x=0, bbox_y=0, bbox_w=50, bbox_h=50,
            confidence=1.0,
            detector_backend="manual",
            crop_path=str(crop),
        )
        session.add(face)
        session.flush()
        return face.id


def test_embed_face_single(tmp_path):
    """embed_face() embeds one already-saved face (used for manual marks)."""
    db_path = tmp_path / "faces.db"
    init_db(db_path)
    face_id = _make_manual_face_with_crop(tmp_path)

    cfg = AppConfig(base_dir=str(tmp_path))
    embedder = _CountingEmbedder()

    with session_scope() as session:
        face = session.get(Face, face_id)
        assert face.get_embedding() is None  # manual face starts without a vector
        svc = EmbeddingService(session=session, embedder=embedder, config=cfg)
        assert svc.embed_face(face) is True

    with session_scope() as session:
        assert session.get(Face, face_id).get_embedding() is not None


def test_embed_manual_face_makes_face_comparable(tmp_path):
    """embed_manual_face() gives a manual face a usable embedding so it can be
    scored for face-match ordering in the person-assign UI."""
    from app.services.recognition_service import RecognitionService

    db_path = tmp_path / "faces.db"
    init_db(db_path)
    face_id = _make_manual_face_with_crop(tmp_path)

    cfg = AppConfig(base_dir=str(tmp_path))

    with session_scope() as session:
        face = session.get(Face, face_id)
        # Best-effort helper builds its own (default) embedder; with no model
        # present it falls back to a stub embedder but still returns a vector.
        assert embed_manual_face(session, face, cfg) is True
        face = session.get(Face, face_id)
        embedding = face.get_embedding()
        assert embedding is not None
        # score_persons must accept the manual face's embedding without error.
        scores = RecognitionService(session, None).score_persons(embedding)
        assert isinstance(scores, dict)


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
