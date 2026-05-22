"""Unit tests for the clustering service."""

from __future__ import annotations

import numpy as np
import pytest

from app.config import ClusteringConfig
from app.db.database import init_db, session_scope
from app.db.models import Face, Image, Person
from app.services.clustering_service import ClusteringService


@pytest.fixture()
def tmp_db(tmp_path):
    """Initialise a fresh SQLite database for each test."""
    db_file = tmp_path / "test.db"
    init_db(db_file)
    return db_file


def _axis_vec(dim: int, index: int, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[index] = 1.0
    if noise > 0:
        v += np.random.default_rng(seed).normal(0, noise, dim).astype(np.float32)
    return v


def _add_face(session, image_id: int, embedding: np.ndarray) -> int:
    face = Face(
        image_id=image_id,
        bbox_x=0, bbox_y=0, bbox_w=20, bbox_h=20,
        confidence=1.0,
        detector_backend="cpu",
    )
    face.set_embedding(embedding)
    session.add(face)
    session.flush()
    return face.id


class TestOrphanCleanup:
    def test_run_removes_orphan_auto_persons(self, tmp_db):
        """Auto-named persons left with no faces must be deleted after a run."""
        with session_scope() as s:
            img = Image(file_path="/i.jpg", file_hash="h", file_mtime=0.0)
            s.add(img)
            s.flush()
            _add_face(s, img.id, _axis_vec(64, 0, noise=0.02, seed=1))
            _add_face(s, img.id, _axis_vec(64, 0, noise=0.02, seed=2))

            # Stale auto-named persons from a previous run — no faces.
            for i in range(1, 4):
                s.add(Person(name=f"Unknown {i}", is_auto_named=True))
            # A manually-named, face-less person must survive.
            s.add(Person(name="Kovács Béla", is_auto_named=False))

        with session_scope() as s:
            ClusteringService(s, ClusteringConfig()).run()

        with session_scope() as s:
            names = {p.name for p in s.query(Person).all()}
            assert "Unknown 1" not in names
            assert "Unknown 2" not in names
            assert "Unknown 3" not in names
            # The manually-named person is kept even with no faces.
            assert "Kovács Béla" in names
            # Every surviving person has at least one face.
            for person in s.query(Person).all():
                assert len(person.faces) >= 1 or not person.is_auto_named

    def test_run_does_not_orphan_assigned_faces(self, tmp_db):
        """Re-running clustering must leave every face assigned to a person."""
        with session_scope() as s:
            img = Image(file_path="/i.jpg", file_hash="h", file_mtime=0.0)
            s.add(img)
            s.flush()
            for seed in range(4):
                _add_face(s, img.id, _axis_vec(64, 0, noise=0.02, seed=seed))

        with session_scope() as s:
            ClusteringService(s, ClusteringConfig()).run()
        with session_scope() as s:
            ClusteringService(s, ClusteringConfig()).run()

        with session_scope() as s:
            faces = s.query(Face).all()
            assert all(f.person_id is not None for f in faces)
            # No face-less auto-named persons remain after a second run.
            for person in s.query(Person).filter(Person.is_auto_named == True).all():  # noqa: E712
                assert len(person.faces) >= 1
