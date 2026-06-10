"""Unit tests for the permanent face ignore list (IgnoredFaceService)."""

from __future__ import annotations

import numpy as np
import pytest

from app.config import ClusteringConfig, IgnoredFaceConfig
from app.db.database import init_db, session_scope
from app.db.models import Face, IgnoredFace, Image, Person
from app.services.clustering_service import ClusteringService
from app.services.identity_service import IdentityService
from app.services.ignored_face_service import IgnoredFaceService


@pytest.fixture()
def tmp_db(tmp_path):
    """Initialise a fresh SQLite database for each test."""
    db_file = tmp_path / "test.db"
    init_db(db_file)
    return db_file


DIM = 64


def _axis_vec(index: int, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[index] = 1.0
    if noise > 0:
        v += np.random.default_rng(seed).normal(0, noise, DIM).astype(np.float32)
    return v


def _add_image(session, path: str = "/img.jpg") -> int:
    img = Image(file_path=path, file_hash=path, file_mtime=0.0)
    session.add(img)
    session.flush()
    return img.id


def _add_face(
    session,
    image_id: int,
    embedding: np.ndarray,
    person_id: int | None = None,
    crop_path: str | None = None,
) -> int:
    face = Face(
        image_id=image_id,
        person_id=person_id,
        bbox_x=0, bbox_y=0, bbox_w=20, bbox_h=20,
        confidence=1.0,
        detector_backend="cpu",
        crop_path=crop_path,
    )
    face.set_embedding(embedding)
    session.add(face)
    session.flush()
    return face.id


def _make_unknown_person(session, name: str, image_id: int, axis: int, n_faces: int = 2):
    person = Person(name=name, is_auto_named=True)
    session.add(person)
    session.flush()
    face_ids = [
        _add_face(
            session, image_id,
            _axis_vec(axis, noise=0.02, seed=i),
            person_id=person.id,
            crop_path=f"/crops/{name}_{i}.jpg",
        )
        for i in range(n_faces)
    ]
    return person.id, face_ids


class TestIgnorePersonForever:
    def test_ignored_person_does_not_reappear_after_rerun(self, tmp_db):
        """Embeddings re-detected after a re-run must be suppressed, not re-clustered."""
        with session_scope() as s:
            img_id = _add_image(s)
            person_id, _ = _make_unknown_person(s, "Unknown 327", img_id, axis=0)

        with session_scope() as s:
            n = IgnoredFaceService(s).ignore_person_forever(person_id)
            assert n == 2

        with session_scope() as s:
            assert s.get(Person, person_id) is None
            assert s.query(IgnoredFace).count() == 2
            # Original faces are hidden.
            for face in s.query(Face).all():
                assert face.is_excluded
                assert face.person_id is None

        # Simulate a re-run: the same physical face is re-detected with a
        # near-identical embedding and no person assignment yet.
        with session_scope() as s:
            img2 = _add_image(s, "/img2.jpg")
            new_ids = [
                _add_face(s, img2, _axis_vec(0, noise=0.02, seed=10 + i))
                for i in range(2)
            ]

        with session_scope() as s:
            stats = IgnoredFaceService(s).suppress_matching_unassigned()
            assert stats.n_suppressed == 2

        with session_scope() as s:
            for fid in new_ids:
                face = s.get(Face, fid)
                assert face.is_excluded
                assert face.assignment_source == "ignored"
            # Clustering must not resurrect the person from suppressed faces.
            cluster_stats = ClusteringService(s, ClusteringConfig()).cluster_unassigned()
            assert cluster_stats.n_new_persons == 0
            assert s.query(Person).count() == 0

    def test_plain_delete_lets_person_reappear(self, tmp_db):
        """Without ignore-forever, re-clustering recreates the Unknown person."""
        with session_scope() as s:
            img_id = _add_image(s)
            person_id, _ = _make_unknown_person(s, "Unknown 1", img_id, axis=0)

        with session_scope() as s:
            IdentityService(s).delete_person(person_id)

        with session_scope() as s:
            stats = IgnoredFaceService(s).suppress_matching_unassigned()
            assert stats.n_suppressed == 0
            cluster_stats = ClusteringService(s, ClusteringConfig()).cluster_unassigned()
            assert cluster_stats.n_new_persons == 1

    def test_named_person_cannot_be_ignored(self, tmp_db):
        with session_scope() as s:
            img_id = _add_image(s)
            person = Person(name="Kati", is_auto_named=False)
            s.add(person)
            s.flush()
            _add_face(s, img_id, _axis_vec(0), person_id=person.id)
            person_id = person.id

        with session_scope() as s:
            with pytest.raises(ValueError):
                IgnoredFaceService(s).ignore_person_forever(person_id)
            assert s.query(IgnoredFace).count() == 0
            assert s.get(Person, person_id) is not None

    def test_protected_person_cannot_be_ignored(self, tmp_db):
        with session_scope() as s:
            person = Person(name="Ismeretlen", is_auto_named=True, is_protected=True)
            s.add(person)
            s.flush()
            person_id = person.id

        with session_scope() as s:
            with pytest.raises(ValueError):
                IgnoredFaceService(s).ignore_person_forever(person_id)


class TestIgnoreFaceForever:
    def test_single_face_ignore_keeps_person_with_remaining_faces(self, tmp_db):
        with session_scope() as s:
            img_id = _add_image(s)
            person_id, face_ids = _make_unknown_person(s, "Unknown 3", img_id, axis=0)

        with session_scope() as s:
            IgnoredFaceService(s).ignore_face_forever(face_ids[0])

        with session_scope() as s:
            assert s.query(IgnoredFace).count() == 1
            face = s.get(Face, face_ids[0])
            assert face.is_excluded and face.person_id is None
            # Person still has its other face, so it survives.
            assert s.get(Person, person_id) is not None
            assert not s.get(Face, face_ids[1]).is_excluded

    def test_ignoring_last_face_removes_empty_unknown_person(self, tmp_db):
        with session_scope() as s:
            img_id = _add_image(s)
            person_id, face_ids = _make_unknown_person(
                s, "Unknown 4", img_id, axis=0, n_faces=1
            )

        with session_scope() as s:
            IgnoredFaceService(s).ignore_face_forever(face_ids[0])

        with session_scope() as s:
            assert s.get(Person, person_id) is None
            assert s.query(IgnoredFace).count() == 1

    def test_named_person_face_cannot_be_ignored(self, tmp_db):
        with session_scope() as s:
            img_id = _add_image(s)
            person = Person(name="Zsuzsi", is_auto_named=False)
            s.add(person)
            s.flush()
            face_id = _add_face(s, img_id, _axis_vec(0), person_id=person.id)

        with session_scope() as s:
            with pytest.raises(ValueError):
                IgnoredFaceService(s).ignore_face_forever(face_id)
            assert s.query(IgnoredFace).count() == 0

    def test_unassigned_face_can_be_ignored(self, tmp_db):
        with session_scope() as s:
            img_id = _add_image(s)
            face_id = _add_face(s, img_id, _axis_vec(2))

        with session_scope() as s:
            entry = IgnoredFaceService(s).ignore_face_forever(face_id)
            assert entry.source_face_id == face_id
            assert entry.source_person_name is None

        with session_scope() as s:
            assert s.get(Face, face_id).is_excluded


class TestSuppressionThreshold:
    def test_dissimilar_new_face_is_not_suppressed(self, tmp_db):
        """A genuinely different person must survive the ignore filter."""
        with session_scope() as s:
            img_id = _add_image(s)
            person_id, _ = _make_unknown_person(s, "Unknown 5", img_id, axis=0)

        with session_scope() as s:
            IgnoredFaceService(s).ignore_person_forever(person_id)

        with session_scope() as s:
            img2 = _add_image(s, "/img2.jpg")
            other_id = _add_face(s, img2, _axis_vec(1, noise=0.02, seed=99))

        with session_scope() as s:
            stats = IgnoredFaceService(s).suppress_matching_unassigned()
            assert stats.n_suppressed == 0
            assert not s.get(Face, other_id).is_excluded

    def test_disabled_filter_suppresses_nothing(self, tmp_db):
        with session_scope() as s:
            img_id = _add_image(s)
            person_id, _ = _make_unknown_person(s, "Unknown 7", img_id, axis=0)

        with session_scope() as s:
            IgnoredFaceService(s).ignore_person_forever(person_id)

        with session_scope() as s:
            img2 = _add_image(s, "/img2.jpg")
            _add_face(s, img2, _axis_vec(0, noise=0.02, seed=42))

        with session_scope() as s:
            cfg = IgnoredFaceConfig(enabled=False)
            stats = IgnoredFaceService(s, cfg).suppress_matching_unassigned()
            assert stats.n_suppressed == 0


class TestUnignore:
    def test_unignore_makes_face_recognisable_again(self, tmp_db):
        with session_scope() as s:
            img_id = _add_image(s)
            person_id, face_ids = _make_unknown_person(s, "Unknown 9", img_id, axis=0)

        with session_scope() as s:
            IgnoredFaceService(s).ignore_person_forever(person_id)

        with session_scope() as s:
            svc = IgnoredFaceService(s)
            entries = svc.list_ignored()
            assert len(entries) == 2
            for entry in entries:
                assert svc.unignore(entry.id)

        with session_scope() as s:
            assert s.query(IgnoredFace).count() == 0
            # Source faces are re-enabled for the next pipeline run.
            for fid in face_ids:
                face = s.get(Face, fid)
                assert not face.is_excluded
                assert face.assignment_source is None

            # New similar faces are no longer suppressed and may cluster again.
            img2 = _add_image(s, "/img2.jpg")
            _add_face(s, img2, _axis_vec(0, noise=0.02, seed=7))
            stats = IgnoredFaceService(s).suppress_matching_unassigned()
            assert stats.n_suppressed == 0

    def test_unignore_missing_entry_returns_false(self, tmp_db):
        with session_scope() as s:
            assert not IgnoredFaceService(s).unignore(12345)


class TestListMetadata:
    def test_entry_keeps_thumbnail_and_source_snapshot(self, tmp_db):
        with session_scope() as s:
            img_id = _add_image(s)
            person_id, _ = _make_unknown_person(s, "Unknown 42", img_id, axis=3)

        with session_scope() as s:
            IgnoredFaceService(s).ignore_person_forever(person_id, note="zavaró háttérarc")

        with session_scope() as s:
            entries = IgnoredFaceService(s).list_ignored()
            assert len(entries) == 2
            for entry in entries:
                assert entry.source_person_name == "Unknown 42"
                assert entry.thumbnail_path and entry.thumbnail_path.startswith("/crops/")
                assert entry.note == "zavaró háttérarc"
                assert entry.created_at is not None
                assert entry.get_embedding() is not None
