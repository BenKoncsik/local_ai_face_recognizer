"""Tests for complete person deletion (IdentityService.delete_person).

Covers the false-detection case from the task: a bogus "Unknown" person built
from mis-detected crops must be removable in one call, taking *all* of its faces
(and their crop thumbnails) with it, while leaving every other person untouched.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.config import ClusteringConfig, IgnoredFaceConfig
from app.db.database import init_db, session_scope
from app.db.models import Face, IgnoredFace, Person
from app.services.clustering_service import ClusteringService
from app.services.identity_service import IdentityService
from app.services.ignored_face_service import IgnoredFaceService

DIM = 64


@pytest.fixture()
def tmp_db(tmp_path):
    init_db(tmp_path / "test.db")
    return tmp_path


def _axis_vec(index: int, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[index] = 1.0
    if noise > 0:
        v += np.random.default_rng(seed).normal(0, noise, DIM).astype(np.float32)
    return v


def _add_image(session, path: str = "/img.jpg") -> int:
    from app.db.models import Image

    img = Image(file_path=path, file_hash=path, file_mtime=0.0)
    session.add(img)
    session.flush()
    return img.id


def _make_unknown_person(session, name, image_id, axis, crop_dir, n_faces=2):
    person = Person(name=name, is_auto_named=True)
    session.add(person)
    session.flush()
    face_ids = []
    for i in range(n_faces):
        crop = crop_dir / f"{name}_{i}.jpg"
        crop.write_bytes(b"fake-jpeg")
        face = Face(
            image_id=image_id,
            person_id=person.id,
            bbox_x=0, bbox_y=0, bbox_w=20, bbox_h=20,
            confidence=1.0,
            detector_backend="cpu",
            crop_path=str(crop),
        )
        face.set_embedding(_axis_vec(axis, noise=0.02, seed=i))
        session.add(face)
        session.flush()
        face_ids.append(face.id)
    return person.id, face_ids


class TestFullDelete:
    def test_hard_delete_removes_person_and_all_faces(self, tmp_db):
        crop_dir = tmp_db / "crops"
        crop_dir.mkdir()
        with session_scope() as s:
            img_id = _add_image(s)
            person_id, face_ids = _make_unknown_person(
                s, "Unknown 101", img_id, axis=0, crop_dir=crop_dir
            )

        with session_scope() as s:
            result = IdentityService(s).delete_person(person_id, remove_faces=True)

        assert result.faces_deleted is True
        assert result.n_faces == 2
        assert result.n_crops_removed == 2

        with session_scope() as s:
            assert s.get(Person, person_id) is None
            for fid in face_ids:
                assert s.get(Face, fid) is None
            # No orphan faces left behind anywhere.
            assert s.query(Face).count() == 0

        # Crop thumbnails are gone from disk.
        assert not any(crop_dir.iterdir())

    def test_hard_deleted_person_does_not_reappear_after_recluster(self, tmp_db):
        crop_dir = tmp_db / "crops"
        crop_dir.mkdir()
        with session_scope() as s:
            img_id = _add_image(s)
            person_id, _ = _make_unknown_person(
                s, "Unknown 101", img_id, axis=0, crop_dir=crop_dir
            )

        with session_scope() as s:
            IdentityService(s).delete_person(person_id, remove_faces=True)

        with session_scope() as s:
            stats = ClusteringService(s, ClusteringConfig()).cluster_unassigned()
            assert stats.n_new_persons == 0
            assert s.query(Person).count() == 0

    def test_other_persons_and_faces_are_untouched(self, tmp_db):
        crop_dir = tmp_db / "crops"
        crop_dir.mkdir()
        with session_scope() as s:
            img_id = _add_image(s)
            bogus_id, _ = _make_unknown_person(
                s, "Unknown 101", img_id, axis=0, crop_dir=crop_dir
            )
            keep_id, keep_faces = _make_unknown_person(
                s, "Unknown 2", img_id, axis=5, crop_dir=crop_dir
            )

        with session_scope() as s:
            IdentityService(s).delete_person(bogus_id, remove_faces=True)

        with session_scope() as s:
            assert s.get(Person, keep_id) is not None
            for fid in keep_faces:
                face = s.get(Face, fid)
                assert face is not None
                assert face.person_id == keep_id

    def test_missing_crop_file_does_not_crash(self, tmp_db):
        crop_dir = tmp_db / "crops"
        crop_dir.mkdir()
        with session_scope() as s:
            img_id = _add_image(s)
            person_id, _ = _make_unknown_person(
                s, "Unknown 101", img_id, axis=0, crop_dir=crop_dir
            )
            # Delete one crop file out from under the DB row.
            next(crop_dir.iterdir()).unlink()

        with session_scope() as s:
            result = IdentityService(s).delete_person(person_id, remove_faces=True)

        assert result.n_faces == 2
        assert result.n_crops_removed == 1  # only the surviving file was unlinked
        with session_scope() as s:
            assert s.get(Person, person_id) is None
            assert s.query(Face).count() == 0

    def test_delete_with_ignore_snapshots_embeddings_and_deletes_faces(self, tmp_db):
        crop_dir = tmp_db / "crops"
        crop_dir.mkdir()
        with session_scope() as s:
            img_id = _add_image(s)
            person_id, face_ids = _make_unknown_person(
                s, "Unknown 101", img_id, axis=0, crop_dir=crop_dir
            )

        with session_scope() as s:
            result = IdentityService(s).delete_person(
                person_id, remove_faces=True, ignore_embeddings=True
            )

        assert result.faces_deleted is True
        assert result.n_faces == 2
        assert result.n_ignored == 2
        with session_scope() as s:
            # Person and faces gone, but the embeddings live on the ignore list.
            assert s.get(Person, person_id) is None
            assert s.query(Face).count() == 0
            assert s.query(IgnoredFace).count() == 2
            # source_face_id was SET NULL by the FK when the face row vanished.
            for entry in s.query(IgnoredFace).all():
                assert entry.source_face_id is None

    def test_ignored_person_suppressed_on_redetection(self, tmp_db):
        """After delete+ignore, a re-detected matching face is suppressed."""
        crop_dir = tmp_db / "crops"
        crop_dir.mkdir()
        with session_scope() as s:
            img_id = _add_image(s)
            person_id, _ = _make_unknown_person(
                s, "Unknown 101", img_id, axis=0, crop_dir=crop_dir
            )

        with session_scope() as s:
            IdentityService(s).delete_person(
                person_id, remove_faces=True, ignore_embeddings=True
            )

        # Simulate the detector re-finding the same face on a later run.
        with session_scope() as s:
            img_id2 = _add_image(s, "/img2.jpg")
            face = Face(
                image_id=img_id2,
                bbox_x=0, bbox_y=0, bbox_w=20, bbox_h=20,
                confidence=1.0, detector_backend="cpu",
            )
            face.set_embedding(_axis_vec(0, noise=0.02, seed=99))
            s.add(face)
            s.flush()
            new_id = face.id

        with session_scope() as s:
            stats = IgnoredFaceService(s, IgnoredFaceConfig()).suppress_matching_unassigned()
            assert stats.n_suppressed == 1
            # And clustering must not spawn a fresh Unknown from it.
            cl = ClusteringService(s, ClusteringConfig()).cluster_unassigned()
            assert cl.n_new_persons == 0
            assert s.get(Face, new_id).is_excluded is True

    def test_delete_without_ignore_adds_nothing_to_ignore_list(self, tmp_db):
        crop_dir = tmp_db / "crops"
        crop_dir.mkdir()
        with session_scope() as s:
            img_id = _add_image(s)
            person_id, _ = _make_unknown_person(
                s, "Unknown 101", img_id, axis=0, crop_dir=crop_dir
            )

        with session_scope() as s:
            result = IdentityService(s).delete_person(person_id, remove_faces=True)

        assert result.n_ignored == 0
        with session_scope() as s:
            assert s.query(IgnoredFace).count() == 0

    def test_protected_person_cannot_be_deleted(self, tmp_db):
        with session_scope() as s:
            protected = Person(name="Ismeretlen", is_auto_named=False, is_protected=True)
            s.add(protected)
            s.flush()
            pid = protected.id

        with session_scope() as s:
            with pytest.raises(ValueError):
                IdentityService(s).delete_person(pid, remove_faces=True)
            assert s.get(Person, pid) is not None

    def test_default_mode_still_unassigns_faces(self, tmp_db):
        """Backward-compat: without remove_faces, faces survive un-assigned."""
        crop_dir = tmp_db / "crops"
        crop_dir.mkdir()
        with session_scope() as s:
            img_id = _add_image(s)
            person_id, face_ids = _make_unknown_person(
                s, "Unknown 1", img_id, axis=0, crop_dir=crop_dir
            )

        with session_scope() as s:
            result = IdentityService(s).delete_person(person_id)

        assert result.faces_deleted is False
        with session_scope() as s:
            assert s.get(Person, person_id) is None
            for fid in face_ids:
                face = s.get(Face, fid)
                assert face is not None
                assert face.person_id is None
        # Crop files are preserved in un-assign mode.
        assert len(list(crop_dir.iterdir())) == 2
