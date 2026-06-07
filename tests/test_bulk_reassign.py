"""Tests for batch face reassignment and its undo (IdentityService)."""

from __future__ import annotations

import pytest

from app.db.database import init_db, session_scope
from app.db.models import Face, Image, Person
from app.services.identity_service import IdentityService


@pytest.fixture()
def db(tmp_path):
    init_db(tmp_path / "bulk.db")
    return tmp_path


def _add_image(session, path: str) -> Image:
    image = Image(file_path=path, file_hash=f"hash_{path}", file_mtime=0.0)
    session.add(image)
    session.flush()
    return image


def _add_person(session, name: str, **kwargs) -> Person:
    person = Person(name=name, is_auto_named=kwargs.pop("is_auto_named", False), **kwargs)
    session.add(person)
    session.flush()
    return person


def _add_face(session, image: Image, person: Person, crop_path=None) -> Face:
    face = Face(
        image_id=image.id,
        person_id=person.id,
        bbox_x=1,
        bbox_y=2,
        bbox_w=3,
        bbox_h=4,
        confidence=0.9,
        detector_backend="cpu",
        crop_path=crop_path,
    )
    session.add(face)
    session.flush()
    return face


def test_bulk_move_some_faces(db):
    with session_scope() as session:
        img = _add_image(session, "/tmp/a.jpg")
        roni = _add_person(session, "Roni")
        aniko = _add_person(session, "Anikó")
        faces = [_add_face(session, img, roni) for _ in range(8)]
        move_ids = [faces[0].id, faces[1].id, faces[2].id]
        roni_id, aniko_id = roni.id, aniko.id

    with session_scope() as session:
        result = IdentityService(session).reassign_faces_bulk(move_ids, aniko_id)

    assert result.moved_count == 3
    assert not result.skipped_face_ids

    with session_scope() as session:
        assert session.query(Face).filter(Face.person_id == roni_id).count() == 5
        assert session.query(Face).filter(Face.person_id == aniko_id).count() == 3
        for fid in move_ids:
            face = session.get(Face, fid)
            assert face.person_id == aniko_id
            assert face.assignment_source == "manual"


def test_bulk_move_is_atomic_with_missing_target(db):
    with session_scope() as session:
        img = _add_image(session, "/tmp/a.jpg")
        roni = _add_person(session, "Roni")
        faces = [_add_face(session, img, roni) for _ in range(3)]
        ids = [f.id for f in faces]
        roni_id = roni.id

    with pytest.raises(ValueError):
        with session_scope() as session:
            IdentityService(session).reassign_faces_bulk(ids, 9999)

    # Nothing moved.
    with session_scope() as session:
        assert session.query(Face).filter(Face.person_id == roni_id).count() == 3


def test_bulk_move_skips_missing_and_already_assigned(db):
    with session_scope() as session:
        img = _add_image(session, "/tmp/a.jpg")
        roni = _add_person(session, "Roni")
        aniko = _add_person(session, "Anikó")
        f_roni = _add_face(session, img, roni)
        f_aniko = _add_face(session, img, aniko)
        roni_fid, aniko_fid, aniko_id = f_roni.id, f_aniko.id, aniko.id

    with session_scope() as session:
        # 12345 does not exist; aniko_fid already belongs to the target.
        result = IdentityService(session).reassign_faces_bulk(
            [roni_fid, aniko_fid, 12345], aniko_id
        )

    assert result.moved_count == 1
    assert set(result.skipped_face_ids) == {aniko_fid, 12345}


def test_undo_restores_original_person(db):
    with session_scope() as session:
        img = _add_image(session, "/tmp/a.jpg")
        roni = _add_person(session, "Roni")
        aniko = _add_person(session, "Anikó")
        faces = [_add_face(session, img, roni) for _ in range(3)]
        ids = [f.id for f in faces]
        roni_id, aniko_id = roni.id, aniko.id

    with session_scope() as session:
        result = IdentityService(session).reassign_faces_bulk(ids, aniko_id)

    with session_scope() as session:
        restored = IdentityService(session).restore_face_assignments(
            result.snapshots, result.removed_persons
        )

    assert restored == 3
    with session_scope() as session:
        assert session.query(Face).filter(Face.person_id == roni_id).count() == 3
        assert session.query(Face).filter(Face.person_id == aniko_id).count() == 0


def test_undo_recreates_emptied_unknown_cluster(db):
    """Moving every face from an auto-named Unknown deletes it; undo brings it back."""
    with session_scope() as session:
        img = _add_image(session, "/tmp/a.jpg")
        unknown = _add_person(session, "Unknown 5", is_auto_named=True)
        target = _add_person(session, "Anikó")
        faces = [_add_face(session, img, unknown) for _ in range(2)]
        ids = [f.id for f in faces]
        unknown_id, target_id = unknown.id, target.id

    with session_scope() as session:
        result = IdentityService(session).reassign_faces_bulk(ids, target_id)

    # Source cluster was cleaned up and captured for undo.
    with session_scope() as session:
        assert session.get(Person, unknown_id) is None
    assert any(p["id"] == unknown_id for p in result.removed_persons)

    with session_scope() as session:
        IdentityService(session).restore_face_assignments(
            result.snapshots, result.removed_persons
        )

    with session_scope() as session:
        restored_person = session.get(Person, unknown_id)
        assert restored_person is not None
        assert restored_person.name == "Unknown 5"
        assert session.query(Face).filter(Face.person_id == unknown_id).count() == 2
