"""Tests for identity management operations."""

from __future__ import annotations

import numpy as np
import pytest

from app.db.database import init_db, session_scope
from app.db.models import Face, FaceCorrection, Image, Person
from app.services.identity_service import IdentityService


@pytest.fixture()
def db(tmp_path):
    init_db(tmp_path / "identity.db")
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


def _add_face(session, image: Image, person: Person | None, **kwargs) -> Face:
    face = Face(
        image_id=image.id,
        person_id=person.id if person else None,
        bbox_x=1,
        bbox_y=2,
        bbox_w=3,
        bbox_h=4,
        confidence=kwargs.pop("confidence", 0.9),
        detector_backend="cpu",
        crop_path=kwargs.pop("crop_path", None),
        **kwargs,
    )
    session.add(face)
    session.flush()
    return face


def test_rename_person(db):
    with session_scope() as session:
        person = _add_person(session, "Régi név", is_auto_named=True)
        pid = person.id
        img = _add_image(session, "/tmp/a.jpg")
        _add_face(session, img, person, assignment_source="auto")

    with session_scope() as session:
        updated = IdentityService(session).rename_person(pid, "  Új név  ")

    assert updated.name == "Új név"
    with session_scope() as session:
        person = session.get(Person, pid)
        assert person.is_auto_named is False
        face = session.query(Face).filter(Face.person_id == pid).one()
        assert face.assignment_source == "manual"


def test_rename_protected_person_raises(db):
    with session_scope() as session:
        person = _add_person(session, "Ismeretlen", is_protected=True)
        pid = person.id

    with session_scope() as session:
        with pytest.raises(ValueError, match="protected"):
            IdentityService(session).rename_person(pid, "Más")


def test_merge_persons_moves_faces_and_deletes_source(db):
    with session_scope() as session:
        img = _add_image(session, "/tmp/a.jpg")
        source = _add_person(session, "Forrás")
        target = _add_person(session, "Cél")
        _add_face(session, img, source)
        _add_face(session, img, source)
        source_id, target_id = source.id, target.id

    with session_scope() as session:
        result = IdentityService(session).merge_persons(source_id, target_id)

    assert result.id == target_id
    with session_scope() as session:
        assert session.get(Person, source_id) is None
        assert session.query(Face).filter(Face.person_id == target_id).count() == 2


def test_merge_persons_respects_merge_exclusions(db):
    with session_scope() as session:
        img = _add_image(session, "/tmp/a.jpg")
        source = _add_person(session, "Unknown 1", is_auto_named=True)
        target = _add_person(session, "Anna")
        keep = _add_face(session, img, source, is_merge_excluded=True)
        move = _add_face(session, img, source)
        source_id, target_id = source.id, target.id
        keep_id, move_id = keep.id, move.id

    with session_scope() as session:
        IdentityService(session).merge_persons(
            source_id, target_id, respect_merge_exclusions=True
        )

    with session_scope() as session:
        assert session.get(Person, source_id) is not None
        assert session.get(Face, keep_id).person_id == source_id
        assert session.get(Face, keep_id).is_merge_excluded is False
        assert session.get(Face, move_id).person_id == target_id


def test_merge_same_person_raises(db):
    with session_scope() as session:
        person = _add_person(session, "Egy")
        pid = person.id

    with session_scope() as session:
        with pytest.raises(ValueError, match="with itself"):
            IdentityService(session).merge_persons(pid, pid)


def test_remove_face_from_cluster_splits_face(db):
    with session_scope() as session:
        img = _add_image(session, "/tmp/a.jpg")
        person = _add_person(session, "Klaster")
        face = _add_face(session, img, person)
        pid, fid = person.id, face.id

    with session_scope() as session:
        updated = IdentityService(session).remove_face_from_cluster(fid)

    assert updated.person_id is None
    assert updated.is_excluded is True
    with session_scope() as session:
        assert session.query(Face).filter(Face.person_id == pid).count() == 0


def test_reassign_face_moves_to_target(db):
    with session_scope() as session:
        img = _add_image(session, "/tmp/a.jpg")
        source = _add_person(session, "Forrás")
        target = _add_person(session, "Cél")
        face = _add_face(session, img, source)
        fid, target_id = face.id, target.id

    with session_scope() as session:
        updated = IdentityService(session).reassign_face(fid, target_id)

    assert updated.person_id == target_id
    assert updated.assignment_source == "manual"


def test_delete_person_unassigns_faces(db):
    with session_scope() as session:
        img = _add_image(session, "/tmp/a.jpg")
        person = _add_person(session, "Törlendő")
        face = _add_face(session, img, person)
        pid, fid = person.id, face.id

    with session_scope() as session:
        result = IdentityService(session).delete_person(pid)

    assert result.n_faces == 1
    assert result.faces_deleted is False
    with session_scope() as session:
        assert session.get(Person, pid) is None
        face = session.get(Face, fid)
        assert face.person_id is None


def test_delete_person_hard_deletes_faces(db, tmp_path):
    crop = tmp_path / "crop.jpg"
    crop.write_bytes(b"x")
    with session_scope() as session:
        img = _add_image(session, "/tmp/a.jpg")
        person = _add_person(session, "Hamis detekció")
        face = _add_face(session, img, person, crop_path=str(crop))
        pid, fid = person.id, face.id

    with session_scope() as session:
        result = IdentityService(session).delete_person(pid, remove_faces=True)

    assert result.faces_deleted is True
    assert result.n_crops_removed == 1
    with session_scope() as session:
        assert session.get(Person, pid) is None
        assert session.get(Face, fid) is None
    assert not crop.exists()


def test_delete_protected_person_raises(db):
    with session_scope() as session:
        person = _add_person(session, "Ismeretlen", is_protected=True)
        pid = person.id

    with session_scope() as session:
        with pytest.raises(ValueError, match="protected"):
            IdentityService(session).delete_person(pid)


def test_exclude_face(db):
    with session_scope() as session:
        img = _add_image(session, "/tmp/a.jpg")
        person = _add_person(session, "Klaster")
        face = _add_face(session, img, person)
        fid = face.id

    with session_scope() as session:
        updated = IdentityService(session).exclude_face(fid)

    assert updated.person_id is None
    assert updated.is_excluded is True


def test_set_face_uncertainty_and_note(db):
    with session_scope() as session:
        img = _add_image(session, "/tmp/a.jpg")
        person = _add_person(session, "Teszt")
        face = _add_face(session, img, person)
        fid = face.id

    with session_scope() as session:
        updated = IdentityService(session).set_face_uncertainty(
            fid, True, note="  lehetséges  "
        )

    assert updated.is_uncertain_identification is True
    assert updated.identification_note == "lehetséges"


def test_set_face_merge_excluded(db):
    with session_scope() as session:
        img = _add_image(session, "/tmp/a.jpg")
        person = _add_person(session, "Teszt")
        face = _add_face(session, img, person)
        fid = face.id

    with session_scope() as session:
        updated = IdentityService(session).set_face_merge_excluded(fid, True)

    assert updated.is_merge_excluded is True


def test_set_and_clear_person_thumbnail(db, tmp_path):
    crop = tmp_path / "crop.jpg"
    crop.write_bytes(b"x")
    with session_scope() as session:
        img = _add_image(session, "/tmp/a.jpg")
        person = _add_person(session, "Teszt")
        face = _add_face(session, img, person, crop_path=str(crop))
        pid, fid = person.id, face.id

    with session_scope() as session:
        person = IdentityService(session).set_person_thumbnail(pid, fid)
        assert person.thumbnail_is_manual is True

    with session_scope() as session:
        person = IdentityService(session).clear_manual_person_thumbnail(pid)

    assert person.thumbnail_is_manual is False
    assert person.thumbnail_path == str(crop)


def test_record_same_and_different(db):
    with session_scope() as session:
        img = _add_image(session, "/tmp/a.jpg")
        person = _add_person(session, "Teszt")
        f1 = _add_face(session, img, person)
        f2 = _add_face(session, img, person)
        a_id, b_id = f1.id, f2.id

    with session_scope() as session:
        same = IdentityService(session).record_same(a_id, b_id)
        assert same.same_person is True
        diff = IdentityService(session).record_different(a_id, b_id)
        assert diff.same_person is False

    with session_scope() as session:
        correction = (
            session.query(FaceCorrection)
            .filter(FaceCorrection.face_id_a == min(a_id, b_id))
            .one()
        )
        assert correction.same_person is False


def test_list_persons_and_get_faces_for_person(db):
    with session_scope() as session:
        img = _add_image(session, "/tmp/a.jpg")
        anna = _add_person(session, "Anna")
        bela = _add_person(session, "Béla", is_auto_named=True)
        good = _add_face(session, img, anna)
        _add_face(session, img, anna, is_excluded=True)
        anna_id = anna.id

    with session_scope() as session:
        svc = IdentityService(session)
        named = svc.list_persons(named_only=True)
        assert [p.name for p in named] == ["Anna"]
        search = svc.list_persons(search="ann")
        assert len(search) == 1
        faces = svc.get_faces_for_person(anna_id)
        assert [f.id for f in faces] == [good.id]


def test_cleanup_empty_unknown_persons(db):
    with session_scope() as session:
        empty = _add_person(session, "Unknown 9", is_auto_named=True)
        empty_id = empty.id

    with session_scope() as session:
        deleted = IdentityService(session).cleanup_empty_unknown_persons()

    assert deleted == 1
    with session_scope() as session:
        assert session.get(Person, empty_id) is None


def test_merge_all_excluded_raises(db):
    with session_scope() as session:
        img = _add_image(session, "/tmp/a.jpg")
        source = _add_person(session, "Unknown 1", is_auto_named=True)
        target = _add_person(session, "Anna")
        _add_face(session, img, source, is_merge_excluded=True)
        source_id, target_id = source.id, target.id

    with session_scope() as session:
        with pytest.raises(ValueError, match="nothing to merge"):
            IdentityService(session).merge_persons(
                source_id, target_id, respect_merge_exclusions=True
            )


def test_reassign_faces_bulk(db):
    with session_scope() as session:
        img = _add_image(session, "/tmp/a.jpg")
        source = _add_person(session, "Forrás")
        target = _add_person(session, "Cél")
        faces = [_add_face(session, img, source) for _ in range(3)]
        move_ids = [f.id for f in faces[:2]]
        source_id, target_id = source.id, target.id

    with session_scope() as session:
        result = IdentityService(session).reassign_faces_bulk(move_ids, target_id)

    assert result.moved_count == 2
    with session_scope() as session:
        assert session.query(Face).filter(Face.person_id == source_id).count() == 1
        assert session.query(Face).filter(Face.person_id == target_id).count() == 2


def test_restore_face_assignments(db):
    with session_scope() as session:
        img = _add_image(session, "/tmp/a.jpg")
        source = _add_person(session, "Unknown 2", is_auto_named=True)
        target = _add_person(session, "Cél")
        face = _add_face(session, img, source)
        source_id, target_id, fid = source.id, target.id, face.id

    with session_scope() as session:
        result = IdentityService(session).reassign_faces_bulk([fid], target_id)

    with session_scope() as session:
        restored = IdentityService(session).restore_face_assignments(
            result.snapshots, result.removed_persons
        )

    assert restored == 1
    with session_scope() as session:
        assert session.get(Face, fid).person_id == source_id
