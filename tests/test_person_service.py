"""Tests for the standalone Persons maintenance service."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.db.database import init_db, session_scope
from app.db.models import Face, Image, Person
from app.services.person_service import PersonFilters, PersonService


@pytest.fixture()
def db(tmp_path):
    db_path = tmp_path / "persons.db"
    init_db(db_path)
    return db_path


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


def _add_face(session, image: Image, person: Person, crop_path=None, conf=0.9) -> Face:
    face = Face(
        image_id=image.id,
        person_id=person.id,
        bbox_x=1,
        bbox_y=2,
        bbox_w=3,
        bbox_h=4,
        confidence=conf,
        detector_backend="cpu",
        crop_path=crop_path,
    )
    session.add(face)
    session.flush()
    return face


def test_list_persons_with_counts(db):
    with session_scope() as session:
        img1 = _add_image(session, "/tmp/a.jpg")
        img2 = _add_image(session, "/tmp/b.jpg")
        person = _add_person(session, "Kovács János", family_code="K1")
        _add_face(session, img1, person)
        _add_face(session, img2, person)

    with session_scope() as session:
        summaries = PersonService(session).list_persons()
        assert len(summaries) == 1
        s = summaries[0]
        assert s.name == "Kovács János"
        assert s.family_code == "K1"
        assert s.face_count == 2
        assert s.image_count == 2


def test_list_persons_thumbnail_fallback_to_crop(db, tmp_path):
    crop_lo = tmp_path / "lo.jpg"
    crop_hi = tmp_path / "hi.jpg"
    crop_lo.write_bytes(b"x")
    crop_hi.write_bytes(b"x")
    with session_scope() as session:
        img = _add_image(session, "/tmp/a.jpg")
        # No explicit thumbnail_path on the person.
        person = _add_person(session, "Unknown 9")
        _add_face(session, img, person, crop_path=str(crop_lo), conf=0.5)
        _add_face(session, img, person, crop_path=str(crop_hi), conf=0.95)
        pid = person.id

    with session_scope() as session:
        summaries = PersonService(session).list_persons()
        s = next(x for x in summaries if x.person_id == pid)
        # Falls back to the highest-confidence crop.
        assert s.thumbnail_path == str(crop_hi)


def test_list_persons_filter_by_name_and_code(db):
    with session_scope() as session:
        _add_person(session, "Anna", nickname="Anci", family_code="A1")
        _add_person(session, "Béla", family_code="B2")

    with session_scope() as session:
        svc = PersonService(session)
        by_name = svc.list_persons(PersonFilters(name="anci"))
        assert [s.name for s in by_name] == ["Anna"]
        by_code = svc.list_persons(PersonFilters(family_code="b2"))
        assert [s.name for s in by_code] == ["Béla"]


def test_update_person_normalises_empty_strings(db):
    with session_scope() as session:
        person = _add_person(session, "Teszt")
        pid = person.id

    with session_scope() as session:
        PersonService(session).update_person(
            pid,
            last_name="Nagy",
            first_name="",       # blank → None
            family_code="  ",     # blank → None (so unique index is not hit)
            notes="megjegyzés",
        )

    with session_scope() as session:
        person = session.get(Person, pid)
        assert person.last_name == "Nagy"
        assert person.first_name is None
        assert person.family_code is None
        assert person.notes == "megjegyzés"


def test_rename_rejects_empty_and_protected(db):
    with session_scope() as session:
        normal = _add_person(session, "Normál")
        protected = _add_person(session, "Ismeretlen", is_protected=True)
        normal_id, protected_id = normal.id, protected.id

    with session_scope() as session:
        svc = PersonService(session)
        with pytest.raises(ValueError):
            svc.rename_person(normal_id, "   ")
        with pytest.raises(ValueError):
            svc.rename_person(protected_id, "Másik név")

    with session_scope() as session:
        svc = PersonService(session)
        svc.rename_person(normal_id, "Új név")
    with session_scope() as session:
        assert session.get(Person, normal_id).name == "Új név"


def test_set_thumbnail_from_face(db, tmp_path):
    crop = tmp_path / "crop.jpg"
    crop.write_bytes(b"fakejpeg")
    with session_scope() as session:
        img = _add_image(session, "/tmp/a.jpg")
        person = _add_person(session, "Teszt")
        face = _add_face(session, img, person, crop_path=str(crop))
        pid, fid = person.id, face.id

    with session_scope() as session:
        PersonService(session).set_thumbnail_from_face(pid, fid)

    with session_scope() as session:
        person = session.get(Person, pid)
        assert person.thumbnail_path == str(crop)
        assert person.thumbnail_is_manual is True


def test_set_thumbnail_missing_crop_raises(db):
    with session_scope() as session:
        img = _add_image(session, "/tmp/a.jpg")
        person = _add_person(session, "Teszt")
        face = _add_face(session, img, person, crop_path="/nonexistent/x.jpg")
        pid, fid = person.id, face.id

    with session_scope() as session:
        with pytest.raises(ValueError):
            PersonService(session).set_thumbnail_from_face(pid, fid)


def test_set_thumbnail_wrong_person_raises(db, tmp_path):
    crop = tmp_path / "crop.jpg"
    crop.write_bytes(b"fakejpeg")
    with session_scope() as session:
        img = _add_image(session, "/tmp/a.jpg")
        p1 = _add_person(session, "Egy")
        p2 = _add_person(session, "Kettő")
        face = _add_face(session, img, p1, crop_path=str(crop))
        p2_id, fid = p2.id, face.id

    with session_scope() as session:
        with pytest.raises(ValueError):
            PersonService(session).set_thumbnail_from_face(p2_id, fid)
