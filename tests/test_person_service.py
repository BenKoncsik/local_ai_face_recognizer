"""Tests for the standalone Persons maintenance service."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.db.database import init_db, session_scope
from app.db.models import Face, Image, Person, PersonGroup, PersonGroupMembership
from app.services.person_service import PersonFilters, PersonService


@pytest.fixture()
def db(tmp_path):
    db_path = tmp_path / "persons.db"
    init_db(db_path)
    return db_path


def _add_image(session, path: str, photo_date: str | None = None) -> Image:
    image = Image(file_path=path, file_hash=f"hash_{path}", file_mtime=0.0, photo_date=photo_date)
    session.add(image)
    session.flush()
    return image


def _add_person(session, name: str, **kwargs) -> Person:
    person = Person(name=name, is_auto_named=kwargs.pop("is_auto_named", False), **kwargs)
    session.add(person)
    session.flush()
    return person


def _add_face(session, image: Image, person: Person, crop_path=None, conf=0.9, **kwargs) -> Face:
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
        **kwargs,
    )
    session.add(face)
    session.flush()
    return face


def _add_group(session, person: Person, group_name: str) -> PersonGroup:
    group = PersonGroup(name=group_name)
    session.add(group)
    session.flush()
    session.add(PersonGroupMembership(person_id=person.id, group_id=group.id))
    session.flush()
    return group


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


def test_set_thumbnail_face_not_found_raises(db):
    with session_scope() as session:
        person = _add_person(session, "Teszt")
        pid = person.id

    with session_scope() as session:
        with pytest.raises(ValueError, match="nem található"):
            PersonService(session).set_thumbnail_from_face(pid, 999_999)


def test_list_face_crops_orders_by_confidence_and_skips_excluded(db, tmp_path):
    crop_lo = tmp_path / "lo.jpg"
    crop_hi = tmp_path / "hi.jpg"
    crop_lo.write_bytes(b"x")
    crop_hi.write_bytes(b"x")
    with session_scope() as session:
        img = _add_image(session, "/tmp/a.jpg")
        person = _add_person(session, "Teszt")
        low = _add_face(
            session,
            img,
            person,
            crop_path=str(crop_lo),
            conf=0.4,
            is_uncertain_identification=True,
            identification_note="bizonytalan",
        )
        high = _add_face(session, img, person, crop_path=str(crop_hi), conf=0.95)
        _add_face(session, img, person, crop_path=str(crop_hi), conf=0.99, is_excluded=True)
        pid = person.id
        low_id, high_id = low.id, high.id

    with session_scope() as session:
        crops = PersonService(session).list_face_crops(pid)
        assert [c.face_id for c in crops] == [high_id, low_id]
        assert crops[0].crop_path == str(crop_hi)
        assert crops[0].image_path == "/tmp/a.jpg"
        assert crops[0].confidence == pytest.approx(0.95)
        assert crops[0].is_uncertain is False
        assert crops[1].is_uncertain is True
        assert crops[1].identification_note == "bizonytalan"


def test_list_images_for_person_returns_distinct_sorted_paths(db):
    with session_scope() as session:
        img_a = _add_image(session, "/tmp/z_last.jpg", photo_date="1950")
        img_b = _add_image(session, "/tmp/a_first.jpg", photo_date="1950")
        img_c = _add_image(session, "/tmp/middle.jpg", photo_date="1960")
        person = _add_person(session, "Teszt")
        _add_face(session, img_a, person)
        _add_face(session, img_b, person)
        _add_face(session, img_c, person)
        _add_face(session, img_a, person, conf=0.5)  # duplicate image
        _add_face(session, img_b, person, conf=0.6, is_excluded=True)
        pid = person.id

    with session_scope() as session:
        paths = PersonService(session).list_images_for_person(pid)
        assert paths == ["/tmp/a_first.jpg", "/tmp/z_last.jpg", "/tmp/middle.jpg"]


def test_list_persons_includes_groups_and_flags(db, tmp_path):
    thumb = tmp_path / "manual.jpg"
    thumb.write_bytes(b"x")
    with session_scope() as session:
        person = _add_person(
            session,
            "Részletes",
            family_code="R1",
            external_family_code="EXT",
            last_name="Nagy",
            first_name="Anna",
            gender="female",
            birth_place="Budapest",
            birth_date="1950",
            notes="jegyzet",
            is_auto_named=True,
            is_protected=True,
            thumbnail_path=str(thumb),
        )
        _add_group(session, person, "Kórus")
        _add_group(session, person, "Munkahely")
        pid = person.id

    with session_scope() as session:
        summary = next(s for s in PersonService(session).list_persons() if s.person_id == pid)
        assert summary.thumbnail_path == str(thumb)
        assert summary.family_code == "R1"
        assert summary.external_family_code == "EXT"
        assert summary.last_name == "Nagy"
        assert summary.first_name == "Anna"
        assert summary.gender == "female"
        assert summary.birth_place == "Budapest"
        assert summary.birth_date == "1950"
        assert summary.notes == "jegyzet"
        assert summary.is_auto_named is True
        assert summary.is_protected is True
        assert summary.groups == ["Kórus", "Munkahely"]


def test_list_persons_thumbnail_fallback_skips_excluded_faces(db, tmp_path):
    excluded_crop = tmp_path / "excluded.jpg"
    included_crop = tmp_path / "included.jpg"
    excluded_crop.write_bytes(b"x")
    included_crop.write_bytes(b"x")
    with session_scope() as session:
        img = _add_image(session, "/tmp/a.jpg")
        person = _add_person(session, "Unknown 1")
        _add_face(session, img, person, crop_path=str(excluded_crop), conf=0.99, is_excluded=True)
        _add_face(session, img, person, crop_path=str(included_crop), conf=0.2)
        pid = person.id

    with session_scope() as session:
        summary = next(s for s in PersonService(session).list_persons() if s.person_id == pid)
        assert summary.thumbnail_path == str(included_crop)


def test_rename_person_clears_auto_named(db):
    with session_scope() as session:
        person = _add_person(session, "Unknown 3", is_auto_named=True)
        pid = person.id

    with session_scope() as session:
        updated = PersonService(session).rename_person(pid, "Valódi név")
        assert updated.name == "Valódi név"

    with session_scope() as session:
        person = session.get(Person, pid)
        assert person.name == "Valódi név"
        assert person.is_auto_named is False


def test_update_person_ignores_unknown_and_name_fields(db):
    with session_scope() as session:
        person = _add_person(session, "Eredeti név")
        pid = person.id

    with session_scope() as session:
        PersonService(session).update_person(
            pid,
            name="Más név",
            unknown_field="skip",
            gender="male",
            married_name="Tót",
        )

    with session_scope() as session:
        person = session.get(Person, pid)
        assert person.name == "Eredeti név"
        assert person.gender == "male"
        assert person.married_name == "Tót"
        assert not hasattr(person, "unknown_field")


def test_update_person_writes_all_editable_fields(db):
    with session_scope() as session:
        person = _add_person(session, "Teszt")
        pid = person.id

    with session_scope() as session:
        PersonService(session).update_person(
            pid,
            gender="male",
            family_code="G1",
            external_family_code="EXT1",
            last_name="Nagy",
            first_name="Péter",
            second_name="János",
            nickname="Peti",
            married_name="Kiss",
            birth_place="Debrecen",
            birth_date="1940",
            death_place="Budapest",
            death_date="2010",
            notes="megjegyzés",
        )

    with session_scope() as session:
        person = session.get(Person, pid)
        assert person.gender == "male"
        assert person.family_code == "G1"
        assert person.external_family_code == "EXT1"
        assert person.last_name == "Nagy"
        assert person.first_name == "Péter"
        assert person.second_name == "János"
        assert person.nickname == "Peti"
        assert person.married_name == "Kiss"
        assert person.birth_place == "Debrecen"
        assert person.birth_date == "1940"
        assert person.death_place == "Budapest"
        assert person.death_date == "2010"
        assert person.notes == "megjegyzés"


@patch("app.services.family_service.FamilyService")
def test_update_person_with_family_code_links_derived_parents(mock_family_cls, db):
    mock_family = MagicMock()
    mock_family_cls.return_value = mock_family
    with session_scope() as session:
        person = _add_person(session, "Gyerek")
        pid = person.id

    with session_scope() as session:
        PersonService(session).update_person(pid, family_code="G1{1}2")

    mock_family_cls.assert_called_once()
    mock_family.link_derived_parents.assert_called_once_with(pid)


@patch("app.services.family_service.FamilyService")
def test_link_derived_parents_swallows_errors(mock_family_cls, db):
    mock_family = MagicMock()
    mock_family.link_derived_parents.side_effect = RuntimeError("cycle")
    mock_family_cls.return_value = mock_family
    with session_scope() as session:
        person = _add_person(session, "Gyerek")
        pid = person.id

    with session_scope() as session:
        # Must not raise — save should still succeed.
        PersonService(session).update_person(pid, family_code="G1")


def test_require_person_raises_for_missing_id(db):
    with session_scope() as session:
        with pytest.raises(ValueError, match="Person id=424242 not found"):
            PersonService(session).update_person(424242, notes="x")
