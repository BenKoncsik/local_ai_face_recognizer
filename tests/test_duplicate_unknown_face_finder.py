from __future__ import annotations

import pytest

from app.db.database import init_db, session_scope
from app.db.models import Face, Image, Person
from app.services.duplicate_unknown_face_finder import (
    DuplicateUnknownFaceFinder,
    is_placeholder_name,
)


@pytest.fixture()
def tmp_db(tmp_path):
    db_file = tmp_path / "faces.db"
    init_db(db_file)
    return db_file


def _add_image(session, path: str = "/tmp/family.jpg") -> Image:
    image = Image(file_path=path, file_hash=path, file_mtime=0.0)
    session.add(image)
    session.flush()
    return image


def _add_person(session, name: str, *, auto: bool = False, protected: bool = False) -> Person:
    person = Person(name=name, is_auto_named=auto, is_protected=protected)
    session.add(person)
    session.flush()
    return person


def _add_face(
    session,
    image: Image,
    bbox: tuple[int, int, int, int],
    person: Person | None = None,
) -> Face:
    x, y, w, h = bbox
    face = Face(
        image_id=image.id,
        person_id=person.id if person is not None else None,
        bbox_x=x,
        bbox_y=y,
        bbox_w=w,
        bbox_h=h,
        confidence=0.9,
        detector_backend="cpu",
    )
    session.add(face)
    session.flush()
    return face


def test_lists_known_face_overlapping_unknown(tmp_db):
    with session_scope() as session:
        image = _add_image(session)
        person = _add_person(session, "Alice")
        known = _add_face(session, image, (10, 10, 100, 100), person)
        unknown = _add_face(session, image, (20, 20, 90, 90))

    with session_scope() as session:
        finder = DuplicateUnknownFaceFinder(session, iou_threshold=0.30)
        matches = finder.find()

    assert finder.images_examined == 1
    assert len(matches) == 1
    assert matches[0].known_face_id == known.id
    assert matches[0].unknown_face_id == unknown.id
    assert matches[0].known_person_name == "Alice"
    assert matches[0].iou >= 0.30


def test_distant_unknown_is_not_listed(tmp_db):
    with session_scope() as session:
        image = _add_image(session)
        person = _add_person(session, "Alice")
        _add_face(session, image, (10, 10, 100, 100), person)
        _add_face(session, image, (220, 220, 80, 80))

    with session_scope() as session:
        matches = DuplicateUnknownFaceFinder(session, iou_threshold=0.30).find()

    assert matches == []


def test_two_known_faces_are_never_listed(tmp_db):
    with session_scope() as session:
        image = _add_image(session)
        alice = _add_person(session, "Alice")
        bob = _add_person(session, "Bob")
        _add_face(session, image, (10, 10, 100, 100), alice)
        _add_face(session, image, (20, 20, 90, 90), bob)

    with session_scope() as session:
        matches = DuplicateUnknownFaceFinder(session, iou_threshold=0.30).find()

    assert matches == []


def test_two_unknown_faces_are_never_listed(tmp_db):
    with session_scope() as session:
        image = _add_image(session)
        _add_face(session, image, (10, 10, 100, 100))
        _add_face(session, image, (20, 20, 90, 90))

    with session_scope() as session:
        matches = DuplicateUnknownFaceFinder(session, iou_threshold=0.30).find()

    assert matches == []


def test_unknown_overlapping_multiple_known_faces_is_listed_once(tmp_db):
    with session_scope() as session:
        image = _add_image(session)
        alice = _add_person(session, "Alice")
        bob = _add_person(session, "Bob")
        _add_face(session, image, (0, 0, 100, 100), alice)
        bob_face = _add_face(session, image, (10, 10, 90, 90), bob)
        unknown = _add_face(session, image, (10, 10, 90, 90))

    with session_scope() as session:
        matches = DuplicateUnknownFaceFinder(session, iou_threshold=0.30).find()

    assert len(matches) == 1
    assert matches[0].unknown_face_id == unknown.id
    assert matches[0].known_face_id == bob_face.id


def test_delete_unknown_faces_deletes_only_still_unknown_records(tmp_db):
    with session_scope() as session:
        image = _add_image(session)
        person = _add_person(session, "Alice")
        known = _add_face(session, image, (10, 10, 100, 100), person)
        unknown = _add_face(session, image, (20, 20, 90, 90))

    with session_scope() as session:
        result = DuplicateUnknownFaceFinder(session).delete_unknown_faces(
            [unknown.id, known.id, 9999]
        )

    assert result.requested == 3
    assert result.deleted == 1
    assert result.image_ids == (image.id,)
    assert set(result.missing_or_changed) == {known.id, 9999}

    with session_scope() as session:
        assert session.get(Face, unknown.id) is None
        assert session.get(Face, known.id) is not None


# ── is_placeholder_name ───────────────────────────────────────────────────────

@pytest.mark.parametrize("name", [
    None, "", "  ", "?", "??", "???",
    "Unknown", "unknown", "UNKNOWN",
    "Unknown 96", "Unknown_96", "unknown 1", "Unknown_003",
    "Ismeretlen", "ismeretlen", "Ismeretlen 5", "ismeretlen_2",
])
def test_is_placeholder_name_true(name):
    assert is_placeholder_name(name) is True


@pytest.mark.parametrize("name", [
    "Alice", "Bob Smith", "Kovács Béla", "John", "Anna-Mária",
])
def test_is_placeholder_name_false(name):
    assert is_placeholder_name(name) is False


# ── Auto-named person treated as unknown ─────────────────────────────────────

def test_auto_named_person_overlapping_known_is_listed(tmp_db):
    """Faces assigned to auto-named persons (e.g. 'Unknown 96') must be found."""
    with session_scope() as session:
        image = _add_image(session)
        alice = _add_person(session, "Alice")
        auto = _add_person(session, "Unknown 96", auto=True)
        known = _add_face(session, image, (10, 10, 100, 100), alice)
        auto_face = _add_face(session, image, (20, 20, 90, 90), auto)

    with session_scope() as session:
        finder = DuplicateUnknownFaceFinder(session, iou_threshold=0.30)
        matches = finder.find()

    assert len(matches) == 1
    assert matches[0].unknown_face_id == auto_face.id
    assert matches[0].known_face_id == known.id
    assert matches[0].known_person_name == "Alice"


def test_auto_named_person_can_be_deleted(tmp_db):
    """delete_unknown_faces must accept auto-named placeholder faces."""
    with session_scope() as session:
        image = _add_image(session)
        auto = _add_person(session, "Unknown 96", auto=True)
        auto_face = _add_face(session, image, (20, 20, 90, 90), auto)

    with session_scope() as session:
        result = DuplicateUnknownFaceFinder(session).delete_unknown_faces([auto_face.id])

    assert result.deleted == 1
    assert result.missing_or_changed == ()

    with session_scope() as session:
        assert session.get(Face, auto_face.id) is None


# ── Same-person duplicate detection ──────────────────────────────────────────

def test_same_person_overlapping_duplicates_are_listed(tmp_db):
    """Two overlapping faces assigned to the same person should be flagged."""
    with session_scope() as session:
        image = _add_image(session)
        alice = _add_person(session, "Alice")
        face_a = _add_face(session, image, (10, 10, 100, 100), alice)
        face_b = _add_face(session, image, (20, 20, 90, 90), alice)

    with session_scope() as session:
        finder = DuplicateUnknownFaceFinder(session, iou_threshold=0.30)
        matches = finder.find()

    assert len(matches) == 1
    ids = {matches[0].unknown_face_id, matches[0].known_face_id}
    assert ids == {face_a.id, face_b.id}


def test_same_person_non_overlapping_not_listed(tmp_db):
    with session_scope() as session:
        image = _add_image(session)
        alice = _add_person(session, "Alice")
        _add_face(session, image, (10, 10, 100, 100), alice)
        _add_face(session, image, (300, 300, 100, 100), alice)

    with session_scope() as session:
        matches = DuplicateUnknownFaceFinder(session, iou_threshold=0.30).find()

    assert matches == []
