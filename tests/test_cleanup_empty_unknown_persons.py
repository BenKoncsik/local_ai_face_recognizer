"""Tests for IdentityService.cleanup_empty_unknown_persons().

Covers the maintenance pass that removes auto-generated "Unknown" persons left
without any associated faces, while protecting:
* the canonical (protected) "Ismeretlen" person,
* manually named persons (even when face-less),
* any Unknown person that still owns at least one face.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.db.database import init_db, session_scope
from app.db.models import Face, Image, Person
from app.services.identity_service import IdentityService


@pytest.fixture()
def tmp_db(tmp_path):
    db_file = tmp_path / "test.db"
    init_db(db_file)
    return db_file


def _add_face(session, image_id: int, person_id: int | None) -> Face:
    face = Face(
        image_id=image_id,
        person_id=person_id,
        bbox_x=0,
        bbox_y=0,
        bbox_w=20,
        bbox_h=20,
        confidence=1.0,
        detector_backend="cpu",
    )
    face.set_embedding(np.array([1.0, 0.0], dtype=np.float32))
    session.add(face)
    session.flush()
    return face


def test_empty_unknown_person_is_deleted(tmp_db):
    with session_scope() as session:
        empty_unknown = Person(name="Unknown 1", is_auto_named=True)
        session.add(empty_unknown)
        session.flush()
        empty_id = empty_unknown.id

    with session_scope() as session:
        deleted = IdentityService(session).cleanup_empty_unknown_persons()

    assert deleted == 1
    with session_scope() as session:
        assert session.get(Person, empty_id) is None


def test_unknown_person_with_a_face_is_kept(tmp_db):
    with session_scope() as session:
        image = Image(file_path="/i.jpg", file_hash="h", file_mtime=0.0)
        unknown = Person(name="Unknown 1", is_auto_named=True)
        session.add_all([image, unknown])
        session.flush()
        _add_face(session, image.id, unknown.id)
        unknown_id = unknown.id

    with session_scope() as session:
        deleted = IdentityService(session).cleanup_empty_unknown_persons()

    assert deleted == 0
    with session_scope() as session:
        assert session.get(Person, unknown_id) is not None


def test_protected_unknown_person_is_never_deleted(tmp_db):
    with session_scope() as session:
        protected = Person(
            name="Ismeretlen", is_auto_named=False, is_protected=True
        )
        session.add(protected)
        session.flush()
        protected_id = protected.id

    with session_scope() as session:
        deleted = IdentityService(session).cleanup_empty_unknown_persons()

    assert deleted == 0
    with session_scope() as session:
        assert session.get(Person, protected_id) is not None


def test_named_person_without_faces_is_kept(tmp_db):
    with session_scope() as session:
        named = Person(name="Anna", is_auto_named=False)
        session.add(named)
        session.flush()
        named_id = named.id

    with session_scope() as session:
        deleted = IdentityService(session).cleanup_empty_unknown_persons()

    assert deleted == 0
    with session_scope() as session:
        assert session.get(Person, named_id) is not None


def test_multiple_empty_unknown_persons_all_deleted(tmp_db):
    with session_scope() as session:
        image = Image(file_path="/i.jpg", file_hash="h", file_mtime=0.0)
        empty_a = Person(name="Unknown 1", is_auto_named=True)
        empty_b = Person(name="Unknown 2", is_auto_named=True)
        empty_c = Person(name="Unknown 3", is_auto_named=True)
        populated = Person(name="Unknown 4", is_auto_named=True)
        named = Person(name="Béla", is_auto_named=False)
        protected = Person(
            name="Ismeretlen", is_auto_named=False, is_protected=True
        )
        session.add_all(
            [image, empty_a, empty_b, empty_c, populated, named, protected]
        )
        session.flush()
        _add_face(session, image.id, populated.id)
        empty_ids = [empty_a.id, empty_b.id, empty_c.id]
        populated_id = populated.id
        named_id = named.id
        protected_id = protected.id

    with session_scope() as session:
        deleted = IdentityService(session).cleanup_empty_unknown_persons()

    assert deleted == 3
    with session_scope() as session:
        for pid in empty_ids:
            assert session.get(Person, pid) is None
        assert session.get(Person, populated_id) is not None
        assert session.get(Person, named_id) is not None
        assert session.get(Person, protected_id) is not None


def test_cleanup_is_a_noop_when_nothing_empty(tmp_db):
    with session_scope() as session:
        deleted = IdentityService(session).cleanup_empty_unknown_persons()

    assert deleted == 0


def test_reassigning_last_face_cleans_up_source_unknown(tmp_db):
    """The face operations auto-trigger cleanup of the drained source."""
    with session_scope() as session:
        image = Image(file_path="/i.jpg", file_hash="h", file_mtime=0.0)
        source = Person(name="Unknown 1", is_auto_named=True)
        target = Person(name="Unknown 2", is_auto_named=True)
        session.add_all([image, source, target])
        session.flush()
        face = _add_face(session, image.id, source.id)
        # target needs a face too so it is not itself considered empty
        _add_face(session, image.id, target.id)
        source_id = source.id
        target_id = target.id
        face_id = face.id

    with session_scope() as session:
        IdentityService(session).reassign_face(face_id, target_id)

    with session_scope() as session:
        assert session.get(Person, source_id) is None
        assert session.get(Person, target_id) is not None
        assert session.get(Face, face_id).person_id == target_id
