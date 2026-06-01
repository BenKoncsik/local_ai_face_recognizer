"""Tests for resetting automatically created Unknown identities."""

from __future__ import annotations

import numpy as np
import pytest

from app.db.database import init_db, session_scope
from app.db.models import Face, Image, MergeSuggestion, Person
from app.services.unknown_person_reset_service import UnknownPersonResetService


@pytest.fixture()
def tmp_db(tmp_path):
    db_file = tmp_path / "test.db"
    init_db(db_file)
    return db_file


def _add_face(session, image_id: int, person_id: int | None, source: str | None) -> Face:
    face = Face(
        image_id=image_id,
        person_id=person_id,
        bbox_x=0,
        bbox_y=0,
        bbox_w=20,
        bbox_h=20,
        confidence=1.0,
        detector_backend="cpu",
        assignment_source=source,
        assignment_confidence=0.8 if source else None,
    )
    face.set_embedding(np.array([1.0, 0.0], dtype=np.float32))
    session.add(face)
    session.flush()
    return face


def test_reset_deletes_auto_named_persons_and_unassigns_their_faces(tmp_db):
    with session_scope() as session:
        image = Image(file_path="/i.jpg", file_hash="h", file_mtime=0.0)
        named = Person(name="Anna", is_auto_named=False)
        unknown = Person(name="Unknown 1", is_auto_named=True)
        protected = Person(name="Ismeretlen", is_auto_named=False, is_protected=True)
        session.add_all([image, named, unknown, protected])
        session.flush()

        unknown_face = _add_face(session, image.id, unknown.id, "clustering")
        named_face = _add_face(session, image.id, named.id, "manual")
        protected_face = _add_face(session, image.id, protected.id, "manual")
        session.add(
            MergeSuggestion(
                source_person_id=min(named.id, unknown.id),
                target_person_id=max(named.id, unknown.id),
                confidence=0.9,
            )
        )
        unknown_id = unknown.id
        unknown_face_id = unknown_face.id
        named_face_id = named_face.id
        protected_face_id = protected_face.id

    with session_scope() as session:
        result = UnknownPersonResetService(session).reset()

    assert result.deleted_persons == 1
    assert result.unassigned_faces == 1
    with session_scope() as session:
        assert session.get(Person, unknown_id) is None
        reset_face = session.get(Face, unknown_face_id)
        assert reset_face is not None
        assert reset_face.person_id is None
        assert reset_face.assignment_source is None
        assert reset_face.assignment_confidence is None
        assert reset_face.get_embedding() is not None
        assert session.get(Face, named_face_id).person_id is not None
        assert session.get(Face, protected_face_id).person_id is not None
        assert session.query(MergeSuggestion).count() == 0


def test_reset_is_a_noop_without_auto_named_persons(tmp_db):
    with session_scope() as session:
        result = UnknownPersonResetService(session).reset()

    assert result.deleted_persons == 0
    assert result.unassigned_faces == 0
