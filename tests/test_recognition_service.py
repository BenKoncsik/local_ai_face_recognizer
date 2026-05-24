"""Unit tests for learned person recognition."""

from __future__ import annotations

import numpy as np
import pytest

from app.config import RecognitionConfig
from app.db.database import init_db, session_scope
from app.db.models import Face, FaceCorrection, Image, Person
from app.services.recognition_service import RecognitionService

DIM = 128


@pytest.fixture()
def tmp_db(tmp_path):
    db_file = tmp_path / "test.db"
    init_db(db_file)
    return db_file


def _axis_vec(dim: int, index: int, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[index] = 1.0
    if noise > 0:
        v += np.random.default_rng(seed).normal(0, noise, dim).astype(np.float32)
    return v


def _add_image(session, path: str = "/img.jpg") -> int:
    image = Image(file_path=path, file_hash=path, file_mtime=0.0)
    session.add(image)
    session.flush()
    return image.id


def _add_person(
    session,
    name: str,
    *,
    auto: bool = False,
    protected: bool = False,
) -> int:
    person = Person(name=name, is_auto_named=auto, is_protected=protected)
    session.add(person)
    session.flush()
    return person.id


def _add_face(
    session,
    image_id: int,
    person_id: int | None,
    embedding: np.ndarray,
    *,
    assignment_source: str | None = None,
    assignment_confidence: float | None = None,
) -> int:
    face = Face(
        image_id=image_id,
        person_id=person_id,
        bbox_x=0,
        bbox_y=0,
        bbox_w=20,
        bbox_h=20,
        confidence=1.0,
        detector_backend="cpu",
        assignment_source=assignment_source,
        assignment_confidence=assignment_confidence,
    )
    face.set_embedding(embedding)
    session.add(face)
    session.flush()
    return face.id


class TestRecognitionAssignments:
    def test_unassigned_face_is_assigned_to_best_known_person(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s)
            kovacs = _add_person(s, "Kovács Béla")
            nagy = _add_person(s, "Nagy Anna")
            candidate = _add_face(
                s,
                img,
                None,
                _axis_vec(DIM, 0, noise=0.02, seed=4),
            )
            _add_face(s, img, kovacs, _axis_vec(DIM, 0, noise=0.02, seed=1))
            _add_face(s, img, kovacs, _axis_vec(DIM, 0, noise=0.02, seed=2))
            _add_face(s, img, nagy, _axis_vec(DIM, 64, noise=0.02, seed=3))

        with session_scope() as s:
            assignments = RecognitionService(s).recognize_pending()

        assert len(assignments) == 1
        assert assignments[0].face_id == candidate
        assert assignments[0].target_person_id == kovacs
        assert assignments[0].score > 0.9

        with session_scope() as s:
            face = s.get(Face, candidate)
            assert face is not None
            assert face.person_id == kovacs
            assert face.assignment_source == "recognition"
            assert face.assignment_confidence is not None

    def test_existing_named_assignment_is_preserved(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s)
            person_a = _add_person(s, "Person A")
            person_b = _add_person(s, "Person B")
            face_id = _add_face(s, img, person_a, _axis_vec(DIM, 1))
            _add_face(s, img, person_a, _axis_vec(DIM, 0))
            _add_face(s, img, person_b, _axis_vec(DIM, 1))

        with session_scope() as s:
            assignments = RecognitionService(s).recognize_pending()

        assert assignments == []
        with session_scope() as s:
            assert s.get(Face, face_id).person_id == person_a

    def test_auto_named_candidate_can_move_and_orphan_is_removed(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s)
            known = _add_person(s, "Known")
            unknown = _add_person(s, "Unknown 1", auto=True)
            face_id = _add_face(s, img, unknown, _axis_vec(DIM, 0))
            _add_face(s, img, known, _axis_vec(DIM, 0))

        with session_scope() as s:
            assignments = RecognitionService(s).recognize_pending()

        assert len(assignments) == 1
        with session_scope() as s:
            assert s.get(Face, face_id).person_id == known
            assert s.get(Person, unknown) is None

    def test_ambiguous_face_is_not_assigned(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s)
            person_a = _add_person(s, "Person A")
            person_b = _add_person(s, "Person B")
            _add_face(s, img, person_a, _axis_vec(DIM, 0))
            _add_face(s, img, person_b, _axis_vec(DIM, 0, noise=0.01, seed=2))
            candidate = _add_face(s, img, None, _axis_vec(DIM, 0, noise=0.01, seed=3))

        config = RecognitionConfig(auto_assign_threshold=0.5, min_margin=0.2)
        with session_scope() as s:
            assignments = RecognitionService(s, config).recognize_pending()

        assert assignments == []
        with session_scope() as s:
            assert s.get(Face, candidate).person_id is None


class TestRecognitionCorrections:
    def test_rejected_face_pair_blocks_assignment(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s)
            known = _add_person(s, "Known")
            known_face = _add_face(s, img, known, _axis_vec(DIM, 0))
            candidate = _add_face(s, img, None, _axis_vec(DIM, 0))
            s.add(
                FaceCorrection(
                    face_id_a=min(known_face, candidate),
                    face_id_b=max(known_face, candidate),
                    same_person=False,
                )
            )

        with session_scope() as s:
            assignments = RecognitionService(s).recognize_pending()

        assert assignments == []
        with session_scope() as s:
            assert s.get(Face, candidate).person_id is None

    def test_protected_unknown_person_is_not_a_training_profile(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s)
            protected = _add_person(s, "Ismeretlen", protected=True)
            candidate = _add_face(s, img, None, _axis_vec(DIM, 0))
            _add_face(s, img, protected, _axis_vec(DIM, 0))

        with session_scope() as s:
            assignments = RecognitionService(s).recognize_pending()

        assert assignments == []
        with session_scope() as s:
            assert s.get(Face, candidate).person_id is None

    def test_low_confidence_auto_assignment_is_not_reused_for_training(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s)
            known = _add_person(s, "Known")
            _add_face(
                s,
                img,
                known,
                _axis_vec(DIM, 0),
                assignment_source="recognition",
                assignment_confidence=0.80,
            )
            candidate = _add_face(s, img, None, _axis_vec(DIM, 0))

        config = RecognitionConfig(profile_auto_min_confidence=0.85)
        with session_scope() as s:
            assignments = RecognitionService(s, config).recognize_pending()

        assert assignments == []
        with session_scope() as s:
            assert s.get(Face, candidate).person_id is None
