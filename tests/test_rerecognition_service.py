"""Unit tests for the image-browser re-recognition workflow service."""

from __future__ import annotations

import numpy as np
import pytest

from app.config import RecognitionConfig
from app.db.database import init_db, session_scope
from app.db.models import Face, Person, RecognitionMergeLog
from app.services.rerecognition_service import (
    KIND_AUTO,
    KIND_NONE,
    KIND_SUGGEST,
    ReRecognitionService,
)

DIM = 128


@pytest.fixture()
def tmp_db(tmp_path):
    db_file = tmp_path / "test.db"
    init_db(db_file)
    return db_file


def _vec(*pairs: "tuple[int, float]") -> np.ndarray:
    """Build a DIM-vector with the given (axis, weight) components."""
    v = np.zeros(DIM, dtype=np.float32)
    for idx, w in pairs:
        v[idx] = w
    return v


def _axis(index: int, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[index] = 1.0
    if noise > 0:
        v += np.random.default_rng(seed).normal(0, noise, DIM).astype(np.float32)
    return v


def _add_image(session, path: str = "/img.jpg") -> int:
    from app.db.models import Image

    image = Image(file_path=path, file_hash=path, file_mtime=0.0)
    session.add(image)
    session.flush()
    return image.id


def _add_person(session, name: str, *, auto: bool = False, protected: bool = False) -> int:
    person = Person(name=name, is_auto_named=auto, is_protected=protected)
    session.add(person)
    session.flush()
    return person.id


def _add_face(session, image_id, person_id, embedding) -> int:
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
    face.set_embedding(embedding)
    session.add(face)
    session.flush()
    return face.id


def _cfg() -> RecognitionConfig:
    return RecognitionConfig()


class TestClassification:
    def test_strong_match_is_auto(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s)
            alice = _add_person(s, "Alice")
            _add_face(s, img, alice, _axis(0, noise=0.01, seed=1))
            unknown = _add_person(s, "Unknown 1", auto=True)
            cand = _add_face(s, img, unknown, _axis(0, noise=0.01, seed=2))

            svc = ReRecognitionService(s, _cfg())
            profiles = svc.load_profiles()
            faces = svc.extract_candidates([img])
            assert {f.face_id for f in faces} == {cand}
            kind, item = svc.classify(faces[0], profiles)
            assert kind == KIND_AUTO
            assert item.target_person_id == alice

    def test_partial_match_is_suggest(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s)
            alice = _add_person(s, "Alice")
            _add_face(s, img, alice, _axis(0))
            unknown = _add_person(s, "Unknown 1", auto=True)
            # cos to Alice's axis 0 == 0.6 (between suggest 0.55 and auto 0.72).
            _add_face(s, img, unknown, _vec((0, 0.6), (5, 0.8)))

            svc = ReRecognitionService(s, _cfg())
            profiles = svc.load_profiles()
            faces = svc.extract_candidates([img])
            kind, item = svc.classify(faces[0], profiles)
            assert kind == KIND_SUGGEST
            assert item.candidates[0].person_id == alice
            assert 0.55 <= item.candidates[0].score < 0.72

    def test_far_match_is_none(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s)
            alice = _add_person(s, "Alice")
            _add_face(s, img, alice, _axis(0))
            unknown = _add_person(s, "Unknown 1", auto=True)
            _add_face(s, img, unknown, _axis(40))  # orthogonal

            svc = ReRecognitionService(s, _cfg())
            profiles = svc.load_profiles()
            faces = svc.extract_candidates([img])
            kind, item = svc.classify(faces[0], profiles)
            assert kind == KIND_NONE
            assert item is None

    def test_named_faces_are_never_candidates(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s)
            alice = _add_person(s, "Alice")
            _add_face(s, img, alice, _axis(0))  # named → must be ignored

            svc = ReRecognitionService(s, _cfg())
            faces = svc.extract_candidates([img])
            assert faces == []


class TestApplyAndUndo:
    def test_auto_merge_moves_face_logs_and_cleans_up(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s)
            alice = _add_person(s, "Alice")
            _add_face(s, img, alice, _axis(0, noise=0.01, seed=1))
            unknown = _add_person(s, "Unknown 1", auto=True)
            cand = _add_face(s, img, unknown, _axis(0, noise=0.01, seed=2))

            svc = ReRecognitionService(s, _cfg())
            profiles = svc.load_profiles()
            faces = svc.extract_candidates([img])
            _, item = svc.classify(faces[0], profiles)
            batch_id = svc.apply_auto_merges([item])

        with session_scope() as s:
            face = s.get(Face, cand)
            assert face.person_id == alice
            # Emptied Unknown cluster was cleaned up.
            assert s.get(Person, unknown) is None
            rows = (
                s.query(RecognitionMergeLog)
                .filter(RecognitionMergeLog.batch_id == batch_id)
                .all()
            )
            assert len(rows) == 1
            assert rows[0].matched_person_id == alice
            assert rows[0].prev_person_was_auto is True
            assert rows[0].undone_at is None

    def test_undo_restores_face_and_recreates_unknown(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s)
            alice = _add_person(s, "Alice")
            _add_face(s, img, alice, _axis(0, noise=0.01, seed=1))
            unknown = _add_person(s, "Unknown 1", auto=True)
            cand = _add_face(s, img, unknown, _axis(0, noise=0.01, seed=2))

            svc = ReRecognitionService(s, _cfg())
            profiles = svc.load_profiles()
            faces = svc.extract_candidates([img])
            _, item = svc.classify(faces[0], profiles)
            batch_id = svc.apply_auto_merges([item])

        with session_scope() as s:
            restored = ReRecognitionService(s).undo_batch(batch_id)
            assert restored == 1

        with session_scope() as s:
            face = s.get(Face, cand)
            assert face.person_id == unknown  # restored to the rebuilt cluster
            assert s.get(Person, unknown) is not None
            rows = (
                s.query(RecognitionMergeLog)
                .filter(RecognitionMergeLog.batch_id == batch_id)
                .all()
            )
            assert all(r.undone_at is not None for r in rows)

    def test_latest_undoable_batch_tracks_state(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s)
            alice = _add_person(s, "Alice")
            _add_face(s, img, alice, _axis(0, noise=0.01, seed=1))
            unknown = _add_person(s, "Unknown 1", auto=True)
            _add_face(s, img, unknown, _axis(0, noise=0.01, seed=2))

            svc = ReRecognitionService(s, _cfg())
            profiles = svc.load_profiles()
            faces = svc.extract_candidates([img])
            _, item = svc.classify(faces[0], profiles)
            batch_id = svc.apply_auto_merges([item])

        with session_scope() as s:
            svc = ReRecognitionService(s)
            assert svc.latest_undoable_batch() == batch_id
            svc.undo_batch(batch_id)
            assert svc.latest_undoable_batch() is None
