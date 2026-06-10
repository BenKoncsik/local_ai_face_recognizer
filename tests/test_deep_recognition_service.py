"""DB-level tests for the deep recognition service."""

from __future__ import annotations

import numpy as np
import pytest

from app.config import DeepRecognitionConfig
from app.db.database import init_db, session_scope
from app.db.models import (
    AUTO_ASSIGN_STATUS_AUTO,
    AUTO_ASSIGN_STATUS_CONFIRMED,
    AUTO_ASSIGN_STATUS_CORRECTED,
    AUTO_ASSIGN_STATUS_REVERTED,
    AutoAssignment,
    Face,
    FaceCorrection,
    Image,
    Person,
    TrainingRun,
)
from app.services.deep_recognition_service import (
    DEEP_ASSIGNMENT_SOURCE,
    DEEP_CONFIRMED_SOURCE,
    DeepRecognitionService,
)

DIM = 64


def _fast_config(tmp_path) -> DeepRecognitionConfig:
    return DeepRecognitionConfig(
        model_dir=str(tmp_path / "deep_model"),
        ensemble_size=2,
        hidden_layers=(32, 16),
        max_iter=300,
        min_class_size=4,
        calibration_folds=2,
        skip_unchanged=True,
    )


@pytest.fixture()
def tmp_db(tmp_path):
    db_file = tmp_path / "test.db"
    init_db(db_file)
    return db_file


def _vec(axis: int, noise: float = 0.05, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = np.zeros(DIM, dtype=np.float32)
    v[axis] = 1.0
    v += rng.normal(0, noise, DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _add_image(session, path: str = "/img.jpg") -> int:
    image = Image(file_path=path, file_hash=path, file_mtime=0.0)
    session.add(image)
    session.flush()
    return image.id


def _add_person(session, name: str, *, auto=False, protected=False) -> int:
    person = Person(name=name, is_auto_named=auto, is_protected=protected)
    session.add(person)
    session.flush()
    return person.id


def _add_face(
    session,
    image_id: int,
    person_id: int | None,
    embedding: np.ndarray | None,
    *,
    source: str | None = "manual",
    confidence: float = 0.95,
) -> int:
    face = Face(
        image_id=image_id,
        person_id=person_id,
        bbox_x=0, bbox_y=0, bbox_w=40, bbox_h=40,
        confidence=confidence,
        detector_backend="cpu",
        assignment_source=source,
    )
    if embedding is not None:
        face.set_embedding(embedding)
    session.add(face)
    session.flush()
    return face.id


def _seed_two_persons(session, img: int) -> tuple[int, int]:
    """Two named people, six trusted faces each."""
    anna = _add_person(session, "Nagy Anna")
    bela = _add_person(session, "Kovács Béla")
    for i in range(6):
        _add_face(session, img, anna, _vec(0, seed=i))
        _add_face(session, img, bela, _vec(30, seed=100 + i))
    return anna, bela


class TestTrainAndRecognize:
    def test_unknown_face_is_assigned_and_logged(self, tmp_db, tmp_path):
        cfg = _fast_config(tmp_path)
        with session_scope() as s:
            img = _add_image(s)
            anna, _ = _seed_two_persons(s, img)
            candidate = _add_face(s, img, None, _vec(0, seed=999), source=None)

        with session_scope() as s:
            result = DeepRecognitionService(s, cfg).train_and_recognize()

        assert result.train.n_persons == 2
        assert result.recognition.n_assigned == 1

        with session_scope() as s:
            face = s.get(Face, candidate)
            assert face.person_id == anna
            assert face.assignment_source == DEEP_ASSIGNMENT_SOURCE
            assert face.assignment_confidence is not None

            log_rows = s.query(AutoAssignment).all()
            assert len(log_rows) == 1
            assert log_rows[0].face_id == candidate
            assert log_rows[0].person_id == anna
            assert log_rows[0].status == AUTO_ASSIGN_STATUS_AUTO
            assert log_rows[0].previous_person_id is None

            runs = s.query(TrainingRun).all()
            assert len(runs) == 1
            assert runs[0].n_persons == 2
            assert runs[0].finished_at is not None

    def test_named_faces_are_never_touched(self, tmp_db, tmp_path):
        """An already recognized face keeps its person — the old one wins."""
        cfg = _fast_config(tmp_path)
        with session_scope() as s:
            img = _add_image(s)
            anna, bela = _seed_two_persons(s, img)
            # A face manually placed on Béla even though it *looks* like Anna.
            tricky = _add_face(s, img, bela, _vec(0, seed=500), source="manual")

        with session_scope() as s:
            DeepRecognitionService(s, cfg).train_and_recognize()

        with session_scope() as s:
            face = s.get(Face, tricky)
            assert face.person_id == bela
            assert face.assignment_source == "manual"

    def test_stranger_face_stays_unknown(self, tmp_db, tmp_path):
        cfg = _fast_config(tmp_path)
        with session_scope() as s:
            img = _add_image(s)
            _seed_two_persons(s, img)
            stranger = _add_face(s, img, None, _vec(55, seed=7), source=None)

        with session_scope() as s:
            result = DeepRecognitionService(s, cfg).train_and_recognize()

        assert result.recognition.n_rejected_outlier >= 1
        with session_scope() as s:
            assert s.get(Face, stranger).person_id is None

    def test_low_confidence_detection_is_never_assigned(self, tmp_db, tmp_path):
        """Boxes that may not even be faces are excluded from auto-assignment."""
        cfg = _fast_config(tmp_path)
        with session_scope() as s:
            img = _add_image(s)
            _seed_two_persons(s, img)
            weak = _add_face(
                s, img, None, _vec(0, seed=42), source=None, confidence=0.30
            )

        with session_scope() as s:
            result = DeepRecognitionService(s, cfg).train_and_recognize()

        assert result.recognition.n_skipped_low_confidence == 1
        with session_scope() as s:
            assert s.get(Face, weak).person_id is None

    def test_correction_veto_blocks_reassignment(self, tmp_db, tmp_path):
        """A user 'different person' judgement is a hard veto."""
        cfg = _fast_config(tmp_path)
        with session_scope() as s:
            img = _add_image(s)
            anna, _ = _seed_two_persons(s, img)
            candidate = _add_face(s, img, None, _vec(0, seed=999), source=None)
            anna_face = (
                s.query(Face).filter(Face.person_id == anna).first()
            )
            s.add(
                FaceCorrection(
                    face_id_a=min(candidate, anna_face.id),
                    face_id_b=max(candidate, anna_face.id),
                    same_person=False,
                )
            )

        with session_scope() as s:
            result = DeepRecognitionService(s, cfg).train_and_recognize()

        assert result.recognition.n_rejected_correction == 1
        with session_scope() as s:
            assert s.get(Face, candidate).person_id is None

    def test_unknown_group_face_can_be_promoted(self, tmp_db, tmp_path):
        """Faces sitting in an auto-named 'Unknown N' group are candidates."""
        cfg = _fast_config(tmp_path)
        with session_scope() as s:
            img = _add_image(s)
            anna, _ = _seed_two_persons(s, img)
            unknown = _add_person(s, "Unknown 1", auto=True)
            member = _add_face(
                s, img, unknown, _vec(0, seed=777), source="clustering"
            )

        with session_scope() as s:
            DeepRecognitionService(s, cfg).train_and_recognize()

        with session_scope() as s:
            face = s.get(Face, member)
            assert face.person_id == anna
            row = s.query(AutoAssignment).filter_by(face_id=member).one()
            # The emptied "Unknown 1" group is cleaned up (FK goes NULL), but
            # the name snapshot keeps the provenance for the review tab.
            assert row.previous_person_name == "Unknown 1"

    def test_unchanged_data_reuses_model(self, tmp_db, tmp_path):
        cfg = _fast_config(tmp_path)
        with session_scope() as s:
            img = _add_image(s)
            _seed_two_persons(s, img)

        with session_scope() as s:
            run1, stats1, _ = DeepRecognitionService(s, cfg).train()
        with session_scope() as s:
            run2, stats2, _ = DeepRecognitionService(s, cfg).train()

        assert not stats1.reused_existing_model
        assert stats2.reused_existing_model
        with session_scope() as s:
            r1 = s.get(TrainingRun, run1.id)
            r2 = s.get(TrainingRun, run2.id)
            assert r1.data_fingerprint == r2.data_fingerprint


class TestReviewActions:
    def _setup_with_assignment(self, tmp_path):
        cfg = _fast_config(tmp_path)
        with session_scope() as s:
            img = _add_image(s)
            anna, bela = _seed_two_persons(s, img)
            candidate = _add_face(s, img, None, _vec(0, seed=999), source=None)
        with session_scope() as s:
            DeepRecognitionService(s, cfg).train_and_recognize()
        with session_scope() as s:
            assignment = s.query(AutoAssignment).one()
            return cfg, assignment.id, candidate, anna, bela

    def test_confirm_marks_face_as_trusted(self, tmp_db, tmp_path):
        cfg, assignment_id, face_id, anna, _ = self._setup_with_assignment(tmp_path)

        with session_scope() as s:
            DeepRecognitionService(s, cfg).confirm_assignment(assignment_id)

        with session_scope() as s:
            face = s.get(Face, face_id)
            assert face.person_id == anna
            assert face.assignment_source == DEEP_CONFIRMED_SOURCE
            row = s.get(AutoAssignment, assignment_id)
            assert row.status == AUTO_ASSIGN_STATUS_CONFIRMED
            assert row.decided_at is not None

    def test_correct_moves_face_and_records_corrections(self, tmp_db, tmp_path):
        cfg, assignment_id, face_id, anna, bela = self._setup_with_assignment(tmp_path)

        with session_scope() as s:
            DeepRecognitionService(s, cfg).correct_assignment(assignment_id, bela)

        with session_scope() as s:
            face = s.get(Face, face_id)
            assert face.person_id == bela
            assert face.assignment_source == "manual"

            row = s.get(AutoAssignment, assignment_id)
            assert row.status == AUTO_ASSIGN_STATUS_CORRECTED
            assert row.corrected_person_id == bela

            # The mistake (anna) is recorded negative; the fix (bela) positive.
            corrections = s.query(FaceCorrection).all()
            sames = [c for c in corrections if c.same_person]
            diffs = [c for c in corrections if not c.same_person]
            assert len(diffs) == 1 and len(sames) == 1

        # The veto must hold on the next run: face stays with Béla.
        with session_scope() as s:
            DeepRecognitionService(s, cfg).train_and_recognize()
        with session_scope() as s:
            assert s.get(Face, face_id).person_id == bela

    def test_correct_to_new_person(self, tmp_db, tmp_path):
        cfg, assignment_id, face_id, _, _ = self._setup_with_assignment(tmp_path)

        with session_scope() as s:
            DeepRecognitionService(s, cfg).correct_assignment_to_new_person(
                assignment_id, "Új Ember"
            )

        with session_scope() as s:
            face = s.get(Face, face_id)
            person = s.get(Person, face.person_id)
            assert person.name == "Új Ember"
            assert not person.is_auto_named

    def test_revert_restores_previous_state(self, tmp_db, tmp_path):
        cfg, assignment_id, face_id, anna, _ = self._setup_with_assignment(tmp_path)

        with session_scope() as s:
            DeepRecognitionService(s, cfg).revert_assignment(assignment_id)

        with session_scope() as s:
            face = s.get(Face, face_id)
            assert face.person_id is None
            assert face.assignment_source is None
            row = s.get(AutoAssignment, assignment_id)
            assert row.status == AUTO_ASSIGN_STATUS_REVERTED
            # Reverting also vetoes the bad person for future runs.
            assert s.query(FaceCorrection).filter_by(same_person=False).count() == 1

    def test_revert_all_open_spares_reviewed_rows(self, tmp_db, tmp_path):
        """Bulk undo reverts every open assignment but not the reviewed ones."""
        cfg = _fast_config(tmp_path)
        with session_scope() as s:
            img = _add_image(s)
            anna, _ = _seed_two_persons(s, img)
            candidates = [
                _add_face(s, img, None, _vec(0, seed=900 + i), source=None)
                for i in range(3)
            ]
        with session_scope() as s:
            DeepRecognitionService(s, cfg).train_and_recognize()
        with session_scope() as s:
            rows = s.query(AutoAssignment).order_by(AutoAssignment.id).all()
            assert len(rows) == 3
            confirmed_id, confirmed_face = rows[0].id, rows[0].face_id
            DeepRecognitionService(s, cfg).confirm_assignment(confirmed_id)

        with session_scope() as s:
            n = DeepRecognitionService(s, cfg).revert_all_open()
            assert n == 2

        with session_scope() as s:
            confirmed = s.get(AutoAssignment, confirmed_id)
            assert confirmed.status == AUTO_ASSIGN_STATUS_CONFIRMED
            assert s.get(Face, confirmed_face).person_id == anna
            for row in s.query(AutoAssignment).filter(
                AutoAssignment.id != confirmed_id
            ):
                assert row.status == AUTO_ASSIGN_STATUS_REVERTED
                assert s.get(Face, row.face_id).person_id is None
            # Each bulk revert is recorded as a mistake the engine learns from.
            assert s.query(FaceCorrection).filter_by(same_person=False).count() == 2
            assert DeepRecognitionService(s, cfg).count_open_assignments() == 0

    def test_list_auto_assignments_returns_dto(self, tmp_db, tmp_path):
        cfg, assignment_id, face_id, anna, _ = self._setup_with_assignment(tmp_path)

        with session_scope() as s:
            svc = DeepRecognitionService(s, cfg)
            dtos = svc.list_auto_assignments()
            assert len(dtos) == 1
            dto = dtos[0]
            assert dto.assignment_id == assignment_id
            assert dto.face_id == face_id
            assert dto.person_name == "Nagy Anna"
            assert dto.status == AUTO_ASSIGN_STATUS_AUTO
            assert svc.count_open_assignments() == 1

    def test_force_retrains_even_when_data_unchanged(self, tmp_db, tmp_path):
        """The explicit "train the model" action must never be skipped."""
        cfg = _fast_config(tmp_path)
        with session_scope() as s:
            img = _add_image(s)
            _seed_two_persons(s, img)
        with session_scope() as s:
            _, first, _ = DeepRecognitionService(s, cfg).train()
            assert not first.reused_existing_model
        with session_scope() as s:
            _, again, _ = DeepRecognitionService(s, cfg).train()
            assert again.reused_existing_model  # skip_unchanged still works
        with session_scope() as s:
            _, forced, _ = DeepRecognitionService(s, cfg).train(
                mode="train", force=True
            )
            assert not forced.reused_existing_model

    def test_review_list_survives_train_only_run(self, tmp_db, tmp_path):
        """A newer train-only run must not hide the open review items."""
        cfg = _fast_config(tmp_path)
        with session_scope() as s:
            img = _add_image(s)
            _seed_two_persons(s, img)
            _add_face(s, img, None, _vec(0, seed=999), source=None)
        with session_scope() as s:
            DeepRecognitionService(s, cfg).train_and_recognize()
        with session_scope() as s:
            assert DeepRecognitionService(s, cfg).count_open_assignments() == 1

        with session_scope() as s:
            DeepRecognitionService(s, cfg).train(mode="train", force=True)

        with session_scope() as s:
            svc = DeepRecognitionService(s, cfg)
            assert svc.count_open_assignments() == 1
            assert len(svc.list_auto_assignments()) == 1
            assert svc.revert_all_open() == 1

    def test_stale_assignment_is_hidden(self, tmp_db, tmp_path):
        """Manually moving the face elsewhere hides the stale review row."""
        cfg, _, face_id, _, bela = self._setup_with_assignment(tmp_path)

        with session_scope() as s:
            face = s.get(Face, face_id)
            face.person_id = bela  # user moved it in another panel meanwhile

        with session_scope() as s:
            assert DeepRecognitionService(s, cfg).list_auto_assignments() == []
