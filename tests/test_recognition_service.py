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
            assignments, stats = RecognitionService(s).recognize_pending()

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
            assignments, stats = RecognitionService(s).recognize_pending()

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
            assignments, stats = RecognitionService(s).recognize_pending()

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
            assignments, stats = RecognitionService(s, config).recognize_pending()

        assert assignments == []
        with session_scope() as s:
            assert s.get(Face, candidate).person_id is None


class TestScorePersons:
    """Tests for :meth:`RecognitionService.score_persons` (face-match ordering)."""

    def test_scores_rank_closest_person_highest(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s)
            kovacs = _add_person(s, "Kovács Béla")
            nagy = _add_person(s, "Nagy Anna")
            _add_face(s, img, kovacs, _axis_vec(DIM, 0, noise=0.02, seed=1))
            _add_face(s, img, kovacs, _axis_vec(DIM, 0, noise=0.02, seed=2))
            _add_face(s, img, nagy, _axis_vec(DIM, 64, noise=0.02, seed=3))
            _add_face(s, img, nagy, _axis_vec(DIM, 64, noise=0.02, seed=4))

            probe = _axis_vec(DIM, 0, noise=0.02, seed=5)
            scores = RecognitionService(s).score_persons(probe)

        assert set(scores) == {kovacs, nagy}
        assert scores[kovacs] > scores[nagy]

    def test_protected_and_auto_named_are_excluded(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s)
            known = _add_person(s, "Known")
            unknown = _add_person(s, "Ismeretlen", protected=True)
            auto = _add_person(s, "Unknown 1", auto=True)
            _add_face(s, img, known, _axis_vec(DIM, 0))
            _add_face(s, img, unknown, _axis_vec(DIM, 0))
            _add_face(s, img, auto, _axis_vec(DIM, 0))

            scores = RecognitionService(s).score_persons(_axis_vec(DIM, 0))

        assert set(scores) == {known}

    def test_missing_embedding_returns_empty(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s)
            known = _add_person(s, "Known")
            _add_face(s, img, known, _axis_vec(DIM, 0))

            assert RecognitionService(s).score_persons(None) == {}


class TestAdaptiveThreshold:
    """Tests for the quality-aware per-face threshold (Pass 1)."""

    def test_threshold_decreases_for_small_low_quality_face(self, tmp_db):
        """Small, low-quality face gets a threshold below the base."""
        config = RecognitionConfig(
            auto_assign_threshold=0.72,
            adaptive_threshold_enabled=True,
            adaptive_min_threshold=0.55,
        )
        with session_scope() as s:
            img = _add_image(s)
            small_face = Face(
                image_id=img,
                bbox_x=0, bbox_y=0, bbox_w=20, bbox_h=20,
                confidence=0.5,
                detector_backend="cpu",
                quality_score=0.2,
            )
            small_face.set_embedding(np.ones(DIM, dtype=np.float32))
            s.add(small_face)
            s.flush()

            svc = RecognitionService(s, config)
            threshold = svc._compute_adaptive_threshold(small_face)

        assert 0.55 <= threshold < 0.72

    def test_threshold_returns_base_when_disabled(self, tmp_db):
        config = RecognitionConfig(
            auto_assign_threshold=0.72,
            adaptive_threshold_enabled=False,
        )
        with session_scope() as s:
            img = _add_image(s)
            face = Face(
                image_id=img,
                bbox_x=0, bbox_y=0, bbox_w=20, bbox_h=20,
                confidence=0.5,
                detector_backend="cpu",
                quality_score=0.1,
            )
            face.set_embedding(np.ones(DIM, dtype=np.float32))
            s.add(face)
            s.flush()

            svc = RecognitionService(s, config)
            assert svc._compute_adaptive_threshold(face) == 0.72

    def test_adaptive_assigns_borderline_face(self, tmp_db):
        """Face below base threshold but above adaptive minimum gets assigned."""
        base_vec = np.zeros(DIM, dtype=np.float32)
        base_vec[0] = 1.0

        # Craft embedding with cosine similarity 0.68 to the profile centroid.
        # cos_sim = v[0] = 0.68, v[1] = sqrt(1 - 0.68^2) to keep it unit-length.
        sim = 0.68
        candidate_emb = np.zeros(DIM, dtype=np.float32)
        candidate_emb[0] = sim
        candidate_emb[1] = float(np.sqrt(1.0 - sim ** 2))

        with session_scope() as s:
            img = _add_image(s)
            known = _add_person(s, "Known")
            _add_face(s, img, known, base_vec)

        with session_scope() as s:
            img2 = _add_image(s, "/img2.jpg")
            # Worst-case quality → lowest possible adaptive threshold
            candidate = Face(
                image_id=img2,
                bbox_x=0, bbox_y=0, bbox_w=20, bbox_h=20,
                confidence=0.5,
                detector_backend="cpu",
                quality_score=0.0,
            )
            candidate.set_embedding(candidate_emb)
            s.add(candidate)
            s.flush()
            candidate_id = candidate.id

        # Expected adaptive threshold (quality=0.0, area=400, frontality=1.0):
        #   combined = 0.0*0.5 + (400/6400)*0.3 + 1.0*0.2 ≈ 0.219
        #   threshold = 0.55 + (0.72 - 0.55)*0.219 ≈ 0.587
        # score = 0.68 > 0.587 → should be assigned
        config = RecognitionConfig(
            auto_assign_threshold=0.72,
            min_margin=0.05,
            adaptive_threshold_enabled=True,
            adaptive_min_threshold=0.55,
            same_image_assist_enabled=False,
        )

        with session_scope() as s:
            assignments, stats = RecognitionService(s, config).recognize_pending()

        assert any(a.face_id == candidate_id for a in assignments)

        with session_scope() as s:
            face = s.get(Face, candidate_id)
            assert face.assignment_source == "recognition"


class TestSameImageAssist:
    """Tests for the same-image context-assisted second pass (Pass 2)."""

    def _make_emb_with_sim(self, target_sim: float) -> np.ndarray:
        """Return a unit vector with cosine similarity *target_sim* to axis-0."""
        v = np.zeros(DIM, dtype=np.float32)
        v[0] = target_sim
        v[1] = float(np.sqrt(max(0.0, 1.0 - target_sim ** 2)))
        return v

    def test_assist_assigns_borderline_face_on_shared_image(self, tmp_db):
        """Unresolved face below base threshold is matched via same-image assist."""
        base_vec = np.zeros(DIM, dtype=np.float32)
        base_vec[0] = 1.0
        # Candidate similarity: above assist threshold (0.65) but below base (0.75)
        candidate_emb = self._make_emb_with_sim(0.70)

        with session_scope() as s:
            img1 = _add_image(s, "/img1.jpg")
            known = _add_person(s, "Known")
            _add_face(s, img1, known, base_vec)
            _add_face(s, img1, known, base_vec)

        with session_scope() as s:
            img2 = _add_image(s, "/img2.jpg")
            known_id = s.query(Person.id).filter(Person.name == "Known").scalar()

            # Confirmed face on img2 (manually labeled)
            confirmed = Face(
                image_id=img2,
                bbox_x=0, bbox_y=0, bbox_w=80, bbox_h=80,
                confidence=0.9,
                detector_backend="cpu",
                assignment_source="manual",
                person_id=known_id,
            )
            confirmed.set_embedding(base_vec)
            s.add(confirmed)

            # Borderline unresolved face on same image
            candidate = Face(
                image_id=img2,
                bbox_x=50, bbox_y=0, bbox_w=20, bbox_h=20,
                confidence=0.6,
                detector_backend="cpu",
            )
            candidate.set_embedding(candidate_emb)
            s.add(candidate)
            s.flush()
            candidate_id = candidate.id

        config = RecognitionConfig(
            auto_assign_threshold=0.75,   # pass 1: 0.70 < 0.75 → not assigned
            min_margin=0.05,
            adaptive_threshold_enabled=False,
            same_image_assist_enabled=True,
            same_image_assist_threshold=0.65,  # pass 2: 0.70 > 0.65 → assigned
            same_image_assist_margin=0.03,
        )

        with session_scope() as s:
            assignments, stats = RecognitionService(s, config).recognize_pending()

        assert any(a.face_id == candidate_id for a in assignments)
        with session_scope() as s:
            face = s.get(Face, candidate_id)
            assert face.person_id is not None
            assert face.assignment_source == "recognition"

    def test_assist_skips_when_no_confirmed_faces_on_image(self, tmp_db):
        """Assist is not triggered when the image has no manually confirmed faces."""
        base_vec = np.zeros(DIM, dtype=np.float32)
        base_vec[0] = 1.0
        candidate_emb = self._make_emb_with_sim(0.70)

        with session_scope() as s:
            img1 = _add_image(s, "/img1.jpg")
            known = _add_person(s, "Known")
            _add_face(s, img1, known, base_vec)

        with session_scope() as s:
            # img2 has only the unresolved candidate — no confirmed face
            img2 = _add_image(s, "/img2_alone.jpg")
            candidate = Face(
                image_id=img2,
                bbox_x=0, bbox_y=0, bbox_w=20, bbox_h=20,
                confidence=0.6,
                detector_backend="cpu",
            )
            candidate.set_embedding(candidate_emb)
            s.add(candidate)
            s.flush()
            candidate_id = candidate.id

        config = RecognitionConfig(
            auto_assign_threshold=0.75,
            adaptive_threshold_enabled=False,
            same_image_assist_enabled=True,
            same_image_assist_threshold=0.65,
        )

        with session_scope() as s:
            assignments, stats = RecognitionService(s, config).recognize_pending()

        assert not any(a.face_id == candidate_id for a in assignments)
        with session_scope() as s:
            assert s.get(Face, candidate_id).person_id is None

    def test_assist_disabled_does_not_assign_borderline_face(self, tmp_db):
        """When assist is off, a borderline face is not assigned even with context."""
        base_vec = np.zeros(DIM, dtype=np.float32)
        base_vec[0] = 1.0
        candidate_emb = self._make_emb_with_sim(0.70)

        with session_scope() as s:
            img1 = _add_image(s, "/img1.jpg")
            known = _add_person(s, "Known")
            _add_face(s, img1, known, base_vec)

        with session_scope() as s:
            img2 = _add_image(s, "/img2.jpg")
            known_id = s.query(Person.id).filter(Person.name == "Known").scalar()
            confirmed = Face(
                image_id=img2,
                bbox_x=0, bbox_y=0, bbox_w=80, bbox_h=80,
                confidence=0.9,
                detector_backend="cpu",
                assignment_source="manual",
                person_id=known_id,
            )
            confirmed.set_embedding(base_vec)
            s.add(confirmed)
            candidate = Face(
                image_id=img2,
                bbox_x=50, bbox_y=0, bbox_w=20, bbox_h=20,
                confidence=0.6,
                detector_backend="cpu",
            )
            candidate.set_embedding(candidate_emb)
            s.add(candidate)
            s.flush()
            candidate_id = candidate.id

        config = RecognitionConfig(
            auto_assign_threshold=0.75,
            adaptive_threshold_enabled=False,
            same_image_assist_enabled=False,  # disabled
        )

        with session_scope() as s:
            assignments, stats = RecognitionService(s, config).recognize_pending()

        assert not any(a.face_id == candidate_id for a in assignments)
        with session_scope() as s:
            assert s.get(Face, candidate_id).person_id is None

    def test_assist_blocked_by_existing_correction(self, tmp_db):
        """A same-image assist match is still blocked by a negative correction."""
        base_vec = np.zeros(DIM, dtype=np.float32)
        base_vec[0] = 1.0
        candidate_emb = self._make_emb_with_sim(0.70)

        with session_scope() as s:
            img1 = _add_image(s, "/img1.jpg")
            known = _add_person(s, "Known")
            training_face_id = _add_face(s, img1, known, base_vec)

        with session_scope() as s:
            img2 = _add_image(s, "/img2.jpg")
            known_id = s.query(Person.id).filter(Person.name == "Known").scalar()
            confirmed = Face(
                image_id=img2,
                bbox_x=0, bbox_y=0, bbox_w=80, bbox_h=80,
                confidence=0.9,
                detector_backend="cpu",
                assignment_source="manual",
                person_id=known_id,
            )
            confirmed.set_embedding(base_vec)
            s.add(confirmed)
            s.flush()
            confirmed_id = confirmed.id

            candidate = Face(
                image_id=img2,
                bbox_x=50, bbox_y=0, bbox_w=20, bbox_h=20,
                confidence=0.6,
                detector_backend="cpu",
            )
            candidate.set_embedding(candidate_emb)
            s.add(candidate)
            s.flush()
            candidate_id = candidate.id

            # Negative correction: candidate ≠ confirmed (≠ Known)
            s.add(FaceCorrection(
                face_id_a=min(confirmed_id, candidate_id),
                face_id_b=max(confirmed_id, candidate_id),
                same_person=False,
            ))

        config = RecognitionConfig(
            auto_assign_threshold=0.75,
            adaptive_threshold_enabled=False,
            same_image_assist_enabled=True,
            same_image_assist_threshold=0.65,
            same_image_assist_margin=0.03,
        )

        with session_scope() as s:
            assignments, stats = RecognitionService(s, config).recognize_pending()

        assert not any(a.face_id == candidate_id for a in assignments)
        with session_scope() as s:
            assert s.get(Face, candidate_id).person_id is None


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
            assignments, stats = RecognitionService(s).recognize_pending()

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
            assignments, stats = RecognitionService(s).recognize_pending()

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
            assignments, stats = RecognitionService(s, config).recognize_pending()

        assert assignments == []
        with session_scope() as s:
            assert s.get(Face, candidate).person_id is None
