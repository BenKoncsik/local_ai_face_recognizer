"""Unit tests for the name-suggestion service."""

from __future__ import annotations

import numpy as np
import pytest

from app.config import SuggestionConfig
from app.db.database import init_db, session_scope
from app.db.models import Face, FaceCorrection, Image, Person
from app.services.suggestion_service import SuggestionService

DIM = 128


@pytest.fixture()
def tmp_db(tmp_path):
    """Initialise a fresh SQLite database for each test."""
    db_file = tmp_path / "test.db"
    init_db(db_file)
    return db_file


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _axis_vec(
    dim: int, index: int, noise: float = 0.0, seed: int = 0, scale: float = 1.0
) -> np.ndarray:
    """An (optionally noisy, optionally non-unit) vector pointing at *index*."""
    v = np.zeros(dim, dtype=np.float32)
    v[index] = 1.0
    if noise > 0:
        v += np.random.default_rng(seed).normal(0, noise, dim).astype(np.float32)
    return (v * scale).astype(np.float32)


def _add_image(session, path: str) -> int:
    img = Image(file_path=path, file_hash="h", file_mtime=0.0)
    session.add(img)
    session.flush()
    return img.id


def _add_person(session, name: str, *, auto: bool) -> int:
    person = Person(name=name, is_auto_named=auto)
    session.add(person)
    session.flush()
    return person.id


def _add_face(
    session,
    image_id: int,
    person_id: int,
    *,
    embedding=None,
    excluded: bool = False,
) -> int:
    face = Face(
        image_id=image_id,
        person_id=person_id,
        bbox_x=0, bbox_y=0, bbox_w=20, bbox_h=20,
        confidence=1.0,
        detector_backend="cpu",
        is_excluded=excluded,
        crop_path="crop.jpg",
    )
    if embedding is not None:
        face.set_embedding(embedding)
    session.add(face)
    session.flush()
    return face.id


# ---------------------------------------------------------------------------
# Centroid
# ---------------------------------------------------------------------------

class TestCentroid:
    def test_centroid_is_unit_length(self):
        vectors = [_axis_vec(64, i) for i in range(5)]
        centroid = SuggestionService.centroid(vectors)
        assert np.linalg.norm(centroid) == pytest.approx(1.0, abs=1e-5)

    def test_centroid_of_identical_vectors(self):
        v = _axis_vec(32, 7)
        centroid = SuggestionService.centroid([v, v, v])
        unit = v / np.linalg.norm(v)
        assert float(np.dot(centroid, unit)) == pytest.approx(1.0, abs=1e-5)

    def test_centroid_normalises_inputs(self):
        """A large-magnitude vector must not dominate the centroid."""
        small = _axis_vec(16, 0, scale=0.1)
        huge = _axis_vec(16, 1, scale=50.0)
        centroid = SuggestionService.centroid([small, huge])
        # After per-vector normalisation both axes contribute equally.
        assert centroid[0] == pytest.approx(centroid[1], abs=1e-5)

    def test_centroid_empty_raises(self):
        with pytest.raises(ValueError):
            SuggestionService.centroid([])


# ---------------------------------------------------------------------------
# Suggestion generation
# ---------------------------------------------------------------------------

class TestGenerateSuggestions:
    def test_unknown_matched_to_correct_person(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s, "/img.jpg")
            kovacs = _add_person(s, "Kovács Béla", auto=False)
            nagy = _add_person(s, "Nagy Anna", auto=False)
            unknown = _add_person(s, "Unknown 12", auto=True)

            _add_face(s, img, kovacs, embedding=_axis_vec(DIM, 0, noise=0.02, seed=1))
            _add_face(s, img, kovacs, embedding=_axis_vec(DIM, 0, noise=0.02, seed=2))
            _add_face(s, img, nagy, embedding=_axis_vec(DIM, 64, noise=0.02, seed=3))
            _add_face(s, img, unknown, embedding=_axis_vec(DIM, 0, noise=0.02, seed=4))

        with session_scope() as s:
            suggestions = SuggestionService(s).generate_suggestions()

        assert len(suggestions) == 1
        top = suggestions[0]
        assert top.candidate_person_id == unknown
        assert top.target_person_id == kovacs
        assert top.target_name == "Kovács Béla"
        assert top.similarity > 0.9

    def test_no_suggestion_below_threshold(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s, "/img.jpg")
            named = _add_person(s, "Kovács Béla", auto=False)
            unknown = _add_person(s, "Unknown 1", auto=True)
            # Orthogonal embeddings → cosine similarity ≈ 0, far below 0.5.
            _add_face(s, img, named, embedding=_axis_vec(DIM, 0))
            _add_face(s, img, unknown, embedding=_axis_vec(DIM, 99))

        with session_scope() as s:
            assert SuggestionService(s).generate_suggestions() == []

    def test_config_threshold_controls_output(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s, "/img.jpg")
            named = _add_person(s, "Kovács Béla", auto=False)
            unknown = _add_person(s, "Unknown 1", auto=True)
            _add_face(s, img, named, embedding=_axis_vec(DIM, 0, noise=0.02, seed=1))
            _add_face(s, img, unknown, embedding=_axis_vec(DIM, 0, noise=0.02, seed=2))

        with session_scope() as s:
            lenient = SuggestionService(s, SuggestionConfig(similarity_threshold=0.5))
            assert len(lenient.generate_suggestions()) == 1

        with session_scope() as s:
            strict = SuggestionService(s, SuggestionConfig(similarity_threshold=0.9999))
            assert strict.generate_suggestions() == []

    def test_multiple_matches_are_ranked(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s, "/img.jpg")
            close = _add_person(s, "Close Match", auto=False)
            far = _add_person(s, "Far Match", auto=False)
            unknown = _add_person(s, "Unknown 1", auto=True)

            _add_face(s, img, unknown, embedding=_axis_vec(DIM, 0, noise=0.02, seed=1))
            _add_face(s, img, close, embedding=_axis_vec(DIM, 0, noise=0.02, seed=2))
            # A blended vector that still leans toward axis 0 enough to pass.
            blended = _axis_vec(DIM, 0) * 0.6 + _axis_vec(DIM, 1) * 0.8
            _add_face(s, img, far, embedding=blended.astype(np.float32))

        with session_scope() as s:
            suggestions = SuggestionService(s).generate_suggestions()

        assert len(suggestions) == 2
        assert suggestions[0].target_person_id == close
        assert suggestions[0].similarity > suggestions[1].similarity

    def test_no_named_persons_yields_no_suggestions(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s, "/img.jpg")
            unknown = _add_person(s, "Unknown 1", auto=True)
            _add_face(s, img, unknown, embedding=_axis_vec(DIM, 0))

        with session_scope() as s:
            assert SuggestionService(s).generate_suggestions() == []


# ---------------------------------------------------------------------------
# Excluded / embedding-less faces
# ---------------------------------------------------------------------------

class TestFaceFiltering:
    def test_excluded_and_unembedded_faces_ignored(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s, "/img.jpg")
            named = _add_person(s, "Kovács Béla", auto=False)
            unknown = _add_person(s, "Unknown 1", auto=True)

            # The only face that should define the named person's profile.
            _add_face(s, img, named, embedding=_axis_vec(DIM, 0, noise=0.02, seed=1))
            # Excluded face on a far axis — must NOT pull the centroid away.
            _add_face(s, img, named, embedding=_axis_vec(DIM, 80), excluded=True)
            # Face without an embedding — must be ignored, not crash.
            _add_face(s, img, named, embedding=None)

            _add_face(s, img, unknown, embedding=_axis_vec(DIM, 0, noise=0.02, seed=2))

        with session_scope() as s:
            suggestions = SuggestionService(s).generate_suggestions()

        assert len(suggestions) == 1
        # Centroid built only from the single usable face → strong match.
        assert suggestions[0].similarity > 0.9
        assert suggestions[0].target_face_count == 1

    def test_person_with_only_excluded_faces_is_not_a_candidate(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s, "/img.jpg")
            named = _add_person(s, "Kovács Béla", auto=False)
            unknown = _add_person(s, "Unknown 1", auto=True)
            _add_face(s, img, named, embedding=_axis_vec(DIM, 0))
            # The unknown person's only face is excluded → no usable embedding.
            _add_face(s, img, unknown, embedding=_axis_vec(DIM, 0), excluded=True)

        with session_scope() as s:
            assert SuggestionService(s).generate_suggestions() == []


# ---------------------------------------------------------------------------
# Approve / reject
# ---------------------------------------------------------------------------

class TestApproveReject:
    def test_rejected_pair_is_not_resuggested(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s, "/img.jpg")
            named = _add_person(s, "Kovács Béla", auto=False)
            unknown = _add_person(s, "Unknown 1", auto=True)
            _add_face(s, img, named, embedding=_axis_vec(DIM, 0, noise=0.02, seed=1))
            _add_face(s, img, unknown, embedding=_axis_vec(DIM, 0, noise=0.02, seed=2))

        with session_scope() as s:
            svc = SuggestionService(s)
            assert len(svc.generate_suggestions()) == 1
            svc.reject(unknown, named)

        with session_scope() as s:
            assert SuggestionService(s).generate_suggestions() == []

    def test_reject_records_different_correction(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s, "/img.jpg")
            named = _add_person(s, "Kovács Béla", auto=False)
            unknown = _add_person(s, "Unknown 1", auto=True)
            _add_face(s, img, named, embedding=_axis_vec(DIM, 0))
            _add_face(s, img, unknown, embedding=_axis_vec(DIM, 0))

        with session_scope() as s:
            SuggestionService(s).reject(unknown, named)

        with session_scope() as s:
            corrections = s.query(FaceCorrection).all()
            assert len(corrections) >= 1
            assert all(c.same_person is False for c in corrections)

    def test_approve_merges_unknown_into_named(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s, "/img.jpg")
            named = _add_person(s, "Kovács Béla", auto=False)
            unknown = _add_person(s, "Unknown 1", auto=True)
            _add_face(s, img, named, embedding=_axis_vec(DIM, 0, noise=0.02, seed=1))
            _add_face(s, img, unknown, embedding=_axis_vec(DIM, 0, noise=0.02, seed=2))

        with session_scope() as s:
            SuggestionService(s).approve(unknown, named)

        with session_scope() as s:
            assert s.get(Person, unknown) is None
            survivor = s.get(Person, named)
            assert survivor is not None
            assert len(survivor.faces) == 2
