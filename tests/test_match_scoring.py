"""Tests for face-match similarity scoring helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.db.database import init_db, session_scope
from app.db.models import Face, Image, Person
from app.services import match_scoring
from app.services.match_scoring import (
    _fallback_scores,
    _normalised,
    _overlay_deep,
    _person_centroid,
    match_scores_for_embedding,
    match_scores_for_face,
    match_scores_for_faces,
    match_scores_for_person,
)

DIM = 8


@pytest.fixture()
def db(tmp_path):
    init_db(tmp_path / "match.db")
    return tmp_path


def _vec(axis: int, scale: float = 1.0) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[axis] = scale
    return v


def _add_image(session, path: str = "/img.jpg") -> Image:
    image = Image(file_path=path, file_hash=path, file_mtime=0.0)
    session.add(image)
    session.flush()
    return image


def _add_person(session, name: str, *, auto: bool = False) -> Person:
    person = Person(name=name, is_auto_named=auto)
    session.add(person)
    session.flush()
    return person


def _add_face(session, image: Image, person: Person, embedding: np.ndarray) -> Face:
    face = Face(
        image_id=image.id,
        person_id=person.id,
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
    return face


def test_normalised_unit_vector():
    v = np.array([3.0, 4.0], dtype=np.float32)
    out = _normalised(v)
    assert out is not None
    assert np.linalg.norm(out) == pytest.approx(1.0)


def test_normalised_none_and_zero():
    assert _normalised(None) is None
    assert _normalised(np.zeros(4, dtype=np.float32)) is None


def test_person_centroid_from_face_embeddings(db):
    with session_scope() as session:
        img = _add_image(session)
        person = _add_person(session, "Anna")
        _add_face(session, img, person, _vec(0))
        _add_face(session, img, person, _vec(0))

        centroid = _person_centroid(person)

    assert centroid is not None
    assert centroid[0] == pytest.approx(1.0)
    assert np.linalg.norm(centroid) == pytest.approx(1.0)


def test_person_centroid_none_without_embeddings(db):
    with session_scope() as session:
        img = _add_image(session)
        person = _add_person(session, "Üres")
        face = Face(
            image_id=img.id,
            person_id=person.id,
            bbox_x=0,
            bbox_y=0,
            bbox_w=20,
            bbox_h=20,
            confidence=1.0,
            detector_backend="cpu",
        )
        session.add(face)
        session.flush()
        person = session.get(Person, person.id)
        assert _person_centroid(person) is None


def test_fallback_scores_ranks_similar_people(db):
    with session_scope() as session:
        img = _add_image(session)
        anna = _add_person(session, "Anna")
        bela = _add_person(session, "Béla")
        _add_face(session, img, anna, _vec(0))
        _add_face(session, img, bela, _vec(1))
        query = _normalised(_vec(0))
        scores = _fallback_scores(session, query)

    assert scores[anna.id] > scores[bela.id]
    assert scores[anna.id] == pytest.approx(1.0, abs=1e-5)


@patch("app.services.match_scoring.RecognitionService")
def test_match_scores_for_embedding_uses_recognition_service(mock_cls, db):
    mock_svc = MagicMock()
    mock_svc.score_persons.return_value = {1: 0.9, 2: 0.1}
    mock_cls.return_value = mock_svc
    embedding = _vec(0)

    with session_scope() as session:
        with patch("app.services.match_scoring._overlay_deep", side_effect=lambda s, e, c: s):
            scores = match_scores_for_embedding(session, embedding)

    assert scores == {1: 0.9, 2: 0.1}
    mock_svc.score_persons.assert_called_once()


@patch("app.services.match_scoring.RecognitionService")
def test_match_scores_for_embedding_falls_back_to_centroids(mock_cls, db):
    mock_svc = MagicMock()
    mock_svc.score_persons.return_value = {}
    mock_cls.return_value = mock_svc

    with session_scope() as session:
        img = _add_image(session)
        person = _add_person(session, "Anna", auto=True)
        _add_face(session, img, person, _vec(0))
        with patch("app.services.match_scoring._overlay_deep", side_effect=lambda s, e, c: s):
            scores = match_scores_for_embedding(session, _vec(0))

    assert person.id in scores
    assert scores[person.id] == pytest.approx(1.0, abs=1e-5)


def test_match_scores_for_face_missing_face_returns_empty(db):
    with session_scope() as session:
        assert match_scores_for_face(session, 999_999) == {}


def test_match_scores_for_face_uses_embedding(db):
    with session_scope() as session:
        img = _add_image(session)
        person = _add_person(session, "Anna")
        face = _add_face(session, img, person, _vec(0))
        fid = face.id

    with session_scope() as session:
        with patch(
            "app.services.match_scoring.match_scores_for_embedding",
            return_value={person.id: 0.88},
        ) as mock_embed:
            scores = match_scores_for_face(session, fid)

    assert scores == {person.id: 0.88}
    mock_embed.assert_called_once()


def test_match_scores_for_faces_averages_selected_faces(db):
    with session_scope() as session:
        img = _add_image(session)
        person = _add_person(session, "Anna")
        f1 = _add_face(session, img, person, _vec(0))
        f2 = _add_face(session, img, person, _vec(0))
        ids = [f1.id, f2.id]

    with session_scope() as session:
        with patch(
            "app.services.match_scoring.match_scores_for_embedding",
            return_value={person.id: 0.75},
        ) as mock_embed:
            scores = match_scores_for_faces(session, ids)

    assert scores == {person.id: 0.75}
    mock_embed.assert_called_once()


@patch("app.services.match_scoring.RecognitionService")
def test_match_scores_for_person_uses_profile_centroid(mock_cls, db):
    profile = MagicMock()
    profile.centroid = _vec(0)
    mock_svc = MagicMock()
    mock_svc.build_profiles.return_value = {5: profile}
    mock_svc.score_persons.return_value = {5: 0.95, 6: 0.2}
    mock_cls.return_value = mock_svc

    with session_scope() as session:
        person = _add_person(session, "Anna")
        person.id = 5
        with patch("app.services.match_scoring._overlay_deep", side_effect=lambda s, e, c: s):
            scores = match_scores_for_person(session, person)

    assert scores == {5: 0.95, 6: 0.2}


def test_overlay_deep_without_model_keeps_scores(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    match_scoring._deep_cache["key"] = None
    base = {1: 0.3, 2: 0.7}
    assert _overlay_deep(base, _vec(0), None) == base
