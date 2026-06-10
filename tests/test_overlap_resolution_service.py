"""Tests for the same-image overlapping-box resolution pass."""

from __future__ import annotations

import numpy as np
import pytest

from app.config import OverlapResolutionConfig
from app.db.database import init_db, session_scope
from app.db.models import Face, Image, Person
from app.services.overlap_resolution_service import OverlapResolutionService

DIM = 32


@pytest.fixture()
def tmp_db(tmp_path):
    db_file = tmp_path / "test.db"
    init_db(db_file)
    return db_file


def _vec(axis: int, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = np.zeros(DIM, dtype=np.float32)
    v[axis] = 1.0
    if noise:
        v += rng.normal(0, noise, DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _add_image(session, path: str = "/img.jpg") -> int:
    image = Image(file_path=path, file_hash=path, file_mtime=0.0)
    session.add(image)
    session.flush()
    return image.id


def _add_person(session, name: str, *, auto=False) -> int:
    person = Person(name=name, is_auto_named=auto)
    session.add(person)
    session.flush()
    return person.id


def _add_face(
    session,
    image_id: int,
    person_id: int | None,
    bbox: tuple[int, int, int, int],
    embedding: np.ndarray | None = None,
    *,
    source: str | None = None,
    backend: str = "cpu",
    quality: float | None = None,
    confidence: float = 0.9,
) -> int:
    face = Face(
        image_id=image_id,
        person_id=person_id,
        bbox_x=bbox[0], bbox_y=bbox[1], bbox_w=bbox[2], bbox_h=bbox[3],
        confidence=confidence,
        detector_backend=backend,
        assignment_source=source,
        quality_score=quality,
    )
    if embedding is not None:
        face.set_embedding(embedding)
    session.add(face)
    session.flush()
    return face.id


def _resolve():
    with session_scope() as s:
        return OverlapResolutionService(s, OverlapResolutionConfig()).resolve()


class TestOverlapResolution:
    def test_assigned_box_wins_over_unknown(self, tmp_db):
        """Duplicate of an already-recognized face: the recognized one stays."""
        with session_scope() as s:
            img = _add_image(s)
            anna = _add_person(s, "Anna")
            kept = _add_face(
                s, img, anna, (10, 10, 50, 50), _vec(0), source="manual"
            )
            dup = _add_face(s, img, None, (14, 12, 52, 50), _vec(0, 0.02, 1))

        stats = _resolve()
        assert stats.faces_removed == 1

        with session_scope() as s:
            assert s.get(Face, kept) is not None
            assert s.get(Face, dup) is None

    def test_two_unknown_boxes_keep_one(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s)
            better = _add_face(
                s, img, None, (10, 10, 50, 50), _vec(0), quality=0.9
            )
            worse = _add_face(
                s, img, None, (12, 11, 50, 50), _vec(0, 0.02, 2), quality=0.4
            )

        stats = _resolve()
        assert stats.faces_removed == 1

        with session_scope() as s:
            assert s.get(Face, better) is not None
            assert s.get(Face, worse) is None

    def test_two_different_named_persons_are_untouchable(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s)
            anna = _add_person(s, "Anna")
            bela = _add_person(s, "Béla")
            f1 = _add_face(s, img, anna, (10, 10, 50, 50), _vec(0), source="manual")
            f2 = _add_face(s, img, bela, (12, 11, 50, 50), _vec(0), source="manual")

        stats = _resolve()
        assert stats.faces_removed == 0
        assert stats.conflicts_skipped >= 1

        with session_scope() as s:
            assert s.get(Face, f1) is not None
            assert s.get(Face, f2) is not None

    def test_non_overlapping_boxes_are_kept(self, tmp_db):
        """A second genuine appearance elsewhere in the frame is never deleted."""
        with session_scope() as s:
            img = _add_image(s)
            anna = _add_person(s, "Anna")
            f1 = _add_face(s, img, anna, (10, 10, 50, 50), _vec(0), source="manual")
            f2 = _add_face(s, img, None, (300, 200, 50, 50), _vec(0, 0.02, 3))

        stats = _resolve()
        assert stats.faces_removed == 0
        with session_scope() as s:
            assert s.get(Face, f1) is not None
            assert s.get(Face, f2) is not None

    def test_embedding_guard_protects_close_faces(self, tmp_db):
        """Moderately overlapping boxes of two different people both survive."""
        with session_scope() as s:
            img = _add_image(s)
            # ~0.39 IoU but clearly different embeddings (cheek-to-cheek faces).
            f1 = _add_face(s, img, None, (10, 10, 50, 50), _vec(0))
            f2 = _add_face(s, img, None, (35, 10, 50, 50), _vec(15))

        stats = _resolve()
        assert stats.faces_removed == 0
        with session_scope() as s:
            assert s.get(Face, f1) is not None
            assert s.get(Face, f2) is not None

    def test_nested_box_is_resolved_by_containment(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s)
            anna = _add_person(s, "Anna")
            big = _add_face(
                s, img, anna, (10, 10, 100, 100), _vec(0), source="manual"
            )
            nested = _add_face(s, img, None, (30, 30, 40, 40), _vec(0, 0.02, 4))

        stats = _resolve()
        assert stats.faces_removed == 1
        with session_scope() as s:
            assert s.get(Face, big) is not None
            assert s.get(Face, nested) is None

    def test_unknown_group_box_loses_to_named(self, tmp_db):
        """A box in an 'Unknown N' group is replaced by the named duplicate."""
        with session_scope() as s:
            img = _add_image(s)
            anna = _add_person(s, "Anna")
            unknown = _add_person(s, "Unknown 3", auto=True)
            named = _add_face(
                s, img, anna, (10, 10, 50, 50), _vec(0), source="recognition"
            )
            grouped = _add_face(
                s, img, unknown, (13, 12, 50, 50), _vec(0, 0.02, 5),
                source="clustering",
            )

        stats = _resolve()
        assert stats.faces_removed == 1
        with session_scope() as s:
            assert s.get(Face, named) is not None
            assert s.get(Face, grouped) is None

    def test_manual_box_beats_assigned_auto_box(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s)
            anna = _add_person(s, "Anna")
            manual = _add_face(
                s, img, anna, (10, 10, 50, 50), _vec(0),
                source="manual", backend="manual",
            )
            auto = _add_face(
                s, img, anna, (12, 12, 50, 50), _vec(0, 0.02, 6),
                source="recognition",
            )

        stats = _resolve()
        assert stats.faces_removed == 1
        with session_scope() as s:
            assert s.get(Face, manual) is not None
            assert s.get(Face, auto) is None
