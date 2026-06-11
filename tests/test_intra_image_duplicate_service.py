"""Unit tests for the same-image duplicate-detection cleanup pass."""

from __future__ import annotations

import numpy as np
import pytest

from app.config import IntraImageDuplicateConfig
from app.db.database import init_db, session_scope
from app.db.models import Face, Image, Person
from app.services.intra_image_duplicate_service import IntraImageDuplicateService


@pytest.fixture()
def tmp_db(tmp_path):
    db_file = tmp_path / "test.db"
    init_db(db_file)
    return db_file


def _unit(dim: int, index: int, tilt: float = 0.0, tilt_index: int = 1) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[index] = 1.0
    if tilt:
        v[tilt_index] = tilt
    return (v / np.linalg.norm(v)).astype(np.float32)


def _add_image(session, path: str) -> int:
    img = Image(file_path=path, file_hash=path, file_mtime=0.0)
    session.add(img)
    session.flush()
    return img.id


def _add_face(
    session,
    image_id: int,
    embedding,
    *,
    person_id=None,
    backend="cpu",
    bbox=(0, 0, 100, 100),
) -> int:
    x, y, w, h = bbox
    face = Face(
        image_id=image_id,
        person_id=person_id,
        bbox_x=x, bbox_y=y, bbox_w=w, bbox_h=h,
        confidence=1.0,
        detector_backend=backend,
    )
    if embedding is not None:
        face.set_embedding(np.asarray(embedding, dtype=np.float32))
    session.add(face)
    session.flush()
    return face.id


def test_overlapping_duplicate_of_assigned_face_is_removed(tmp_db):
    """A new unassigned box overlapping an assigned, embedding-identical face
    on the same image is deleted."""
    with session_scope() as session:
        person = Person(name="Anna", is_auto_named=False)
        session.add(person)
        session.flush()
        image_id = _add_image(session, "/p.jpg")
        emb = _unit(8, 0)
        anchor_id = _add_face(
            session, image_id, emb, person_id=person.id, bbox=(0, 0, 100, 100)
        )
        # Shifted box (IoU ~0.18 < detection's 0.35 dedup) but same embedding.
        dup_id = _add_face(
            session, image_id, emb, person_id=None, bbox=(55, 0, 100, 100)
        )

    with session_scope() as session:
        stats = IntraImageDuplicateService(session).remove_duplicates()

    assert stats.faces_removed == 1
    with session_scope() as session:
        assert session.get(Face, dup_id) is None
        assert session.get(Face, anchor_id) is not None


def test_duplicate_of_manual_face_is_removed(tmp_db):
    """Manually drawn faces are anchors too — overlapping duplicates go away."""
    with session_scope() as session:
        image_id = _add_image(session, "/p.jpg")
        emb = _unit(8, 2)
        manual_id = _add_face(
            session, image_id, emb, backend="manual", bbox=(10, 10, 80, 80)
        )
        dup_id = _add_face(
            session, image_id, emb, person_id=None, backend="cpu",
            bbox=(45, 15, 80, 80),
        )

    with session_scope() as session:
        stats = IntraImageDuplicateService(session).remove_duplicates()

    assert stats.faces_removed == 1
    with session_scope() as session:
        assert session.get(Face, dup_id) is None
        assert session.get(Face, manual_id) is not None


def test_non_overlapping_same_person_is_kept(tmp_db):
    """A genuine second appearance of the same person elsewhere in the frame
    (identical embedding but no overlap) must NOT be removed."""
    with session_scope() as session:
        person = Person(name="Béla", is_auto_named=False)
        session.add(person)
        session.flush()
        image_id = _add_image(session, "/p.jpg")
        emb = _unit(8, 3)
        _add_face(session, image_id, emb, person_id=person.id, bbox=(0, 0, 80, 80))
        far_id = _add_face(
            session, image_id, emb, person_id=None, bbox=(500, 500, 80, 80)
        )

    with session_scope() as session:
        stats = IntraImageDuplicateService(session).remove_duplicates()

    assert stats.faces_removed == 0
    with session_scope() as session:
        assert session.get(Face, far_id) is not None


def test_overlapping_different_person_is_kept(tmp_db):
    """Overlapping but embedding-dissimilar faces are different people — kept."""
    with session_scope() as session:
        person = Person(name="Cili", is_auto_named=False)
        session.add(person)
        session.flush()
        image_id = _add_image(session, "/p.jpg")
        _add_face(
            session, image_id, _unit(8, 0), person_id=person.id, bbox=(0, 0, 100, 100)
        )
        other_id = _add_face(
            session, image_id, _unit(8, 5), person_id=None, bbox=(40, 0, 100, 100)
        )

    with session_scope() as session:
        stats = IntraImageDuplicateService(session).remove_duplicates()

    assert stats.faces_removed == 0
    with session_scope() as session:
        assert session.get(Face, other_id) is not None


def test_duplicate_across_different_images_is_kept(tmp_db):
    """The collage case: identical face on a *different* image is never touched."""
    with session_scope() as session:
        person = Person(name="Dóra", is_auto_named=False)
        session.add(person)
        session.flush()
        emb = _unit(8, 1)
        img_a = _add_image(session, "/a.jpg")
        img_b = _add_image(session, "/collage.jpg")
        _add_face(session, img_a, emb, person_id=person.id, bbox=(0, 0, 100, 100))
        collage_id = _add_face(
            session, img_b, emb, person_id=None, bbox=(0, 0, 100, 100)
        )

    with session_scope() as session:
        stats = IntraImageDuplicateService(session).remove_duplicates()

    assert stats.faces_removed == 0
    with session_scope() as session:
        assert session.get(Face, collage_id) is not None


def test_two_unassigned_duplicates_keeps_the_better_box(tmp_db):
    """The reported bug: one face auto-detected twice, both still Unknown.

    Neither is an anchor, yet the embedding-identical overlapping pair must
    collapse to a single face — and the larger/higher-confidence box survives.
    """
    with session_scope() as session:
        image_id = _add_image(session, "/p.jpg")
        emb = _unit(8, 0)
        # Larger box (the keeper) and a smaller differently-sized box (low IoU
        # vs detection's 0.35 bar) over the same face.
        big_id = _add_face(
            session, image_id, emb, person_id=None, bbox=(0, 0, 120, 120)
        )
        small_id = _add_face(
            session, image_id, emb, person_id=None, bbox=(20, 20, 70, 70)
        )

    with session_scope() as session:
        stats = IntraImageDuplicateService(session).remove_duplicates()

    assert stats.faces_removed == 1
    with session_scope() as session:
        assert session.get(Face, big_id) is not None
        assert session.get(Face, small_id) is None


def test_two_unassigned_different_people_overlap_are_kept(tmp_db):
    """Two distinct, unassigned faces that happen to overlap are both kept."""
    with session_scope() as session:
        image_id = _add_image(session, "/p.jpg")
        a_id = _add_face(
            session, image_id, _unit(8, 0), person_id=None, bbox=(0, 0, 100, 100)
        )
        b_id = _add_face(
            session, image_id, _unit(8, 5), person_id=None, bbox=(40, 0, 100, 100)
        )

    with session_scope() as session:
        stats = IntraImageDuplicateService(session).remove_duplicates()

    assert stats.faces_removed == 0
    with session_scope() as session:
        assert session.get(Face, a_id) is not None
        assert session.get(Face, b_id) is not None


def test_disabled_config_is_a_noop(tmp_db):
    with session_scope() as session:
        person = Person(name="Anna", is_auto_named=False)
        session.add(person)
        session.flush()
        image_id = _add_image(session, "/p.jpg")
        emb = _unit(8, 0)
        _add_face(session, image_id, emb, person_id=person.id, bbox=(0, 0, 100, 100))
        dup_id = _add_face(session, image_id, emb, person_id=None, bbox=(50, 0, 100, 100))

    cfg = IntraImageDuplicateConfig(enabled=False)
    with session_scope() as session:
        stats = IntraImageDuplicateService(session, config=cfg).remove_duplicates()

    assert stats.faces_removed == 0
    with session_scope() as session:
        assert session.get(Face, dup_id) is not None


def test_multiple_duplicates_all_removed(tmp_db):
    with session_scope() as session:
        person = Person(name="Anna", is_auto_named=False)
        session.add(person)
        session.flush()
        image_id = _add_image(session, "/p.jpg")
        emb = _unit(8, 0)
        anchor_id = _add_face(
            session, image_id, emb, person_id=person.id, bbox=(0, 0, 100, 100)
        )
        dup1 = _add_face(session, image_id, emb, person_id=None, bbox=(40, 0, 100, 100))
        dup2 = _add_face(session, image_id, emb, person_id=None, bbox=(20, 20, 100, 100))

    with session_scope() as session:
        stats = IntraImageDuplicateService(session).remove_duplicates()

    assert stats.faces_removed == 2
    with session_scope() as session:
        assert session.get(Face, dup1) is None
        assert session.get(Face, dup2) is None
        assert session.get(Face, anchor_id) is not None
