"""Tests for ImageBrowserService — folder grouping, face counts, edge cases."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.db.database import init_db, session_scope
from app.db.models import Face, Image, Person
from app.services.image_browser_service import ImageBrowserService


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def db(tmp_path):
    """Initialised empty SQLite database."""
    db_path = tmp_path / "test.db"
    init_db(db_path)
    return db_path


def _add_image(session, path: str, detection_done: bool = False) -> Image:
    img = Image(
        file_path=path,
        file_hash=f"hash_{path}",
        file_mtime=0.0,
        detection_done=detection_done,
        embedding_done=False,
    )
    session.add(img)
    session.flush()
    return img


def _add_face(
    session,
    image: Image,
    person: Person | None = None,
    excluded: bool = False,
) -> Face:
    face = Face(
        image_id=image.id,
        bbox_x=10, bbox_y=10, bbox_w=50, bbox_h=50,
        confidence=0.9,
        detector_backend="cpu",
        is_excluded=excluded,
        person=person,
    )
    session.add(face)
    session.flush()
    return face


def _add_person(session, name: str) -> Person:
    p = Person(name=name, is_auto_named=False)
    session.add(p)
    session.flush()
    return p


# ──────────────────────────────────────────────────────────────────────────────
# list_folders tests
# ──────────────────────────────────────────────────────────────────────────────

def test_list_folders_empty_db(db):
    with session_scope() as session:
        folders = ImageBrowserService(session).list_folders()
    assert folders == []


def test_list_folders_groups_by_parent(db, tmp_path):
    """Images from different parent directories must appear as separate folders."""
    folder_a = tmp_path / "trip"
    folder_b = tmp_path / "birthday"

    with session_scope() as session:
        _add_image(session, str(folder_a / "img1.jpg"))
        _add_image(session, str(folder_a / "img2.jpg"))
        _add_image(session, str(folder_b / "img3.jpg"))

    with session_scope() as session:
        folders = ImageBrowserService(session).list_folders()

    folder_paths = [f.folder_path for f in folders]
    assert str(folder_a) in folder_paths
    assert str(folder_b) in folder_paths
    assert len(folders) == 2


def test_list_folders_display_name_is_leaf(db, tmp_path):
    path = tmp_path / "2024" / "summer" / "greece"
    with session_scope() as session:
        _add_image(session, str(path / "beach.jpg"))

    with session_scope() as session:
        folders = ImageBrowserService(session).list_folders()

    assert len(folders) == 1
    assert folders[0].display_name == "greece"


def test_list_folders_total_counts(db, tmp_path):
    folder = tmp_path / "docs"
    with session_scope() as session:
        _add_image(session, str(folder / "a.jpg"), detection_done=True)
        _add_image(session, str(folder / "b.jpg"), detection_done=True)
        _add_image(session, str(folder / "c.jpg"), detection_done=False)

    with session_scope() as session:
        folders = ImageBrowserService(session).list_folders()

    assert len(folders) == 1
    f = folders[0]
    assert f.total_images == 3
    assert f.processed_images == 2
    assert f.unprocessed_images == 1


def test_list_folders_face_counts(db, tmp_path):
    folder = tmp_path / "faces"
    with session_scope() as session:
        alice = _add_person(session, "Alice")
        img = _add_image(session, str(folder / "photo.jpg"), detection_done=True)
        _add_face(session, img, person=alice)      # assigned
        _add_face(session, img, person=None)       # unknown
        _add_face(session, img, person=None)       # unknown

    with session_scope() as session:
        folders = ImageBrowserService(session).list_folders()

    f = folders[0]
    assert f.detected_faces_count == 3
    assert f.assigned_faces_count == 1
    assert f.unknown_faces_count == 2


def test_list_folders_excluded_faces_not_counted(db, tmp_path):
    folder = tmp_path / "exc"
    with session_scope() as session:
        img = _add_image(session, str(folder / "x.jpg"), detection_done=True)
        _add_face(session, img, excluded=False)
        _add_face(session, img, excluded=True)   # must NOT be counted

    with session_scope() as session:
        folders = ImageBrowserService(session).list_folders()

    f = folders[0]
    assert f.detected_faces_count == 1


# ──────────────────────────────────────────────────────────────────────────────
# list_images tests
# ──────────────────────────────────────────────────────────────────────────────

def test_list_images_returns_only_matching_folder(db, tmp_path):
    folder_a = tmp_path / "alpha"
    folder_b = tmp_path / "beta"
    with session_scope() as session:
        _add_image(session, str(folder_a / "img1.jpg"))
        _add_image(session, str(folder_b / "img2.jpg"))

    with session_scope() as session:
        images = ImageBrowserService(session).list_images(str(folder_a))

    assert len(images) == 1
    assert images[0].filename == "img1.jpg"


def test_list_images_does_not_include_subfolders(db, tmp_path):
    """Images inside a sub-folder must not appear in the parent's listing."""
    parent = tmp_path / "photos"
    child = parent / "raw"
    with session_scope() as session:
        _add_image(session, str(parent / "shot.jpg"))
        _add_image(session, str(child / "raw.jpg"))   # sub-folder — must be excluded

    with session_scope() as session:
        images = ImageBrowserService(session).list_images(str(parent))

    assert len(images) == 1
    assert images[0].filename == "shot.jpg"


def test_list_images_face_stats(db, tmp_path):
    folder = tmp_path / "stats"
    with session_scope() as session:
        alice = _add_person(session, "Alice")
        img = _add_image(session, str(folder / "p.jpg"), detection_done=True)
        _add_face(session, img, person=alice)   # assigned
        _add_face(session, img, person=None)    # unknown

    with session_scope() as session:
        images = ImageBrowserService(session).list_images(str(folder))

    assert len(images) == 1
    s = images[0]
    assert s.face_count == 2
    assert s.assigned_face_count == 1
    assert s.unknown_face_count == 1


def test_list_images_empty_folder(db, tmp_path):
    with session_scope() as session:
        images = ImageBrowserService(session).list_images(str(tmp_path / "nope"))
    assert images == []


def test_list_images_thumbnail_cache_key(db, tmp_path):
    folder = tmp_path / "keys"
    with session_scope() as session:
        img = _add_image(session, str(folder / "a.jpg"))
        expected_key = f"thumb_{img.id}"

    with session_scope() as session:
        images = ImageBrowserService(session).list_images(str(folder))

    assert images[0].thumbnail_cache_key == expected_key


# ──────────────────────────────────────────────────────────────────────────────
# get_image_faces tests
# ──────────────────────────────────────────────────────────────────────────────

def test_get_image_faces_returns_tuples(db, tmp_path):
    folder = tmp_path / "faces"
    with session_scope() as session:
        alice = _add_person(session, "Alice")
        img = _add_image(session, str(folder / "f.jpg"), detection_done=True)
        f1 = _add_face(session, img, person=alice)
        _add_face(session, img, person=None)
        image_id = img.id

    with session_scope() as session:
        faces = ImageBrowserService(session).get_image_faces(image_id)

    assert len(faces) == 2
    face_ids = [fd[0] for fd in faces]
    assert f1.id in face_ids
    # Tuple structure: (face_id, x, y, w, h, name_or_None, person_id_or_None)
    for fd in faces:
        assert len(fd) == 7


def test_get_image_faces_excludes_excluded(db, tmp_path):
    folder = tmp_path / "excl"
    with session_scope() as session:
        img = _add_image(session, str(folder / "x.jpg"), detection_done=True)
        _add_face(session, img, excluded=False)
        _add_face(session, img, excluded=True)
        image_id = img.id

    with session_scope() as session:
        faces = ImageBrowserService(session).get_image_faces(image_id)

    assert len(faces) == 1


def test_get_image_faces_missing_image(db):
    with session_scope() as session:
        faces = ImageBrowserService(session).get_image_faces(9999)
    assert faces == []


# ──────────────────────────────────────────────────────────────────────────────
# Selection-reset and assignment-safety tests (service-level)
# ──────────────────────────────────────────────────────────────────────────────

def test_assignment_does_not_change_other_faces(db, tmp_path):
    """Reassigning one face must not modify another face's person_id."""
    folder = tmp_path / "assign"
    with session_scope() as session:
        alice = _add_person(session, "Alice")
        bob = _add_person(session, "Bob")
        img = _add_image(session, str(folder / "g.jpg"), detection_done=True)
        f1 = _add_face(session, img, person=alice)
        f2 = _add_face(session, img, person=None)
        from app.services.identity_service import IdentityService
        IdentityService(session).reassign_face(f2.id, bob.id)
        face1_pid = session.get(Face, f1.id).person_id
        face2_pid = session.get(Face, f2.id).person_id

    assert face1_pid == alice.id   # unchanged
    assert face2_pid == bob.id     # only this one changed


def test_face_count_after_reassignment_stays_consistent(db, tmp_path):
    """After reassignment, unknown_face_count should decrease by 1."""
    folder = tmp_path / "recount"
    with session_scope() as session:
        alice = _add_person(session, "Alice")
        img = _add_image(session, str(folder / "r.jpg"), detection_done=True)
        f1 = _add_face(session, img, person=None)
        _add_face(session, img, person=None)
        from app.services.identity_service import IdentityService
        IdentityService(session).reassign_face(f1.id, alice.id)

    with session_scope() as session:
        images = ImageBrowserService(session).list_images(str(folder))

    s = images[0]
    assert s.assigned_face_count == 1
    assert s.unknown_face_count == 1
    assert s.face_count == 2
