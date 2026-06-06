"""Tests for manual thumbnail selection on Person and Place."""

from __future__ import annotations

import cv2
import numpy as np

from app.db.database import init_db, session_scope
from app.db.models import Face, Image, Person, Place
from app.services.identity_service import IdentityService
from app.services.place_service import PlaceService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(path) -> None:
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


# ---------------------------------------------------------------------------
# Person / face thumbnail tests
# ---------------------------------------------------------------------------

def test_set_person_thumbnail_marks_manual(tmp_path):
    db_path = tmp_path / "db.sqlite"
    init_db(db_path)

    img_path = tmp_path / "img.jpg"
    _make_image(img_path)
    crop_path = tmp_path / "crop.jpg"
    _make_image(crop_path)

    with session_scope() as session:
        img = Image(file_path=str(img_path), file_hash="h1", file_mtime=0)
        alice = Person(name="Alice", is_auto_named=False)
        session.add_all([img, alice])
        session.flush()
        face = Face(
            image_id=img.id, person_id=alice.id,
            bbox_x=0, bbox_y=0, bbox_w=50, bbox_h=50,
            confidence=0.9, detector_backend="cpu",
            crop_path=str(crop_path),
        )
        session.add(face)
        session.flush()
        person_id = alice.id
        face_id = face.id

    with session_scope() as session:
        svc = IdentityService(session)
        person = svc.set_person_thumbnail(person_id, face_id)
        assert person.thumbnail_path == str(crop_path)
        assert person.thumbnail_is_manual is True


def test_set_person_thumbnail_wrong_person_raises(tmp_path):
    db_path = tmp_path / "db.sqlite"
    init_db(db_path)

    img_path = tmp_path / "img.jpg"
    _make_image(img_path)
    crop_path = tmp_path / "crop.jpg"
    _make_image(crop_path)

    with session_scope() as session:
        img = Image(file_path=str(img_path), file_hash="h1", file_mtime=0)
        alice = Person(name="Alice", is_auto_named=False)
        bob   = Person(name="Bob",   is_auto_named=False)
        session.add_all([img, alice, bob])
        session.flush()
        face = Face(
            image_id=img.id, person_id=alice.id,
            bbox_x=0, bbox_y=0, bbox_w=50, bbox_h=50,
            confidence=0.9, detector_backend="cpu",
            crop_path=str(crop_path),
        )
        session.add(face)
        session.flush()
        bob_id = bob.id
        face_id = face.id

    import pytest
    with pytest.raises(ValueError, match="does not belong"):
        with session_scope() as session:
            IdentityService(session).set_person_thumbnail(bob_id, face_id)


def test_set_person_thumbnail_missing_crop_raises(tmp_path):
    db_path = tmp_path / "db.sqlite"
    init_db(db_path)

    img_path = tmp_path / "img.jpg"
    _make_image(img_path)

    with session_scope() as session:
        img = Image(file_path=str(img_path), file_hash="h1", file_mtime=0)
        alice = Person(name="Alice", is_auto_named=False)
        session.add_all([img, alice])
        session.flush()
        face = Face(
            image_id=img.id, person_id=alice.id,
            bbox_x=0, bbox_y=0, bbox_w=50, bbox_h=50,
            confidence=0.9, detector_backend="cpu",
            crop_path=str(tmp_path / "nonexistent.jpg"),
        )
        session.add(face)
        session.flush()
        person_id = alice.id
        face_id = face.id

    import pytest
    with pytest.raises(ValueError, match="no valid crop file"):
        with session_scope() as session:
            IdentityService(session).set_person_thumbnail(person_id, face_id)


def test_clear_manual_person_thumbnail_resets_to_auto(tmp_path):
    db_path = tmp_path / "db.sqlite"
    init_db(db_path)

    img_path = tmp_path / "img.jpg"
    _make_image(img_path)
    crop1 = tmp_path / "crop1.jpg"
    crop2 = tmp_path / "crop2.jpg"
    _make_image(crop1)
    _make_image(crop2)

    with session_scope() as session:
        img = Image(file_path=str(img_path), file_hash="h1", file_mtime=0)
        alice = Person(name="Alice", is_auto_named=False)
        session.add_all([img, alice])
        session.flush()
        f1 = Face(
            image_id=img.id, person_id=alice.id,
            bbox_x=0, bbox_y=0, bbox_w=40, bbox_h=40,
            confidence=0.9, detector_backend="cpu",
            crop_path=str(crop1),
        )
        f2 = Face(
            image_id=img.id, person_id=alice.id,
            bbox_x=50, bbox_y=0, bbox_w=40, bbox_h=40,
            confidence=0.9, detector_backend="cpu",
            crop_path=str(crop2),
        )
        session.add_all([f1, f2])
        session.flush()
        person_id = alice.id
        face1_id = f1.id

    # Set manual thumbnail to face1
    with session_scope() as session:
        IdentityService(session).set_person_thumbnail(person_id, face1_id)

    # Clear it
    with session_scope() as session:
        person = IdentityService(session).clear_manual_person_thumbnail(person_id)
        assert person.thumbnail_is_manual is False
        assert person.thumbnail_path is not None  # some crop was selected as fallback


def test_auto_thumbnail_not_overwritten_by_face_crop_refresh(tmp_path):
    """Automatic thumbnail refresh must not overwrite a manual thumbnail."""
    from app.services.face_crop_service import _update_matching_person_thumbnail

    db_path = tmp_path / "db.sqlite"
    init_db(db_path)

    img_path = tmp_path / "img.jpg"
    _make_image(img_path)
    crop_old = tmp_path / "crop_old.jpg"
    crop_new = tmp_path / "crop_new.jpg"
    _make_image(crop_old)
    _make_image(crop_new)

    with session_scope() as session:
        img = Image(file_path=str(img_path), file_hash="h1", file_mtime=0)
        alice = Person(
            name="Alice",
            is_auto_named=False,
            thumbnail_path=str(crop_old),
            thumbnail_is_manual=True,
        )
        session.add_all([img, alice])
        session.flush()
        face = Face(
            image_id=img.id, person_id=alice.id,
            bbox_x=0, bbox_y=0, bbox_w=50, bbox_h=50,
            confidence=0.9, detector_backend="cpu",
            crop_path=str(crop_old),
        )
        session.add(face)
        session.flush()

        # Simulate a crop path change — manual flag should block the update
        _update_matching_person_thumbnail(face, str(crop_old), str(crop_new))
        assert alice.thumbnail_path == str(crop_old), (
            "Manual thumbnail must not be overwritten by automatic crop refresh"
        )


def test_exclude_face_clears_manual_thumbnail_fallback(tmp_path):
    db_path = tmp_path / "db.sqlite"
    init_db(db_path)

    img_path = tmp_path / "img.jpg"
    _make_image(img_path)
    crop1 = tmp_path / "crop1.jpg"
    crop2 = tmp_path / "crop2.jpg"
    _make_image(crop1)
    _make_image(crop2)

    with session_scope() as session:
        img = Image(file_path=str(img_path), file_hash="h1", file_mtime=0)
        alice = Person(
            name="Alice",
            is_auto_named=False,
            thumbnail_path=str(crop1),
            thumbnail_is_manual=True,
        )
        session.add_all([img, alice])
        session.flush()
        f1 = Face(
            image_id=img.id, person_id=alice.id,
            bbox_x=0, bbox_y=0, bbox_w=40, bbox_h=40,
            confidence=0.9, detector_backend="cpu",
            crop_path=str(crop1),
        )
        f2 = Face(
            image_id=img.id, person_id=alice.id,
            bbox_x=50, bbox_y=0, bbox_w=40, bbox_h=40,
            confidence=0.9, detector_backend="cpu",
            crop_path=str(crop2),
        )
        session.add_all([f1, f2])
        session.flush()
        person_id = alice.id
        face1_id = f1.id

    # Exclude the face that was the manual thumbnail
    with session_scope() as session:
        svc = IdentityService(session)
        svc.exclude_face(face1_id)
        person = session.get(Person, person_id)
        assert person.thumbnail_is_manual is False
        # Fallback to the remaining face's crop
        assert person.thumbnail_path == str(crop2)


# ---------------------------------------------------------------------------
# Place thumbnail tests
# ---------------------------------------------------------------------------

def test_set_place_thumbnail_marks_manual(tmp_path):
    db_path = tmp_path / "db.sqlite"
    init_db(db_path)

    img_path = tmp_path / "photo.jpg"
    _make_image(img_path)

    with session_scope() as session:
        place = Place(name="Home")
        session.add(place)
        session.flush()
        img = Image(
            file_path=str(img_path), file_hash="h1", file_mtime=0,
            place_id=place.id,
        )
        session.add(img)
        session.flush()
        place_id = place.id

    with session_scope() as session:
        svc = PlaceService(session)
        place = svc.set_place_thumbnail(place_id, str(img_path))
        assert place.thumbnail_path == str(img_path)
        assert place.thumbnail_is_manual is True


def test_set_place_thumbnail_missing_file_raises(tmp_path):
    db_path = tmp_path / "db.sqlite"
    init_db(db_path)

    with session_scope() as session:
        place = Place(name="Nowhere")
        session.add(place)
        session.flush()
        place_id = place.id

    import pytest
    with pytest.raises(ValueError, match="not found"):
        with session_scope() as session:
            PlaceService(session).set_place_thumbnail(
                place_id, str(tmp_path / "ghost.jpg")
            )


def test_ensure_thumbnail_skips_manual(tmp_path):
    """_ensure_thumbnail must not replace a valid manual thumbnail."""
    db_path = tmp_path / "db.sqlite"
    init_db(db_path)

    manual_img  = tmp_path / "manual.jpg"
    auto_img    = tmp_path / "auto.jpg"
    _make_image(manual_img)
    _make_image(auto_img)

    with session_scope() as session:
        place = Place(
            name="Park",
            thumbnail_path=str(manual_img),
            thumbnail_is_manual=True,
        )
        session.add(place)
        session.flush()
        img = Image(
            file_path=str(auto_img), file_hash="h2", file_mtime=0,
            place_id=place.id, detection_done=True,
        )
        session.add(img)
        session.flush()

        # Calling _ensure_thumbnail should leave the manual path untouched
        svc = PlaceService(session)
        svc._ensure_thumbnail(place)
        assert place.thumbnail_path == str(manual_img)


def test_ensure_thumbnail_falls_back_when_manual_file_gone(tmp_path):
    """When the manual source file is deleted, fall back to auto."""
    db_path = tmp_path / "db.sqlite"
    init_db(db_path)

    deleted_img = tmp_path / "gone.jpg"
    auto_img    = tmp_path / "auto.jpg"
    _make_image(auto_img)
    # Intentionally do NOT create deleted_img

    with session_scope() as session:
        place = Place(
            name="Beach",
            thumbnail_path=str(deleted_img),
            thumbnail_is_manual=True,
        )
        session.add(place)
        session.flush()
        img = Image(
            file_path=str(auto_img), file_hash="h3", file_mtime=0,
            place_id=place.id, detection_done=True,
        )
        session.add(img)
        session.flush()

        svc = PlaceService(session)
        svc._ensure_thumbnail(place)
        assert place.thumbnail_is_manual is False
        assert place.thumbnail_path == str(auto_img)


def test_clear_manual_place_thumbnail(tmp_path):
    db_path = tmp_path / "db.sqlite"
    init_db(db_path)

    manual_img = tmp_path / "manual.jpg"
    auto_img   = tmp_path / "auto.jpg"
    _make_image(manual_img)
    _make_image(auto_img)

    with session_scope() as session:
        place = Place(name="Lake")
        session.add(place)
        session.flush()
        img = Image(
            file_path=str(auto_img), file_hash="h4", file_mtime=0,
            place_id=place.id, detection_done=True,
        )
        session.add(img)
        session.flush()
        place_id = place.id

    with session_scope() as session:
        PlaceService(session).set_place_thumbnail(place_id, str(manual_img))

    with session_scope() as session:
        place = PlaceService(session).clear_manual_place_thumbnail(place_id)
        assert place.thumbnail_is_manual is False
        assert place.thumbnail_path == str(auto_img)


# ---------------------------------------------------------------------------
# reassign_face invalidates manual thumbnail on old person
# ---------------------------------------------------------------------------

def test_reassign_face_invalidates_manual_thumbnail(tmp_path):
    """Moving a face to another person must clear the old person's manual thumb."""
    db_path = tmp_path / "db.sqlite"
    init_db(db_path)

    img_path = tmp_path / "img.jpg"
    _make_image(img_path)
    crop1 = tmp_path / "crop1.jpg"
    crop2 = tmp_path / "crop2.jpg"
    _make_image(crop1)
    _make_image(crop2)

    with session_scope() as session:
        img = Image(file_path=str(img_path), file_hash="h1", file_mtime=0)
        alice = Person(name="Alice", is_auto_named=False)
        bob   = Person(name="Bob",   is_auto_named=False)
        session.add_all([img, alice, bob])
        session.flush()
        f1 = Face(
            image_id=img.id, person_id=alice.id,
            bbox_x=0, bbox_y=0, bbox_w=40, bbox_h=40,
            confidence=0.9, detector_backend="cpu",
            crop_path=str(crop1),
        )
        f2 = Face(
            image_id=img.id, person_id=alice.id,
            bbox_x=50, bbox_y=0, bbox_w=40, bbox_h=40,
            confidence=0.9, detector_backend="cpu",
            crop_path=str(crop2),
        )
        session.add_all([f1, f2])
        session.flush()
        alice_id = alice.id
        bob_id   = bob.id
        face1_id = f1.id

    # Make face1 Alice's manual thumbnail
    with session_scope() as session:
        IdentityService(session).set_person_thumbnail(alice_id, face1_id)

    # Reassign face1 to Bob — Alice should lose the manual thumbnail flag
    with session_scope() as session:
        IdentityService(session).reassign_face(face1_id, bob_id)

    with session_scope() as session:
        alice = session.get(Person, alice_id)
        assert alice.thumbnail_is_manual is False
        # Fallback crop must be the remaining face (f2)
        assert alice.thumbnail_path == str(crop2)


# ---------------------------------------------------------------------------
# merge_places preserves manual thumbnail flag from source
# ---------------------------------------------------------------------------

def test_merge_places_inherits_manual_thumbnail_flag(tmp_path):
    """When target has no thumbnail, merge should copy source's manual flag."""
    db_path = tmp_path / "db.sqlite"
    init_db(db_path)

    img_a = tmp_path / "a.jpg"
    img_b = tmp_path / "b.jpg"
    _make_image(img_a)
    _make_image(img_b)

    with session_scope() as session:
        place_a = Place(name="A")
        place_b = Place(name="B")
        session.add_all([place_a, place_b])
        session.flush()
        ia = Image(file_path=str(img_a), file_hash="ha", file_mtime=0, place_id=place_a.id)
        ib = Image(file_path=str(img_b), file_hash="hb", file_mtime=0, place_id=place_b.id)
        session.add_all([ia, ib])
        session.flush()
        a_id = place_a.id
        b_id = place_b.id

    # Give place_b a manual thumbnail
    with session_scope() as session:
        PlaceService(session).set_place_thumbnail(b_id, str(img_b))

    # Merge place_b INTO place_a (a_id is target, b_id is source)
    # target (a) has no thumbnail yet → should inherit b's manual thumbnail
    with session_scope() as session:
        target = PlaceService(session).merge_places([b_id], a_id)
        assert target.thumbnail_path == str(img_b)
        assert target.thumbnail_is_manual is True


def test_merge_places_keeps_target_manual_thumbnail(tmp_path):
    """When target already has a manual thumbnail, merge must not overwrite it."""
    db_path = tmp_path / "db.sqlite"
    init_db(db_path)

    img_a = tmp_path / "a.jpg"
    img_b = tmp_path / "b.jpg"
    _make_image(img_a)
    _make_image(img_b)

    with session_scope() as session:
        place_a = Place(name="A")
        place_b = Place(name="B")
        session.add_all([place_a, place_b])
        session.flush()
        ia = Image(file_path=str(img_a), file_hash="ha", file_mtime=0, place_id=place_a.id)
        ib = Image(file_path=str(img_b), file_hash="hb", file_mtime=0, place_id=place_b.id)
        session.add_all([ia, ib])
        session.flush()
        a_id = place_a.id
        b_id = place_b.id

    # Both places get a manual thumbnail
    with session_scope() as session:
        PlaceService(session).set_place_thumbnail(a_id, str(img_a))
    with session_scope() as session:
        PlaceService(session).set_place_thumbnail(b_id, str(img_b))

    # Merge b INTO a — a's manual thumbnail must survive
    with session_scope() as session:
        target = PlaceService(session).merge_places([b_id], a_id)
        assert target.thumbnail_path == str(img_a)
        assert target.thumbnail_is_manual is True
