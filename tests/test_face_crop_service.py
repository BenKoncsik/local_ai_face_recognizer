"""Tests for FaceId-keyed crop preview safety."""

from __future__ import annotations

import cv2
import numpy as np

from app.db.database import init_db, session_scope
from app.db.models import Face, Image, Person
from app.services.face_crop_service import (
    canonical_face_crop_path,
    ensure_unique_face_crops,
    save_crop_for_face,
)
from app.services.identity_service import IdentityService


def _make_test_image(path) -> None:
    img = np.zeros((120, 240, 3), dtype=np.uint8)
    img[:, :120] = (0, 0, 255)
    img[:, 120:] = (255, 0, 0)
    assert cv2.imwrite(str(path), img)


def test_duplicate_crop_paths_are_repaired_per_face_id(tmp_path):
    db_path = tmp_path / "faces.db"
    init_db(db_path)

    image_path = tmp_path / "family.jpg"
    _make_test_image(image_path)
    crops_dir = tmp_path / "crops"
    crops_dir.mkdir()
    shared_crop = crops_dir / "legacy_shared.jpg"
    shared_crop.write_bytes(b"legacy")

    with session_scope() as session:
        img = Image(
            file_path=str(image_path),
            file_hash="hash",
            file_mtime=image_path.stat().st_mtime,
        )
        alice = Person(name="Alice", is_auto_named=False)
        bob = Person(name="Bob", is_auto_named=False)
        session.add_all([img, alice, bob])
        session.flush()
        f1 = Face(
            image_id=img.id,
            person=alice,
            bbox_x=10,
            bbox_y=10,
            bbox_w=60,
            bbox_h=60,
            confidence=0.9,
            detector_backend="cpu",
            crop_path=str(shared_crop),
        )
        f2 = Face(
            image_id=img.id,
            person=bob,
            bbox_x=150,
            bbox_y=10,
            bbox_w=60,
            bbox_h=60,
            confidence=0.9,
            detector_backend="cpu",
            crop_path=str(shared_crop),
        )
        session.add_all([f1, f2])
        session.flush()
        repaired = ensure_unique_face_crops(
            session,
            [f1, f2],
            crops_dir,
            (64, 64),
        )
        f1_id, f2_id = f1.id, f2.id
        f1_crop, f2_crop = f1.crop_path, f2.crop_path
        image_id = img.id

    assert repaired == 2
    assert f1_crop == str(canonical_face_crop_path(crops_dir, image_id, f1_id))
    assert f2_crop == str(canonical_face_crop_path(crops_dir, image_id, f2_id))
    assert f1_crop != f2_crop
    assert f1_crop is not None and cv2.imread(f1_crop) is not None
    assert f2_crop is not None and cv2.imread(f2_crop) is not None


def test_save_crop_for_face_does_not_reuse_another_face_path(tmp_path):
    db_path = tmp_path / "faces.db"
    init_db(db_path)

    image_path = tmp_path / "family.jpg"
    _make_test_image(image_path)
    crops_dir = tmp_path / "crops"
    crops_dir.mkdir()

    with session_scope() as session:
        img = Image(
            file_path=str(image_path),
            file_hash="hash",
            file_mtime=image_path.stat().st_mtime,
        )
        session.add(img)
        session.flush()
        f1 = Face(
            image_id=img.id,
            bbox_x=10,
            bbox_y=10,
            bbox_w=60,
            bbox_h=60,
            confidence=0.9,
            detector_backend="cpu",
        )
        f2 = Face(
            image_id=img.id,
            bbox_x=150,
            bbox_y=10,
            bbox_w=60,
            bbox_h=60,
            confidence=0.9,
            detector_backend="cpu",
        )
        session.add_all([f1, f2])
        session.flush()
        f1.crop_path = str(canonical_face_crop_path(crops_dir, img.id, f1.id))
        f2.crop_path = f1.crop_path

        saved = save_crop_for_face(f2, crops_dir, (64, 64))
        f1_crop, f2_crop = f1.crop_path, f2.crop_path

    assert saved is not None
    assert f1_crop != f2_crop
    assert f2_crop == str(saved)


def test_legacy_noncanonical_crop_path_is_migrated(tmp_path):
    db_path = tmp_path / "faces.db"
    init_db(db_path)

    image_path = tmp_path / "family.jpg"
    _make_test_image(image_path)
    crops_dir = tmp_path / "crops"
    crops_dir.mkdir()
    legacy_crop = crops_dir / "img000001_face000000.jpg"
    legacy_crop.write_bytes(b"legacy")

    with session_scope() as session:
        img = Image(
            file_path=str(image_path),
            file_hash="hash",
            file_mtime=image_path.stat().st_mtime,
        )
        session.add(img)
        session.flush()
        face = Face(
            image_id=img.id,
            bbox_x=10,
            bbox_y=10,
            bbox_w=60,
            bbox_h=60,
            confidence=0.9,
            detector_backend="cpu",
            crop_path=str(legacy_crop),
        )
        session.add(face)
        session.flush()
        face_id = face.id
        repaired = ensure_unique_face_crops(session, [face], crops_dir, (64, 64))
        migrated_path = face.crop_path
        image_id = img.id

    assert repaired == 1
    assert migrated_path == str(canonical_face_crop_path(crops_dir, image_id, face_id))
    assert migrated_path != str(legacy_crop)


def test_rapid_reassignments_keep_face_crop_sources_stable(tmp_path):
    db_path = tmp_path / "faces.db"
    init_db(db_path)

    image_path = tmp_path / "family.jpg"
    _make_test_image(image_path)
    crops_dir = tmp_path / "crops"
    crops_dir.mkdir()

    with session_scope() as session:
        img = Image(
            file_path=str(image_path),
            file_hash="hash",
            file_mtime=image_path.stat().st_mtime,
        )
        alice = Person(name="Alice", is_auto_named=False)
        bob = Person(name="Bob", is_auto_named=False)
        session.add_all([img, alice, bob])
        session.flush()
        faces = []
        for idx, x in enumerate((10, 90, 150)):
            face = Face(
                image_id=img.id,
                bbox_x=x,
                bbox_y=10,
                bbox_w=50,
                bbox_h=50,
                confidence=0.9,
                detector_backend="cpu",
            )
            session.add(face)
            session.flush()
            save_crop_for_face(face, crops_dir, (64, 64))
            faces.append(face)

        before = {face.id: face.crop_path for face in faces}
        svc = IdentityService(session)
        svc.reassign_face(faces[0].id, alice.id)
        svc.reassign_face(faces[1].id, bob.id)
        svc.reassign_face(faces[2].id, alice.id)

        after = {
            face.id: (
                session.get(Face, face.id).person_id,
                session.get(Face, face.id).crop_path,
            )
            for face in faces
        }
        alice_id, bob_id = alice.id, bob.id

    assert after[faces[0].id] == (alice_id, before[faces[0].id])
    assert after[faces[1].id] == (bob_id, before[faces[1].id])
    assert after[faces[2].id] == (alice_id, before[faces[2].id])
