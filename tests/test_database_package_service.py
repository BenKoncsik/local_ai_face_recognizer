"""Tests for the database-only .facedb export / import."""

from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path

import numpy as np
import pytest

from app.db.database import init_db, session_scope
from app.db.models import Face, Image, Person
from app.services.database_package_service import (
    ARCHIVE_DB_PATH,
    MANIFEST_NAME,
    PACKAGE_EXTENSION,
    DatabasePackageError,
    DatabasePackageService,
)
from app.services.image_library_service import get_image_library
from app.utils.image_utils import save_image_bgr


@pytest.fixture()
def project(tmp_path):
    """A minimal project: db + two images under a library root + one crop."""
    db_path = tmp_path / "faces.db"
    init_db(db_path)

    images_root = tmp_path / "libraryA"
    (images_root / "1998" / "nyar").mkdir(parents=True)
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[:, :] = (10, 20, 30)

    photo = images_root / "1998" / "nyar" / "photo.jpg"
    other = images_root / "other.jpg"
    assert save_image_bgr(photo, arr)
    assert save_image_bgr(other, arr)

    crops_dir = tmp_path / "crops"
    crops_dir.mkdir()
    crop_file = crops_dir / "img000001_face000001.jpg"
    assert save_image_bgr(crop_file, arr)

    get_image_library().set_library_root(images_root)

    with session_scope() as session:
        person = Person(name="Alice", is_auto_named=False)
        session.add(person)
        session.flush()
        img = Image(
            file_path=str(photo),
            relative_path="1998/nyar/photo.jpg",
            file_hash="hash1",
            file_mtime=0.0,
            width=32,
            height=32,
            detection_done=True,
        )
        # No relative_path: exercises the "derive it at export time" path.
        legacy = Image(
            file_path=str(other),
            file_hash="hash2",
            file_mtime=0.0,
        )
        session.add_all([img, legacy])
        session.flush()
        session.add(
            Face(
                image_id=img.id,
                person_id=person.id,
                bbox_x=1, bbox_y=1, bbox_w=10, bbox_h=10,
                confidence=0.9,
                detector_backend="cpu",
                crop_path=str(crop_file),
            )
        )

    return {
        "db_path": db_path,
        "images_root": images_root,
        "crops_dir": crops_dir,
        "tmp_path": tmp_path,
    }


def _export(project, name="pkg") -> Path:
    svc = DatabasePackageService(
        db_path=project["db_path"], library_root=project["images_root"]
    )
    return svc.export_database(project["tmp_path"] / name).package_path


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def test_export_contains_only_database_and_manifest(project):
    package = _export(project)

    assert package.suffix == PACKAGE_EXTENSION
    with zipfile.ZipFile(package) as zf:
        names = sorted(zf.namelist())
        assert names == sorted([MANIFEST_NAME, ARCHIVE_DB_PATH])
        manifest = json.loads(zf.read(MANIFEST_NAME).decode("utf-8"))

    assert manifest["kind"] == "database"
    assert manifest["source_library_root"] == str(project["images_root"])
    assert manifest["images"]["total"] == 2
    assert manifest["images"]["with_relative_path"] == 2


def test_export_derives_missing_relative_paths_without_touching_live_db(project):
    package = _export(project)

    with zipfile.ZipFile(package) as zf:
        copy = project["tmp_path"] / "extracted.db"
        copy.write_bytes(zf.read(ARCHIVE_DB_PATH))

    conn = sqlite3.connect(str(copy))
    rels = dict(conn.execute("SELECT file_hash, relative_path FROM images"))
    conn.close()
    assert rels["hash2"] == "other.jpg"

    # The live database keeps its NULL — the export must not mutate it.
    with session_scope() as session:
        legacy = session.query(Image).filter(Image.file_hash == "hash2").one()
        assert legacy.relative_path is None


def test_export_reports_images_outside_the_library_root(tmp_path):
    db_path = tmp_path / "faces.db"
    init_db(db_path)
    root = tmp_path / "lib"
    root.mkdir()
    with session_scope() as session:
        session.add(
            Image(
                file_path=str(tmp_path / "elsewhere" / "x.jpg"),
                file_hash="h",
                file_mtime=0.0,
            )
        )

    svc = DatabasePackageService(db_path=db_path, library_root=root)
    result = svc.export_database(tmp_path / "pkg")

    assert result.outside_root_count == 1
    assert result.has_warnings


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_validate_rejects_wrong_extension(tmp_path):
    bogus = tmp_path / "not_a_package.zip"
    bogus.write_bytes(b"x")
    assert not DatabasePackageService.validate_package(bogus).ok


def test_validate_rejects_archive_without_manifest(tmp_path):
    package = tmp_path / "broken.facedb"
    with zipfile.ZipFile(package, "w") as zf:
        zf.writestr(ARCHIVE_DB_PATH, b"not really a db")
    result = DatabasePackageService.validate_package(package)
    assert not result.ok
    assert result.errors


def test_validate_accepts_a_real_package(project):
    package = _export(project)
    result = DatabasePackageService.validate_package(package)
    assert result.ok
    assert result.manifest["database"]["sha256"]


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def test_import_rebases_paths_onto_a_different_base_folder(project, tmp_path):
    package = _export(project)

    # The "other machine": same layout below the root, different base folder.
    root_b = tmp_path / "libraryB"
    (root_b / "1998" / "nyar").mkdir(parents=True)
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    assert save_image_bgr(root_b / "1998" / "nyar" / "photo.jpg", arr)
    assert save_image_bgr(root_b / "other.jpg", arr)

    result = DatabasePackageService.import_database(
        package, tmp_path / "imported", root_b
    )

    assert result.db_path.exists()
    assert result.rebased == 2
    assert result.resolved == 2
    assert result.unresolved == 0

    conn = sqlite3.connect(str(result.db_path))
    rows = dict(conn.execute("SELECT file_hash, file_path FROM images"))
    conn.close()
    assert rows["hash1"] == str(root_b / "1998" / "nyar" / "photo.jpg")
    assert rows["hash2"] == str(root_b / "other.jpg")

    local_config = json.loads(
        (result.project_dir / "project.local.json").read_text(encoding="utf-8")
    )
    assert local_config["image_library_root"] == str(root_b)


def test_import_counts_files_missing_at_the_rebased_location(project, tmp_path):
    package = _export(project)
    root_b = tmp_path / "empty_library"
    root_b.mkdir()

    result = DatabasePackageService.import_database(
        package, tmp_path / "imported", root_b
    )

    assert result.resolved == 0
    assert result.unresolved == 2
    assert result.needs_matching


def test_import_clears_crop_paths_that_have_no_local_file(project, tmp_path):
    package = _export(project)
    root_b = tmp_path / "libraryB"
    root_b.mkdir()

    result = DatabasePackageService.import_database(
        package, tmp_path / "imported", root_b
    )

    conn = sqlite3.connect(str(result.db_path))
    crop_paths = [r[0] for r in conn.execute("SELECT crop_path FROM faces")]
    conn.close()
    # Crops are not bundled, so the stale absolute path is cleared and the
    # startup crop-repair pass regenerates it.
    assert crop_paths == [None]
    assert result.cleared_crops >= 1


def test_import_refuses_a_non_empty_destination(project, tmp_path):
    package = _export(project)
    dest = tmp_path / "occupied"
    dest.mkdir()
    (dest / "keepme.txt").write_text("data", encoding="utf-8")

    with pytest.raises(DatabasePackageError):
        DatabasePackageService.import_database(package, dest, tmp_path)

    assert (dest / "keepme.txt").exists()


def test_export_import_then_match_reattaches_a_moved_photo(tmp_path):
    """The full user story: same photos, different base folder *and* layout."""
    from app.services.image_path_matcher import ImagePathMatcher
    from app.services.scan_service import hash_file

    # --- machine A ---
    db_a = tmp_path / "a" / "faces.db"
    db_a.parent.mkdir()
    init_db(db_a)
    root_a = tmp_path / "a" / "kepek"
    (root_a / "1998" / "nyar").mkdir(parents=True)
    arr = np.full((32, 32, 3), 77, dtype=np.uint8)
    photo_a = root_a / "1998" / "nyar" / "photo.jpg"
    assert save_image_bgr(photo_a, arr)
    get_image_library().set_library_root(root_a)
    with session_scope() as session:
        session.add(
            Image(
                file_path=str(photo_a),
                relative_path="1998/nyar/photo.jpg",
                file_hash=hash_file(photo_a),
                file_mtime=0.0,
            )
        )

    package = DatabasePackageService(
        db_path=db_a, library_root=root_a
    ).export_database(tmp_path / "trip").package_path

    # --- machine B: different drive/base folder AND a renamed sub-folder ---
    root_b = tmp_path / "b" / "kepek"
    (root_b / "1998" / "nyaralas").mkdir(parents=True)
    photo_b = root_b / "1998" / "nyaralas" / "photo.jpg"
    assert save_image_bgr(photo_b, arr)

    result = DatabasePackageService.import_database(
        package, tmp_path / "b" / "project", root_b
    )
    # Re-basing alone cannot find it — the folder below the root was renamed.
    assert result.unresolved == 1

    init_db(result.db_path)
    get_image_library().set_library_root(root_b)

    matcher = ImagePathMatcher([root_b])
    with session_scope() as session:
        report = matcher.run(session)
        assert report.confident_count == 1
        proposal = report.proposals[0]
        assert proposal.best.path == str(photo_b)
        ImagePathMatcher.apply(
            session,
            {proposal.missing.image_id: proposal.best.path},
            library_root=root_b,
        )

    with session_scope() as session:
        image = session.query(Image).one()
        assert image.file_path == str(photo_b)
        assert image.relative_path == "1998/nyaralas/photo.jpg"


def test_import_rejects_a_traversal_member(tmp_path):
    package = tmp_path / "evil.facedb"
    with zipfile.ZipFile(package, "w") as zf:
        zf.writestr(
            MANIFEST_NAME,
            json.dumps({"database": {"archive_path": "../escape.db"}}),
        )
        zf.writestr("../escape.db", b"payload")

    result = DatabasePackageService.validate_package(package)
    assert not result.ok
    assert any("traversal" in e.lower() or "biztonságos" in e for e in result.errors)
