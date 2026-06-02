"""Tests for the portable .facepack project export / import."""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

from app.db.database import init_db, session_scope
from app.db.models import Face, Image, Person
from app.services.image_library_service import get_image_library
from app.services.project_package_service import (
    ARCHIVE_CROPS_DIR,
    ARCHIVE_DB_PATH,
    MANIFEST_NAME,
    PACKAGE_EXTENSION,
    ProjectPackageError,
    ProjectPackageService,
)
from app.utils.image_utils import save_image_bgr


@pytest.fixture()
def project(tmp_path):
    """Build a minimal project: db + one image + one face + one crop file."""
    db_path = tmp_path / "faces.db"
    init_db(db_path)

    images_root = tmp_path / "library"
    images_root.mkdir()
    image_path = images_root / "sub" / "photo.jpg"
    image_path.parent.mkdir(parents=True)
    arr = np.zeros((40, 40, 3), dtype=np.uint8)
    arr[:, :] = (10, 20, 30)
    assert save_image_bgr(image_path, arr)

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
            file_path=str(image_path),
            relative_path="sub/photo.jpg",
            file_hash="hash1",
            file_mtime=0.0,
            width=40,
            height=40,
            detection_done=True,
        )
        session.add(img)
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


def _service(project):
    return ProjectPackageService(
        db_path=project["db_path"],
        crops_dir=project["crops_dir"],
        library_root=project["images_root"],
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def test_export_creates_facepack_with_expected_layout(project):
    target = project["tmp_path"] / "out.facepack"
    result = _service(project).export_package(target, app_version="9.9.9")

    assert result.package_path.exists()
    assert result.package_path.suffix == PACKAGE_EXTENSION
    assert result.image_count == 1
    assert result.crop_count == 1

    with zipfile.ZipFile(result.package_path) as zf:
        names = set(zf.namelist())
        # manifest present
        assert MANIFEST_NAME in names
        # database present
        assert ARCHIVE_DB_PATH in names
        # crop folder present
        assert any(n.startswith(ARCHIVE_CROPS_DIR + "/") for n in names)
        # image bundled under its portable relative path
        assert "images/sub/photo.jpg" in names

        manifest = json.loads(zf.read(MANIFEST_NAME).decode("utf-8"))

    assert manifest["app_version"] == "9.9.9"
    assert manifest["database"]["sha256"]
    assert manifest["database"]["size"] > 0
    assert manifest["images"]["count"] == 1


def test_export_appends_extension_when_missing(project):
    target = project["tmp_path"] / "noext"
    result = _service(project).export_package(target)
    assert result.package_path.name == "noext.facepack"


def test_export_reports_missing_image_without_aborting(project):
    # Delete the source image so it cannot be bundled.
    (project["images_root"] / "sub" / "photo.jpg").unlink()
    target = project["tmp_path"] / "out.facepack"
    result = _service(project).export_package(target)

    assert result.package_path.exists()
    assert result.image_count == 0
    assert len(result.missing_images) == 1
    # The crop is still bundled even though the original image is gone.
    assert result.crop_count == 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_validate_rejects_wrong_extension(project, tmp_path):
    bogus = tmp_path / "not_a_package.zip"
    bogus.write_bytes(b"PK\x03\x04 not really")
    res = ProjectPackageService.validate_package(bogus)
    assert not res.ok
    assert any("kiterjeszt" in e.lower() for e in res.errors)


def test_validate_rejects_missing_manifest(tmp_path):
    pkg = tmp_path / "broken.facepack"
    with zipfile.ZipFile(pkg, "w") as zf:
        zf.writestr("database/faces.db", b"x")
    res = ProjectPackageService.validate_package(pkg)
    assert not res.ok
    assert any("manifest" in e.lower() for e in res.errors)


def test_validate_rejects_bad_manifest_json(tmp_path):
    pkg = tmp_path / "broken.facepack"
    with zipfile.ZipFile(pkg, "w") as zf:
        zf.writestr(MANIFEST_NAME, b"{not json")
        zf.writestr("database/faces.db", b"x")
        zf.writestr("crops/x.jpg", b"x")
        zf.writestr("images/x.jpg", b"x")
    res = ProjectPackageService.validate_package(pkg)
    assert not res.ok


def test_validate_accepts_real_package(project):
    target = project["tmp_path"] / "out.facepack"
    _service(project).export_package(target)
    res = ProjectPackageService.validate_package(target)
    assert res.ok
    assert res.manifest is not None


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def test_import_round_trip(project, tmp_path):
    target = project["tmp_path"] / "out.facepack"
    _service(project).export_package(target)

    dest = tmp_path / "restored"
    result = ProjectPackageService.import_package(target, dest)

    assert result.db_path.exists()
    assert (result.images_dir / "sub" / "photo.jpg").exists()
    assert result.crops_dir.is_dir()
    # Local config rewritten (next to the db) to point at the extracted images.
    cfg = json.loads(
        (result.db_path.parent / "project.local.json").read_text(encoding="utf-8")
    )
    assert cfg["image_library_root"] == str(result.images_dir.resolve())


def test_import_rejects_wrong_extension(tmp_path):
    bogus = tmp_path / "thing.zip"
    bogus.write_bytes(b"data")
    with pytest.raises(ProjectPackageError):
        ProjectPackageService.import_package(bogus, tmp_path / "dest")


def test_import_rejects_non_empty_destination(project, tmp_path):
    target = project["tmp_path"] / "out.facepack"
    _service(project).export_package(target)
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "existing.txt").write_text("keep me")
    with pytest.raises(ProjectPackageError):
        ProjectPackageService.import_package(target, dest)


def test_import_refuses_path_traversal_archive(tmp_path):
    """A .facepack containing a ../ member must never be extracted."""
    pkg = tmp_path / "evil.facepack"
    manifest = {
        "package_format_version": 1,
        "database": {"archive_path": "database/faces.db"},
    }
    with zipfile.ZipFile(pkg, "w") as zf:
        zf.writestr(MANIFEST_NAME, json.dumps(manifest))
        zf.writestr("database/faces.db", b"db")
        zf.writestr("crops/c.jpg", b"c")
        zf.writestr("images/i.jpg", b"i")
        zf.writestr("../escape.txt", b"pwned")

    dest = tmp_path / "dest"
    with pytest.raises(ProjectPackageError):
        ProjectPackageService.import_package(pkg, dest)

    # The traversal target outside dest must not have been written.
    assert not (tmp_path / "escape.txt").exists()


def test_import_remaps_crop_paths(project, tmp_path):
    target = project["tmp_path"] / "out.facepack"
    _service(project).export_package(target)
    dest = tmp_path / "restored"
    result = ProjectPackageService.import_package(target, dest)

    # Reopen the imported DB and remap crop paths into the new crops dir.
    init_db(result.db_path)
    with session_scope() as session:
        remapped = ProjectPackageService.remap_crop_paths(session, result.crops_dir)
        assert remapped == 1
        face = session.query(Face).first()
        assert face.crop_path.startswith(str(result.crops_dir))
        assert (result.crops_dir / "img000001_face000001.jpg").exists()
