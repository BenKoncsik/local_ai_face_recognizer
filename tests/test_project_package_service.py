"""Tests for the portable .facepack project export / import."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

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


def test_export_handles_pre_1980_timestamp(project):
    """A source file dated before 1980 must not abort the ZIP export."""
    import os

    image_path = project["images_root"] / "sub" / "photo.jpg"
    crop_file = project["crops_dir"] / "img000001_face000001.jpg"
    # 1970-01-01 — earlier than the ZIP epoch (1980).
    os.utime(image_path, (0, 0))
    os.utime(crop_file, (0, 0))

    target = project["tmp_path"] / "old.facepack"
    result = _service(project).export_package(target)

    assert result.package_path.exists()
    assert result.image_count == 1
    assert result.crop_count == 1

    # Archive is still a valid ZIP and its entries carry the clamped 1980 date.
    with zipfile.ZipFile(result.package_path) as zf:
        assert zf.testzip() is None
        info = zf.getinfo("images/sub/photo.jpg")
        assert info.date_time == (1980, 1, 1, 0, 0, 0)
        manifest = json.loads(zf.read(MANIFEST_NAME).decode("utf-8"))

    # The real (pre-1980) mtime is still recorded in the manifest.
    entry = manifest["images"]["entries"][0]
    assert entry["mtime"] is not None
    assert entry["mtime"].startswith("19")
    assert entry["mtime_epoch"] == 0.0


def test_import_restores_pre_1980_timestamp(project, tmp_path):
    """A pre-1980 image's original mtime survives the export/import round-trip."""
    import os

    image_path = project["images_root"] / "sub" / "photo.jpg"
    os.utime(image_path, (0, 0))  # 1970-01-01

    target = project["tmp_path"] / "old.facepack"
    _service(project).export_package(target)

    dest = tmp_path / "restored"
    result = ProjectPackageService.import_package(target, dest)

    restored_img = result.images_dir / "sub" / "photo.jpg"
    assert restored_img.exists()
    # Original epoch-0 modification time is re-applied (not the clamped 1980).
    assert restored_img.stat().st_mtime == 0.0


def test_external_image_with_windows_path_and_long_name(project, tmp_path):
    """An image with a Windows absolute path and a very long basename must
    bundle under a sane, truncated, POSIX-safe archive name and import cleanly."""
    long_name = (
        "193X Szilvay Mária (1883-1966), Szilvay Géza (1920-1992), "
        "Szilvay Mária (Kőmama) (1921-2011)-Balatonöszödi képeslap "
        "Babától és Gézától, melyben kottákat kérnek.jpg"
    )
    # Real source on disk (outside the library root → no relative_path).
    ext_dir = tmp_path / "external"
    ext_dir.mkdir()
    src = ext_dir / "src.jpg"
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    assert save_image_bgr(src, arr)

    with session_scope() as session:
        img = Image(
            # A Windows-style absolute path stored in the DB.
            file_path="D:\\Családi archiv képek\\_Táblák\\" + long_name,
            relative_path=None,
            file_hash="winhash",
            file_mtime=0.0,
            width=8, height=8,
            detection_done=True,
        )
        session.add(img)

    # The library service resolves DB images by file_path; redirect that single
    # row's resolution by pointing file_path at the real temp file via a stub is
    # overkill — instead verify archive-naming directly.
    from app.services.project_package_service import (
        _basename_any,
        _safe_component,
    )

    base = _basename_any("D:\\Családi archiv képek\\_Táblák\\" + long_name)
    assert base == long_name  # no backslashes survive
    assert "\\" not in base and "/" not in base
    safe = _safe_component(base)
    assert len(safe.encode("utf-8")) <= 200
    assert safe.endswith(".jpg")

    # End-to-end: export then import must not raise ENAMETOOLONG.
    target = tmp_path / "win.facepack"
    _service(project).export_package(target)
    with zipfile.ZipFile(target) as zf:
        for name in zf.namelist():
            # No archive member is an absolute / drive-letter path.
            assert "\\" not in name
            assert not (len(name) >= 2 and name[1] == ":")
    dest = tmp_path / "restored_win"
    result = ProjectPackageService.import_package(target, dest)
    assert result.db_path.exists()


def test_export_skips_foreign_windows_path_without_crashing(project, tmp_path):
    """A Windows absolute path with a very long name must be skipped (not
    crash with ENAMETOOLONG) when exporting on macOS/Linux."""
    win_path = (
        "D:\\Családi archiv képek\\Szemesi fényképválogatás\\_Táblák\\"
        "Nagy1-1 (62[60]x140[138,5]) A\\Felhasználva\\"
        "193X Szilvay Mária (1883-1966), Szilvay Géza (1920-1992), "
        "Szilvay Mária (Kőmama) (1921-2011)-Balatonöszödi képeslap "
        "Babától és Gézától, melyben kottákat kérnek.jpg"
    )
    with session_scope() as session:
        session.add(
            Image(
                file_path=win_path,
                relative_path=None,
                file_hash="winhash-long",
                file_mtime=0.0,
                width=8, height=8,
                detection_done=True,
            )
        )

    target = tmp_path / "win.facepack"
    # Must not raise OSError / ENAMETOOLONG.
    result = _service(project).export_package(target)

    assert result.package_path.exists()
    # The fixture's real image is still bundled; the Windows-only one is skipped.
    assert result.image_count == 1
    assert any("Szilvay" in s for s in result.skipped_images)
    assert any("Szilvay" in s for s in result.missing_images)
    assert result.warning_count >= 1

    with zipfile.ZipFile(result.package_path) as zf:
        names = set(zf.namelist())
        assert ARCHIVE_DB_PATH in names                       # db present
        assert any(n.startswith(ARCHIVE_CROPS_DIR + "/") for n in names)  # crop present
        manifest = json.loads(zf.read(MANIFEST_NAME).decode("utf-8"))

    assert manifest["images"]["skipped_count"] >= 1
    assert any("Szilvay" in s for s in manifest["images"]["skipped"])


def test_is_foreign_windows_path_detection():
    import os as _os

    from app.services.project_package_service import _is_foreign_windows_path

    if _os.name == "nt":
        pytest.skip("Foreign-path detection only applies off Windows")

    assert _is_foreign_windows_path("D:\\dir\\file.jpg")
    assert _is_foreign_windows_path("C:/dir/file.jpg")
    assert _is_foreign_windows_path("\\\\server\\share\\file.jpg")
    # Genuine local POSIX paths must NOT be treated as foreign.
    assert not _is_foreign_windows_path("/Users/me/photos/file.jpg")
    assert not _is_foreign_windows_path("relative/file.jpg")
    assert not _is_foreign_windows_path("")


def test_safe_file_exists_swallows_oserror(monkeypatch):
    from pathlib import Path as _Path

    from app.services.project_package_service import _safe_file_exists

    def boom(self):
        raise OSError(63, "File name too long")

    monkeypatch.setattr(_Path, "exists", boom)
    # Must not raise — returns False instead.
    assert _safe_file_exists("D:\\whatever\\x.jpg") is False


def test_safe_component_truncates_to_byte_budget():
    from app.services.project_package_service import _safe_component

    name = ("á" * 300) + ".jpg"  # 600+ bytes of accented chars
    out = _safe_component(name)
    assert len(out.encode("utf-8")) <= 200
    assert out.endswith(".jpg")
    # Multi-byte char never split → still decodes cleanly.
    out.encode("utf-8").decode("utf-8")


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


def test_import_remaps_image_paths_with_relative_path(project, tmp_path):
    """A portable image resolves against the new library root after import."""
    target = project["tmp_path"] / "out.facepack"
    _service(project).export_package(target)
    dest = tmp_path / "restored"
    result = ProjectPackageService.import_package(target, dest)

    init_db(result.db_path)
    with session_scope() as session:
        remapped = ProjectPackageService.remap_image_paths(
            session, result.manifest, result.images_dir
        )
        # The relative_path already matches the bundled layout, but file_path
        # still points at the original machine → it gets repointed to the new
        # absolute location so direct file_path readers resolve locally.
        assert remapped == 1
        img = session.query(Image).first()
        assert img.relative_path == "sub/photo.jpg"
        expected = (result.images_dir / "sub" / "photo.jpg").resolve()
        assert expected.exists()
        assert Path(img.file_path) == expected


def test_import_remaps_legacy_absolute_only_image(tmp_path):
    """A legacy image with only a foreign absolute path resolves after import.

    This is the cross-machine case: the database row carries an absolute
    ``file_path`` from another machine and no ``relative_path``.  The file lives
    *outside* the library root, so it is bundled under ``images/_external/...``;
    after import :meth:`remap_image_paths` must repoint it so it loads locally.
    """
    from app.services.image_library_service import (
        get_image_library,
        resolve_image_path,
    )

    db_path = tmp_path / "faces.db"
    init_db(db_path)

    images_root = tmp_path / "library"
    images_root.mkdir()
    get_image_library().set_library_root(images_root)

    # The real source sits outside the library root (no portable relative path).
    external = tmp_path / "external" / "legacy.jpg"
    external.parent.mkdir(parents=True)
    arr = np.zeros((20, 20, 3), dtype=np.uint8)
    assert save_image_bgr(external, arr)

    crops_dir = tmp_path / "crops"
    crops_dir.mkdir()
    assert save_image_bgr(crops_dir / "img000001_face000001.jpg", arr)

    with session_scope() as session:
        img = Image(
            file_path=str(external),
            relative_path=None,
            file_hash="legacy",
            file_mtime=0.0,
            width=20,
            height=20,
            detection_done=True,
        )
        session.add(img)

    service = ProjectPackageService(
        db_path=db_path,
        crops_dir=crops_dir,
        library_root=images_root,
    )
    target = tmp_path / "legacy.facepack"
    service.export_package(target)

    dest = tmp_path / "restored"
    result = ProjectPackageService.import_package(target, dest)

    init_db(result.db_path)
    get_image_library().set_library_root(result.images_dir)
    with session_scope() as session:
        remapped = ProjectPackageService.remap_image_paths(
            session, result.manifest, result.images_dir
        )
        assert remapped == 1
        img = session.query(Image).first()
        # relative_path now points under the extracted images/ dir...
        assert img.relative_path is not None
        # ...and resolves to a file that actually exists on this machine.
        resolved = resolve_image_path(img)
        assert resolved is not None and resolved.exists()
        # file_path is repointed at the same on-disk location (so callers that
        # read it directly no longer chase the original machine's path).
        assert Path(img.file_path).resolve() == resolved.resolve()
        assert str(result.images_dir.resolve()) in str(Path(img.file_path).resolve())


# ---------------------------------------------------------------------------
# Cross-OS path sanitisation (macOS/Linux export → Windows import)
# ---------------------------------------------------------------------------

def test_sanitize_component_handles_windows_illegal_names():
    from app.services.project_package_service import _sanitize_component

    # Illegal characters are replaced, the extension is kept.
    assert _sanitize_component('pho:to?.jpg') == "pho_to_.jpg"
    # Trailing dot / space (forbidden on Windows) are stripped.
    assert _sanitize_component("folder. ") == "folder"
    # Reserved DOS device names are escaped, even with an extension.
    assert _sanitize_component("CON") == "_CON"
    assert _sanitize_component("nul.txt") == "_nul.txt"
    # Ordinary unicode photo names pass through untouched.
    assert _sanitize_component("nyár_2024.JPG") == "nyár_2024.JPG"


def test_sanitize_archive_path_splits_both_separators():
    from app.services.project_package_service import _sanitize_archive_path

    assert (
        _sanitize_archive_path("images\\a:b/c?.jpg") == "images/a_b/c_.jpg"
    )


def test_safe_extract_sanitizes_illegal_member_name(tmp_path):
    """A member with Windows-illegal chars lands at the sanitised path."""
    from app.services.project_package_service import _safe_extract_member

    archive = tmp_path / "src.zip"
    member_name = "images/sub:dir/pho?to.jpg"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(member_name, b"jpeg-bytes")

    dest = tmp_path / "out"
    dest.mkdir()
    with zipfile.ZipFile(archive, "r") as zf:
        _safe_extract_member(zf, zf.getinfo(member_name), dest)

    landed = dest / "images" / "sub_dir" / "pho_to.jpg"
    assert landed.exists()
    assert landed.read_bytes() == b"jpeg-bytes"
    # The original, illegal-char path was NOT created.
    assert not (dest / "images" / "sub:dir").exists()


def test_remap_image_paths_matches_sanitized_extraction(tmp_path):
    """remap_image_paths repoints to the sanitised on-disk path, not the raw one."""
    init_db(tmp_path / "faces.db")

    images_dir = tmp_path / "images"
    # The file landed under the sanitised name (as extraction would write it).
    landed = images_dir / "sub_dir" / "pho_to.jpg"
    landed.parent.mkdir(parents=True)
    landed.write_bytes(b"x")

    with session_scope() as session:
        session.add(
            Image(
                file_path="/Users/someone/lib/sub:dir/pho?to.jpg",
                relative_path=None,
                file_hash="h",
                file_mtime=0.0,
                width=10,
                height=10,
                detection_done=True,
            )
        )

    # Manifest still references the raw (pre-sanitisation) archive path.
    with session_scope() as session:
        image_id = session.query(Image).first().id
    manifest = {
        "images": {
            "entries": [
                {"id": image_id, "archive_path": "images/sub:dir/pho?to.jpg"}
            ]
        }
    }

    with session_scope() as session:
        remapped = ProjectPackageService.remap_image_paths(
            session, manifest, images_dir
        )
        assert remapped == 1
        img = session.query(Image).first()
        assert img.relative_path == "sub_dir/pho_to.jpg"
