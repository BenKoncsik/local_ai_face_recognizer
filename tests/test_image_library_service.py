"""Tests for ImageLibraryService and related helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.db.database import init_db, session_scope
from app.db.models import Image
from app.services.image_library_service import (
    ImageLibraryService,
    MigrationResult,
    _from_posix,
    get_image_library_optional,
    init_image_library,
    resolve_image_path,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from types import SimpleNamespace


def _make_image(file_path: str, relative_path: str | None = None) -> SimpleNamespace:
    """Lightweight stand-in for an Image ORM object for unit tests."""
    return SimpleNamespace(
        id=1,
        file_path=file_path,
        relative_path=relative_path,
    )


# ---------------------------------------------------------------------------
# _from_posix
# ---------------------------------------------------------------------------

class TestFromPosix:
    def test_simple(self):
        p = _from_posix("family/IMG_001.jpg")
        assert p == Path("family", "IMG_001.jpg")

    def test_backslash(self):
        p = _from_posix("family\\IMG_001.jpg")
        assert p == Path("family", "IMG_001.jpg")

    def test_deep(self):
        p = _from_posix("a/b/c/d.jpg")
        assert p == Path("a", "b", "c", "d.jpg")

    def test_empty(self):
        p = _from_posix("")
        assert str(p) == "."

    def test_unicode(self):
        p = _from_posix("nyár/IMG_ékezetes.jpg")
        assert p == Path("nyár", "IMG_ékezetes.jpg")


# ---------------------------------------------------------------------------
# ImageLibraryService — basic
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_path(tmp_path) -> Path:
    path = tmp_path / "test.db"
    init_db(path)
    return path


@pytest.fixture()
def svc(db_path) -> ImageLibraryService:
    return ImageLibraryService(db_path)


@pytest.fixture()
def image_root(tmp_path) -> Path:
    root = tmp_path / "images"
    root.mkdir()
    (root / "family").mkdir()
    (root / "family" / "IMG_001.jpg").write_bytes(b"\xff\xd8\xff")
    (root / "family" / "IMG_002.jpg").write_bytes(b"\xff\xd8\xff")
    (root / "vacation").mkdir()
    (root / "vacation" / "photo.jpg").write_bytes(b"\xff\xd8\xff")
    return root


class TestLibraryRootAccess:
    def test_not_configured_by_default(self, svc):
        assert svc.library_root is None
        assert not svc.is_configured()
        assert not svc.is_available()

    def test_set_and_get(self, svc, image_root):
        svc.set_library_root(image_root)
        assert svc.library_root == image_root.resolve()
        assert svc.is_configured()
        assert svc.is_available()

    def test_set_invalid_raises(self, svc, tmp_path):
        with pytest.raises(NotADirectoryError):
            svc.set_library_root(tmp_path / "nonexistent")

    def test_persists_to_json(self, svc, image_root):
        svc.set_library_root(image_root)
        config = json.loads(svc.local_config_path().read_text())
        assert "image_library_root" in config
        assert config["image_library_root"] == str(image_root.resolve())

    def test_loads_from_json(self, db_path, image_root):
        # Write config manually then create a new service instance.
        config_path = db_path.parent / "project.local.json"
        config_path.write_text(
            json.dumps({"image_library_root": str(image_root.resolve())}),
            encoding="utf-8",
        )
        svc2 = ImageLibraryService(db_path)
        assert svc2.library_root == image_root.resolve()

    def test_missing_root_returns_path_not_none(self, db_path):
        config_path = db_path.parent / "project.local.json"
        config_path.write_text(
            json.dumps({"image_library_root": "/nonexistent/path/xyz"}),
            encoding="utf-8",
        )
        svc2 = ImageLibraryService(db_path)
        assert svc2.library_root is not None
        assert not svc2.is_available()  # Path doesn't exist
        assert svc2.is_configured()     # But it is configured


# ---------------------------------------------------------------------------
# make_relative
# ---------------------------------------------------------------------------

class TestMakeRelative:
    def test_under_root(self, svc, image_root):
        svc.set_library_root(image_root)
        rel = svc.make_relative(image_root / "family" / "IMG_001.jpg")
        assert rel == "family/IMG_001.jpg"

    def test_deep_path(self, svc, image_root):
        svc.set_library_root(image_root)
        rel = svc.make_relative(image_root / "vacation" / "photo.jpg")
        assert rel == "vacation/photo.jpg"

    def test_outside_root_returns_none(self, svc, image_root, tmp_path):
        svc.set_library_root(image_root)
        outside = tmp_path / "other" / "file.jpg"
        assert svc.make_relative(outside) is None

    def test_no_root_returns_none(self, svc, image_root):
        assert svc.make_relative(image_root / "family" / "IMG_001.jpg") is None

    def test_forward_slashes(self, svc, image_root):
        svc.set_library_root(image_root)
        rel = svc.make_relative(image_root / "family" / "IMG_001.jpg")
        assert rel is not None
        assert "\\" not in rel

    def test_unicode_filename(self, tmp_path):
        root = tmp_path / "képek"
        root.mkdir()
        (root / "nyár").mkdir()
        f = root / "nyár" / "kép_001.jpg"
        f.write_bytes(b"")
        svc2 = ImageLibraryService(tmp_path / "test.db")
        svc2.set_library_root(root)
        rel = svc2.make_relative(f)
        assert rel == "nyár/kép_001.jpg"


# ---------------------------------------------------------------------------
# resolve_path
# ---------------------------------------------------------------------------

class TestResolvePath:
    def test_relative_path_with_root(self, svc, image_root):
        svc.set_library_root(image_root)
        img = _make_image(
            file_path="/old/machine/path.jpg",
            relative_path="family/IMG_001.jpg",
        )
        resolved = svc.resolve_path(img)
        assert resolved == (image_root / "family" / "IMG_001.jpg").resolve()

    def test_legacy_absolute_path(self, svc, image_root):
        abs_path = str(image_root / "family" / "IMG_001.jpg")
        img = _make_image(file_path=abs_path, relative_path=None)
        resolved = svc.resolve_path(img)
        assert resolved == Path(abs_path)

    def test_relative_without_root_falls_through_to_file_path(self, svc):
        img = _make_image(
            file_path="/some/path.jpg",
            relative_path="family/photo.jpg",
        )
        resolved = svc.resolve_path(img)
        # relative_path is set but no root — should return file_path as fallback
        assert resolved == Path("/some/path.jpg")

    def test_no_path_returns_none(self, svc):
        img = _make_image(file_path="", relative_path=None)
        img.file_path = ""
        resolved = svc.resolve_path(img)
        assert resolved is None


# ---------------------------------------------------------------------------
# is_path_under_root
# ---------------------------------------------------------------------------

class TestIsPathUnderRoot:
    def test_under(self, svc, image_root):
        svc.set_library_root(image_root)
        assert svc.is_path_under_root(image_root / "family" / "photo.jpg")

    def test_not_under(self, svc, image_root, tmp_path):
        svc.set_library_root(image_root)
        assert not svc.is_path_under_root(tmp_path / "other.jpg")

    def test_no_root(self, svc, image_root):
        assert not svc.is_path_under_root(image_root / "photo.jpg")


# ---------------------------------------------------------------------------
# detect_common_root
# ---------------------------------------------------------------------------

class TestDetectCommonRoot:
    def test_detects_common_prefix(self, db_path, image_root):
        with session_scope() as session:
            for i in range(3):
                session.add(Image(
                    file_path=str(image_root / "family" / f"img_{i}.jpg"),
                    file_hash=f"hash{i}",
                    file_mtime=0.0,
                ))

        svc = ImageLibraryService(db_path)
        with session_scope() as session:
            common = svc.detect_common_root(session)
        assert common is not None
        # Should be the family subfolder or images root
        assert str(image_root) in common or str(image_root / "family") == common

    def test_no_records_returns_none(self, db_path):
        svc = ImageLibraryService(db_path)
        with session_scope() as session:
            assert svc.detect_common_root(session) is None

    def test_ignores_already_migrated(self, db_path, image_root):
        with session_scope() as session:
            session.add(Image(
                file_path=str(image_root / "family" / "img.jpg"),
                file_hash="h1",
                file_mtime=0.0,
                relative_path="family/img.jpg",
            ))

        svc = ImageLibraryService(db_path)
        with session_scope() as session:
            # Only images without relative_path are considered
            assert svc.detect_common_root(session) is None


# ---------------------------------------------------------------------------
# migrate_to_relative_paths
# ---------------------------------------------------------------------------

class TestMigration:
    def _populate_db(self, image_root: Path) -> None:
        with session_scope() as session:
            for name in ["family/IMG_001.jpg", "family/IMG_002.jpg", "vacation/photo.jpg"]:
                session.add(Image(
                    file_path=str(image_root / name),
                    file_hash=name.replace("/", "_"),
                    file_mtime=0.0,
                ))

    def test_basic_migration(self, db_path, image_root):
        self._populate_db(image_root)
        svc = ImageLibraryService(db_path)
        with session_scope() as session:
            result = svc.migrate_to_relative_paths(session, image_root, validate_files=True)

        assert result.migrated == 3
        assert result.skipped == 0
        assert result.errors == []

        with session_scope() as session:
            imgs = session.query(Image).all()
            rel_paths = {img.relative_path for img in imgs}
        assert "family/IMG_001.jpg" in rel_paths
        assert "family/IMG_002.jpg" in rel_paths
        assert "vacation/photo.jpg" in rel_paths

    def test_uses_posix_separators(self, db_path, image_root):
        self._populate_db(image_root)
        svc = ImageLibraryService(db_path)
        with session_scope() as session:
            svc.migrate_to_relative_paths(session, image_root, validate_files=True)
        with session_scope() as session:
            for img in session.query(Image).all():
                assert img.relative_path is not None
                assert "\\" not in img.relative_path

    def test_skips_outside_root(self, db_path, image_root, tmp_path):
        with session_scope() as session:
            session.add(Image(
                file_path=str(tmp_path / "outside.jpg"),
                file_hash="outside",
                file_mtime=0.0,
            ))
        svc = ImageLibraryService(db_path)
        with session_scope() as session:
            result = svc.migrate_to_relative_paths(session, image_root, validate_files=False)
        assert result.skipped == 1
        assert result.migrated == 0

    def test_validate_missing_files(self, db_path, image_root):
        with session_scope() as session:
            session.add(Image(
                file_path=str(image_root / "family" / "NONEXISTENT.jpg"),
                file_hash="nope",
                file_mtime=0.0,
            ))
        svc = ImageLibraryService(db_path)
        with session_scope() as session:
            result = svc.migrate_to_relative_paths(session, image_root, validate_files=True)
        assert len(result.errors) == 1
        assert result.migrated == 0

    def test_no_validate_accepts_missing_files(self, db_path, image_root):
        with session_scope() as session:
            session.add(Image(
                file_path=str(image_root / "family" / "NONEXISTENT.jpg"),
                file_hash="nope",
                file_mtime=0.0,
            ))
        svc = ImageLibraryService(db_path)
        with session_scope() as session:
            result = svc.migrate_to_relative_paths(session, image_root, validate_files=False)
        assert result.migrated == 1
        assert result.errors == []

    def test_idempotent(self, db_path, image_root):
        self._populate_db(image_root)
        svc = ImageLibraryService(db_path)
        with session_scope() as session:
            r1 = svc.migrate_to_relative_paths(session, image_root, validate_files=False)
        with session_scope() as session:
            r2 = svc.migrate_to_relative_paths(session, image_root, validate_files=False)
        assert r1.migrated == 3
        assert r2.migrated == 0  # Already migrated — nothing to do
        assert r2.skipped == 0   # relative_path already set, not counted

    def test_persists_library_root(self, db_path, image_root):
        svc = ImageLibraryService(db_path)
        with session_scope() as session:
            svc.migrate_to_relative_paths(session, image_root, validate_files=False)
        assert svc.library_root is not None

    def test_progress_callback(self, db_path, image_root):
        self._populate_db(image_root)
        svc = ImageLibraryService(db_path)
        calls = []
        with session_scope() as session:
            svc.migrate_to_relative_paths(
                session, image_root, validate_files=False,
                progress_cb=lambda c, t, p: calls.append((c, t)),
            )
        assert len(calls) == 3
        assert calls[-1][0] == 3


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_init_and_get(self, tmp_path):
        db = tmp_path / "singleton.db"
        init_db(db)
        svc = get_image_library_optional()
        assert svc is not None
        assert isinstance(svc, ImageLibraryService)

    def test_reinit_on_db_change(self, tmp_path):
        db1 = tmp_path / "db1.db"
        db2 = tmp_path / "db2.db"
        init_db(db1)
        svc1 = get_image_library_optional()
        init_db(db2)
        svc2 = get_image_library_optional()
        assert svc1 is not svc2

    def test_resolve_image_path_no_root(self, tmp_path):
        db = tmp_path / "resolve.db"
        init_db(db)
        img = _make_image("/absolute/path.jpg")
        result = resolve_image_path(img)
        assert result == Path("/absolute/path.jpg")

    def test_resolve_image_path_with_root(self, tmp_path):
        root = tmp_path / "images"
        root.mkdir()
        db = tmp_path / "test.db"
        init_db(db)
        svc = get_image_library_optional()
        assert svc is not None
        svc.set_library_root(root)

        img = _make_image(
            file_path="/old/machine/path.jpg",
            relative_path="family/photo.jpg",
        )
        result = resolve_image_path(img)
        assert result == root / "family" / "photo.jpg"

    def test_resolve_none_when_no_path(self, tmp_path):
        db = tmp_path / "resolve_none.db"
        init_db(db)
        img = _make_image("")
        img.file_path = ""
        result = resolve_image_path(img)
        assert result is None


# ---------------------------------------------------------------------------
# Scan service integration: cross-machine re-linking
# ---------------------------------------------------------------------------

class TestScanServiceIntegration:
    """Test that ScanService recognises images by relative_path on a new machine."""

    def test_relink_by_relative_path(self, tmp_path):
        from app.config import ScanConfig
        from app.services.scan_service import ScanService

        # Set up: old machine paths stored in DB.
        db_path = tmp_path / "test.db"
        init_db(db_path)

        image_root = tmp_path / "images"
        image_root.mkdir()
        (image_root / "family").mkdir()
        img_file = image_root / "family" / "photo.jpg"
        img_file.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)

        svc = get_image_library_optional()
        assert svc is not None
        svc.set_library_root(image_root)

        # Simulate old machine: insert record with a different file_path but correct relative_path.
        with session_scope() as session:
            session.add(Image(
                file_path="/old/machine/images/family/photo.jpg",
                file_hash="old_hash",
                file_mtime=0.0,
                relative_path="family/photo.jpg",
            ))

        # Scan on "new machine" with the actual path.
        with session_scope() as session:
            scan_svc = ScanService(
                session=session,
                config=ScanConfig(),
                image_library_svc=svc,
            )
            new_ids = scan_svc.scan(str(image_root))

        # Should update the existing record's file_path, not create a duplicate.
        with session_scope() as session:
            images = session.query(Image).all()
        assert len(images) == 1
        assert images[0].file_path == str(img_file)
        assert images[0].relative_path == "family/photo.jpg"

    def test_backfills_relative_path_on_existing_records(self, tmp_path):
        from app.config import ScanConfig
        from app.services.scan_service import ScanService

        db_path = tmp_path / "test.db"
        init_db(db_path)

        image_root = tmp_path / "images"
        image_root.mkdir()
        img_file = image_root / "photo.jpg"
        img_file.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)

        # Insert a record WITHOUT relative_path (legacy record).
        with session_scope() as session:
            session.add(Image(
                file_path=str(img_file),
                file_hash="old",
                file_mtime=0.0,
                relative_path=None,
            ))

        svc = get_image_library_optional()
        assert svc is not None
        svc.set_library_root(image_root)

        with session_scope() as session:
            scan_svc = ScanService(
                session=session,
                config=ScanConfig(),
                image_library_svc=svc,
            )
            scan_svc.scan(str(image_root))

        with session_scope() as session:
            img = session.query(Image).first()
        assert img is not None
        assert img.relative_path == "photo.jpg"
