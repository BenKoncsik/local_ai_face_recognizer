"""End-to-end tests for :class:`GDriveProjectSession` using the fake client."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.gdrive.cache import GDriveCacheManager
from app.gdrive.project_session import (
    GDriveProjectSession,
    LockInfo,
    ProjectLocked,
)

from tests._gdrive_fakes import FakeDriveClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cache(tmp_path: Path) -> GDriveCacheManager:
    return GDriveCacheManager(tmp_path / "cache")


@pytest.fixture
def client() -> FakeDriveClient:
    return FakeDriveClient()


# ---------------------------------------------------------------------------
# open()
# ---------------------------------------------------------------------------

class TestOpen:
    def test_first_open_creates_folder_structure(
        self, client: FakeDriveClient, cache: GDriveCacheManager
    ) -> None:
        # Start from an empty project root — no database/ or metadata/ yet.
        session = GDriveProjectSession(
            client=client, project_root_id=client.root_id, cache=cache,
            heartbeat_interval=999,
        )
        local_db = session.open()
        try:
            folders = session.folders
            assert folders is not None
            assert folders.root_id == client.root_id
            assert folders.database_folder_id is not None
            assert folders.metadata_folder_id is not None
            # Lock file should exist now.
            assert folders.lock_file_id is not None
            # Local DB file should be present (empty placeholder for fresh project).
            assert local_db.exists()
            # Project descriptor was uploaded to metadata/.
            descriptor = client.find_child(folders.metadata_folder_id, "project.json")
            assert descriptor is not None
        finally:
            session.close(upload_pending=False)

    def test_existing_db_is_downloaded(
        self, client: FakeDriveClient, cache: GDriveCacheManager
    ) -> None:
        # Pre-populate the project with a folder layout + DB blob.
        db_folder = client.create_folder(client.root_id, "database")
        client.create_folder(client.root_id, "metadata")
        sample_bytes = b"SQLite format 3\x00fake-payload"
        client.add_file_bytes(db_folder.file_id, "faces.db", sample_bytes)

        session = GDriveProjectSession(
            client=client, project_root_id=client.root_id, cache=cache,
            heartbeat_interval=999,
        )
        local_db = session.open()
        try:
            assert local_db.read_bytes() == sample_bytes
            assert len(client.download_calls) >= 1
        finally:
            session.close(upload_pending=False)


# ---------------------------------------------------------------------------
# Lock semantics
# ---------------------------------------------------------------------------

class TestLocking:
    def test_fresh_lock_blocks_second_session(
        self, client: FakeDriveClient, cache: GDriveCacheManager
    ) -> None:
        session_a = GDriveProjectSession(
            client=client, project_root_id=client.root_id, cache=cache,
            heartbeat_interval=999,
        )
        session_a.open()
        try:
            session_b = GDriveProjectSession(
                client=client, project_root_id=client.root_id, cache=cache,
                heartbeat_interval=999,
            )
            with pytest.raises(ProjectLocked) as exc_info:
                session_b.open()
            assert exc_info.value.info.lock_id  # lock metadata is surfaced
        finally:
            session_a.close(upload_pending=False)

    def test_stale_lock_is_overridden(
        self, client: FakeDriveClient, cache: GDriveCacheManager
    ) -> None:
        # Pre-create a lock with an old heartbeat.
        db_folder = client.create_folder(client.root_id, "database")
        client.create_folder(client.root_id, "metadata")
        old = LockInfo(
            lock_id="ghost",
            device_name="dead-host",
            user_email="ghost@example.com",
            app_version="0.0",
            started_at="2020-01-01T00:00:00+00:00",
            last_heartbeat=(
                datetime.now(timezone.utc) - timedelta(hours=1)
            ).isoformat(),
        )
        client.add_file_bytes(
            db_folder.file_id, "faces.db.lock", old.to_json().encode("utf-8"),
            mime="application/json",
        )

        session = GDriveProjectSession(
            client=client, project_root_id=client.root_id, cache=cache,
            heartbeat_interval=999,
        )
        # Should succeed silently — old lock is treated as abandoned.
        session.open()
        try:
            assert session.folders is not None
            assert session.folders.lock_file_id is not None
        finally:
            session.close(upload_pending=False)

    def test_close_releases_lock(
        self, client: FakeDriveClient, cache: GDriveCacheManager
    ) -> None:
        session = GDriveProjectSession(
            client=client, project_root_id=client.root_id, cache=cache,
            heartbeat_interval=999,
        )
        session.open()
        folders = session.folders
        assert folders is not None
        lock_id = folders.lock_file_id
        assert lock_id is not None

        session.close(upload_pending=False)

        # Lock file gone from Drive
        with pytest.raises(FileNotFoundError):
            client.get_metadata(lock_id)


# ---------------------------------------------------------------------------
# sync()
# ---------------------------------------------------------------------------

class TestSync:
    def test_sync_uploads_dirty_db(
        self, client: FakeDriveClient, cache: GDriveCacheManager
    ) -> None:
        session = GDriveProjectSession(
            client=client, project_root_id=client.root_id, cache=cache,
            heartbeat_interval=999,
        )
        local_db = session.open()
        try:
            # Simulate a real schema write (so the WAL checkpoint has work to do).
            conn = sqlite3.connect(str(local_db))
            conn.execute("CREATE TABLE t (id INTEGER)")
            conn.execute("INSERT INTO t VALUES (1)")
            conn.commit()
            conn.close()
            session.mark_dirty()

            session.sync()

            assert session.status.dirty is False
            assert session.status.last_sync_succeeded is True
            assert session.folders.db_file_id is not None
        finally:
            session.close(upload_pending=False)

    def test_sync_no_op_when_clean(
        self, client: FakeDriveClient, cache: GDriveCacheManager
    ) -> None:
        session = GDriveProjectSession(
            client=client, project_root_id=client.root_id, cache=cache,
            heartbeat_interval=999,
        )
        # Pre-populate with an existing DB so first open is not dirty.
        db_folder = client.create_folder(client.root_id, "database")
        client.create_folder(client.root_id, "metadata")
        db_file_id = client.add_file_bytes(
            db_folder.file_id, "faces.db", b"\x01\x02\x03"
        )

        session.open()
        try:
            db_uploads_before = client.upload_calls.count(db_file_id)
            session.sync()  # heartbeat-only refresh; no DB upload
            # The DB file id was NOT re-uploaded (only the lock was touched).
            assert client.upload_calls.count(db_file_id) == db_uploads_before
            # And the on-disk DB bytes are unchanged.
            assert client._bytes[db_file_id] == b"\x01\x02\x03"
        finally:
            session.close(upload_pending=False)

    def test_failed_upload_keeps_local_db(
        self, client: FakeDriveClient, cache: GDriveCacheManager, monkeypatch
    ) -> None:
        session = GDriveProjectSession(
            client=client, project_root_id=client.root_id, cache=cache,
            heartbeat_interval=999,
        )
        local_db = session.open()
        try:
            session.mark_dirty()

            # Force the upload to fail.
            def boom(*_a, **_kw):
                raise RuntimeError("simulated network failure")

            monkeypatch.setattr(client, "upload_file", boom)
            with pytest.raises(RuntimeError):
                session.sync()

            # Local DB must still be there and still marked dirty.
            assert local_db.exists()
            assert session.status.dirty is True
            assert session.status.last_sync_succeeded is False
            assert "simulated" in (session.status.last_error or "")
        finally:
            # close() will also try to sync; let it fail silently.
            session.close(upload_pending=False)


# ---------------------------------------------------------------------------
# LockInfo helpers
# ---------------------------------------------------------------------------

class TestLockInfoStaleDetection:
    def test_recent_heartbeat_is_fresh(self) -> None:
        info = LockInfo(
            lock_id="x", device_name="d", user_email="u",
            app_version="v", started_at="",
            last_heartbeat=datetime.now(timezone.utc).isoformat(),
        )
        assert info.is_stale() is False

    def test_old_heartbeat_is_stale(self) -> None:
        info = LockInfo(
            lock_id="x", device_name="d", user_email="u",
            app_version="v", started_at="",
            last_heartbeat=(
                datetime.now(timezone.utc) - timedelta(minutes=30)
            ).isoformat(),
        )
        assert info.is_stale() is True

    def test_garbled_heartbeat_treated_as_stale(self) -> None:
        info = LockInfo(
            lock_id="x", device_name="d", user_email="u",
            app_version="v", started_at="",
            last_heartbeat="not-a-date",
        )
        assert info.is_stale() is True


# Suppress unused-import warning.
_ = json
