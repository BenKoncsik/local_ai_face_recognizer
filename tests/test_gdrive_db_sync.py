"""Tests for Google Drive database synchronisation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.gdrive.cache import GDriveCacheManager
from app.gdrive.connectivity import GDriveOfflineError
from app.gdrive.db_sync import GDriveDbSync
from tests._gdrive_fakes import FakeDriveClient


@pytest.fixture()
def cache(tmp_path: Path) -> GDriveCacheManager:
    return GDriveCacheManager(tmp_path / "cache")


@pytest.fixture()
def client() -> FakeDriveClient:
    return FakeDriveClient()


@pytest.fixture()
def always_online(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.gdrive.db_sync.require_online", lambda: None)


class TestGDriveDbSyncOpen:
    def test_downloads_db_into_cache(
        self,
        cache: GDriveCacheManager,
        client: FakeDriveClient,
        always_online,
    ) -> None:
        db_bytes = b"SQLite format 3\x00"
        file_id = client.add_file_bytes(client.root_id, "faces.db", db_bytes)
        sync = GDriveDbSync(file_id, cache_manager=cache)

        local = sync.open(client)

        assert local.exists()
        assert local.read_bytes() == db_bytes
        assert sync.local_db_path == local

    def test_cleans_up_cache_on_download_failure(
        self,
        cache: GDriveCacheManager,
        client: FakeDriveClient,
        always_online,
    ) -> None:
        sync = GDriveDbSync("missing-file-id", cache_manager=cache)
        with pytest.raises(FileNotFoundError):
            sync.open(client)
        assert sync.local_db_path is None
        assert list(cache.cache_dir.rglob("*")) == []

    def test_raises_when_offline(
        self,
        cache: GDriveCacheManager,
        client: FakeDriveClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _offline() -> None:
            raise GDriveOfflineError("offline")

        monkeypatch.setattr("app.gdrive.db_sync.require_online", _offline)
        sync = GDriveDbSync("db-id", cache_manager=cache)
        with pytest.raises(GDriveOfflineError):
            sync.open(client)


class TestGDriveDbSyncSync:
    def test_uploads_local_copy(
        self,
        cache: GDriveCacheManager,
        client: FakeDriveClient,
        always_online,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        db_bytes = b"SQLite format 3\x00"
        file_id = client.add_file_bytes(client.root_id, "faces.db", db_bytes)
        sync = GDriveDbSync(file_id, cache_manager=cache)
        local = sync.open(client)
        local.write_bytes(b"updated-db")

        mock_client = MagicMock()
        sync.sync(mock_client)
        mock_client.upload_file.assert_called_once_with(str(local), file_id)

    def test_raises_without_open_session(
        self, cache: GDriveCacheManager, always_online
    ) -> None:
        sync = GDriveDbSync("db-id", cache_manager=cache)
        with pytest.raises(RuntimeError, match="without an open session"):
            sync.sync(MagicMock())

    def test_raises_when_local_file_missing(
        self,
        cache: GDriveCacheManager,
        client: FakeDriveClient,
        always_online,
    ) -> None:
        file_id = client.add_file_bytes(client.root_id, "faces.db", b"db")
        sync = GDriveDbSync(file_id, cache_manager=cache)
        local = sync.open(client)
        local.unlink()
        with pytest.raises(RuntimeError, match="without an open session"):
            sync.sync(client)


class TestGDriveDbSyncClose:
    def test_releases_active_reference(
        self,
        cache: GDriveCacheManager,
        client: FakeDriveClient,
        always_online,
    ) -> None:
        file_id = client.add_file_bytes(client.root_id, "faces.db", b"db")
        sync = GDriveDbSync(file_id, cache_manager=cache)
        local = sync.open(client)
        sync.close()
        assert sync.local_db_path is None
        assert local.exists()
