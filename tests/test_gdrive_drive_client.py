"""Tests for the concrete Google Drive client."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.gdrive.connectivity import GDriveOfflineError
from app.gdrive.credential_store import InMemoryCredentialStore, StoredCredential
from app.gdrive.drive_client import GoogleDriveClient, _from_raw, build_drive_client
from app.gdrive.protocols import DriveFile


@pytest.fixture()
def stored_cred() -> StoredCredential:
    return StoredCredential(
        account_email="alice@example.com",
        refresh_token="rt",
        access_token="at",
        token_expiry="2030-01-01T00:00:00",
        client_id="cid",
        client_secret="csecret",
        scopes=["openid", "drive"],
    )


@pytest.fixture()
def cred_store(stored_cred: StoredCredential) -> InMemoryCredentialStore:
    store = InMemoryCredentialStore()
    store.save(stored_cred)
    return store


@pytest.fixture()
def always_online(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.gdrive.drive_client.require_online", lambda: None)


class TestFromRaw:
    def test_builds_drive_file(self) -> None:
        raw = {
            "id": "fid",
            "name": "photo.jpg",
            "mimeType": "image/jpeg",
            "size": "1234",
            "modifiedTime": "2025-01-01T00:00:00Z",
            "md5Checksum": "abc",
            "parents": ["parent"],
        }
        f = _from_raw(raw)
        assert f == DriveFile(
            file_id="fid",
            name="photo.jpg",
            mime_type="image/jpeg",
            size=1234,
            modified_time="2025-01-01T00:00:00Z",
            md5_checksum="abc",
            parents=("parent",),
        )

    def test_invalid_size_becomes_none(self) -> None:
        f = _from_raw({"id": "x", "name": "n", "size": "not-a-number"})
        assert f.size is None


class TestBuildDriveClient:
    def test_returns_client_for_stored_account(
        self, cred_store: InMemoryCredentialStore, stored_cred: StoredCredential
    ) -> None:
        client = build_drive_client(stored_cred.account_email, credential_store=cred_store)
        assert isinstance(client, GoogleDriveClient)
        assert client.account.email == stored_cred.account_email

    def test_raises_when_no_credentials(self, cred_store: InMemoryCredentialStore) -> None:
        with pytest.raises(RuntimeError, match="No stored Google credentials"):
            build_drive_client("missing@example.com", credential_store=cred_store)


class TestGoogleDriveClient:
    def _make_client(
        self,
        stored_cred: StoredCredential,
        cred_store: InMemoryCredentialStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[GoogleDriveClient, MagicMock]:
        service = MagicMock()
        monkeypatch.setattr(
            "app.gdrive.drive_client.GoogleDriveClient._ensure_service",
            lambda self: service,
        )
        monkeypatch.setattr(
            "app.gdrive.drive_client.GoogleDriveClient._persist_refreshed_token",
            lambda self: None,
        )
        return GoogleDriveClient(stored_cred, credential_store=cred_store), service

    def test_account_property(
        self, stored_cred: StoredCredential, cred_store: InMemoryCredentialStore
    ) -> None:
        client = GoogleDriveClient(stored_cred, credential_store=cred_store)
        assert client.account.email == "alice@example.com"

    def test_list_folder_returns_drive_files(
        self,
        stored_cred: StoredCredential,
        cred_store: InMemoryCredentialStore,
        always_online,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client, service = self._make_client(stored_cred, cred_store, monkeypatch)
        service.files.return_value.list.return_value.execute.return_value = {
            "files": [
                {
                    "id": "f1",
                    "name": "a.jpg",
                    "mimeType": "image/jpeg",
                    "size": "10",
                }
            ]
        }
        files = client.list_folder("parent-id")
        assert len(files) == 1
        assert files[0].name == "a.jpg"

    def test_find_child_returns_none_when_missing(
        self,
        stored_cred: StoredCredential,
        cred_store: InMemoryCredentialStore,
        always_online,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client, service = self._make_client(stored_cred, cred_store, monkeypatch)
        service.files.return_value.list.return_value.execute.return_value = {"files": []}
        assert client.find_child("parent-id", "missing.txt") is None

    def test_get_metadata(
        self,
        stored_cred: StoredCredential,
        cred_store: InMemoryCredentialStore,
        always_online,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client, service = self._make_client(stored_cred, cred_store, monkeypatch)
        service.files.return_value.get.return_value.execute.return_value = {
            "id": "fid",
            "name": "db.sqlite",
            "mimeType": "application/octet-stream",
        }
        meta = client.get_metadata("fid")
        assert meta.file_id == "fid"

    def test_create_folder(
        self,
        stored_cred: StoredCredential,
        cred_store: InMemoryCredentialStore,
        always_online,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client, service = self._make_client(stored_cred, cred_store, monkeypatch)
        service.files.return_value.create.return_value.execute.return_value = {
            "id": "new-folder",
            "name": "database",
            "mimeType": "application/vnd.google-apps.folder",
        }
        folder = client.create_folder("parent-id", "database")
        assert folder.is_folder
        assert folder.name == "database"

    def test_upload_file_creates_new_remote_file(
        self,
        stored_cred: StoredCredential,
        cred_store: InMemoryCredentialStore,
        always_online,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        client, service = self._make_client(stored_cred, cred_store, monkeypatch)
        local = tmp_path / "upload.bin"
        local.write_bytes(b"payload")
        service.files.return_value.create.return_value.execute.return_value = {"id": "new-id"}
        with patch("googleapiclient.http.MediaFileUpload", MagicMock()):
            file_id = client.upload_file(
                str(local),
                parent_folder_id="parent-id",
                remote_name="upload.bin",
            )
        assert file_id == "new-id"

    def test_upload_file_updates_existing(
        self,
        stored_cred: StoredCredential,
        cred_store: InMemoryCredentialStore,
        always_online,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        client, service = self._make_client(stored_cred, cred_store, monkeypatch)
        local = tmp_path / "upload.bin"
        local.write_bytes(b"payload")
        service.files.return_value.update.return_value.execute.return_value = {
            "id": "existing-id"
        }
        with patch("googleapiclient.http.MediaFileUpload", MagicMock()):
            file_id = client.upload_file(
                str(local),
                parent_folder_id="parent-id",
                existing_file_id="existing-id",
            )
        assert file_id == "existing-id"

    def test_upload_file_missing_source_raises(
        self,
        stored_cred: StoredCredential,
        cred_store: InMemoryCredentialStore,
        always_online,
    ) -> None:
        client = GoogleDriveClient(stored_cred, credential_store=cred_store)
        with pytest.raises(FileNotFoundError, match="Upload source not found"):
            client.upload_file("/no/such/file.bin", parent_folder_id="parent-id")

    def test_download_file_writes_bytes(
        self,
        stored_cred: StoredCredential,
        cred_store: InMemoryCredentialStore,
        always_online,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        client, service = self._make_client(stored_cred, cred_store, monkeypatch)
        local = tmp_path / "out" / "file.bin"
        downloader = MagicMock()
        downloader.next_chunk.side_effect = [(None, True)]
        with patch("googleapiclient.http.MediaIoBaseDownload", return_value=downloader):
            service.files.return_value.get_media.return_value = MagicMock()
            client.download_file("fid", str(local))
        assert local.exists()

    def test_delete_file_ignores_404(
        self,
        stored_cred: StoredCredential,
        cred_store: InMemoryCredentialStore,
        always_online,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client, service = self._make_client(stored_cred, cred_store, monkeypatch)
        from googleapiclient.errors import HttpError

        resp = SimpleNamespace(status=404, reason="Not Found")
        service.files.return_value.delete.return_value.execute.side_effect = HttpError(
            resp, b"not found"
        )
        client.delete_file("missing-id")  # idempotent

    def test_offline_raises_without_retry(
        self,
        stored_cred: StoredCredential,
        cred_store: InMemoryCredentialStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = GoogleDriveClient(stored_cred, credential_store=cred_store)

        def _offline() -> None:
            raise GDriveOfflineError("offline")

        monkeypatch.setattr("app.gdrive.drive_client.require_online", _offline)
        with pytest.raises(GDriveOfflineError):
            client.get_metadata("fid")
