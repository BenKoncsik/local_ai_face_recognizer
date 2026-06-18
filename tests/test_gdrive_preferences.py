"""Tests for persisted Google Drive preferences."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.gdrive import preferences as prefs_module
from app.gdrive.preferences import GDrivePrefs, clear_account, load, save, update_last_sync


@pytest.fixture()
def settings_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "settings"
    d.mkdir()
    monkeypatch.setattr("app.paths.ensure_settings_dir", lambda: d)
    return d


class TestGDrivePrefs:
    def test_is_ready_requires_account_and_folder(self) -> None:
        assert GDrivePrefs().is_ready is False
        assert GDrivePrefs(account_email="a@b.com").is_ready is False
        assert GDrivePrefs(folder_id="fid").is_ready is False
        assert GDrivePrefs(account_email="a@b.com", folder_id="fid").is_ready is True


class TestLoadSave:
    def test_load_returns_defaults(self, settings_dir: Path) -> None:
        loaded = load()
        assert loaded == GDrivePrefs()

    def test_save_and_load_roundtrip(self, settings_dir: Path) -> None:
        original = GDrivePrefs(
            enabled=True,
            account_email="user@gmail.com",
            folder_id="folder-123",
            folder_name="My Photos",
            selected_at="2025-01-01T12:00:00",
            db_sync_enabled=False,
            last_sync_at="2025-06-01T08:30:00",
        )
        save(original)
        loaded = load()
        assert loaded == original

    def test_load_normalises_empty_strings_to_none(self, settings_dir: Path) -> None:
        save(GDrivePrefs(account_email="", folder_id=""))
        loaded = load()
        assert loaded.account_email is None
        assert loaded.folder_id is None


class TestUpdateLastSync:
    def test_writes_explicit_timestamp(self, settings_dir: Path) -> None:
        update_last_sync("2025-06-17T10:00:00")
        assert load().last_sync_at == "2025-06-17T10:00:00"

    def test_writes_utc_now_when_omitted(
        self, settings_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _FixedDatetime:
            @staticmethod
            def utcnow():
                from datetime import datetime

                return datetime(2025, 6, 17, 9, 0, 0)

        monkeypatch.setattr(prefs_module, "datetime", _FixedDatetime)
        update_last_sync()
        assert load().last_sync_at == "2025-06-17T09:00:00"


class TestClearAccount:
    def test_clears_account_and_folder_fields(self, settings_dir: Path) -> None:
        save(
            GDrivePrefs(
                enabled=True,
                account_email="user@gmail.com",
                folder_id="folder-123",
                folder_name="My Photos",
                selected_at="2025-01-01T12:00:00",
                last_sync_at="2025-06-01T08:30:00",
            )
        )
        clear_account()
        loaded = load()
        assert loaded.enabled is True
        assert loaded.account_email is None
        assert loaded.folder_id is None
        assert loaded.folder_name is None
        assert loaded.selected_at is None
        assert loaded.last_sync_at == "2025-06-01T08:30:00"
