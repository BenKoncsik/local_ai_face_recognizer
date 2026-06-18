"""Tests for centralized QSettings helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QSettings

from app.app_settings import _LEGACY_APP, _LEGACY_ORG, _SETTINGS_FILENAME, app_qsettings, migrate_legacy_settings


@pytest.fixture()
def settings_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "settings"
    d.mkdir()
    monkeypatch.setattr("app.paths.ensure_settings_dir", lambda: d)
    return d


class TestAppQsettings:
    def test_returns_ini_backed_settings(self, settings_dir: Path) -> None:
        qs = app_qsettings()
        assert isinstance(qs, QSettings)
        qs.setValue("test/key", "value")
        qs.sync()
        assert (settings_dir / _SETTINGS_FILENAME).exists()

    def test_same_file_on_repeated_calls(self, settings_dir: Path) -> None:
        first = app_qsettings()
        first.setValue("persisted", True)
        first.sync()
        second = app_qsettings()
        assert second.value("persisted", False, type=bool) is True


@pytest.fixture()
def legacy_settings(tmp_path: Path) -> QSettings:
    legacy_file = tmp_path / "legacy.ini"
    return QSettings(str(legacy_file), QSettings.Format.IniFormat)


@pytest.fixture()
def patch_legacy_qsettings(
    legacy_settings: QSettings, monkeypatch: pytest.MonkeyPatch
) -> QSettings:
    real_qsettings = QSettings

    def _qsettings(*args, **kwargs):
        if len(args) >= 2 and args[0] == _LEGACY_ORG and args[1] == _LEGACY_APP:
            return legacy_settings
        return real_qsettings(*args, **kwargs)

    _qsettings.Format = real_qsettings.Format
    _qsettings.IniFormat = real_qsettings.IniFormat
    monkeypatch.setattr("PySide6.QtCore.QSettings", _qsettings)
    return legacy_settings


class TestMigrateLegacySettings:
    def test_no_op_when_new_file_exists(
        self,
        settings_dir: Path,
        patch_legacy_qsettings: QSettings,
        qtbot,
    ) -> None:
        new_file = settings_dir / _SETTINGS_FILENAME
        new_file.write_text("[General]\nexisting=true\n", encoding="utf-8")
        patch_legacy_qsettings.setValue("legacy/key", "should-not-copy")
        patch_legacy_qsettings.sync()
        migrate_legacy_settings()
        qs = app_qsettings()
        assert qs.value("legacy/key") is None

    def test_no_op_when_legacy_store_empty(
        self, settings_dir: Path, patch_legacy_qsettings, qtbot
    ) -> None:
        migrate_legacy_settings()
        assert not (settings_dir / _SETTINGS_FILENAME).exists()

    def test_migrates_keys_from_legacy_store(
        self, settings_dir: Path, patch_legacy_qsettings: QSettings, qtbot
    ) -> None:
        patch_legacy_qsettings.setValue("ui/theme", "dark")
        patch_legacy_qsettings.setValue("gdrive/enabled", True)
        patch_legacy_qsettings.sync()

        migrate_legacy_settings()

        new_file = settings_dir / _SETTINGS_FILENAME
        assert new_file.exists()
        qs = app_qsettings()
        assert qs.value("ui/theme") == "dark"
        assert qs.value("gdrive/enabled", False, type=bool) is True

    def test_second_call_is_idempotent(
        self, settings_dir: Path, patch_legacy_qsettings: QSettings, qtbot
    ) -> None:
        patch_legacy_qsettings.setValue("only/once", 1)
        patch_legacy_qsettings.sync()
        migrate_legacy_settings()
        new_file = settings_dir / _SETTINGS_FILENAME
        mtime_first = new_file.stat().st_mtime
        migrate_legacy_settings()
        assert new_file.stat().st_mtime == mtime_first
