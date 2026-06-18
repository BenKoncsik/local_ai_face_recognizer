"""Tests for the Sparkle updater bridge."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.updater import sparkle_bridge


class TestBundlePath:
    def test_returns_none_when_not_frozen(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delattr(sys, "frozen", raising=False)
        assert sparkle_bridge._bundle_path() is None

    def test_returns_app_root_when_frozen(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        app = tmp_path / "Face-Local.app"
        macos = app / "Contents" / "MacOS"
        macos.mkdir(parents=True)
        exe = macos / "Face-Local"
        exe.touch()
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "executable", str(exe))
        assert sparkle_bridge._bundle_path() == app.resolve()


class TestIsSparkleAvailable:
    def test_false_off_macos(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "platform", "linux")
        assert sparkle_bridge.is_sparkle_available() is False

    def test_false_on_macos_when_not_frozen(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.delattr(sys, "frozen", raising=False)
        assert sparkle_bridge.is_sparkle_available() is False

    def test_false_when_helper_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        app = tmp_path / "Face-Local.app"
        macos = app / "Contents" / "MacOS"
        macos.mkdir(parents=True)
        exe = macos / "Face-Local"
        exe.touch()
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "executable", str(exe))
        assert sparkle_bridge.is_sparkle_available() is False

    def test_true_when_helper_present(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        app = tmp_path / "Face-Local.app"
        helper = app / "Contents" / "MacOS" / "SparkleHelper"
        helper.parent.mkdir(parents=True)
        helper.touch()
        exe = helper.parent / "Face-Local"
        exe.touch()
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "executable", str(exe))
        assert sparkle_bridge.is_sparkle_available() is True


class TestStartBackgroundUpdateCheck:
    def test_false_off_macos(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "platform", "win32")
        assert sparkle_bridge.start_background_update_check() is False

    def test_false_when_not_in_bundle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(sparkle_bridge, "_bundle_path", lambda: None)
        assert sparkle_bridge.start_background_update_check() is False

    def test_launches_helper_when_available(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        app = tmp_path / "Face-Local.app"
        helper = app / "Contents" / "MacOS" / "SparkleHelper"
        helper.parent.mkdir(parents=True)
        helper.touch()
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(sparkle_bridge, "_bundle_path", lambda: app)
        popen = MagicMock()
        monkeypatch.setattr(sparkle_bridge.subprocess, "Popen", popen)
        assert sparkle_bridge.start_background_update_check() is True
        popen.assert_called_once()
        assert popen.call_args.args[0] == [str(helper), "--background"]


class TestCheckForUpdates:
    def test_false_off_macos(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "platform", "linux")
        assert sparkle_bridge.check_for_updates() is False

    def test_launches_manual_check(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        app = tmp_path / "Face-Local.app"
        helper = app / "Contents" / "MacOS" / "SparkleHelper"
        helper.parent.mkdir(parents=True)
        helper.touch()
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(sparkle_bridge, "_bundle_path", lambda: app)
        popen = MagicMock()
        monkeypatch.setattr(sparkle_bridge.subprocess, "Popen", popen)
        assert sparkle_bridge.check_for_updates() is True
        assert popen.call_args.args[0] == [str(helper), "--check"]
