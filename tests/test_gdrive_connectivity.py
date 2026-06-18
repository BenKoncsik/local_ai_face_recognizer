"""Tests for Google Drive connectivity checks."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.gdrive.connectivity import GDriveOfflineError, is_online, require_online


class TestIsOnline:
    def test_returns_true_when_socket_connects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = None
        mock_cm.__exit__.return_value = False
        monkeypatch.setattr(
            "app.gdrive.connectivity.socket.create_connection",
            lambda *_a, **_k: mock_cm,
        )
        assert is_online() is True

    def test_returns_false_on_os_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _fail(*_a, **_k):
            raise OSError("network down")

        monkeypatch.setattr("app.gdrive.connectivity.socket.create_connection", _fail)
        assert is_online() is False


class TestRequireOnline:
    def test_passes_when_online(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("app.gdrive.connectivity.is_online", lambda: True)
        require_online()  # no exception

    def test_raises_gdrive_offline_error_when_offline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("app.gdrive.connectivity.is_online", lambda: False)
        with pytest.raises(GDriveOfflineError, match="internetkapcsolat"):
            require_online()
