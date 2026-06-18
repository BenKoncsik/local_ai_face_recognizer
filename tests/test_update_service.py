"""Tests for the GitHub release update helper (pure logic, no network)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.services import update_service
from app.services.update_service import (
    ReleaseInfo,
    _parse_version,
    _pick_asset,
    fetch_latest_release,
    is_newer,
)


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("1.2.3", (1, 2, 3)),
        ("v2.0.1", (2, 0, 1)),
        ("bad", (0,)),
        ("1.2", (0,)),
    ],
)
def test_parse_version(version, expected):
    assert _parse_version(version) == expected


@pytest.mark.parametrize(
    ("remote", "local", "expected"),
    [
        ("1.2.3", "1.2.2", True),
        ("1.2.3", "1.2.3", False),
        ("1.2.2", "1.2.3", False),
        ("v2.0.0", "1.9.9", True),
    ],
)
def test_is_newer(remote, local, expected):
    assert is_newer(remote, local) is expected


def test_pick_asset_macos_dmg_preferred(monkeypatch):
    monkeypatch.setattr(update_service.sys, "platform", "darwin")
    assets = [
        {"name": "face-local-macos.zip"},
        {"name": "face-local-macos.dmg"},
    ]
    picked = _pick_asset(assets)
    assert picked["name"] == "face-local-macos.dmg"


def test_pick_asset_windows_exe_preferred(monkeypatch):
    monkeypatch.setattr(update_service.sys, "platform", "win32")
    assets = [
        {"name": "face-local-windows.zip"},
        {"name": "face-local-windows.exe"},
    ]
    picked = _pick_asset(assets)
    assert picked["name"] == "face-local-windows.exe"


def test_pick_asset_linux_arch_match(monkeypatch):
    monkeypatch.setattr(update_service.sys, "platform", "linux")
    monkeypatch.setattr(update_service.platform, "machine", lambda: "x86_64")
    assets = [
        {"name": "face-local-linux-arm64.tar.gz"},
        {"name": "face-local-linux-x64.deb"},
    ]
    picked = _pick_asset(assets)
    assert picked["name"] == "face-local-linux-x64.deb"


def test_pick_asset_no_match_returns_none(monkeypatch):
    monkeypatch.setattr(update_service.sys, "platform", "darwin")
    assert _pick_asset([{"name": "face-local-linux-x64.deb"}]) is None


def test_fetch_latest_release_success(monkeypatch):
    payload = {
        "tag_name": "v1.4.0",
        "html_url": "https://github.com/example/repo/releases/tag/v1.4.0",
        "assets": [{"name": "face-local-macos.dmg", "browser_download_url": "https://dl/dmg", "size": 123}],
    }
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(payload).encode()
    mock_resp.__enter__ = lambda self: self
    mock_resp.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr(update_service.sys, "platform", "darwin")
    monkeypatch.setattr(update_service.urllib.request, "urlopen", lambda *a, **k: mock_resp)

    info = fetch_latest_release()

    assert isinstance(info, ReleaseInfo)
    assert info.version == "1.4.0"
    assert info.tag == "v1.4.0"
    assert info.asset_name == "face-local-macos.dmg"
    assert info.asset_url == "https://dl/dmg"
    assert info.asset_size == 123


def test_fetch_latest_release_network_error_returns_none(monkeypatch):
    monkeypatch.setattr(
        update_service.urllib.request,
        "urlopen",
        MagicMock(side_effect=OSError("offline")),
    )
    assert fetch_latest_release() is None


def test_fetch_latest_release_no_matching_asset_returns_none(monkeypatch):
    payload = {
        "tag_name": "v1.0.0",
        "html_url": "https://github.com/example/repo/releases/tag/v1.0.0",
        "assets": [{"name": "readme.txt", "browser_download_url": "https://dl/readme", "size": 1}],
    }
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(payload).encode()
    mock_resp.__enter__ = lambda self: self
    mock_resp.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr(update_service.sys, "platform", "darwin")
    monkeypatch.setattr(update_service.urllib.request, "urlopen", lambda *a, **k: mock_resp)

    assert fetch_latest_release() is None
