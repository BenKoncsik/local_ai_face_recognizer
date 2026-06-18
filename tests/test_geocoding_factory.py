"""Tests for geocoding service factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.db.database import init_db, session_scope
from app.services.geocoding.factory import (
    SETTING_BASE_URL,
    SETTING_COUNTRY_CODES,
    SETTING_ENABLED,
    _read_bool,
    create_geocoding_service,
)
from app.services.geocoding_service import GeocodingService


@pytest.fixture()
def db(tmp_path):
    init_db(tmp_path / "geo_factory.db")
    return tmp_path


class _FakeQS:
    def __init__(self, values: dict):
        self._values = values

    def value(self, key, default=None, type=None):
        if key not in self._values:
            if type is bool:
                return default
            return default
        val = self._values[key]
        if type is bool and not isinstance(val, bool):
            return _read_bool(self, key, default)
        return val


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        (True, True),
        (False, False),
        ("1", True),
        ("true", True),
        ("YES", True),
        ("on", True),
        ("0", False),
        ("off", False),
    ],
)
def test_read_bool(stored, expected):
    qs = _FakeQS({"test/key": stored})
    assert _read_bool(qs, "test/key", False) is expected


def test_read_bool_missing_uses_default():
    qs = _FakeQS({})
    assert _read_bool(qs, "missing", True) is True
    assert _read_bool(qs, "missing", False) is False


def test_create_geocoding_service_force_offline(db):
    with session_scope() as session:
        svc = create_geocoding_service(session, force_online=False)

    assert isinstance(svc, GeocodingService)
    assert svc._online is False
    assert svc._provider is None


@patch("app.services.geocoding.nominatim_provider.NominatimProvider")
def test_create_geocoding_service_force_online(mock_provider_cls, db):
    mock_provider_cls.return_value = MagicMock()
    with session_scope() as session:
        svc = create_geocoding_service(session, force_online=True)

    assert isinstance(svc, GeocodingService)
    assert svc._online is True
    mock_provider_cls.assert_called_once_with()


@patch("app.services.geocoding.nominatim_provider.NominatimProvider")
def test_create_geocoding_service_reads_settings_when_not_forced(mock_provider_cls, db):
    mock_provider_cls.return_value = MagicMock()
    qs = _FakeQS(
        {
            SETTING_ENABLED: "true",
            SETTING_BASE_URL: "https://example.test/nominatim",
            SETTING_COUNTRY_CODES: "hu",
        }
    )
    with patch("app.app_settings.app_qsettings", return_value=qs):
        with session_scope() as session:
            svc = create_geocoding_service(session)

    assert svc._online is True
    mock_provider_cls.assert_called_once_with(
        base_url="https://example.test/nominatim",
        country_codes="hu",
    )


@patch("app.services.geocoding.nominatim_provider.NominatimProvider", side_effect=RuntimeError("no provider"))
def test_create_geocoding_service_provider_failure_stays_offline(mock_provider_cls, db):
    with session_scope() as session:
        svc = create_geocoding_service(session, force_online=True)

    assert svc._online is False
    assert svc._provider is None
