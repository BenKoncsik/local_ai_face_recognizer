"""Tests for background geocoding QRunnable workers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from app.services.geocoding.provider import GeocodeResult, SettlementSuggestion, StreetSuggestion
from app.workers.geocoding_worker import (
    GeocodeWorker,
    SettlementSuggestWorker,
    StreetSuggestWorker,
    _to_dicts,
)


@dataclass
class _FakeItem:
    name: str
    lat: float
    lon: float


def _patch_session():
    @contextmanager
    def _scope():
        yield object()

    return patch("app.workers.geocoding_worker.session_scope", _scope)


def test_to_dicts():
    items = [_FakeItem("Town", 1.0, 2.0)]
    assert _to_dicts(items) == [{"name": "Town", "lat": 1.0, "lon": 2.0}]


def _mock_service(**methods):
    svc = MagicMock()
    for name, value in methods.items():
        setattr(svc, name, MagicMock(return_value=value))
    return svc


class TestSettlementSuggestWorker:
    def test_emits_settlements(self, qtbot):
        settlements = [SettlementSuggestion("Balaton", 46.8, 17.7)]
        svc = _mock_service(suggest_settlements=settlements)
        with _patch_session(), patch(
            "app.workers.geocoding_worker.create_geocoding_service",
            return_value=svc,
        ):
            worker = SettlementSuggestWorker("Bal", request_id=7)
            with qtbot.waitSignal(worker.signals.settlements_ready, timeout=3000) as blocker:
                worker.run()
        assert blocker.args[0] == 7
        assert blocker.args[1][0]["name"] == "Balaton"
        assert blocker.args[1][0]["latitude"] == 46.8

    def test_emits_failed_on_error(self, qtbot):
        with _patch_session(), patch(
            "app.workers.geocoding_worker.create_geocoding_service",
            side_effect=RuntimeError("offline"),
        ):
            worker = SettlementSuggestWorker("x", request_id=3)
            with qtbot.waitSignal(worker.signals.failed, timeout=3000) as blocker:
                worker.run()
        assert list(blocker.args) == [3, "offline"]


class TestStreetSuggestWorker:
    def test_emits_streets(self, qtbot):
        streets = [StreetSuggestion("Main St", 46.8, 17.7)]
        svc = _mock_service(suggest_streets=streets)
        with _patch_session(), patch(
            "app.workers.geocoding_worker.create_geocoding_service",
            return_value=svc,
        ):
            worker = StreetSuggestWorker("Town", "Ma", request_id=9)
            with qtbot.waitSignal(worker.signals.streets_ready, timeout=3000) as blocker:
                worker.run()
        assert blocker.args[0] == 9
        assert blocker.args[1][0]["name"] == "Main St"


class TestGeocodeWorker:
    def test_emits_geocode_result(self, qtbot):
        result = GeocodeResult(46.8, 17.7, "address", 10.0, "display")
        svc = _mock_service(geocode=result)
        with _patch_session(), patch(
            "app.workers.geocoding_worker.create_geocoding_service",
            return_value=svc,
        ):
            worker = GeocodeWorker("Town", "Main", "1", request_id=4)
            with qtbot.waitSignal(worker.signals.geocode_ready, timeout=3000) as blocker:
                worker.run()
        payload = blocker.args[1]
        assert payload["latitude"] == 46.8
        assert payload["matched_level"] == "address"

    def test_emits_none_when_not_found(self, qtbot):
        svc = _mock_service(geocode=None)
        with _patch_session(), patch(
            "app.workers.geocoding_worker.create_geocoding_service",
            return_value=svc,
        ):
            worker = GeocodeWorker("Town", None, None, request_id=1)
            with qtbot.waitSignal(worker.signals.geocode_ready, timeout=3000) as blocker:
                worker.run()
        assert list(blocker.args) == [1, None]
