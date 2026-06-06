"""Tests for GeocodingService cache-first orchestration and the address store."""

from __future__ import annotations

import pytest

from app.db.database import init_db, session_scope
from app.services.geocoding.provider import (
    GeocodeResult,
    SettlementSuggestion,
    StreetSuggestion,
)
from app.services.geocoding_service import GeocodingService


class FakeProvider:
    """In-memory provider that counts calls (no network)."""

    def __init__(self):
        self.calls = 0

    def suggest_settlements(self, prefix, *, limit=8):
        self.calls += 1
        return [
            SettlementSuggestion("Balatonszemes", 46.81, 17.78),
            SettlementSuggestion("Balatonlelle", 46.79, 17.69),
        ]

    def suggest_streets(self, settlement, prefix, *, limit=8):
        self.calls += 1
        return [StreetSuggestion("Bajcsy-Zsilinszky utca", 46.81, 17.78)]

    def geocode_address(self, settlement, street=None, house_number=None):
        self.calls += 1
        level = "settlement" if not street else ("address" if house_number else "street")
        return GeocodeResult(46.81, 17.78, level, 5000.0, "BSzemes")

    def reverse_geocode(self, lat, lon):
        return None


@pytest.fixture()
def db(tmp_path):
    init_db(tmp_path / "geo.db")
    return tmp_path


def test_online_suggest_then_cache_hit(db):
    fp = FakeProvider()
    with session_scope() as s:
        g = GeocodingService(s, fp, online=True)
        first = g.suggest_settlements("Balaton")
        assert [x.name for x in first] == ["Balatonszemes", "Balatonlelle"]
        assert fp.calls == 1
        # Second identical query is served from cache — provider not called again.
        second = g.suggest_settlements("Balaton")
        assert [x.name for x in second] == ["Balatonszemes", "Balatonlelle"]
        assert fp.calls == 1


def test_offline_uses_local_suggestions_only(db):
    with session_scope() as s:
        # Seed the local store, then go offline (no provider).
        seed = GeocodingService(s, None, online=False)
        seed.record_address_use("Balatonszemes", "Bajcsy-Zsilinszky utca")
    with session_scope() as s:
        g = GeocodingService(s, None, online=False)
        settlements = g.suggest_settlements("Bala")
        assert [x.name for x in settlements] == ["Balatonszemes"]
        streets = g.suggest_streets("Balatonszemes", "Bajc")
        assert [x.name for x in streets] == ["Bajcsy-Zsilinszky utca"]


def test_offline_never_calls_provider(db):
    fp = FakeProvider()
    with session_scope() as s:
        # online=False forces offline even though a provider exists.
        g = GeocodingService(s, fp, online=False)
        g.suggest_settlements("Balaton")
        g.geocode("Balatonszemes")
        assert fp.calls == 0


def test_provider_error_degrades_gracefully(db):
    class BoomProvider(FakeProvider):
        def suggest_settlements(self, prefix, *, limit=8):
            raise RuntimeError("network down")

    with session_scope() as s:
        g = GeocodingService(s, BoomProvider(), online=True)
        # Should not raise; falls back to (empty) local suggestions.
        assert g.suggest_settlements("Balaton") == []


def test_geocode_cached(db):
    fp = FakeProvider()
    with session_scope() as s:
        g = GeocodingService(s, fp, online=True)
        r1 = g.geocode("Balatonszemes", "Bajcsy-Zsilinszky utca", "12")
        assert r1.matched_level == "address"
        assert fp.calls == 1
        r2 = g.geocode("Balatonszemes", "Bajcsy-Zsilinszky utca", "12")
        assert r2.matched_level == "address"
        assert fp.calls == 1  # cache hit


def test_record_address_use_upsert(db):
    with session_scope() as s:
        g = GeocodingService(s, None, online=False)
        g.record_address_use("Pécs", "Király utca")
        g.record_address_use("Pécs", "Király utca")  # upsert, not duplicate
        rows = g.suggest_streets("Pécs", "Kir")
        assert [x.name for x in rows] == ["Király utca"]
