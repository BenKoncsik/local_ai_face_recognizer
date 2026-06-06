"""Parser tests for NominatimProvider (no live network — _get is stubbed)."""

from __future__ import annotations

from app.services.geocoding.nominatim_provider import NominatimProvider


def _provider(payload):
    p = NominatimProvider()
    p._get = lambda path, params: payload  # type: ignore[assignment]
    return p


def test_parse_settlements_dedup_and_names():
    payload = [
        {"lat": "46.81", "lon": "17.78", "name": "Balatonszemes",
         "address": {"village": "Balatonszemes", "country": "Magyarország"}},
        {"lat": "46.81", "lon": "17.78", "name": "Balatonszemes dup",
         "address": {"village": "Balatonszemes"}},
        {"lat": "46.79", "lon": "17.69", "address": {"town": "Balatonlelle"}},
    ]
    res = _provider(payload).suggest_settlements("Balaton")
    assert [r.name for r in res] == ["Balatonszemes", "Balatonlelle"]
    assert res[0].latitude == 46.81


def test_parse_streets():
    payload = [
        {"lat": "46.81", "lon": "17.78", "address": {"road": "Bajcsy-Zsilinszky utca"}},
    ]
    res = _provider(payload).suggest_streets("Balatonszemes", "Bajc")
    assert [r.name for r in res] == ["Bajcsy-Zsilinszky utca"]


def test_geocode_downgrades_level_when_house_missing():
    # Asked for a full address but the response has no house_number → street.
    payload = [
        {"lat": "46.81", "lon": "17.78", "display_name": "Bajcsy…",
         "address": {"road": "Bajcsy-Zsilinszky utca"}},
    ]
    res = _provider(payload).geocode_address("Balatonszemes", "Bajcsy-Zsilinszky utca", "12")
    assert res is not None
    assert res.matched_level == "street"


def test_geocode_full_address_level():
    payload = [
        {"lat": "46.81", "lon": "17.78",
         "address": {"road": "Bajcsy-Zsilinszky utca", "house_number": "12"}},
    ]
    res = _provider(payload).geocode_address("Balatonszemes", "Bajcsy-Zsilinszky utca", "12")
    assert res.matched_level == "address"
    assert res.accuracy_radius_meters == 30.0


def test_geocode_empty_result():
    assert _provider([]).geocode_address("Nowhere") is None
