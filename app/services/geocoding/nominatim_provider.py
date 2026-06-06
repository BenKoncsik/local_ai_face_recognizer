"""Nominatim / OpenStreetMap geocoding provider.

Uses the stdlib ``urllib`` (matching app.services.update_service — no extra
dependency). Honours the Nominatim usage policy: a descriptive User-Agent and
a minimum 1 request/second throttle. Network/parse errors propagate to the
caller; GeocodingService turns them into graceful empty results.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
from typing import List, Optional

from app.services.geocoding.provider import (
    AddressParts,
    GeocodeResult,
    SettlementSuggestion,
    StreetSuggestion,
)

log = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://nominatim.openstreetmap.org"
_DEFAULT_USER_AGENT = "Face-Local/1.0 (https://github.com/HuKonTech/local_ai_face_recognizer)"
# Nominatim accuracy radius estimate (metres) per matched level.
_LEVEL_RADIUS = {"address": 30.0, "street": 300.0, "settlement": 5_000.0}


class NominatimProvider:
    """Online geocoder backed by a Nominatim endpoint.

    Args:
        base_url:    Nominatim service root (override for a self-hosted instance).
        user_agent:  Required by the Nominatim policy; identify the app.
        country_codes: Optional comma-separated ISO codes to bias results (e.g. "hu").
        min_interval_s: Minimum spacing between requests (rate-limit guard).
        timeout_s:   Per-request timeout.
    """

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        user_agent: str = _DEFAULT_USER_AGENT,
        country_codes: Optional[str] = None,
        min_interval_s: float = 1.0,
        timeout_s: float = 8.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._user_agent = user_agent
        self._country_codes = country_codes
        self._min_interval_s = min_interval_s
        self._timeout_s = timeout_s
        self._last_request_at = 0.0

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self._min_interval_s:
            time.sleep(self._min_interval_s - elapsed)

    def _get(self, path: str, params: dict) -> list | dict:
        params = {k: v for k, v in params.items() if v not in (None, "")}
        params.setdefault("format", "jsonv2")
        if self._country_codes:
            params.setdefault("countrycodes", self._country_codes)
        url = f"{self._base_url}/{path}?{urllib.parse.urlencode(params)}"
        self._throttle()
        req = urllib.request.Request(url, headers={"User-Agent": self._user_agent})
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                payload = resp.read().decode("utf-8")
        finally:
            self._last_request_at = time.monotonic()
        return json.loads(payload)

    # ------------------------------------------------------------------
    # Provider protocol
    # ------------------------------------------------------------------

    def suggest_settlements(
        self, prefix: str, *, limit: int = 8
    ) -> List[SettlementSuggestion]:
        prefix = (prefix or "").strip()
        if not prefix:
            return []
        rows = self._get(
            "search",
            {
                "q": prefix,
                "addressdetails": 1,
                "limit": limit,
                # Towns/villages/cities live in these OSM place classes.
                "featuretype": "settlement",
            },
        )
        out: List[SettlementSuggestion] = []
        seen: set = set()
        for row in rows if isinstance(rows, list) else []:
            addr = row.get("address", {})
            name = (
                addr.get("city")
                or addr.get("town")
                or addr.get("village")
                or addr.get("municipality")
                or row.get("name")
            )
            if not name or name.casefold() in seen:
                continue
            seen.add(name.casefold())
            out.append(
                SettlementSuggestion(
                    name=name,
                    latitude=_to_float(row.get("lat")),
                    longitude=_to_float(row.get("lon")),
                    country=addr.get("country"),
                    raw=row,
                )
            )
        return out

    def suggest_streets(
        self, settlement: str, prefix: str, *, limit: int = 8
    ) -> List[StreetSuggestion]:
        settlement = (settlement or "").strip()
        prefix = (prefix or "").strip()
        if not settlement or not prefix:
            return []
        rows = self._get(
            "search",
            {
                "street": prefix,
                "city": settlement,
                "addressdetails": 1,
                "limit": limit,
            },
        )
        out: List[StreetSuggestion] = []
        seen: set = set()
        for row in rows if isinstance(rows, list) else []:
            addr = row.get("address", {})
            name = addr.get("road") or row.get("name")
            if not name or name.casefold() in seen:
                continue
            seen.add(name.casefold())
            out.append(
                StreetSuggestion(
                    name=name,
                    latitude=_to_float(row.get("lat")),
                    longitude=_to_float(row.get("lon")),
                    raw=row,
                )
            )
        return out

    def geocode_address(
        self,
        settlement: str,
        street: Optional[str] = None,
        house_number: Optional[str] = None,
    ) -> Optional[GeocodeResult]:
        settlement = (settlement or "").strip()
        if not settlement:
            return None
        # Build the structured query and remember the most precise level asked.
        street_q = street or ""
        if street and house_number:
            street_q = f"{house_number} {street}"
            level = "address"
        elif street:
            level = "street"
        else:
            level = "settlement"
        rows = self._get(
            "search",
            {
                "city": settlement,
                "street": street_q or None,
                "addressdetails": 1,
                "limit": 1,
            },
        )
        if not isinstance(rows, list) or not rows:
            return None
        row = rows[0]
        lat = _to_float(row.get("lat"))
        lon = _to_float(row.get("lon"))
        if lat is None or lon is None:
            return None
        # Downgrade the claimed level if the response lacks the requested detail.
        addr = row.get("address", {})
        if level == "address" and not addr.get("house_number"):
            level = "street" if addr.get("road") else "settlement"
        elif level == "street" and not addr.get("road"):
            level = "settlement"
        return GeocodeResult(
            latitude=lat,
            longitude=lon,
            matched_level=level,
            accuracy_radius_meters=_LEVEL_RADIUS.get(level),
            display_name=row.get("display_name"),
            raw=row,
        )

    def reverse_geocode(
        self, latitude: float, longitude: float
    ) -> Optional[AddressParts]:
        row = self._get(
            "reverse",
            {"lat": latitude, "lon": longitude, "addressdetails": 1},
        )
        if not isinstance(row, dict):
            return None
        addr = row.get("address", {})
        settlement = (
            addr.get("city")
            or addr.get("town")
            or addr.get("village")
            or addr.get("municipality")
        )
        return AddressParts(
            settlement_name=settlement,
            street_name=addr.get("road"),
            house_number=addr.get("house_number"),
            raw=row,
        )


def _to_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
