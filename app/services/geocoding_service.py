"""Cache-first geocoding orchestration.

Layers, in order, so the UI keeps working offline and the network is a last
resort:

  1. ``geocoding_cache`` — previously fetched provider responses.
  2. ``place_address_suggestions`` — settlement/street pairs the user has
     actually used (always available offline).
  3. the online provider — only when ``online`` is True (opt-in; default off),
     and any result is written back to the cache.

The service never raises on transport errors: a failed provider call logs a
warning and yields an empty list / ``None`` so callers degrade gracefully.
The service does not commit; the caller owns the transaction.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from app.db.models import GeocodingCache, PlaceAddressSuggestion
from app.services.geocoding.provider import (
    AddressParts,
    GeocodeResult,
    GeocodingProvider,
    SettlementSuggestion,
    StreetSuggestion,
)
from app.utils.person_search import normalize

log = logging.getLogger(__name__)

# Sentinel stored for "settlement only" suggestion rows (NULL would break the
# (settlement, street) unique constraint under SQLite's NULL-distinct rule).
_NO_STREET = ""


class GeocodingService:
    def __init__(
        self,
        session: Session,
        provider: Optional[GeocodingProvider] = None,
        *,
        online: bool = False,
    ) -> None:
        self._session = session
        self._provider = provider
        self._online = online and provider is not None

    @property
    def online(self) -> bool:
        return self._online

    # ------------------------------------------------------------------
    # Suggestions
    # ------------------------------------------------------------------

    def suggest_settlements(
        self, prefix: str, *, limit: int = 8
    ) -> List[SettlementSuggestion]:
        prefix = (prefix or "").strip()
        if not prefix:
            return []
        key = self._key("settlement", prefix)
        cached = self._read_cache("settlement", key)
        if cached is not None:
            return [SettlementSuggestion(**row) for row in cached][:limit]

        local = self._local_settlements(prefix, limit)
        if not self._online:
            return local

        try:
            results = self._provider.suggest_settlements(prefix, limit=limit)
        except Exception as exc:  # noqa: BLE001
            log.warning("Settlement suggest failed (%s); using local only", exc)
            return local
        self._write_cache("settlement", key, [_strip_raw(asdict(r)) for r in results])
        return results or local

    def suggest_streets(
        self, settlement: str, prefix: str, *, limit: int = 8
    ) -> List[StreetSuggestion]:
        settlement = (settlement or "").strip()
        prefix = (prefix or "").strip()
        if not settlement or not prefix:
            return []
        key = self._key("street", f"{settlement}|{prefix}")
        cached = self._read_cache("street", key)
        if cached is not None:
            return [StreetSuggestion(**row) for row in cached][:limit]

        local = self._local_streets(settlement, prefix, limit)
        if not self._online:
            return local

        try:
            results = self._provider.suggest_streets(settlement, prefix, limit=limit)
        except Exception as exc:  # noqa: BLE001
            log.warning("Street suggest failed (%s); using local only", exc)
            return local
        self._write_cache("street", key, [_strip_raw(asdict(r)) for r in results])
        return results or local

    # ------------------------------------------------------------------
    # Geocode / reverse
    # ------------------------------------------------------------------

    def geocode(
        self,
        settlement: str,
        street: Optional[str] = None,
        house_number: Optional[str] = None,
    ) -> Optional[GeocodeResult]:
        settlement = (settlement or "").strip()
        if not settlement:
            return None
        key = self._key("address", f"{settlement}|{street or ''}|{house_number or ''}")
        cached = self._read_cache("address", key)
        if cached is not None:
            return GeocodeResult(**cached[0]) if cached else None
        if not self._online:
            return None
        try:
            result = self._provider.geocode_address(settlement, street, house_number)
        except Exception as exc:  # noqa: BLE001
            log.warning("Geocode failed (%s)", exc)
            return None
        self._write_cache(
            "address", key, [_strip_raw(asdict(result))] if result else [],
            settlement=settlement, street=street, house_number=house_number,
        )
        return result

    def reverse_geocode(
        self, latitude: float, longitude: float
    ) -> Optional[AddressParts]:
        key = self._key("reverse", f"{latitude:.5f}|{longitude:.5f}")
        cached = self._read_cache("reverse", key)
        if cached is not None:
            return AddressParts(**cached[0]) if cached else None
        if not self._online:
            return None
        try:
            result = self._provider.reverse_geocode(latitude, longitude)
        except Exception as exc:  # noqa: BLE001
            log.warning("Reverse geocode failed (%s)", exc)
            return None
        self._write_cache("reverse", key, [_strip_raw(asdict(result))] if result else [])
        return result

    # ------------------------------------------------------------------
    # Local suggestion store
    # ------------------------------------------------------------------

    def record_address_use(
        self, settlement: str, street: Optional[str] = None, *, source: str = "user"
    ) -> None:
        """Upsert a settlement/street pair into the local suggestion store."""
        settlement = (settlement or "").strip()
        if not settlement:
            return
        street_key = (street or "").strip() or _NO_STREET
        existing = (
            self._session.query(PlaceAddressSuggestion)
            .filter(
                PlaceAddressSuggestion.settlement_name == settlement,
                PlaceAddressSuggestion.street_name == street_key,
            )
            .first()
        )
        if existing is not None:
            existing.last_used_at = datetime.utcnow()
            existing.source = source
            return
        self._session.add(
            PlaceAddressSuggestion(
                settlement_name=settlement,
                street_name=street_key,
                source=source,
            )
        )
        self._session.flush()

    def _local_settlements(self, prefix: str, limit: int) -> List[SettlementSuggestion]:
        term = f"{prefix}%"
        rows = (
            self._session.query(PlaceAddressSuggestion.settlement_name)
            .filter(PlaceAddressSuggestion.settlement_name.ilike(term))
            .distinct()
            .order_by(PlaceAddressSuggestion.settlement_name)
            .limit(limit)
            .all()
        )
        return [SettlementSuggestion(name=r[0]) for r in rows]

    def _local_streets(
        self, settlement: str, prefix: str, limit: int
    ) -> List[StreetSuggestion]:
        term = f"{prefix}%"
        rows = (
            self._session.query(PlaceAddressSuggestion.street_name)
            .filter(
                PlaceAddressSuggestion.settlement_name == settlement,
                PlaceAddressSuggestion.street_name.ilike(term),
                PlaceAddressSuggestion.street_name != _NO_STREET,
            )
            .distinct()
            .order_by(PlaceAddressSuggestion.street_name)
            .limit(limit)
            .all()
        )
        return [StreetSuggestion(name=r[0]) for r in rows if r[0]]

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(query_type: str, text: str) -> str:
        return normalize(text)

    def _read_cache(self, query_type: str, key: str) -> Optional[list]:
        row = (
            self._session.query(GeocodingCache)
            .filter(
                GeocodingCache.query_type == query_type,
                GeocodingCache.query_text == key,
            )
            .first()
        )
        if row is None:
            return None
        try:
            return json.loads(row.result_json)
        except (ValueError, TypeError):
            return None

    def _write_cache(
        self,
        query_type: str,
        key: str,
        payload: list,
        *,
        settlement: Optional[str] = None,
        street: Optional[str] = None,
        house_number: Optional[str] = None,
    ) -> None:
        existing = (
            self._session.query(GeocodingCache)
            .filter(
                GeocodingCache.query_type == query_type,
                GeocodingCache.query_text == key,
            )
            .first()
        )
        blob = json.dumps(payload, ensure_ascii=False)
        if existing is not None:
            existing.result_json = blob
            existing.created_at = datetime.utcnow()
            return
        self._session.add(
            GeocodingCache(
                query_type=query_type,
                query_text=key,
                settlement_name=settlement,
                street_name=street,
                house_number=house_number,
                result_json=blob,
            )
        )
        self._session.flush()


def _strip_raw(d: dict) -> dict:
    """Drop the bulky provider ``raw`` payload before caching."""
    d.pop("raw", None)
    return d
