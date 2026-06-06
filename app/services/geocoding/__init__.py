"""Geocoding providers and value types.

The UI and :class:`~app.services.geocoding_service.GeocodingService` depend only
on the :class:`GeocodingProvider` protocol and the dataclasses here — never on a
concrete provider or an external API directly, so providers stay swappable
(online Nominatim today, an offline/cached one later).
"""

from app.services.geocoding.provider import (
    AddressParts,
    GeocodeResult,
    GeocodingProvider,
    SettlementSuggestion,
    StreetSuggestion,
)

__all__ = [
    "AddressParts",
    "GeocodeResult",
    "GeocodingProvider",
    "SettlementSuggestion",
    "StreetSuggestion",
]
