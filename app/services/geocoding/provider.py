"""Provider protocol and value types for geocoding.

A provider turns address text into coordinates and suggestions. It performs no
caching and no persistence — that is the GeocodingService's job. Concrete
providers (e.g. Nominatim) implement this protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol


@dataclass(frozen=True)
class SettlementSuggestion:
    """A settlement (town/village) autocomplete candidate."""

    name: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    country: Optional[str] = None
    # Provider-specific raw payload, kept for caching / debugging.
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class StreetSuggestion:
    """A street autocomplete candidate within a settlement."""

    name: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GeocodeResult:
    """The resolved coordinate for an address query.

    ``matched_level`` records how precise the match was: an exact address, a
    street, or only the settlement centre. ``accuracy_radius_meters`` is the
    provider's confidence radius for that level.
    """

    latitude: float
    longitude: float
    matched_level: str  # "address" | "street" | "settlement"
    accuracy_radius_meters: Optional[float] = None
    display_name: Optional[str] = None
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AddressParts:
    """Structured address from a reverse-geocode lookup."""

    settlement_name: Optional[str] = None
    street_name: Optional[str] = None
    house_number: Optional[str] = None
    raw: dict = field(default_factory=dict)


class GeocodingProvider(Protocol):
    """Pluggable geocoding backend.

    Implementations must be side-effect-free apart from issuing network calls,
    and must raise on transport errors (the GeocodingService catches and
    degrades gracefully — see its docstring).
    """

    def suggest_settlements(
        self, prefix: str, *, limit: int = 8
    ) -> List[SettlementSuggestion]:
        ...

    def suggest_streets(
        self, settlement: str, prefix: str, *, limit: int = 8
    ) -> List[StreetSuggestion]:
        ...

    def geocode_address(
        self,
        settlement: str,
        street: Optional[str] = None,
        house_number: Optional[str] = None,
    ) -> Optional[GeocodeResult]:
        ...

    def reverse_geocode(
        self, latitude: float, longitude: float
    ) -> Optional[AddressParts]:
        ...
