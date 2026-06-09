"""Service layer for reusable Places / Locations."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from sqlalchemy import case, distinct, func, or_
from sqlalchemy.orm import Session

from app.db.models import Face, Image, Person, Place, PlaceAlias
from app.utils.person_search import normalize

log = logging.getLogger(__name__)

ANONYMOUS_GPS_PLACE_NAME = "Névtelen hely GPS alapján"
DEFAULT_NEARBY_RADIUS_METERS = 100.0

# Place granularity. "exact" = a few metres (house, church, beach),
# "area" = town/settlement sized, "region" = large geographic unit.
PLACE_TYPE_EXACT = "exact"
PLACE_TYPE_AREA = "area"
PLACE_TYPE_REGION = "region"
PLACE_TYPES = (PLACE_TYPE_EXACT, PLACE_TYPE_AREA, PLACE_TYPE_REGION)
DEFAULT_PLACE_TYPE = PLACE_TYPE_AREA

# Default radius (metres) within which a GPS coordinate is considered part of
# a place of the given type. Used when a place has no explicit radius.
DEFAULT_RADIUS_BY_TYPE = {
    PLACE_TYPE_EXACT: 50.0,
    PLACE_TYPE_AREA: 5_000.0,
    PLACE_TYPE_REGION: 50_000.0,
}

# Order in which EXIF GPS coordinates are linked: try the most precise first.
EXIF_LINK_TYPE_ORDER = (PLACE_TYPE_EXACT, PLACE_TYPE_AREA, PLACE_TYPE_REGION)

# Where a place's coordinate came from.
COORD_SOURCE_GEOCODED_SETTLEMENT = "geocoded_settlement"
COORD_SOURCE_GEOCODED_STREET = "geocoded_street"
COORD_SOURCE_GEOCODED_ADDRESS = "geocoded_address"
COORD_SOURCE_MANUAL_MAP_PIN = "manual_map_pin"
COORD_SOURCE_EXIF = "exif"
COORD_SOURCE_IMPORTED = "imported"
COORDINATE_SOURCES = (
    COORD_SOURCE_GEOCODED_SETTLEMENT,
    COORD_SOURCE_GEOCODED_STREET,
    COORD_SOURCE_GEOCODED_ADDRESS,
    COORD_SOURCE_MANUAL_MAP_PIN,
    COORD_SOURCE_EXIF,
    COORD_SOURCE_IMPORTED,
)


def normalize_place_type(value: Optional[str]) -> str:
    """Return a valid place_type, falling back to the default for bad input."""
    clean = (value or "").strip().lower()
    return clean if clean in PLACE_TYPES else DEFAULT_PLACE_TYPE


def normalize_coordinate_source(value: Optional[str]) -> Optional[str]:
    """Return a valid coordinate_source, or None for unknown/empty input."""
    clean = (value or "").strip().lower()
    return clean if clean in COORDINATE_SOURCES else None


def build_display_name(
    settlement: Optional[str],
    street: Optional[str] = None,
    house_number: Optional[str] = None,
) -> str:
    """Compose the human label from structured address parts.

    - "Balatonszemes"
    - "Balatonszemes, Bajcsy-Zsilinszky utca"
    - "Balatonszemes, Bajcsy-Zsilinszky utca 12."
    Returns "" when there is no settlement (caller keeps the legacy name).
    """
    settlement = (settlement or "").strip()
    if not settlement:
        return ""
    street = (street or "").strip()
    house_number = (house_number or "").strip()
    if not street:
        return settlement
    if not house_number:
        return f"{settlement}, {street}"
    return f"{settlement}, {street} {house_number}."


def build_place_label(
    name: Optional[str],
    settlement: Optional[str] = None,
) -> str:
    """Compose the list label, keeping the original place name intact.

    The original ``name`` is always preserved; the settlement is only used as a
    prefix so the name keeps its context without being overwritten:

    - settlement "Balatonszemes" + name "Fagylaltkert" -> "Balatonszemes, Fagylaltkert"
    - settlement "Balatonszemes" + name "Balatonszemes, Kikötő" -> "Balatonszemes, Kikötő"
      (no duplication when the name already starts with the settlement)
    - no settlement -> just the name
    Empty parts never produce a stray comma or a "None" fragment.
    """
    name = (name or "").strip()
    settlement = (settlement or "").strip()
    if not settlement:
        return name
    if not name:
        return settlement
    if name.casefold().startswith(settlement.casefold()):
        return name
    return f"{settlement}, {name}"


@dataclass(frozen=True)
class AddressClassification:
    """Auto-derived place_type / coordinate_source / is_exact for an address."""

    place_type: str
    coordinate_source: str
    is_exact_coordinate: bool


def classify_address(
    settlement: Optional[str],
    street: Optional[str] = None,
    house_number: Optional[str] = None,
) -> AddressClassification:
    """Apply the settlement/street/house → type+source+exact rules.

    - settlement only           → area  / geocoded_settlement / not exact
    - settlement + street       → area  / geocoded_street      / not exact
    - settlement + street + no. → exact / geocoded_address     / exact
    A later manual map pin overrides the source (handled by the caller).
    """
    has_street = bool((street or "").strip())
    has_house = bool((house_number or "").strip())
    if has_street and has_house:
        return AddressClassification(PLACE_TYPE_EXACT, COORD_SOURCE_GEOCODED_ADDRESS, True)
    if has_street:
        return AddressClassification(PLACE_TYPE_AREA, COORD_SOURCE_GEOCODED_STREET, False)
    return AddressClassification(PLACE_TYPE_AREA, COORD_SOURCE_GEOCODED_SETTLEMENT, False)


def default_radius_for(place_type: str) -> float:
    """Return the default accuracy radius (metres) for a place type."""
    return DEFAULT_RADIUS_BY_TYPE.get(normalize_place_type(place_type), DEFAULT_RADIUS_BY_TYPE[DEFAULT_PLACE_TYPE])


@dataclass(frozen=True)
class PlaceFilters:
    name: str = ""
    person_id: Optional[int] = None
    date_from: str = ""
    date_to: str = ""
    min_images: Optional[int] = None
    max_images: Optional[int] = None
    has_coordinates: Optional[bool] = None
    anonymous_exif_only: bool = False
    place_types: Optional[tuple] = None


@dataclass(frozen=True)
class PlaceSummary:
    place_id: int
    name: str
    latitude: Optional[float]
    longitude: Optional[float]
    thumbnail_path: Optional[str]
    is_anonymous: bool
    source: Optional[str]
    image_count: int
    person_count: int
    place_type: str = DEFAULT_PLACE_TYPE
    accuracy_radius_meters: Optional[float] = None
    parent_id: Optional[int] = None
    display_name: Optional[str] = None
    settlement_name: Optional[str] = None
    street_name: Optional[str] = None
    house_number: Optional[str] = None
    coordinate_source: Optional[str] = None
    is_exact_coordinate: bool = False


class PlaceService:
    """Operations for creating, linking, filtering, and merging places.

    The service does not commit; callers control the session transaction.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Creation / assignment
    # ------------------------------------------------------------------

    def create_place(
        self,
        name: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        thumbnail_path: Optional[str] = None,
        *,
        is_anonymous: bool = False,
        source: Optional[str] = None,
        place_type: str = DEFAULT_PLACE_TYPE,
        accuracy_radius_meters: Optional[float] = None,
        parent_id: Optional[int] = None,
    ) -> Place:
        clean_name = (name or "").strip()
        if not clean_name:
            clean_name = ANONYMOUS_GPS_PLACE_NAME
            is_anonymous = True
        ptype = normalize_place_type(place_type)
        radius = accuracy_radius_meters
        if radius is None:
            radius = default_radius_for(ptype)
        place = Place(
            name=clean_name,
            latitude=latitude,
            longitude=longitude,
            thumbnail_path=thumbnail_path,
            is_anonymous=is_anonymous,
            source=source,
            place_type=ptype,
            accuracy_radius_meters=radius,
            parent_id=parent_id,
            display_name=clean_name,
        )
        self._session.add(place)
        self._session.flush()
        return place

    def find_by_name(self, name: str) -> Optional[Place]:
        clean = name.strip()
        if not clean:
            return None
        # Accent-insensitive match ("Pecs" → "Pécs") still needs a Python-side
        # normalize comparison, but scan only the id/name columns instead of
        # hydrating a full Place ORM object per row — with many (e.g. anonymous
        # GPS) places the old full-object load made every add noticeably slow.
        wanted = normalize(clean)
        for place_id, place_name in (
            self._session.query(Place.id, Place.name).order_by(Place.id)
        ):
            if normalize(place_name) == wanted:
                return self._session.get(Place, place_id)
        return None

    def get_or_create_by_name(self, name: str) -> Place:
        existing = self.find_by_name(name)
        if existing is not None:
            return existing
        return self.create_place(name=name, source="manual")

    def assign_place_to_image_by_name(self, image_id: int, name: str) -> Place:
        image = self._require_image(image_id)
        place = self.get_or_create_by_name(name)
        image.place_id = place.id
        self._ensure_thumbnail(place)
        return place

    def assign_place_to_image(self, image_id: int, place_id: Optional[int]) -> None:
        image = self._require_image(image_id)
        if place_id is None:
            image.place_id = None
            return
        place = self._require_place(place_id)
        image.place_id = place.id
        self._ensure_thumbnail(place)

    def name_place(self, place_id: int, name: str) -> Place:
        place = self._require_place(place_id)
        clean = name.strip()
        if not clean:
            raise ValueError("Place name cannot be empty")
        place.name = clean
        place.is_anonymous = False
        if place.source == "exif":
            place.source = "manual"
        return place

    def set_place_type(
        self,
        place_id: int,
        place_type: str,
        *,
        accuracy_radius_meters: Optional[float] = None,
        reset_radius_to_default: bool = False,
    ) -> Place:
        """Change a place's granularity (and optionally its accuracy radius).

        ``reset_radius_to_default`` snaps the radius to the new type's default;
        otherwise an explicit ``accuracy_radius_meters`` wins, and a falsy/None
        existing radius is filled with the new type's default.
        """
        place = self._require_place(place_id)
        place.place_type = normalize_place_type(place_type)
        if accuracy_radius_meters is not None:
            place.accuracy_radius_meters = accuracy_radius_meters
        elif reset_radius_to_default or not place.accuracy_radius_meters:
            place.accuracy_radius_meters = default_radius_for(place.place_type)
        return place

    def set_coordinates(
        self,
        place_id: int,
        latitude: Optional[float],
        longitude: Optional[float],
    ) -> Place:
        """Set (or clear) a place's coordinates.

        Pass ``None`` for both to clear the coordinates. Latitude/longitude must
        be provided together and fall within valid geographic ranges.
        """
        place = self._require_place(place_id)
        if latitude is None and longitude is None:
            place.latitude = None
            place.longitude = None
            return place
        if latitude is None or longitude is None:
            raise ValueError("Latitude and longitude must be provided together")
        if not -90.0 <= latitude <= 90.0:
            raise ValueError(f"Latitude out of range: {latitude}")
        if not -180.0 <= longitude <= 180.0:
            raise ValueError(f"Longitude out of range: {longitude}")
        place.latitude = latitude
        place.longitude = longitude
        return place

    def set_parent(self, place_id: int, parent_id: Optional[int]) -> Place:
        """Attach a place to a parent, guarding against cycles."""
        place = self._require_place(place_id)
        if parent_id is None:
            place.parent_id = None
            return place
        if parent_id == place_id:
            raise ValueError("A place cannot be its own parent")
        # Walk up the prospective parent chain to ensure place_id is not an
        # ancestor of parent_id (which would create a cycle).
        cursor: Optional[int] = parent_id
        seen: set = set()
        while cursor is not None:
            if cursor == place_id:
                raise ValueError("Cannot set parent: would create a cycle")
            if cursor in seen:
                break
            seen.add(cursor)
            parent = self._session.get(Place, cursor)
            if parent is None:
                raise ValueError(f"Parent place not found: {cursor}")
            cursor = parent.parent_id
        place.parent_id = parent_id
        return place

    def list_children(self, place_id: int) -> List[Place]:
        return (
            self._session.query(Place)
            .filter(Place.parent_id == place_id)
            .order_by(Place.name)
            .all()
        )

    def list_descendants(self, place_id: int) -> List[Place]:
        """Return all transitive descendants of a place (breadth-first)."""
        result: List[Place] = []
        frontier = [place_id]
        seen: set = {place_id}
        while frontier:
            children = (
                self._session.query(Place)
                .filter(Place.parent_id.in_(frontier))
                .order_by(Place.name)
                .all()
            )
            frontier = []
            for child in children:
                if child.id in seen:
                    continue
                seen.add(child.id)
                result.append(child)
                frontier.append(child.id)
        return result

    # ------------------------------------------------------------------
    # Structured address
    # ------------------------------------------------------------------

    def create_place_from_address(
        self,
        settlement: str,
        street: Optional[str] = None,
        house_number: Optional[str] = None,
        *,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        place_type: Optional[str] = None,
        coordinate_source: Optional[str] = None,
        is_exact_coordinate: Optional[bool] = None,
        accuracy_radius_meters: Optional[float] = None,
        parent_id: Optional[int] = None,
        name: Optional[str] = None,
    ) -> Place:
        """Create a place from structured address parts.

        ``settlement`` is required. When ``place_type`` / ``coordinate_source``
        / ``is_exact_coordinate`` are omitted they are derived from the address
        via :func:`classify_address`; an explicit value always wins (e.g. a
        manual map pin sets ``coordinate_source='manual_map_pin'``).
        """
        settlement = (settlement or "").strip()
        if not settlement:
            raise ValueError("settlement is required")
        street = (street or "").strip() or None
        house_number = (house_number or "").strip() or None
        auto = classify_address(settlement, street, house_number)
        ptype = normalize_place_type(place_type or auto.place_type)
        display = build_display_name(settlement, street, house_number)
        place = self.create_place(
            name=(name or "").strip() or display,
            latitude=latitude,
            longitude=longitude,
            source="manual",
            place_type=ptype,
            accuracy_radius_meters=accuracy_radius_meters,
            parent_id=parent_id,
        )
        place.settlement_name = settlement
        place.street_name = street
        place.house_number = house_number
        place.display_name = display
        place.coordinate_source = (
            normalize_coordinate_source(coordinate_source) or auto.coordinate_source
        )
        place.is_exact_coordinate = (
            auto.is_exact_coordinate if is_exact_coordinate is None else bool(is_exact_coordinate)
        )
        self._session.flush()
        return place

    def update_place_address(
        self,
        place_id: int,
        settlement: str,
        street: Optional[str] = None,
        house_number: Optional[str] = None,
        *,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        place_type: Optional[str] = None,
        coordinate_source: Optional[str] = None,
        is_exact_coordinate: Optional[bool] = None,
        accuracy_radius_meters: Optional[float] = None,
        name: Optional[str] = None,
    ) -> Place:
        """Update the structured address of an existing place.

        ``display_name`` is always re-derived from the address parts. The
        human-facing ``name`` comes from the explicit ``name`` argument (e.g.
        what the user typed in the "Name" field); only when no name is given
        does it fall back to the derived display name. This keeps a manually
        entered name (e.g. "Koppány villa") from being overwritten by the
        settlement ("Balatonszemes")."""
        place = self._require_place(place_id)
        settlement = (settlement or "").strip()
        if not settlement:
            raise ValueError("settlement is required")
        street = (street or "").strip() or None
        house_number = (house_number or "").strip() or None
        auto = classify_address(settlement, street, house_number)
        place.settlement_name = settlement
        place.street_name = street
        place.house_number = house_number
        display = build_display_name(settlement, street, house_number)
        place.display_name = display
        place.name = (name or "").strip() or display
        place.is_anonymous = False
        place.place_type = normalize_place_type(place_type or auto.place_type)
        if accuracy_radius_meters is not None:
            place.accuracy_radius_meters = accuracy_radius_meters
        elif not place.accuracy_radius_meters:
            place.accuracy_radius_meters = default_radius_for(place.place_type)
        if latitude is not None:
            place.latitude = latitude
        if longitude is not None:
            place.longitude = longitude
        place.coordinate_source = (
            normalize_coordinate_source(coordinate_source) or auto.coordinate_source
        )
        place.is_exact_coordinate = (
            auto.is_exact_coordinate if is_exact_coordinate is None else bool(is_exact_coordinate)
        )
        self._session.flush()
        return place

    def set_manual_coordinates(
        self,
        place_id: int,
        latitude: float,
        longitude: float,
        *,
        is_exact_coordinate: bool = True,
    ) -> Place:
        """Override a place's coordinate from a hand-placed map pin."""
        place = self._require_place(place_id)
        place.latitude = latitude
        place.longitude = longitude
        place.coordinate_source = COORD_SOURCE_MANUAL_MAP_PIN
        place.is_exact_coordinate = bool(is_exact_coordinate)
        return place

    def find_places_by_settlement(self, settlement: str) -> List[Place]:
        settlement = (settlement or "").strip()
        if not settlement:
            return []
        return (
            self._session.query(Place)
            .filter(Place.settlement_name.ilike(settlement))
            .order_by(Place.display_name, Place.name)
            .all()
        )

    def find_places_near_coordinate(
        self, latitude: float, longitude: float, *, radius_meters: float = DEFAULT_NEARBY_RADIUS_METERS
    ) -> List[Place]:
        """Convenience alias for :meth:`find_nearby` with a flat radius."""
        return self.find_nearby(latitude, longitude, radius_meters=radius_meters)

    # ------------------------------------------------------------------
    # EXIF GPS
    # ------------------------------------------------------------------

    def link_exif_gps_to_image(
        self,
        image_id: int,
        latitude: float,
        longitude: float,
        *,
        nearby_radius_meters: float = DEFAULT_NEARBY_RADIUS_METERS,
    ) -> Place:
        image = self._require_image(image_id)
        image.exif_latitude = latitude
        image.exif_longitude = longitude

        # Try to attach to an existing place, most precise type first: an exact
        # spot, then a town-sized area, then a region. Each type uses its own
        # accuracy radius so a region "captures" points much farther out than an
        # exact place would.
        for place_type in EXIF_LINK_TYPE_ORDER:
            nearby = self.find_nearby(
                latitude,
                longitude,
                radius_meters=nearby_radius_meters,
                place_types=(place_type,),
                use_place_radius=True,
            )
            if nearby:
                image.place_id = nearby[0].id
                self._ensure_thumbnail(nearby[0])
                return nearby[0]

        # No existing place matched — record the precise GPS spot as an anonymous
        # "exact" place the user can later name, merge, or re-type.
        place = self.create_place(
            name=ANONYMOUS_GPS_PLACE_NAME,
            latitude=latitude,
            longitude=longitude,
            is_anonymous=True,
            source="exif",
            place_type=PLACE_TYPE_EXACT,
        )
        place.coordinate_source = COORD_SOURCE_EXIF
        place.is_exact_coordinate = True
        image.place_id = place.id
        self._ensure_thumbnail(place)
        return place

    def find_nearby(
        self,
        latitude: float,
        longitude: float,
        *,
        radius_meters: float = DEFAULT_NEARBY_RADIUS_METERS,
        place_types: Optional[Iterable[str]] = None,
        use_place_radius: bool = False,
    ) -> List[Place]:
        """Return coordinate-bearing places near a point, nearest first.

        Args:
            radius_meters: Upper bound on the match distance.
            place_types:   When given, only consider places of these types.
            use_place_radius: When True, each place matches within its own
                ``accuracy_radius_meters`` (or the type default) instead of the
                flat ``radius_meters`` — so exact places match close up while
                regions match far out. When False the flat ``radius_meters``
                applies to every candidate.
        """
        wanted_types = (
            {normalize_place_type(t) for t in place_types}
            if place_types is not None
            else None
        )
        query = self._session.query(Place).filter(
            Place.latitude.isnot(None), Place.longitude.isnot(None)
        )
        if wanted_types is not None:
            query = query.filter(Place.place_type.in_(wanted_types))
        places = query.all()

        ranked = []
        for p in places:
            if p.latitude is None or p.longitude is None:
                continue
            distance = self.distance_meters(latitude, longitude, p.latitude, p.longitude)
            if use_place_radius:
                limit = p.accuracy_radius_meters or default_radius_for(p.place_type)
            else:
                limit = radius_meters
            if distance <= limit:
                ranked.append((distance, p))
        # Nearest first; for equal distance prefer the more precise type.
        type_rank = {PLACE_TYPE_EXACT: 0, PLACE_TYPE_AREA: 1, PLACE_TYPE_REGION: 2}
        ranked.sort(key=lambda item: (item[0], type_rank.get(item[1].place_type, 1)))
        return [p for _, p in ranked]

    @staticmethod
    def distance_meters(
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        radius = 6_371_000.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = (
            math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        )
        return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def list_places(self, filters: Optional[PlaceFilters] = None) -> List[PlaceSummary]:
        filters = filters or PlaceFilters()
        image_count = func.count(distinct(Image.id)).label("image_count")
        person_count = func.count(distinct(Face.person_id)).label("person_count")

        q = (
            self._session.query(
                Place,
                image_count,
                person_count,
            )
            .outerjoin(Image, Image.place_id == Place.id)
            .outerjoin(Face, Face.image_id == Image.id)
        )

        if filters.name.strip():
            term = f"%{filters.name.strip()}%"
            q = q.filter(
                or_(
                    Place.name.ilike(term),
                    Place.display_name.ilike(term),
                    Place.settlement_name.ilike(term),
                    Place.street_name.ilike(term),
                    Place.house_number.ilike(term),
                    Place.aliases.any(PlaceAlias.name.ilike(term)),
                )
            )
        if filters.person_id is not None:
            q = q.filter(Face.person_id == filters.person_id)
        if filters.date_from.strip():
            q = q.filter(Image.photo_date >= filters.date_from.strip())
        if filters.date_to.strip():
            q = q.filter(Image.photo_date <= filters.date_to.strip())
        if filters.has_coordinates is True:
            q = q.filter(Place.latitude.isnot(None), Place.longitude.isnot(None))
        elif filters.has_coordinates is False:
            q = q.filter(or_(Place.latitude.is_(None), Place.longitude.is_(None)))
        if filters.anonymous_exif_only:
            q = q.filter(Place.is_anonymous == True, Place.source == "exif")  # noqa: E712
        if filters.place_types:
            wanted = [normalize_place_type(t) for t in filters.place_types]
            q = q.filter(Place.place_type.in_(wanted))

        q = q.group_by(Place.id)

        if filters.min_images is not None:
            q = q.having(image_count >= filters.min_images)
        if filters.max_images is not None:
            q = q.having(image_count <= filters.max_images)

        rows = q.order_by(Place.is_anonymous.desc(), Place.name).all()
        return [
            PlaceSummary(
                place_id=place.id,
                name=place.name,
                latitude=place.latitude,
                longitude=place.longitude,
                thumbnail_path=place.thumbnail_path,
                is_anonymous=place.is_anonymous,
                source=place.source,
                image_count=int(img_count or 0),
                person_count=int(p_count or 0),
                place_type=place.place_type or DEFAULT_PLACE_TYPE,
                accuracy_radius_meters=place.accuracy_radius_meters,
                parent_id=place.parent_id,
                display_name=place.display_name,
                settlement_name=place.settlement_name,
                street_name=place.street_name,
                house_number=place.house_number,
                coordinate_source=place.coordinate_source,
                is_exact_coordinate=bool(place.is_exact_coordinate),
            )
            for place, img_count, p_count in rows
        ]

    def list_images_for_place(self, place_id: int) -> List[Image]:
        return (
            self._session.query(Image)
            .filter(Image.place_id == place_id)
            .order_by(Image.photo_date, Image.file_path)
            .all()
        )

    def list_persons_for_place(self, place_id: int) -> List[Person]:
        return (
            self._session.query(Person)
            .join(Face, Face.person_id == Person.id)
            .join(Image, Image.id == Face.image_id)
            .filter(Image.place_id == place_id)
            .filter(Face.is_excluded == False)  # noqa: E712
            .group_by(Person.id)
            .order_by(Person.name)
            .all()
        )

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def merge_places(
        self,
        source_ids: Iterable[int],
        target_id: int,
        *,
        name: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        thumbnail_path: Optional[str] = None,
    ) -> Place:
        target = self._require_place(target_id)
        ids = [sid for sid in dict.fromkeys(source_ids) if sid != target_id]
        if not ids:
            return target

        sources = (
            self._session.query(Place)
            .filter(Place.id.in_(ids))
            .order_by(Place.id)
            .all()
        )
        if len(sources) != len(ids):
            raise ValueError("One or more source places do not exist")

        for source in sources:
            self._session.add(
                PlaceAlias(
                    place_id=target.id,
                    source_place_id=source.id,
                    name=source.name,
                    latitude=source.latitude,
                    longitude=source.longitude,
                    thumbnail_path=source.thumbnail_path,
                )
            )
            for alias in source.aliases:
                self._session.add(
                    PlaceAlias(
                        place_id=target.id,
                        source_place_id=alias.source_place_id,
                        name=alias.name,
                        latitude=alias.latitude,
                        longitude=alias.longitude,
                        thumbnail_path=alias.thumbnail_path,
                    )
                )

        self._session.query(Image).filter(Image.place_id.in_(ids)).update(
            {Image.place_id: target.id},
            synchronize_session="fetch",
        )

        if name is not None and name.strip():
            target.name = name.strip()
        target.is_anonymous = False
        if latitude is not None:
            target.latitude = latitude
        elif target.latitude is None:
            target.latitude = next((p.latitude for p in sources if p.latitude is not None), None)
        if longitude is not None:
            target.longitude = longitude
        elif target.longitude is None:
            target.longitude = next((p.longitude for p in sources if p.longitude is not None), None)
        if thumbnail_path is not None:
            # Explicit caller-supplied path (e.g. merge dialog): just store it.
            # Do NOT mark as thumbnail_is_manual — that flag is reserved for
            # selections made through the UI thumbnail picker.
            target.thumbnail_path = thumbnail_path or None
        elif target.thumbnail_is_manual and target.thumbnail_path:
            # Target already has a user-chosen thumbnail — leave it alone.
            pass
        elif target.thumbnail_path is None:
            # Inherit the best available thumbnail from sources.
            # Preserve the manual flag when the winning source had one.
            winning_source = next(
                (p for p in sources if p.thumbnail_path), None
            )
            if winning_source is not None:
                target.thumbnail_path = winning_source.thumbnail_path
                target.thumbnail_is_manual = winning_source.thumbnail_is_manual

        self._ensure_thumbnail(target)
        for source in sources:
            self._session.delete(source)
        self._session.flush()
        log.info("Merged places %s into %d", ids, target_id)
        return target

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_image(self, image_id: int) -> Image:
        image = self._session.get(Image, image_id)
        if image is None:
            raise ValueError(f"Image not found: {image_id}")
        return image

    def _require_place(self, place_id: int) -> Place:
        place = self._session.get(Place, place_id)
        if place is None:
            raise ValueError(f"Place not found: {place_id}")
        return place

    def set_place_thumbnail(self, place_id: int, image_path: str) -> Place:
        """Set an image as the manual thumbnail for a place.

        Args:
            place_id:   Place to update.
            image_path: Absolute path to the image file.

        Returns:
            The updated :class:`Place` row.
        """
        if not Path(image_path).exists():
            raise ValueError(f"Image file not found: {image_path}")
        place = self._require_place(place_id)
        place.thumbnail_path = image_path
        place.thumbnail_is_manual = True
        self._session.commit()
        log.info("Set manual thumbnail for place %d → %s", place_id, image_path)
        return place

    def clear_manual_place_thumbnail(self, place_id: int) -> Place:
        """Reset a place's thumbnail to automatic selection.

        Args:
            place_id: Place to reset.

        Returns:
            The updated :class:`Place` row.
        """
        place = self._require_place(place_id)
        place.thumbnail_is_manual = False
        place.thumbnail_path = None
        self._ensure_thumbnail(place)
        self._session.commit()
        log.info("Cleared manual thumbnail for place %d", place_id)
        return place

    def _ensure_thumbnail(self, place: Place) -> None:
        # Honour manual selection unless the file has been deleted
        if place.thumbnail_is_manual:
            if place.thumbnail_path and Path(place.thumbnail_path).exists():
                return
            # Source file gone → fall back to auto
            place.thumbnail_is_manual = False
            place.thumbnail_path = None

        if place.thumbnail_path:
            return
        image = (
            self._session.query(Image)
            .filter(Image.place_id == place.id)
            .order_by(
                case((Image.detection_done == True, 0), else_=1),  # noqa: E712
                Image.photo_date,
                Image.file_path,
            )
            .first()
        )
        if image is not None:
            place.thumbnail_path = image.file_path
