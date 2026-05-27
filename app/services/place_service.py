"""Service layer for reusable Places / Locations."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional

from sqlalchemy import case, distinct, func, or_
from sqlalchemy.orm import Session

from app.db.models import Face, Image, Person, Place, PlaceAlias
from app.utils.person_search import normalize

log = logging.getLogger(__name__)

ANONYMOUS_GPS_PLACE_NAME = "Névtelen hely GPS alapján"
DEFAULT_NEARBY_RADIUS_METERS = 100.0


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
    ) -> Place:
        clean_name = (name or "").strip()
        if not clean_name:
            clean_name = ANONYMOUS_GPS_PLACE_NAME
            is_anonymous = True
        place = Place(
            name=clean_name,
            latitude=latitude,
            longitude=longitude,
            thumbnail_path=thumbnail_path,
            is_anonymous=is_anonymous,
            source=source,
        )
        self._session.add(place)
        self._session.flush()
        return place

    def find_by_name(self, name: str) -> Optional[Place]:
        clean = name.strip()
        if not clean:
            return None
        wanted = normalize(clean)
        for place in self._session.query(Place).order_by(Place.id).all():
            if normalize(place.name) == wanted:
                return place
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

        nearby = self.find_nearby(latitude, longitude, radius_meters=nearby_radius_meters)
        if nearby:
            image.place_id = nearby[0].id
            self._ensure_thumbnail(nearby[0])
            return nearby[0]

        place = self.create_place(
            name=ANONYMOUS_GPS_PLACE_NAME,
            latitude=latitude,
            longitude=longitude,
            is_anonymous=True,
            source="exif",
        )
        image.place_id = place.id
        self._ensure_thumbnail(place)
        return place

    def find_nearby(
        self,
        latitude: float,
        longitude: float,
        *,
        radius_meters: float = DEFAULT_NEARBY_RADIUS_METERS,
    ) -> List[Place]:
        places = (
            self._session.query(Place)
            .filter(Place.latitude.isnot(None), Place.longitude.isnot(None))
            .all()
        )
        ranked = [
            (self.distance_meters(latitude, longitude, p.latitude, p.longitude), p)
            for p in places
            if p.latitude is not None and p.longitude is not None
        ]
        ranked = [(d, p) for d, p in ranked if d <= radius_meters]
        ranked.sort(key=lambda item: item[0])
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
            target.thumbnail_path = thumbnail_path or None
        elif target.thumbnail_path is None:
            target.thumbnail_path = next(
                (p.thumbnail_path for p in sources if p.thumbnail_path),
                None,
            )

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

    def _ensure_thumbnail(self, place: Place) -> None:
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
