"""Tests for place types (exact / area / region), hierarchy and EXIF linking."""

from __future__ import annotations

import pytest
from sqlalchemy import text

from app.db.database import init_db, session_scope
from app.db import database as db_module
from app.db.models import Image, Place
from app.services.place_service import (
    ANONYMOUS_GPS_PLACE_NAME,
    DEFAULT_RADIUS_BY_TYPE,
    PLACE_TYPE_AREA,
    PLACE_TYPE_EXACT,
    PLACE_TYPE_REGION,
    PlaceFilters,
    PlaceService,
    default_radius_for,
    normalize_place_type,
)


@pytest.fixture()
def db(tmp_path):
    init_db(tmp_path / "places.db")
    return tmp_path


def _add_image(session, path: str) -> Image:
    image = Image(file_path=path, file_hash=f"h_{path}", file_mtime=0.0)
    session.add(image)
    session.flush()
    return image


# ---------------------------------------------------------------------------
# Helpers / constants
# ---------------------------------------------------------------------------

def test_normalize_place_type_falls_back_to_area():
    assert normalize_place_type("EXACT") == PLACE_TYPE_EXACT
    assert normalize_place_type("nonsense") == PLACE_TYPE_AREA
    assert normalize_place_type(None) == PLACE_TYPE_AREA


def test_default_radius_for_each_type():
    assert default_radius_for(PLACE_TYPE_EXACT) == DEFAULT_RADIUS_BY_TYPE[PLACE_TYPE_EXACT]
    assert default_radius_for(PLACE_TYPE_REGION) == DEFAULT_RADIUS_BY_TYPE[PLACE_TYPE_REGION]
    assert default_radius_for("bad") == DEFAULT_RADIUS_BY_TYPE[PLACE_TYPE_AREA]


# ---------------------------------------------------------------------------
# create_place / defaults
# ---------------------------------------------------------------------------

def test_create_place_defaults_to_area(db):
    with session_scope() as session:
        place = PlaceService(session).create_place("Anywhere")
        assert place.place_type == PLACE_TYPE_AREA
        assert place.accuracy_radius_meters == DEFAULT_RADIUS_BY_TYPE[PLACE_TYPE_AREA]


def test_create_place_explicit_type_and_radius(db):
    with session_scope() as session:
        place = PlaceService(session).create_place(
            "Bazilika", 46.08, 18.23, place_type="exact", accuracy_radius_meters=20.0
        )
        assert place.place_type == PLACE_TYPE_EXACT
        assert place.accuracy_radius_meters == 20.0


# ---------------------------------------------------------------------------
# set_coordinates
# ---------------------------------------------------------------------------

def test_set_coordinates_updates_place(db):
    with session_scope() as session:
        svc = PlaceService(session)
        place = svc.create_place("Somewhere")
        svc.set_coordinates(place.id, 47.4979, 19.0402)
        assert place.latitude == 47.4979
        assert place.longitude == 19.0402


def test_set_coordinates_clears_when_both_none(db):
    with session_scope() as session:
        svc = PlaceService(session)
        place = svc.create_place("Somewhere", 47.0, 19.0)
        svc.set_coordinates(place.id, None, None)
        assert place.latitude is None
        assert place.longitude is None


def test_set_coordinates_rejects_partial(db):
    with session_scope() as session:
        svc = PlaceService(session)
        place = svc.create_place("Somewhere")
        with pytest.raises(ValueError):
            svc.set_coordinates(place.id, 47.0, None)


def test_set_coordinates_rejects_out_of_range(db):
    with session_scope() as session:
        svc = PlaceService(session)
        place = svc.create_place("Somewhere")
        with pytest.raises(ValueError):
            svc.set_coordinates(place.id, 200.0, 19.0)
        with pytest.raises(ValueError):
            svc.set_coordinates(place.id, 47.0, 400.0)


# ---------------------------------------------------------------------------
# find_nearby type awareness
# ---------------------------------------------------------------------------

def test_find_nearby_filters_by_type(db):
    with session_scope() as session:
        svc = PlaceService(session)
        svc.create_place("ExactSpot", 47.0, 19.0, place_type="exact")
        svc.create_place("BigArea", 47.0, 19.0, place_type="area")
        exact_only = svc.find_nearby(47.0, 19.0, radius_meters=10.0, place_types=("exact",))
        assert [p.name for p in exact_only] == ["ExactSpot"]


def test_find_nearby_use_place_radius(db):
    with session_scope() as session:
        svc = PlaceService(session)
        # An area place 2 km away: outside flat 100m, inside its 5km radius.
        svc.create_place("Town", 47.0, 19.0, place_type="area")
        # ~2 km north
        far_lat = 47.018
        flat = svc.find_nearby(far_lat, 19.0, radius_meters=100.0)
        assert flat == []
        by_radius = svc.find_nearby(far_lat, 19.0, use_place_radius=True)
        assert [p.name for p in by_radius] == ["Town"]


# ---------------------------------------------------------------------------
# EXIF two-stage linking
# ---------------------------------------------------------------------------

def test_exif_prefers_exact_over_area(db):
    with session_scope() as session:
        svc = PlaceService(session)
        area = svc.create_place("Town", 47.0, 19.0, place_type="area")
        exact = svc.create_place("House", 47.0001, 19.0001, place_type="exact")
        img = _add_image(session, "/a.jpg")
        linked = svc.link_exif_gps_to_image(img.id, 47.00005, 19.00005)
        assert linked.id == exact.id


def test_exif_falls_back_to_area_when_no_exact(db):
    with session_scope() as session:
        svc = PlaceService(session)
        area = svc.create_place("Town", 47.0, 19.0, place_type="area")
        img = _add_image(session, "/b.jpg")
        # ~1 km away: no exact, but within the area radius.
        linked = svc.link_exif_gps_to_image(img.id, 47.009, 19.0)
        assert linked.id == area.id


def test_exif_creates_anonymous_exact_when_nothing_matches(db):
    with session_scope() as session:
        svc = PlaceService(session)
        img = _add_image(session, "/c.jpg")
        linked = svc.link_exif_gps_to_image(img.id, 10.0, 10.0)
        assert linked.is_anonymous is True
        assert linked.place_type == PLACE_TYPE_EXACT
        assert linked.name == ANONYMOUS_GPS_PLACE_NAME


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------

def test_set_parent_and_descendants(db):
    with session_scope() as session:
        svc = PlaceService(session)
        region = svc.create_place("Baranya", place_type="region")
        town = svc.create_place("Pécs", place_type="area")
        spot = svc.create_place("Bazilika", place_type="exact")
        svc.set_parent(town.id, region.id)
        svc.set_parent(spot.id, town.id)
        desc = svc.list_descendants(region.id)
        assert {p.name for p in desc} == {"Pécs", "Bazilika"}
        assert [c.name for c in svc.list_children(region.id)] == ["Pécs"]


def test_set_parent_rejects_cycles(db):
    with session_scope() as session:
        svc = PlaceService(session)
        a = svc.create_place("A", place_type="region")
        b = svc.create_place("B", place_type="area")
        svc.set_parent(b.id, a.id)
        with pytest.raises(ValueError):
            svc.set_parent(a.id, b.id)
        with pytest.raises(ValueError):
            svc.set_parent(a.id, a.id)


# ---------------------------------------------------------------------------
# set_place_type
# ---------------------------------------------------------------------------

def test_set_place_type_resets_radius_to_default(db):
    with session_scope() as session:
        svc = PlaceService(session)
        p = svc.create_place("X", place_type="exact")
        svc.set_place_type(p.id, "region", reset_radius_to_default=True)
        assert p.place_type == PLACE_TYPE_REGION
        assert p.accuracy_radius_meters == DEFAULT_RADIUS_BY_TYPE[PLACE_TYPE_REGION]


def test_set_place_type_explicit_radius_wins(db):
    with session_scope() as session:
        svc = PlaceService(session)
        p = svc.create_place("X", place_type="area")
        svc.set_place_type(p.id, "exact", accuracy_radius_meters=7.0)
        assert p.place_type == PLACE_TYPE_EXACT
        assert p.accuracy_radius_meters == 7.0


# ---------------------------------------------------------------------------
# list_places filtering + summary fields
# ---------------------------------------------------------------------------

def test_list_places_type_filter_and_summary(db):
    with session_scope() as session:
        svc = PlaceService(session)
        region = svc.create_place("R", place_type="region")
        svc.create_place("E", place_type="exact")
        regions = svc.list_places(PlaceFilters(place_types=("region",)))
        assert [s.name for s in regions] == ["R"]
        assert regions[0].place_type == PLACE_TYPE_REGION
        assert regions[0].accuracy_radius_meters == DEFAULT_RADIUS_BY_TYPE[PLACE_TYPE_REGION]


# ---------------------------------------------------------------------------
# Migration backfill
# ---------------------------------------------------------------------------

def test_migrate_place_types_backfills_nulls(db, tmp_path):
    from sqlalchemy import create_engine

    legacy = tmp_path / "legacy.db"
    eng = create_engine(f"sqlite:///{legacy}")
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE places (id INTEGER PRIMARY KEY, name TEXT, place_type VARCHAR(16), accuracy_radius_meters FLOAT)"))
        conn.execute(text("INSERT INTO places (name, place_type) VALUES ('Old', NULL)"))
    db_module._migrate_place_types(eng)
    db_module._migrate_place_types(eng)  # idempotent
    with eng.connect() as conn:
        row = conn.execute(text("SELECT place_type, accuracy_radius_meters FROM places")).fetchone()
    assert row[0] == PLACE_TYPE_AREA
    assert row[1] == DEFAULT_RADIUS_BY_TYPE[PLACE_TYPE_AREA]
