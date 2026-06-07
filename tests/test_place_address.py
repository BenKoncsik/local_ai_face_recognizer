"""Tests for structured-address places: display name, classification, search."""

from __future__ import annotations

import pytest
from sqlalchemy import text

from app.db import database as db_module
from app.db.database import init_db, session_scope
from app.services.place_service import (
    COORD_SOURCE_GEOCODED_ADDRESS,
    COORD_SOURCE_GEOCODED_SETTLEMENT,
    COORD_SOURCE_GEOCODED_STREET,
    COORD_SOURCE_MANUAL_MAP_PIN,
    PLACE_TYPE_AREA,
    PLACE_TYPE_EXACT,
    PlaceFilters,
    PlaceService,
    build_display_name,
    classify_address,
)


@pytest.fixture()
def db(tmp_path):
    init_db(tmp_path / "addr.db")
    return tmp_path


# --- display name -----------------------------------------------------------

def test_build_display_name_levels():
    assert build_display_name("Balatonszemes") == "Balatonszemes"
    assert build_display_name("Balatonszemes", "Bajcsy-Zsilinszky utca") == (
        "Balatonszemes, Bajcsy-Zsilinszky utca"
    )
    assert build_display_name("Balatonszemes", "Bajcsy-Zsilinszky utca", "12") == (
        "Balatonszemes, Bajcsy-Zsilinszky utca 12."
    )
    assert build_display_name("") == ""


# --- classification state machine ------------------------------------------

def test_classify_settlement_only():
    c = classify_address("Balatonszemes")
    assert c.place_type == PLACE_TYPE_AREA
    assert c.coordinate_source == COORD_SOURCE_GEOCODED_SETTLEMENT
    assert c.is_exact_coordinate is False


def test_classify_with_street():
    c = classify_address("Balatonszemes", "Fő utca")
    assert c.place_type == PLACE_TYPE_AREA
    assert c.coordinate_source == COORD_SOURCE_GEOCODED_STREET
    assert c.is_exact_coordinate is False


def test_classify_full_address():
    c = classify_address("Balatonszemes", "Fő utca", "12")
    assert c.place_type == PLACE_TYPE_EXACT
    assert c.coordinate_source == COORD_SOURCE_GEOCODED_ADDRESS
    assert c.is_exact_coordinate is True


# --- create / update --------------------------------------------------------

def test_create_place_from_full_address(db):
    with session_scope() as s:
        svc = PlaceService(s)
        p = svc.create_place_from_address(
            "Balatonszemes", "Bajcsy-Zsilinszky utca", "12",
            latitude=46.8, longitude=17.8,
        )
        assert p.display_name == "Balatonszemes, Bajcsy-Zsilinszky utca 12."
        assert p.name == p.display_name  # legacy name kept in sync
        assert p.place_type == PLACE_TYPE_EXACT
        assert p.coordinate_source == COORD_SOURCE_GEOCODED_ADDRESS
        assert p.is_exact_coordinate is True


def test_create_requires_settlement(db):
    with session_scope() as s:
        with pytest.raises(ValueError):
            PlaceService(s).create_place_from_address("")


def test_set_manual_coordinates_overrides_source(db):
    with session_scope() as s:
        svc = PlaceService(s)
        p = svc.create_place_from_address("Balatonlelle", latitude=46.79, longitude=17.69)
        assert p.coordinate_source == COORD_SOURCE_GEOCODED_SETTLEMENT
        svc.set_manual_coordinates(p.id, 46.80, 17.70)
        assert p.coordinate_source == COORD_SOURCE_MANUAL_MAP_PIN
        assert p.latitude == 46.80
        assert p.is_exact_coordinate is True
        # Structured address is preserved.
        assert p.settlement_name == "Balatonlelle"


def test_update_place_address_rebuilds_display(db):
    with session_scope() as s:
        svc = PlaceService(s)
        p = svc.create_place_from_address("Pécs", latitude=46.07, longitude=18.23)
        svc.update_place_address(p.id, "Pécs", "Király utca", "5")
        assert p.display_name == "Pécs, Király utca 5."
        assert p.place_type == PLACE_TYPE_EXACT


def test_custom_name_survives_address_save(db):
    """A hand-entered name must not be overwritten by the settlement.

    Regression: editing a place with Name="Koppány villa" and
    Settlement="Balatonszemes" used to leave name == "Balatonszemes".
    """
    with session_scope() as s:
        svc = PlaceService(s)
        # Create with a custom name distinct from the settlement.
        p = svc.create_place_from_address(
            "Balatonszemes", latitude=46.8, longitude=17.8, name="Koppány villa"
        )
        assert p.name == "Koppány villa"
        assert p.settlement_name == "Balatonszemes"
        assert p.display_name == "Balatonszemes"
        pid = p.id

    # Re-edit the place keeping the same custom name.
    with session_scope() as s:
        svc = PlaceService(s)
        svc.update_place_address(pid, "Balatonszemes", name="Koppány villa")
        p = svc._require_place(pid)
        assert p.name == "Koppány villa"
        assert p.settlement_name == "Balatonszemes"
        assert p.display_name == "Balatonszemes"

    # And it shows up correctly in the list the table is built from.
    with session_scope() as s:
        summaries = PlaceService(s).list_places(PlaceFilters(name="Koppány"))
        assert len(summaries) == 1
        assert summaries[0].name == "Koppány villa"
        assert summaries[0].settlement_name == "Balatonszemes"


def test_update_place_address_without_name_falls_back_to_display(db):
    """Without an explicit name, the legacy sync (name == display) holds."""
    with session_scope() as s:
        svc = PlaceService(s)
        p = svc.create_place_from_address("Pécs", latitude=46.07, longitude=18.23)
        svc.update_place_address(p.id, "Pécs", "Király utca", "5")
        assert p.name == "Pécs, Király utca 5."
        assert p.name == p.display_name


# --- search -----------------------------------------------------------------

def test_search_matches_structured_fields(db):
    with session_scope() as s:
        svc = PlaceService(s)
        svc.create_place_from_address("Balatonszemes", "Bajcsy-Zsilinszky utca", "12",
                                      latitude=46.8, longitude=17.8)
        by_street = svc.list_places(PlaceFilters(name="Bajcsy"))
        assert len(by_street) == 1
        by_settlement = svc.list_places(PlaceFilters(name="Balatonszemes"))
        assert len(by_settlement) == 1
        assert by_settlement[0].settlement_name == "Balatonszemes"


def test_find_places_by_settlement(db):
    with session_scope() as s:
        svc = PlaceService(s)
        svc.create_place_from_address("Pécs", "Király utca", latitude=46.0, longitude=18.0)
        svc.create_place_from_address("Pécs", "Rákóczi út", latitude=46.0, longitude=18.0)
        svc.create_place_from_address("Szeged", latitude=46.2, longitude=20.1)
        pecs = svc.find_places_by_settlement("Pécs")
        assert len(pecs) == 2


# --- migration backfill -----------------------------------------------------

def test_migrate_display_name_backfill(db, tmp_path):
    from sqlalchemy import create_engine

    legacy = tmp_path / "legacy.db"
    eng = create_engine(f"sqlite:///{legacy}")
    with eng.begin() as conn:
        conn.execute(text(
            "CREATE TABLE places (id INTEGER PRIMARY KEY, name TEXT, source VARCHAR(32), "
            "display_name VARCHAR(512), coordinate_source VARCHAR(32))"
        ))
        conn.execute(text("INSERT INTO places (name, source) VALUES ('Old Place', NULL)"))
        conn.execute(text("INSERT INTO places (name, source) VALUES ('Exif Spot', 'exif')"))
    db_module._migrate_place_display_name(eng)
    db_module._migrate_place_display_name(eng)  # idempotent
    with eng.connect() as conn:
        rows = conn.execute(text(
            "SELECT name, display_name, coordinate_source FROM places ORDER BY id"
        )).fetchall()
    assert rows[0] == ("Old Place", "Old Place", "imported")
    assert rows[1] == ("Exif Spot", "Exif Spot", "exif")
