"""Tests for reusable Places / Locations."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import ScanConfig
from app.db.database import init_db, session_scope
from app.db.models import Face, Image, Person, Place, PlaceAlias
from app.services.image_browser_service import ImageBrowserService
from app.services.place_service import (
    ANONYMOUS_GPS_PLACE_NAME,
    PlaceFilters,
    PlaceService,
)
from app.services.scan_service import ScanService


@pytest.fixture()
def db(tmp_path):
    db_path = tmp_path / "places.db"
    init_db(db_path)
    return db_path


def _add_image(session, path: str = "/tmp/photo.jpg", photo_date: str | None = None) -> Image:
    image = Image(
        file_path=path,
        file_hash=f"hash_{path}",
        file_mtime=0.0,
        photo_date=photo_date,
    )
    session.add(image)
    session.flush()
    return image


def _add_person_face(session, image: Image, name: str) -> Person:
    person = Person(name=name, is_auto_named=False)
    session.add(person)
    session.flush()
    face = Face(
        image_id=image.id,
        person_id=person.id,
        bbox_x=1,
        bbox_y=2,
        bbox_w=3,
        bbox_h=4,
        confidence=0.9,
        detector_backend="cpu",
    )
    session.add(face)
    session.flush()
    return person


def test_place_create_and_image_link(db):
    with session_scope() as session:
        image = _add_image(session)
        place = PlaceService(session).create_place("Budapest", 47.4979, 19.0402)
        PlaceService(session).assign_place_to_image(image.id, place.id)
        image_id = image.id
        place_id = place.id

    with session_scope() as session:
        image = session.get(Image, image_id)
        assert image.place_id == place_id
        assert image.place.name == "Budapest"


def test_assign_existing_place_by_name(db):
    with session_scope() as session:
        image_a = _add_image(session, "/tmp/a.jpg")
        image_b = _add_image(session, "/tmp/b.jpg")
        svc = PlaceService(session)
        first = svc.assign_place_to_image_by_name(image_a.id, "Balaton")
        second = svc.assign_place_to_image_by_name(image_b.id, "balaton")

    assert first.id == second.id


def test_scan_processes_exif_gps_without_blocking_import(db, tmp_path, monkeypatch):
    folder = tmp_path / "photos"
    folder.mkdir()
    image_path = folder / "gps.jpg"
    image_path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)
    monkeypatch.setattr(
        "app.services.scan_service.read_exif_gps",
        lambda _path: (47.4979, 19.0402),
    )

    with session_scope() as session:
        ids = ScanService(session, ScanConfig()).scan(str(folder))

    assert len(ids) == 1
    with session_scope() as session:
        image = session.get(Image, ids[0])
        assert image.exif_latitude == pytest.approx(47.4979)
        assert image.exif_longitude == pytest.approx(19.0402)
        assert image.place is not None
        assert image.place.name == ANONYMOUS_GPS_PLACE_NAME
        assert image.place.is_anonymous is True
        assert image.place.source == "exif"


def test_exif_gps_processing_error_does_not_abort_scan(db, tmp_path, monkeypatch):
    folder = tmp_path / "photos"
    folder.mkdir()
    image_path = folder / "broken.jpg"
    image_path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)

    def _boom(_path):
        raise ValueError("bad gps")

    monkeypatch.setattr("app.services.scan_service.read_exif_gps", _boom)

    with session_scope() as session:
        ids = ScanService(session, ScanConfig()).scan(str(folder))

    assert len(ids) == 1


def test_merge_places_moves_images_and_preserves_source_data(db):
    with session_scope() as session:
        svc = PlaceService(session)
        target = svc.create_place("Target", 47.0, 19.0, "/tmp/target.jpg")
        source = svc.create_place("Old Name", 48.0, 20.0, "/tmp/source.jpg")
        image = _add_image(session)
        image.place_id = source.id
        target_id = target.id
        source_id = source.id
        image_id = image.id

    with session_scope() as session:
        PlaceService(session).merge_places(
            [source_id],
            target_id,
            name="Chosen Name",
            latitude=48.0,
            longitude=20.0,
            thumbnail_path="/tmp/source.jpg",
        )

    with session_scope() as session:
        image = session.get(Image, image_id)
        target = session.get(Place, target_id)
        assert image.place_id == target_id
        assert target.name == "Chosen Name"
        assert target.latitude == pytest.approx(48.0)
        assert target.thumbnail_path == "/tmp/source.jpg"
        assert session.get(Place, source_id) is None
        alias = session.query(PlaceAlias).filter(PlaceAlias.place_id == target_id).one()
        assert alias.name == "Old Name"
        assert alias.latitude == pytest.approx(48.0)


def test_list_persons_for_place_uses_face_assignments(db):
    with session_scope() as session:
        image = _add_image(session)
        person = _add_person_face(session, image, "Anna")
        place = PlaceService(session).assign_place_to_image_by_name(image.id, "Szeged")
        place_id = place.id
        person_id = person.id

    with session_scope() as session:
        persons = PlaceService(session).list_persons_for_place(place_id)
        assert [p.id for p in persons] == [person_id]


def test_place_filters_by_date_and_image_count(db):
    with session_scope() as session:
        svc = PlaceService(session)
        old = svc.create_place("Old")
        new = svc.create_place("New")
        _add_image(session, "/tmp/old.jpg", "1954.01.01").place_id = old.id
        _add_image(session, "/tmp/new.jpg", "1970.01.01").place_id = new.id

    with session_scope() as session:
        summaries = PlaceService(session).list_places(
            PlaceFilters(date_from="1960", min_images=1)
        )

    assert [s.name for s in summaries] == ["New"]


def test_image_browser_filters_by_place(db, tmp_path):
    folder = tmp_path / "photos"
    with session_scope() as session:
        svc = PlaceService(session)
        budapest = svc.create_place("Budapest")
        other = svc.create_place("Other")
        _add_image(session, str(folder / "a.jpg")).place_id = budapest.id
        _add_image(session, str(folder / "b.jpg")).place_id = other.id
        place_id = budapest.id

    with session_scope() as session:
        images = ImageBrowserService(session).list_images(str(folder), place_id=place_id)

    assert [img.filename for img in images] == ["a.jpg"]
