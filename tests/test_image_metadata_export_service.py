"""Tests for ImageMetadataExportService (CSV and XLSX export)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from app.db.database import init_db, session_scope
from app.db.models import Face, Image, Person, Place
from app.services.image_metadata_export_service import (
    ALL_FIELDS,
    FIELD_DATE,
    FIELD_FILENAME,
    FIELD_GPS,
    FIELD_LOCATION,
    FIELD_PERSONS,
    FIELD_RELPATH,
    PERSON_MODE_COLS,
    PERSON_MODE_LIST,
    ImageMetadataExportService,
    UNKNOWN,
)


@pytest.fixture()
def db(tmp_path):
    db_path = tmp_path / "test_meta_export.db"
    init_db(db_path)
    return db_path


def _make_image(session, file_path, **kwargs) -> Image:
    img = Image(
        file_path=str(file_path),
        file_hash="abc123",
        file_mtime=0.0,
        detection_done=True,
        **kwargs,
    )
    session.add(img)
    session.flush()
    return img


def _make_person(session, name: str) -> Person:
    p = Person(name=name, is_auto_named=False)
    session.add(p)
    session.flush()
    return p


def _make_face(session, image: Image, person: Person) -> Face:
    f = Face(
        image_id=image.id,
        person_id=person.id,
        bbox_x=0, bbox_y=0, bbox_w=10, bbox_h=10,
        confidence=0.9,
        detector_backend="cpu",
        is_excluded=False,
    )
    session.add(f)
    session.flush()
    return f


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _read_csv_headers(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return next(csv.reader(fh))


# ---------------------------------------------------------------------------
# CSV tests
# ---------------------------------------------------------------------------

def test_export_csv_single_person_list_mode(db, tmp_path):
    with session_scope() as session:
        alice = _make_person(session, "Alice")
        img = _make_image(session, tmp_path / "photo.jpg", photo_date="2020-06-15")
        _make_face(session, img, alice)

    out = tmp_path / "out.csv"
    with session_scope() as session:
        ImageMetadataExportService(session).export_csv(out, ALL_FIELDS, PERSON_MODE_LIST)

    rows = _read_csv(out)
    assert len(rows) == 1
    row = rows[0]
    assert row["Filename"] == "photo.jpg"
    assert row["Persons"] == "Alice"
    assert row["Date"] == "2020-06-15"


def test_export_csv_multi_persons_list_mode(db, tmp_path):
    with session_scope() as session:
        panni = _make_person(session, "Panni")
        kati = _make_person(session, "Kati")
        bela = _make_person(session, "Béla")
        img = _make_image(session, tmp_path / "group.jpg")
        _make_face(session, img, panni)
        _make_face(session, img, kati)
        _make_face(session, img, bela)

    out = tmp_path / "out.csv"
    with session_scope() as session:
        ImageMetadataExportService(session).export_csv(out, ALL_FIELDS, PERSON_MODE_LIST)

    rows = _read_csv(out)
    assert len(rows) == 1
    persons_str = rows[0]["Persons"]
    for name in ["Panni", "Kati", "Béla"]:
        assert name in persons_str


def test_export_csv_multi_persons_cols_mode(db, tmp_path):
    with session_scope() as session:
        panni = _make_person(session, "Panni")
        kati = _make_person(session, "Kati")
        img = _make_image(session, tmp_path / "group.jpg")
        _make_face(session, img, panni)
        _make_face(session, img, kati)

    out = tmp_path / "out.csv"
    with session_scope() as session:
        ImageMetadataExportService(session).export_csv(out, ALL_FIELDS, PERSON_MODE_COLS)

    rows = _read_csv(out)
    assert len(rows) == 1
    row = rows[0]
    assert "Person1" in row
    assert "Person2" in row
    assert "Persons" not in row
    person_values = {row.get("Person1"), row.get("Person2")}
    assert "Panni" in person_values
    assert "Kati" in person_values


def test_export_csv_cols_mode_auto_expands_columns(db, tmp_path):
    with session_scope() as session:
        persons = [_make_person(session, f"Person {i}") for i in range(4)]
        img = _make_image(session, tmp_path / "big.jpg")
        for p in persons:
            _make_face(session, img, p)

    out = tmp_path / "out.csv"
    with session_scope() as session:
        ImageMetadataExportService(session).export_csv(out, ALL_FIELDS, PERSON_MODE_COLS)

    rows = _read_csv(out)
    assert "Person4" in rows[0]


def test_export_csv_empty_persons(db, tmp_path):
    with session_scope() as session:
        _make_image(session, tmp_path / "noperson.jpg")

    out = tmp_path / "out.csv"
    with session_scope() as session:
        ImageMetadataExportService(session).export_csv(out, ALL_FIELDS, PERSON_MODE_LIST)

    rows = _read_csv(out)
    assert len(rows) == 1
    assert rows[0]["Persons"] == ""


def test_export_csv_no_gps_exports_empty(db, tmp_path):
    with session_scope() as session:
        _make_image(session, tmp_path / "nogps.jpg")

    out = tmp_path / "out.csv"
    with session_scope() as session:
        ImageMetadataExportService(session).export_csv(out, ALL_FIELDS, PERSON_MODE_LIST)

    rows = _read_csv(out)
    assert rows[0]["Lat"] == ""
    assert rows[0]["Lon"] == ""


def test_export_csv_gps_from_exif(db, tmp_path):
    with session_scope() as session:
        _make_image(
            session, tmp_path / "gps.jpg",
            exif_latitude=46.253010,
            exif_longitude=20.141425,
        )

    out = tmp_path / "out.csv"
    with session_scope() as session:
        ImageMetadataExportService(session).export_csv(out, ALL_FIELDS, PERSON_MODE_LIST)

    rows = _read_csv(out)
    assert float(rows[0]["Lat"]) == pytest.approx(46.253010, abs=1e-5)
    assert float(rows[0]["Lon"]) == pytest.approx(20.141425, abs=1e-5)


def test_export_csv_gps_image_coords_take_priority_over_exif(db, tmp_path):
    with session_scope() as session:
        _make_image(
            session, tmp_path / "gps2.jpg",
            exif_latitude=46.0,
            exif_longitude=20.0,
            image_latitude=47.5,
            image_longitude=19.0,
        )

    out = tmp_path / "out.csv"
    with session_scope() as session:
        ImageMetadataExportService(session).export_csv(out, ALL_FIELDS, PERSON_MODE_LIST)

    rows = _read_csv(out)
    assert float(rows[0]["Lat"]) == pytest.approx(47.5)
    assert float(rows[0]["Lon"]) == pytest.approx(19.0)


def test_export_csv_no_exif_fallback_to_filename_date(db, tmp_path):
    with session_scope() as session:
        _make_image(session, tmp_path / "2019_08_15_vacation.jpg")

    out = tmp_path / "out.csv"
    with session_scope() as session:
        ImageMetadataExportService(session).export_csv(out, ALL_FIELDS, PERSON_MODE_LIST)

    rows = _read_csv(out)
    assert rows[0]["Date"] == "2019-08-15"


def test_export_csv_no_date_returns_unknown(db, tmp_path):
    with session_scope() as session:
        _make_image(session, tmp_path / "nodatephoto.jpg")

    out = tmp_path / "out.csv"
    with session_scope() as session:
        ImageMetadataExportService(session).export_csv(out, ALL_FIELDS, PERSON_MODE_LIST)

    rows = _read_csv(out)
    assert rows[0]["Date"] == UNKNOWN


def test_export_csv_location_from_place(db, tmp_path):
    with session_scope() as session:
        place = Place(name="Budapest", is_anonymous=False)
        session.add(place)
        session.flush()
        _make_image(session, tmp_path / "city.jpg", place_id=place.id)

    out = tmp_path / "out.csv"
    with session_scope() as session:
        ImageMetadataExportService(session).export_csv(out, ALL_FIELDS, PERSON_MODE_LIST)

    rows = _read_csv(out)
    assert rows[0]["Location"] == "Budapest"


def test_export_csv_anonymous_place_not_exported_as_location(db, tmp_path):
    with session_scope() as session:
        place = Place(name="Anonymous place", is_anonymous=True)
        session.add(place)
        session.flush()
        _make_image(session, tmp_path / "anon.jpg", place_id=place.id)

    out = tmp_path / "out.csv"
    with session_scope() as session:
        ImageMetadataExportService(session).export_csv(out, ALL_FIELDS, PERSON_MODE_LIST)

    rows = _read_csv(out)
    assert rows[0]["Location"] == ""


def test_export_csv_utf8_accents(db, tmp_path):
    with session_scope() as session:
        p = _make_person(session, "Érző Ádám")
        img = _make_image(session, tmp_path / "hu.jpg")
        _make_face(session, img, p)

    out = tmp_path / "out.csv"
    with session_scope() as session:
        ImageMetadataExportService(session).export_csv(out, ALL_FIELDS, PERSON_MODE_LIST)

    rows = _read_csv(out)
    assert "Érző Ádám" in rows[0]["Persons"]


def test_export_csv_escaping_comma_in_name(db, tmp_path):
    with session_scope() as session:
        _make_image(session, tmp_path / "photo.jpg", photo_date='Contains, comma "and" quotes')

    out = tmp_path / "out.csv"
    with session_scope() as session:
        ImageMetadataExportService(session).export_csv(out, ALL_FIELDS, PERSON_MODE_LIST)

    rows = _read_csv(out)
    assert rows[0]["Date"] == 'Contains, comma "and" quotes'


def test_export_csv_selective_fields(db, tmp_path):
    with session_scope() as session:
        _make_image(session, tmp_path / "photo.jpg", photo_date="2021-01-01")

    out = tmp_path / "out.csv"
    fields = {FIELD_FILENAME, FIELD_DATE}
    with session_scope() as session:
        ImageMetadataExportService(session).export_csv(out, fields, PERSON_MODE_LIST)

    rows = _read_csv(out)
    assert "Filename" in rows[0]
    assert "Date" in rows[0]
    assert "Persons" not in rows[0]
    assert "Path" not in rows[0]
    assert "Lat" not in rows[0]


def test_export_csv_places_date_location_gps_before_person_columns(db, tmp_path):
    with session_scope() as session:
        alice = _make_person(session, "Alice")
        place = Place(name="Budapest", is_anonymous=False)
        session.add(place)
        session.flush()
        img = _make_image(
            session,
            tmp_path / "ordered.jpg",
            photo_date="2023-04-05",
            place_id=place.id,
            image_latitude=47.4979,
            image_longitude=19.0402,
        )
        _make_face(session, img, alice)

    out = tmp_path / "out.csv"
    with session_scope() as session:
        ImageMetadataExportService(session).export_csv(out, ALL_FIELDS, PERSON_MODE_COLS)

    headers = _read_csv_headers(out)
    assert headers == ["Filename", "Path", "Date", "Location", "Lat", "Lon", "Person1"]
    assert "Latitude" not in headers
    assert "Longitude" not in headers


def test_export_csv_multiple_images(db, tmp_path):
    with session_scope() as session:
        for i in range(50):
            _make_image(session, tmp_path / f"photo_{i:03d}.jpg")

    out = tmp_path / "out.csv"
    with session_scope() as session:
        ImageMetadataExportService(session).export_csv(out, ALL_FIELDS, PERSON_MODE_LIST)

    rows = _read_csv(out)
    assert len(rows) == 50


def test_export_csv_progress_callback(db, tmp_path):
    with session_scope() as session:
        for i in range(5):
            _make_image(session, tmp_path / f"img{i}.jpg")

    calls: list[tuple[int, int]] = []

    def on_progress(done: int, total: int) -> None:
        calls.append((done, total))

    out = tmp_path / "out.csv"
    with session_scope() as session:
        ImageMetadataExportService(session).export_csv(
            out, ALL_FIELDS, PERSON_MODE_LIST, progress_cb=on_progress
        )

    assert len(calls) == 5
    assert calls[-1] == (5, 5)


# ---------------------------------------------------------------------------
# XLSX tests
# ---------------------------------------------------------------------------

def test_export_xlsx_basic(db, tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    with session_scope() as session:
        alice = _make_person(session, "Alice")
        img = _make_image(
            session, tmp_path / "photo.jpg",
            photo_date="2022-03-10",
            exif_latitude=47.5,
            exif_longitude=19.0,
        )
        _make_face(session, img, alice)

    out = tmp_path / "out.xlsx"
    with session_scope() as session:
        ImageMetadataExportService(session).export_xlsx(out, ALL_FIELDS, PERSON_MODE_LIST)

    assert out.exists()
    wb = openpyxl.load_workbook(out)
    ws = wb.active
    headers = [cell.value for cell in ws[1]]
    assert "Filename" in headers
    assert "Persons" in headers
    assert "Lat" in headers

    data_row = [cell.value for cell in ws[2]]
    row_dict = dict(zip(headers, data_row))
    assert row_dict["Filename"] == "photo.jpg"
    assert row_dict["Persons"] == "Alice"
    assert row_dict["Date"] == "2022-03-10"


def test_export_xlsx_gps_numeric(db, tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    with session_scope() as session:
        _make_image(
            session, tmp_path / "gps.jpg",
            exif_latitude=46.253010,
            exif_longitude=20.141425,
        )

    out = tmp_path / "out.xlsx"
    with session_scope() as session:
        ImageMetadataExportService(session).export_xlsx(out, ALL_FIELDS, PERSON_MODE_LIST)

    wb = openpyxl.load_workbook(out)
    ws = wb.active
    headers = [cell.value for cell in ws[1]]
    data_row = [cell.value for cell in ws[2]]
    row_dict = dict(zip(headers, data_row))

    assert isinstance(row_dict["Lat"], float)
    assert isinstance(row_dict["Lon"], float)
    assert row_dict["Lat"] == pytest.approx(46.253010, abs=1e-5)


def test_export_xlsx_utf8_accents(db, tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    with session_scope() as session:
        p = _make_person(session, "Árvíztűrő Tükörfúrógép")
        img = _make_image(session, tmp_path / "hu.jpg")
        _make_face(session, img, p)

    out = tmp_path / "out.xlsx"
    with session_scope() as session:
        ImageMetadataExportService(session).export_xlsx(out, ALL_FIELDS, PERSON_MODE_LIST)

    wb = openpyxl.load_workbook(out)
    ws = wb.active
    headers = [cell.value for cell in ws[1]]
    data_row = [cell.value for cell in ws[2]]
    row_dict = dict(zip(headers, data_row))
    assert "Árvíztűrő Tükörfúrógép" in (row_dict.get("Persons") or "")


def test_export_xlsx_cols_mode_separate_person_columns(db, tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    with session_scope() as session:
        panni = _make_person(session, "Panni")
        kati = _make_person(session, "Kati")
        img = _make_image(session, tmp_path / "group.jpg")
        _make_face(session, img, panni)
        _make_face(session, img, kati)

    out = tmp_path / "out.xlsx"
    with session_scope() as session:
        ImageMetadataExportService(session).export_xlsx(out, ALL_FIELDS, PERSON_MODE_COLS)

    wb = openpyxl.load_workbook(out)
    ws = wb.active
    headers = [cell.value for cell in ws[1]]
    assert "Person1" in headers
    assert "Person2" in headers
    assert "Persons" not in headers


def test_export_xlsx_uses_same_column_order_as_csv(db, tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    with session_scope() as session:
        alice = _make_person(session, "Alice")
        place = Place(name="Szeged", is_anonymous=False)
        session.add(place)
        session.flush()
        img = _make_image(
            session,
            tmp_path / "ordered.xlsx.jpg",
            photo_date="2024-01-02",
            place_id=place.id,
            image_latitude=46.253,
            image_longitude=20.141,
        )
        _make_face(session, img, alice)

    out = tmp_path / "out.xlsx"
    with session_scope() as session:
        ImageMetadataExportService(session).export_xlsx(out, ALL_FIELDS, PERSON_MODE_COLS)

    wb = openpyxl.load_workbook(out)
    ws = wb.active
    headers = [cell.value for cell in ws[1]]
    assert headers == ["Filename", "Path", "Date", "Location", "Lat", "Lon", "Person1"]
    assert "Latitude" not in headers
    assert "Longitude" not in headers


def test_export_xlsx_normalizes_extension(db, tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    with session_scope() as session:
        _make_image(session, tmp_path / "photo.jpg")

    requested = tmp_path / "metadata.csv"
    with session_scope() as session:
        out = ImageMetadataExportService(session).export_xlsx(
            requested, ALL_FIELDS, PERSON_MODE_LIST
        )

    assert out == tmp_path / "metadata.xlsx"
    assert out.exists()
    assert not requested.exists()
    openpyxl.load_workbook(out)


def test_export_xlsx_large_batch(db, tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    with session_scope() as session:
        for i in range(200):
            _make_image(session, tmp_path / f"img_{i:04d}.jpg")

    out = tmp_path / "big.xlsx"
    with session_scope() as session:
        ImageMetadataExportService(session).export_xlsx(out, ALL_FIELDS, PERSON_MODE_LIST)

    wb = openpyxl.load_workbook(out)
    ws = wb.active
    assert ws.max_row == 201  # 1 header + 200 data rows
