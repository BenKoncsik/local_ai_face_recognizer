"""Tests for HTML export coordinate metadata."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.db.database import init_db, session_scope
from app.db.models import Face, Image, Person, Place
from app.services.export_service import ExportService, _face_bbox_for_export
from app.utils.exif import _dms_to_decimal
from app.utils.image_utils import save_image_bgr


@pytest.fixture()
def db(tmp_path):
    db_path = tmp_path / "test_export.db"
    init_db(db_path)
    return db_path


def _read_json_const(index_html: Path, const_name: str):
    html = index_html.read_text(encoding="utf-8")
    marker = f"const {const_name} = "
    start = html.index(marker) + len(marker)
    end = html.index(";\n", start)
    return json.loads(html[start:end])


def test_export_html_uses_percentage_boxes_for_responsive_overlay(db, tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    image_path = photos / "wide.jpg"
    thumb_path = photos / "alice_thumb.jpg"

    image = np.zeros((400, 800, 3), dtype=np.uint8)
    image[:, :] = (40, 80, 120)
    thumb = np.zeros((64, 64, 3), dtype=np.uint8)
    thumb[:, :] = (30, 120, 30)
    assert save_image_bgr(image_path, image)
    assert save_image_bgr(thumb_path, thumb)

    with session_scope() as session:
        alice = Person(name="Alice", is_auto_named=False)
        bob = Person(name="Bob", is_auto_named=False)
        session.add_all([alice, bob])
        session.flush()

        img = Image(
            file_path=str(image_path),
            file_hash="hash-wide",
            file_mtime=0.0,
            width=800,
            height=400,
            detection_done=True,
        )
        session.add(img)
        session.flush()

        session.add_all(
            [
                Face(
                    image_id=img.id,
                    person_id=alice.id,
                    bbox_x=80,
                    bbox_y=40,
                    bbox_w=160,
                    bbox_h=100,
                    confidence=0.98,
                    detector_backend="cpu",
                    crop_path=str(thumb_path),
                ),
                Face(
                    image_id=img.id,
                    person_id=bob.id,
                    bbox_x=720,
                    bbox_y=300,
                    bbox_w=70,
                    bbox_h=80,
                    confidence=0.94,
                    detector_backend="cpu",
                ),
            ]
        )

    with session_scope() as session:
        out = ExportService(session).export_html(str(tmp_path / "html_export"))

    index_html = out / "index.html"
    html = index_html.read_text(encoding="utf-8")
    images = _read_json_const(index_html, "IMAGES")
    image_order = _read_json_const(index_html, "IMAGE_ORDER")
    assert '<meta charset="utf-8">' in html
    assert "setPercentStyle(box,'left',bbox.left)" in html
    assert "showRelativeImage(-1)" in html
    assert "showRelativeImage(1)" in html

    assert len(images) == 1
    record = next(iter(images.values()))
    assert record["width"] == 800
    assert record["height"] == 400
    assert image_order == [record["id"]]
    assert (out / record["src"]).exists()

    boxes_by_name = {face["name"]: face["bbox"] for face in record["faces"]}
    assert boxes_by_name["Alice"] == {
        "left": 10.0,
        "top": 10.0,
        "width": 20.0,
        "height": 25.0,
    }
    assert boxes_by_name["Bob"] == {
        "left": 90.0,
        "top": 75.0,
        "width": 8.75,
        "height": 20.0,
    }


def test_export_html_preserves_hungarian_text_and_script_safe_json(db, tmp_path):
    photos = tmp_path / "Családi archív képek"
    photos.mkdir()
    image_path = photos / "Őszi kép.jpg"
    image = np.zeros((300, 500, 3), dtype=np.uint8)
    image[:, :] = (70, 90, 110)
    assert save_image_bgr(image_path, image)

    names = [
        "Ági",
        "Őri Éva",
        "Tóth Károly",
        "Szűcs Zsófia </script><script>alert(1)</script>",
    ]

    with session_scope() as session:
        people = [Person(name=name, is_auto_named=False) for name in names]
        session.add_all(people)
        session.flush()

        img = Image(
            file_path=str(image_path),
            file_hash="hash-hu",
            file_mtime=0.0,
            width=500,
            height=300,
            detection_done=True,
        )
        session.add(img)
        session.flush()

        for idx, person in enumerate(people):
            session.add(
                Face(
                    image_id=img.id,
                    person_id=person.id,
                    bbox_x=20 + idx * 90,
                    bbox_y=30 + idx * 30,
                    bbox_w=50,
                    bbox_h=70,
                    confidence=0.9,
                    detector_backend="cpu",
                )
            )

    with session_scope() as session:
        out = ExportService(session).export_html(str(tmp_path / "html_export"))

    html_bytes = (out / "index.html").read_bytes()
    assert b'<meta charset="utf-8">' in html_bytes
    html = html_bytes.decode("utf-8")
    for name in names[:-1]:
        assert name in html
    assert "Családi archív képek" in html
    assert "Őszi kép.jpg" in html
    assert "</script><script>alert(1)</script>" not in html
    assert "<\\/script><script>alert(1)<\\/script>" in html

    persons = _read_json_const(out / "index.html", "PERSONS")
    images = _read_json_const(out / "index.html", "IMAGES")
    exported_names = {person["name"] for person in persons}
    assert set(names).issubset(exported_names)
    record = next(iter(images.values()))
    face_names = {face["name"] for face in record["faces"]}
    assert set(names).issubset(face_names)


def test_export_html_includes_faceless_images_in_all_person_export(db, tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    image_path = photos / "no_face.jpg"
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    image[:, :] = (20, 20, 20)
    assert save_image_bgr(image_path, image)

    with session_scope() as session:
        session.add(
            Image(
                file_path=str(image_path),
                file_hash="hash-no-face",
                file_mtime=0.0,
                width=300,
                height=200,
                detection_done=True,
            )
        )

    with session_scope() as session:
        out = ExportService(session).export_html(str(tmp_path / "html_export"))

    persons = _read_json_const(out / "index.html", "PERSONS")
    images = _read_json_const(out / "index.html", "IMAGES")
    assert persons == [
        {
            "name": "Arc nélküli képek",
            "thumbs": [],
            "images": list(images.keys()),
        }
    ]
    record = next(iter(images.values()))
    assert record["faces"] == []


def test_face_bbox_for_export_clamps_pixel_boxes_to_image_bounds():
    face = Face(
        image_id=1,
        bbox_x=750,
        bbox_y=380,
        bbox_w=100,
        bbox_h=50,
        confidence=0.9,
        detector_backend="cpu",
    )

    assert _face_bbox_for_export(face, 800, 400) == {
        "left": 93.75,
        "top": 95.0,
        "width": 6.25,
        "height": 5.0,
    }


def test_face_bbox_for_export_accepts_normalized_legacy_boxes():
    face = Face(
        image_id=1,
        bbox_x=0.25,
        bbox_y=0.5,
        bbox_w=0.25,
        bbox_h=0.1,
        confidence=0.9,
        detector_backend="legacy",
    )

    assert _face_bbox_for_export(face, 800, 400) == {
        "left": 25.0,
        "top": 50.0,
        "width": 25.0,
        "height": 10.0,
    }


def test_exif_gps_dms_conversion_handles_direction_refs():
    values = ((46, 1), (15, 1), (108, 10))

    assert _dms_to_decimal(values, "N") == pytest.approx(46.253)
    assert _dms_to_decimal(values, "S") == pytest.approx(-46.253)


def test_export_html_writes_map_page_and_utf8_data(db, tmp_path):
    photos = tmp_path / "1984 Szemes"
    photos.mkdir()
    image_path = photos / "Őszi kép.jpg"
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:, :] = (90, 80, 70)
    assert save_image_bgr(image_path, image)

    with session_scope() as session:
        place = Place(name="Balatonszemes", latitude=46.808, longitude=17.77)
        person = Person(name="Panni", is_auto_named=False)
        session.add_all([place, person])
        session.flush()

        img = Image(
            file_path=str(image_path),
            file_hash="hash-map",
            file_mtime=0.0,
            width=320,
            height=240,
            photo_date="1984.07.12",
            image_latitude=46.253,
            image_longitude=20.148,
            place_id=place.id,
            detection_done=True,
        )
        session.add(img)
        session.flush()
        session.add(
            Face(
                image_id=img.id,
                person_id=person.id,
                bbox_x=20,
                bbox_y=30,
                bbox_w=50,
                bbox_h=60,
                confidence=0.9,
                detector_backend="cpu",
            )
        )

    with session_scope() as session:
        out = ExportService(session).export_html(str(tmp_path / "html_export"))

    assert (out / "map.html").exists()
    assert (out / "map.css").exists()
    assert (out / "map.js").exists()
    assert (out / "map-data.json").exists()
    assert (out / "map-data.js").exists()
    assert 'href="map.html">Térkép</a>' in (out / "index.html").read_text(encoding="utf-8")

    data_bytes = (out / "map-data.json").read_bytes()
    data = json.loads(data_bytes.decode("utf-8"))
    assert data == [
        {
            "imageId": 1,
            "fileName": "Őszi kép.jpg",
            "relativePath": next(iter(_read_json_const(out / "index.html", "IMAGES").values()))["src"],
            "thumbnailPath": next(iter(_read_json_const(out / "index.html", "IMAGES").values()))["src"],
            "detailPage": "index.html#image=" + next(iter(_read_json_const(out / "index.html", "IMAGES").keys())),
            "latitude": 46.253,
            "longitude": 20.148,
            "dateTaken": "1984.07.12",
            "people": ["Panni"],
            "placeName": "Balatonszemes",
            "folder": "1984 Szemes",
        }
    ]
    assert "Őszi kép.jpg" in data_bytes.decode("utf-8")


def test_export_html_map_skips_missing_bad_and_zero_gps(db, tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    good_path = photos / "good.jpg"
    zero_path = photos / "zero.jpg"
    bad_path = photos / "bad.jpg"
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    assert save_image_bgr(good_path, image)
    assert save_image_bgr(zero_path, image)
    assert save_image_bgr(bad_path, image)

    with session_scope() as session:
        session.add_all(
            [
                Image(
                    file_path=str(good_path),
                    file_hash="good",
                    file_mtime=0.0,
                    width=120,
                    height=80,
                    exif_latitude=47.5,
                    exif_longitude=19.0,
                    detection_done=True,
                ),
                Image(
                    file_path=str(zero_path),
                    file_hash="zero",
                    file_mtime=0.0,
                    width=120,
                    height=80,
                    exif_latitude=0.0,
                    exif_longitude=0.0,
                    detection_done=True,
                ),
                Image(
                    file_path=str(bad_path),
                    file_hash="bad",
                    file_mtime=0.0,
                    width=120,
                    height=80,
                    exif_latitude=95.0,
                    exif_longitude=19.0,
                    detection_done=True,
                ),
            ]
        )

    with session_scope() as session:
        out = ExportService(session).export_html(str(tmp_path / "html_export"))

    data = json.loads((out / "map-data.json").read_text(encoding="utf-8"))
    assert [record["fileName"] for record in data] == ["good.jpg"]


def test_export_html_writes_slideshow_page_with_rich_records(db, tmp_path):
    photos = tmp_path / "1984 Szemes"
    photos.mkdir()
    image_path = photos / "Nyaralás.jpg"
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:, :] = (90, 80, 70)
    assert save_image_bgr(image_path, image)

    with session_scope() as session:
        place = Place(name="Balatonszemes", latitude=46.808, longitude=17.77)
        named = Person(name="Panni", is_auto_named=False)
        auto = Person(name="Unknown 4", is_auto_named=True)
        session.add_all([place, named, auto])
        session.flush()

        img = Image(
            file_path=str(image_path),
            file_hash="hash-tour",
            file_mtime=0.0,
            width=320,
            height=240,
            photo_date="1984.07.12",
            note="Nyári emlék a Balatonnál",
            image_latitude=46.253,
            image_longitude=20.148,
            place_id=place.id,
            detection_done=True,
        )
        session.add(img)
        session.flush()
        session.add_all(
            [
                Face(
                    image_id=img.id, person_id=named.id,
                    bbox_x=20, bbox_y=30, bbox_w=50, bbox_h=60,
                    confidence=0.9, detector_backend="cpu",
                ),
                Face(
                    image_id=img.id, person_id=auto.id,
                    bbox_x=120, bbox_y=40, bbox_w=40, bbox_h=50,
                    confidence=0.8, detector_backend="cpu",
                ),
            ]
        )

    with session_scope() as session:
        out = ExportService(session).export_html(str(tmp_path / "html_export"))

    assert (out / "slideshow.html").exists()
    assert (out / "slideshow.css").exists()
    assert (out / "slideshow.js").exists()
    assert (out / "slideshow-data.json").exists()
    assert (out / "slideshow-data.js").exists()

    # Cross-links exist between the pages without breaking the old gallery page.
    index_html = (out / "index.html").read_text(encoding="utf-8")
    assert 'href="slideshow.html">Diavetítés</a>' in index_html
    assert 'href="map.html">Térkép</a>' in index_html

    data = json.loads((out / "slideshow-data.json").read_text(encoding="utf-8"))
    assert len(data) == 1
    record = data[0]
    assert record["fileName"] == "Nyaralás.jpg"
    assert record["folder"] == "1984 Szemes"
    assert record["date"] == "1984.07.12"
    assert record["year"] == "1984"
    assert record["decade"] == "1980-as évek"
    assert record["locationName"] == "Balatonszemes"
    assert record["city"] == "Balatonszemes"
    assert record["gpsLatitude"] == 46.253
    assert record["gpsLongitude"] == 20.148
    assert record["caption"] == "Nyári emlék a Balatonnál"
    assert record["description"] == "Nyári emlék a Balatonnál"
    assert record["hasCaption"] is True
    assert record["hasLocation"] is True
    assert record["isFavorite"] is False
    assert record["hasIdentifiedPersons"] is True
    assert set(record["persons"]) == {"Panni", "Unknown 4"}
    assert len(record["faces"]) == 2
    assert all("bbox" in face and "name" in face for face in record["faces"])


def test_export_html_slideshow_handles_missing_metadata(db, tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    image_path = photos / "bare.jpg"
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    assert save_image_bgr(image_path, image)

    with session_scope() as session:
        session.add(
            Image(
                file_path=str(image_path),
                file_hash="bare",
                file_mtime=0.0,
                width=120,
                height=100,
                detection_done=True,
            )
        )

    with session_scope() as session:
        out = ExportService(session).export_html(str(tmp_path / "html_export"))

    data = json.loads((out / "slideshow-data.json").read_text(encoding="utf-8"))
    assert len(data) == 1
    record = data[0]
    assert record["year"] == ""
    assert record["decade"] == ""
    assert record["locationName"] == ""
    assert record["gpsLatitude"] is None
    assert record["gpsLongitude"] is None
    assert record["persons"] == []
    assert record["faces"] == []
    assert record["hasCaption"] is False
    assert record["hasLocation"] is False
    assert record["hasIdentifiedPersons"] is False


def test_export_html_creates_empty_map_page_without_gps(db, tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    image_path = photos / "no_gps.jpg"
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    assert save_image_bgr(image_path, image)

    with session_scope() as session:
        session.add(
            Image(
                file_path=str(image_path),
                file_hash="no-gps",
                file_mtime=0.0,
                width=120,
                height=80,
                detection_done=True,
            )
        )

    with session_scope() as session:
        out = ExportService(session).export_html(str(tmp_path / "html_export"))

    assert json.loads((out / "map-data.json").read_text(encoding="utf-8")) == []
    assert "Nincs GPS-koordinátával rendelkező kép" in (out / "map.js").read_text(encoding="utf-8")


def test_export_html_links_deoldified_pair_with_compare_slider(db, tmp_path):
    """An exported image with a deoldified counterpart gets a bw/color pair."""
    bw_dir = tmp_path / "bw"
    color_dir = tmp_path / "color"
    bw_dir.mkdir()
    color_dir.mkdir()
    bw_path = bw_dir / "portrait.jpg"
    color_path = color_dir / "portrait-deoldified (artistic).jpg"
    thumb_path = bw_dir / "thumb.jpg"

    bw = np.zeros((200, 300, 3), dtype=np.uint8)
    bw[:, :] = (90, 90, 90)
    color = np.zeros((200, 300, 3), dtype=np.uint8)
    color[:, :] = (20, 120, 200)
    thumb = np.zeros((64, 64, 3), dtype=np.uint8)
    assert save_image_bgr(bw_path, bw)
    assert save_image_bgr(color_path, color)
    assert save_image_bgr(thumb_path, thumb)

    with session_scope() as session:
        person = Person(name="Anna", is_auto_named=False)
        session.add(person)
        session.flush()

        bw_img = Image(
            file_path=str(bw_path),
            file_hash="bw-hash",
            file_mtime=0.0,
            width=300,
            height=200,
            detection_done=True,
        )
        color_img = Image(
            file_path=str(color_path),
            file_hash="color-hash",
            file_mtime=0.0,
            width=300,
            height=200,
        )
        session.add_all([bw_img, color_img])
        session.flush()
        session.add(
            Face(
                image_id=bw_img.id,
                person_id=person.id,
                bbox_x=30,
                bbox_y=20,
                bbox_w=80,
                bbox_h=80,
                confidence=0.95,
                detector_backend="cpu",
                crop_path=str(thumb_path),
            )
        )

    with session_scope() as session:
        out = ExportService(session).export_html(str(tmp_path / "html_export"))

    index_html = out / "index.html"
    html = index_html.read_text(encoding="utf-8")
    images = _read_json_const(index_html, "IMAGES")

    # The B&W image (the one with the face) carries a pair pointing at both sides.
    record = next(r for r in images.values() if r["faces"])
    assert "pair" in record
    pair = record["pair"]
    assert pair["bw"] == record["src"]
    assert pair["color"] != pair["bw"]
    # Both normalized image files were exported.
    assert (out / pair["bw"]).exists()
    assert (out / pair["color"]).exists()

    # Viewer wiring for the toggle + drag-compare slider is present.
    assert 'onclick="toggleCompare()"' in html
    assert 'onclick="setPairView(' in html
    assert "function setCompareSplit(pct)" in html
    assert "initCompareDrag" in html

    # The slideshow record carries the same pair, and slideshow.js wires up the
    # slider-only compare (no toggle buttons there).
    tour = json.loads(
        (out / "slideshow-data.json").read_text(encoding="utf-8")
    )
    tour_record = next(r for r in tour if r["faces"])
    assert tour_record["pair"] == pair
    tour_js = (out / "slideshow.js").read_text(encoding="utf-8")
    assert "function setupPairMedia(record)" in tour_js
    assert "function setTourSplit(pct)" in tour_js
    assert "initTourCompareDrag" in tour_js
    tour_css = (out / "slideshow.css").read_text(encoding="utf-8")
    assert "#tour-slider" in tour_css


def test_export_html_slideshow_person_panel_data_and_wiring(db, tmp_path):
    """Clicking a face opens a person panel; rich person data is exported."""
    photos = tmp_path / "photos"
    photos.mkdir()
    image_path = photos / "group.jpg"
    thumb_path = photos / "anna_thumb.jpg"
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    thumb = np.zeros((64, 64, 3), dtype=np.uint8)
    assert save_image_bgr(image_path, image)
    assert save_image_bgr(thumb_path, thumb)

    with session_scope() as session:
        anna = Person(
            name="Anna Kovács",
            is_auto_named=False,
            gender="female",
            birth_date="1950",
            birth_place="Szeged",
            nickname="Anci",
            notes="Nagymama.",
            thumbnail_path=str(thumb_path),
        )
        session.add(anna)
        session.flush()

        img = Image(
            file_path=str(image_path),
            file_hash="grp-hash",
            file_mtime=0.0,
            width=300,
            height=200,
            detection_done=True,
        )
        session.add(img)
        session.flush()
        session.add(
            Face(
                image_id=img.id,
                person_id=anna.id,
                bbox_x=20,
                bbox_y=20,
                bbox_w=60,
                bbox_h=60,
                confidence=0.97,
                detector_backend="cpu",
                crop_path=str(thumb_path),
            )
        )
        anna_id = anna.id

    with session_scope() as session:
        out = ExportService(session).export_html(str(tmp_path / "html_export"))

    # Person details are embedded in slideshow-data.js as window.TOUR_PERSONS.
    data_js = (out / "slideshow-data.js").read_text(encoding="utf-8")
    marker = "window.TOUR_PERSONS = "
    start = data_js.index(marker) + len(marker)
    end = data_js.index(";\n", start)
    persons = json.loads(data_js[start:end])

    detail = persons[str(anna_id)]
    assert detail["name"] == "Anna Kovács"
    assert detail["gender"] == "female"
    assert detail["birthDate"] == "1950"
    assert detail["nickname"] == "Anci"
    assert detail["notes"] == "Nagymama."
    assert detail["thumb"] == "thumbs/person_%d.jpg" % anna_id
    assert (out / detail["thumb"]).exists()

    # Front-end wiring: clickable panel, close handler, drag guard for faces.
    tour_js = (out / "slideshow.js").read_text(encoding="utf-8")
    assert "function openPersonPanel(personId" in tour_js
    assert "function closePersonPanel()" in tour_js
    assert "openPersonPanel(face.person_id" in tour_js
    assert "closest('.tb')) return" in tour_js
    assert 'id="person-panel"' in (out / "slideshow.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Astro export bundle
# ---------------------------------------------------------------------------


def _read_bundle(data_dir: Path, name: str):
    return json.loads((data_dir / name).read_text(encoding="utf-8"))


def test_export_astro_bundle_structure(db, tmp_path):
    from app.db.models import Relationship
    from app.services.export_service import ASTRO_THUMB_EDGE

    photos = tmp_path / "photos"
    photos.mkdir()
    image_path = photos / "family.jpg"
    crop_path = photos / "alice_crop.jpg"
    # A wide image so the medium variant (1280px long edge) actually downscales.
    image = np.zeros((1000, 2000, 3), dtype=np.uint8)
    image[:, :] = (40, 80, 120)
    crop = np.zeros((80, 80, 3), dtype=np.uint8)
    crop[:, :] = (30, 120, 30)
    assert save_image_bgr(image_path, image)
    assert save_image_bgr(crop_path, crop)

    with session_scope() as session:
        alice = Person(name="Alice Kovács", is_auto_named=False, birth_date="1950-01-01")
        bob = Person(name="Bob Kovács", is_auto_named=False)
        session.add_all([alice, bob])
        session.flush()
        # Bob is Alice's child.
        session.add(Relationship(relationship_type="ParentChild",
                                 person_a_id=alice.id, person_b_id=bob.id))
        img = Image(file_path=str(image_path), file_hash="h", file_mtime=0.0,
                    width=2000, height=1000, detection_done=True)
        session.add(img)
        session.flush()
        session.add(Face(image_id=img.id, person_id=alice.id, bbox_x=100, bbox_y=80,
                         bbox_w=200, bbox_h=160, confidence=0.98,
                         detector_backend="cpu", crop_path=str(crop_path)))

    project = tmp_path / "astro"
    (project / "src" / "data").mkdir(parents=True)
    with session_scope() as session:
        data_dir = ExportService(session).export_astro(
            str(tmp_path / "out"), run_build=False, astro_project_dir=str(project))

    # Bundle files exist.
    manifest = _read_bundle(data_dir, "manifest.json")
    assert manifest["personCount"] == 2
    assert manifest["photoCount"] == 1
    assert manifest["pageSize"] >= 1

    persons = _read_bundle(data_dir, "persons.json")
    alice_rec = next(p for p in persons if p["name"] == "Alice Kovács")
    assert alice_rec["birthYear"] == "1950"
    assert alice_rec["imageCount"] == 1
    assert any(r["label"] == "Gyermek" and r["relatedName"] == "Bob Kovács"
               for r in alice_rec["relationships"])
    bob_rec = next(p for p in persons if p["name"] == "Bob Kovács")
    assert any(r["label"] == "Szülő" for r in bob_rec["relationships"])

    photos_json = _read_bundle(data_dir, "photos.json")
    assert len(photos_json) == 1
    photo = photos_json[0]
    assert photo["thumb"].startswith("/assets/images/thumbs/")
    assert photo["medium"].startswith("/assets/images/medium/")
    assert photo["original"].startswith("/assets/images/original/")
    assert photo["faces"][0]["name"] == "Alice Kovács"
    assert alice_rec["id"] in photo["personIds"]

    # Search index ships under public/ (served at runtime), minimal and typed.
    index = json.loads(
        (project / "public" / "assets" / "data" / "search-index.json").read_text(encoding="utf-8"))
    types = {e["type"] for e in index}
    assert types == {"person", "photo"}
    person_entry = next(e for e in index if e["type"] == "person")
    assert set(person_entry) >= {"id", "name", "type", "url"}

    # Three image variants physically written; medium downscaled, thumb small.
    images_root = project / "public" / "assets" / "images"
    image_id = Path(photo["thumb"]).stem
    assert (images_root / "thumbs" / f"{image_id}.jpg").exists()
    assert (images_root / "medium" / f"{image_id}.jpg").exists()
    assert (images_root / "original" / f"{image_id}.jpg").exists()
    assert photo["thumbW"] <= ASTRO_THUMB_EDGE
    assert photo["mediumW"] <= 1280
    assert photo["width"] == 2000  # original dimensions preserved in metadata
