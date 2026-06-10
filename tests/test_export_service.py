"""Tests for the Astro static-site export (faces, objects, parity pages).

The legacy single-page ``export_html`` gallery was removed; all gallery export
now goes through :meth:`ExportService.export_astro`. These tests exercise the
data bundle it writes (``src/data/*.json``) plus the parity map/slideshow pages
it drops into ``public/`` — without running the ``npm`` build (``run_build=False``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.db.database import init_db, session_scope
from app.db.models import Face, Image, Person, Place
from app.services.export_service import ExportService, _face_bbox_for_export
from app.services.object_service import ObjectService
from app.utils.exif import _dms_to_decimal
from app.utils.image_utils import save_image_bgr


@pytest.fixture()
def db(tmp_path):
    db_path = tmp_path / "test_export.db"
    init_db(db_path)
    return db_path


def _read_bundle(data_dir: Path, name: str):
    return json.loads((data_dir / name).read_text(encoding="utf-8"))


def _run_astro(session, tmp_path):
    """Run the data-only Astro export; return ``(data_dir, project)``."""
    project = tmp_path / "astro"
    (project / "src" / "data").mkdir(parents=True, exist_ok=True)
    data_dir = ExportService(session).export_astro(
        str(tmp_path / "out"), run_build=False, astro_project_dir=str(project)
    )
    return Path(data_dir), project


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


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


def test_legacy_export_html_is_removed():
    """The non-Astro single-page gallery export no longer exists."""
    assert not hasattr(ExportService, "export_html")


# ---------------------------------------------------------------------------
# Astro export bundle — persons / faces
# ---------------------------------------------------------------------------


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

    with session_scope() as session:
        data_dir, project = _run_astro(session, tmp_path)

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
    # Objects key is always present (empty here), so the frontend never guards.
    assert photo["objects"] == []

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


def test_export_astro_compare_group_with_multiple_variants(db, tmp_path):
    """A B&W original with two colorized variants exports a 3-member group."""
    photos = tmp_path / "photos"
    photos.mkdir()
    bw = photos / "family.jpg"
    artistic = photos / "family-deoldified (artistic).jpg"
    stable = photos / "family-deoldified (stable).jpg"
    img = np.zeros((600, 900, 3), dtype=np.uint8)
    for p, colr in ((bw, (10, 10, 10)), (artistic, (40, 80, 120)), (stable, (90, 60, 30))):
        frame = img.copy()
        frame[:, :] = colr
        assert save_image_bgr(p, frame)

    with session_scope() as session:
        for p in (bw, artistic, stable):
            session.add(Image(file_path=str(p), file_hash=p.name, file_mtime=0.0,
                              width=900, height=600, detection_done=True))

    with session_scope() as session:
        data_dir, project = _run_astro(session, tmp_path)

    photos_json = _read_bundle(data_dir, "photos.json")
    by_name = {Path(p["fileName"]).name: p for p in photos_json}

    bw_photo = by_name["family.jpg"]
    members = bw_photo["compare"]
    assert members is not None and len(members) == 3
    # B&W original first, then colorized variants by label.
    assert members[0]["isBw"] is True
    assert [m["label"] for m in members[1:]] == ["(artistic)", "(stable)"]
    assert all(m["src"].startswith("/assets/images/medium/") for m in members)
    # Legacy pair still derived (B&W + first colorized) for the slideshow slider.
    assert bw_photo["pair"]["bw"] == members[0]["src"]
    assert bw_photo["pair"]["color"] == members[1]["src"]

    # Each colorized variant resolves to the same 3-member group.
    assert len(by_name["family-deoldified (artistic).jpg"]["compare"]) == 3
    assert len(by_name["family-deoldified (stable).jpg"]["compare"]) == 3


def test_export_astro_no_compare_group_for_lone_image(db, tmp_path):
    """An image without colorized siblings has compare=None (simple view)."""
    photos = tmp_path / "photos"
    photos.mkdir()
    lone = photos / "solo.jpg"
    assert save_image_bgr(lone, np.full((400, 600, 3), 50, dtype=np.uint8))
    with session_scope() as session:
        session.add(Image(file_path=str(lone), file_hash="solo", file_mtime=0.0,
                          width=600, height=400, detection_done=True))
    with session_scope() as session:
        data_dir, _ = _run_astro(session, tmp_path)
    photo = _read_bundle(data_dir, "photos.json")[0]
    assert photo["compare"] is None
    assert photo["pair"] is None


def test_export_astro_faces_use_percentage_boxes(db, tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    image_path = photos / "wide.jpg"
    image = np.zeros((400, 800, 3), dtype=np.uint8)
    image[:, :] = (40, 80, 120)
    assert save_image_bgr(image_path, image)

    with session_scope() as session:
        alice = Person(name="Alice", is_auto_named=False)
        bob = Person(name="Bob", is_auto_named=False)
        session.add_all([alice, bob])
        session.flush()
        img = Image(file_path=str(image_path), file_hash="hash-wide", file_mtime=0.0,
                    width=800, height=400, detection_done=True)
        session.add(img)
        session.flush()
        session.add_all([
            Face(image_id=img.id, person_id=alice.id, bbox_x=80, bbox_y=40,
                 bbox_w=160, bbox_h=100, confidence=0.98, detector_backend="cpu"),
            Face(image_id=img.id, person_id=bob.id, bbox_x=720, bbox_y=300,
                 bbox_w=70, bbox_h=80, confidence=0.94, detector_backend="cpu"),
        ])

    with session_scope() as session:
        data_dir, _project = _run_astro(session, tmp_path)

    photo = _read_bundle(data_dir, "photos.json")[0]
    boxes = {f["name"]: f["bbox"] for f in photo["faces"]}
    assert boxes["Alice"] == {"left": 10.0, "top": 10.0, "width": 20.0, "height": 25.0}
    assert boxes["Bob"] == {"left": 90.0, "top": 75.0, "width": 8.75, "height": 20.0}


def test_export_astro_includes_faceless_images(db, tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    image_path = photos / "no_face.jpg"
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    assert save_image_bgr(image_path, image)

    with session_scope() as session:
        session.add(Image(file_path=str(image_path), file_hash="hash-no-face",
                          file_mtime=0.0, width=300, height=200, detection_done=True))

    with session_scope() as session:
        data_dir, _project = _run_astro(session, tmp_path)

    photos_json = _read_bundle(data_dir, "photos.json")
    assert len(photos_json) == 1
    assert photos_json[0]["faces"] == []
    assert photos_json[0]["objects"] == []


def test_export_astro_writes_map_and_slideshow_parity_pages(db, tmp_path):
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
        img = Image(file_path=str(image_path), file_hash="hash-map", file_mtime=0.0,
                    width=320, height=240, photo_date="1984.07.12",
                    note="Nyári emlék", image_latitude=46.253, image_longitude=20.148,
                    place_id=place.id, detection_done=True)
        session.add(img)
        session.flush()
        session.add(Face(image_id=img.id, person_id=person.id, bbox_x=20, bbox_y=30,
                         bbox_w=50, bbox_h=60, confidence=0.9, detector_backend="cpu"))

    with session_scope() as session:
        _data_dir, project = _run_astro(session, tmp_path)

    public = project / "public"
    for name in ("map.html", "map.js", "map-data.json",
                 "slideshow.html", "slideshow.js", "slideshow-data.json"):
        assert (public / name).exists(), name

    map_data = json.loads((public / "map-data.json").read_text(encoding="utf-8"))
    assert [r["fileName"] for r in map_data] == ["Őszi kép.jpg"]
    assert map_data[0]["people"] == ["Panni"]

    tour = json.loads((public / "slideshow-data.json").read_text(encoding="utf-8"))
    assert tour[0]["fileName"] == "Őszi kép.jpg"
    assert tour[0]["year"] == "1984"
    assert tour[0]["caption"] == "Nyári emlék"


# ---------------------------------------------------------------------------
# Astro export bundle — tagged objects
# ---------------------------------------------------------------------------


def test_export_astro_objects_bbox_and_point(db, tmp_path):
    """Objects export as image-relative percentages, matching the face format."""
    photos = tmp_path / "photos"
    photos.mkdir()
    image_path = photos / "street.jpg"
    image = np.zeros((400, 800, 3), dtype=np.uint8)
    assert save_image_bgr(image_path, image)

    with session_scope() as session:
        img = Image(file_path=str(image_path), file_hash="obj-hash", file_mtime=0.0,
                    width=800, height=400, detection_done=True)
        session.add(img)
        session.flush()
        svc = ObjectService(session)
        car = svc.create_object(name="BMW E91", description="330D Touring")
        sign = svc.create_object(name="Útjelző tábla")
        # A bbox object and a point-only object on the same image.
        svc.add_occurrence_bbox(car.id, img.id, x=80, y=40, w=160, h=100,
                                note="Frissen mosva")
        svc.add_occurrence(sign.id, img.id, point_x=400, point_y=200)

    with session_scope() as session:
        data_dir, _project = _run_astro(session, tmp_path)

    photo = _read_bundle(data_dir, "photos.json")[0]
    objects = {o["name"]: o for o in photo["objects"]}
    assert set(objects) == {"BMW E91", "Útjelző tábla"}

    car = objects["BMW E91"]
    assert car["bbox"] == {"left": 10.0, "top": 10.0, "width": 20.0, "height": 25.0}
    assert car["point"] is None
    assert car["note"] == "Frissen mosva"

    sign = objects["Útjelző tábla"]
    assert sign["bbox"] is None
    assert sign["point"] == {"left": 50.0, "top": 50.0}
    assert sign["note"] == ""

    # Faces remain unaffected (none here) and persons still export normally.
    assert photo["faces"] == []


def test_export_astro_handles_image_with_no_objects(db, tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    img_a = photos / "a.jpg"
    img_b = photos / "b.jpg"
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    assert save_image_bgr(img_a, image)
    assert save_image_bgr(img_b, image)

    with session_scope() as session:
        a = Image(file_path=str(img_a), file_hash="a", file_mtime=0.0,
                  width=300, height=200, detection_done=True)
        b = Image(file_path=str(img_b), file_hash="b", file_mtime=0.0,
                  width=300, height=200, detection_done=True)
        session.add_all([a, b])
        session.flush()
        svc = ObjectService(session)
        obj = svc.create_object(name="Csónak")
        svc.add_occurrence(obj.id, a.id, point_x=150, point_y=100)

    with session_scope() as session:
        data_dir, _project = _run_astro(session, tmp_path)

    by_file = {p["fileName"]: p for p in _read_bundle(data_dir, "photos.json")}
    assert len(by_file["a.jpg"]["objects"]) == 1
    # The image with no tagged objects gets an empty list, not a crash.
    assert by_file["b.jpg"]["objects"] == []


def test_export_astro_object_collection_records(db, tmp_path):
    """objects.json mirrors persons.json: identity, thumb, counts, linked people."""
    photos = tmp_path / "photos"
    photos.mkdir()
    ip1 = photos / "a.jpg"
    ip2 = photos / "b.jpg"
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    assert save_image_bgr(ip1, image)
    assert save_image_bgr(ip2, image)

    with session_scope() as session:
        anna = Person(name="Anna", is_auto_named=False)
        bob = Person(name="Bob", is_auto_named=False)
        session.add_all([anna, bob])
        session.flush()
        im1 = Image(file_path=str(ip1), file_hash="a", file_mtime=0.0,
                    width=600, height=400, detection_done=True)
        im2 = Image(file_path=str(ip2), file_hash="b", file_mtime=0.0,
                    width=600, height=400, detection_done=True)
        session.add_all([im1, im2])
        session.flush()
        svc = ObjectService(session)
        car = svc.create_object(name="BMW E91", description="330D Touring", notes="Kék")
        svc.add_occurrence_bbox(car.id, im1.id, x=10, y=10, w=100, h=80)
        svc.add_occurrence(car.id, im2.id, point_x=50, point_y=50)
        svc.add_person_link(car.id, anna.id, "owner", note="2010 óta")
        svc.add_person_link(car.id, bob.id, "driver")

    with session_scope() as session:
        data_dir, _project = _run_astro(session, tmp_path)

    objects = _read_bundle(data_dir, "objects.json")
    assert len(objects) == 1
    car = objects[0]
    assert car["name"] == "BMW E91"
    assert car["description"] == "330D Touring"
    assert car["notes"] == "Kék"
    assert car["thumb"].startswith("/assets/images/objects/")
    assert car["imageCount"] == 2
    assert len(car["imageIds"]) == 2
    assert car["occurrenceCount"] == 2
    roles = {p["name"]: p["roleLabel"] for p in car["persons"]}
    assert roles == {"Anna": "Tulajdonos", "Bob": "Sofőr"}
    assert next(p for p in car["persons"] if p["name"] == "Anna")["note"] == "2010 óta"

    manifest = _read_bundle(data_dir, "manifest.json")
    assert manifest["objectCount"] == 1
    assert manifest["hasObjects"] is True

    # The object also lands in the runtime search index, typed as "object".
    index = json.loads(
        (_project / "public" / "assets" / "data" / "search-index.json").read_text(encoding="utf-8"))
    obj_entry = next(e for e in index if e["type"] == "object")
    assert obj_entry["name"] == "BMW E91"
    assert obj_entry["url"] == "objects/1/"


def test_export_astro_slideshow_carries_objects_and_panel_data(db, tmp_path):
    """Slideshow records list per-image objects; TOUR_OBJECTS holds descriptions."""
    photos = tmp_path / "photos"
    photos.mkdir()
    image_path = photos / "scene.jpg"
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    assert save_image_bgr(image_path, image)

    with session_scope() as session:
        img = Image(file_path=str(image_path), file_hash="s", file_mtime=0.0,
                    width=600, height=400, detection_done=True)
        session.add(img)
        session.flush()
        svc = ObjectService(session)
        boat = svc.create_object(name="Klotild", description="Fa vitorlás")
        svc.add_occurrence(boat.id, img.id, point_x=300, point_y=200, note="Kikötve")

    with session_scope() as session:
        data_dir, project = _run_astro(session, tmp_path)

    bundle = _read_bundle(data_dir, "slideshow-data.json")
    rec = bundle["records"][0]
    assert rec["objects"][0]["name"] == "Klotild"
    assert rec["objects"][0]["point"] == {"left": 50.0, "top": 50.0}
    assert rec["objects"][0]["note"] == "Kikötve"
    detail = bundle["objects"]["1"]
    assert detail["name"] == "Klotild"
    assert detail["description"] == "Fa vitorlás"

    # The standalone slideshow page ships the object data + click-panel wiring.
    data_js = (project / "public" / "slideshow-data.js").read_text(encoding="utf-8")
    assert "window.TOUR_OBJECTS = " in data_js
    tour_js = (project / "public" / "slideshow.js").read_text(encoding="utf-8")
    assert "function openObjectPanel" in tour_js
    assert "record.objects" in tour_js


def test_export_astro_multiple_occurrences_same_object(db, tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    image_path = photos / "twins.jpg"
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    assert save_image_bgr(image_path, image)

    with session_scope() as session:
        img = Image(file_path=str(image_path), file_hash="twins", file_mtime=0.0,
                    width=400, height=400, detection_done=True)
        session.add(img)
        session.flush()
        svc = ObjectService(session)
        obj = svc.create_object(name="Lampion")
        svc.add_occurrence(obj.id, img.id, point_x=100, point_y=100)
        svc.add_occurrence(obj.id, img.id, point_x=300, point_y=300)

    with session_scope() as session:
        data_dir, _project = _run_astro(session, tmp_path)

    objects = _read_bundle(data_dir, "photos.json")[0]["objects"]
    assert len(objects) == 2
    points = sorted((o["point"]["left"], o["point"]["top"]) for o in objects)
    assert points == [(25.0, 25.0), (75.0, 75.0)]
