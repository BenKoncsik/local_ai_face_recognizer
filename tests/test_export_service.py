"""Tests for HTML export coordinate metadata."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.db.database import init_db, session_scope
from app.db.models import Face, Image, Person
from app.services.export_service import ExportService, _face_bbox_for_export
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
