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


def _read_images_json(index_html: Path) -> dict:
    html = index_html.read_text(encoding="utf-8")
    start = html.index("const IMAGES = ") + len("const IMAGES = ")
    end = html.index(";\nlet currentImage", start)
    return json.loads(html[start:end])


def test_export_html_keeps_original_pixel_boxes_for_overlay(db, tmp_path):
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
    images = _read_images_json(index_html)
    assert "function transformBox(box, originalSize, display)" in html
    assert "display.offsetX+Number(box.x)*scaleX" in html

    assert len(images) == 1
    record = next(iter(images.values()))
    assert record["width"] == 800
    assert record["height"] == 400
    assert (out / record["src"]).exists()

    boxes_by_name = {face["name"]: face["bbox"] for face in record["faces"]}
    assert boxes_by_name["Alice"] == {"x": 80, "y": 40, "width": 160, "height": 100}
    assert boxes_by_name["Bob"] == {"x": 720, "y": 300, "width": 70, "height": 80}


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
        "x": 750,
        "y": 380,
        "width": 50,
        "height": 20,
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
        "x": 200,
        "y": 200,
        "width": 200,
        "height": 40,
    }
