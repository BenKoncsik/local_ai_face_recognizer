"""Tests for face quality evaluation."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.db.database import init_db, session_scope
from app.db.models import Face, Image, Person
from app.services.face_quality_service import (
    BLUR_THRESHOLD,
    CONFIDENCE_THRESHOLD,
    FaceQualityBatchService,
    FaceQualityEvaluator,
    MIN_FACE_AREA,
    QUALITY_THRESHOLD,
)


@pytest.fixture()
def db(tmp_path):
    init_db(tmp_path / "quality.db")
    return tmp_path


def _make_face(**kwargs) -> Face:
    defaults = {
        "bbox_x": 0,
        "bbox_y": 0,
        "bbox_w": 100,
        "bbox_h": 100,
        "confidence": 0.9,
        "detector_backend": "cpu",
    }
    defaults.update(kwargs)
    return Face(**defaults)


def _write_sharp_crop(path) -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (80, 80), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _write_blurry_crop(path) -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (80, 80), dtype=np.uint8)
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    cv2.imwrite(str(path), blurred)


def test_evaluate_high_quality_face(tmp_path):
    crop = tmp_path / "sharp.jpg"
    _write_sharp_crop(crop)
    face = _make_face(crop_path=str(crop))

    score, reasons = FaceQualityEvaluator().evaluate(face)

    assert score == pytest.approx(1.0)
    assert reasons == []


def test_evaluate_flags_low_confidence():
    face = _make_face(confidence=CONFIDENCE_THRESHOLD - 0.1)

    score, reasons = FaceQualityEvaluator().evaluate(face)

    assert "low_confidence" in reasons
    assert score < 1.0


def test_evaluate_flags_too_small():
    side = int(MIN_FACE_AREA ** 0.5) - 1
    face = _make_face(bbox_w=side, bbox_h=side)

    score, reasons = FaceQualityEvaluator().evaluate(face)

    assert "too_small" in reasons
    assert score < 1.0


def test_evaluate_flags_bad_aspect_ratio():
    face = _make_face(bbox_w=100, bbox_h=10)

    score, reasons = FaceQualityEvaluator().evaluate(face)

    assert "bad_aspect_ratio" in reasons
    assert score < 1.0


def test_evaluate_flags_blurry_crop(tmp_path):
    crop = tmp_path / "blur.jpg"
    _write_blurry_crop(crop)
    face = _make_face(crop_path=str(crop))

    score, reasons = FaceQualityEvaluator().evaluate(face)

    blur = FaceQualityEvaluator._laplacian_variance(str(crop))
    assert blur is not None
    assert blur < BLUR_THRESHOLD
    assert "blurry" in reasons
    assert score < 1.0


def test_evaluate_and_update_writes_orm_fields():
    face = _make_face(confidence=0.2, bbox_w=30, bbox_h=30)

    FaceQualityEvaluator().evaluate_and_update(face)

    assert face.quality_score is not None
    assert face.quality_score < QUALITY_THRESHOLD
    assert face.is_low_quality is True
    assert face.quality_reasons is not None
    assert "low_confidence" in face.quality_reasons.split(",")
    assert "too_small" in face.quality_reasons.split(",")


def test_laplacian_variance_missing_file_returns_none():
    assert FaceQualityEvaluator._laplacian_variance("/nonexistent/crop.jpg") is None


def test_laplacian_variance_invalid_image_returns_none(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    assert FaceQualityEvaluator._laplacian_variance(str(bad)) is None


def test_laplacian_variance_sharp_image(tmp_path):
    crop = tmp_path / "sharp.jpg"
    _write_sharp_crop(crop)
    var = FaceQualityEvaluator._laplacian_variance(str(crop))
    assert var is not None
    assert var >= BLUR_THRESHOLD


def test_batch_evaluate_all_persists_and_reports_progress(db, tmp_path):
    crop = tmp_path / "sharp.jpg"
    _write_sharp_crop(crop)
    with session_scope() as session:
        img = Image(file_path="/tmp/a.jpg", file_hash="h1", file_mtime=0.0)
        person = Person(name="Teszt")
        session.add_all([img, person])
        session.flush()
        good = Face(
            image_id=img.id,
            person_id=person.id,
            bbox_x=0,
            bbox_y=0,
            bbox_w=100,
            bbox_h=100,
            confidence=0.95,
            detector_backend="cpu",
            crop_path=str(crop),
        )
        bad = Face(
            image_id=img.id,
            person_id=person.id,
            bbox_x=0,
            bbox_y=0,
            bbox_w=20,
            bbox_h=20,
            confidence=0.1,
            detector_backend="cpu",
        )
        session.add_all([good, bad])
        session.commit()
        good_id, bad_id = good.id, bad.id

    progress: list[tuple[int, int]] = []
    with session_scope() as session:
        total = FaceQualityBatchService(session).evaluate_all(
            progress_cb=lambda c, t: progress.append((c, t))
        )

    assert total == 2
    assert progress == [(1, 2), (2, 2)]

    with session_scope() as session:
        good_face = session.get(Face, good_id)
        bad_face = session.get(Face, bad_id)
        assert good_face.quality_score is not None
        assert good_face.is_low_quality is False
        assert bad_face.quality_score is not None
        assert bad_face.is_low_quality is True
