"""Unit tests for the analysis-only AI face-detection service."""

from __future__ import annotations

import cv2
import numpy as np

from app.config import AiFaceDetectionConfig
from app.db.database import init_db, session_scope
from app.db.models import AiFaceDetection, Face, Image
from app.detectors.base import Detection, FaceDetector
from app.services.ai_face_detection_service import (
    SOURCE_PIPELINE,
    AiFaceDetectionService,
)


class _FakeAiDetector(FaceDetector):
    """Deterministic stand-in for the YuNet detector; records its call args."""

    def __init__(self, detections=None):
        self._detections = detections if detections is not None else [
            Detection(x=10, y=12, w=64, h=64, confidence=0.91),
            Detection(x=120, y=20, w=58, h=60, confidence=0.74),
        ]
        self.calls: list[dict] = []

    @property
    def backend_name(self) -> str:
        return "fake-ai"

    def detect(
        self,
        image_bgr: np.ndarray,
        confidence_threshold: float = 0.5,
        min_face_size: int = 50,
    ):
        self.calls.append(
            {
                "confidence_threshold": confidence_threshold,
                "min_face_size": min_face_size,
            }
        )
        return list(self._detections)


def _seed_image(tmp_path, name="photo.jpg", write_file=True) -> int:
    """Insert one Image row (and optionally the file on disk); return its id."""
    image_path = tmp_path / name
    if write_file:
        assert cv2.imwrite(str(image_path), np.full((240, 320, 3), 255, np.uint8))
    with session_scope() as session:
        image = Image(
            file_path=str(image_path),
            file_hash=f"hash-{name}",
            file_mtime=image_path.stat().st_mtime if write_file else 0.0,
        )
        session.add(image)
        session.flush()
        return image.id


def test_detect_images_persists_results(tmp_path):
    init_db(tmp_path / "faces.db")
    image_id = _seed_image(tmp_path)

    with session_scope() as session:
        svc = AiFaceDetectionService(
            session, AiFaceDetectionConfig(), detector=_FakeAiDetector()
        )
        stats = svc.detect_images([image_id], source=SOURCE_PIPELINE)

    assert stats.available
    assert stats.images_processed == 1
    assert stats.images_failed == 0
    assert stats.faces_found == 2
    assert stats.detector_name == "fake-ai"
    assert stats.run_id

    with session_scope() as session:
        rows = AiFaceDetectionService(
            session, AiFaceDetectionConfig()
        ).list_for_image(image_id)
        assert len(rows) == 2
        # Strongest first.
        assert rows[0].confidence > rows[1].confidence
        top = rows[0]
        assert (top.bbox_x, top.bbox_y, top.bbox_w, top.bbox_h) == (10, 12, 64, 64)
        assert top.detector_name == "fake-ai"
        assert top.run_id == stats.run_id
        assert top.source == SOURCE_PIPELINE
        assert top.image_path and top.image_path.endswith("photo.jpg")
        assert top.detected_at is not None


def test_rerun_replaces_previous_rows(tmp_path):
    init_db(tmp_path / "faces.db")
    image_id = _seed_image(tmp_path)
    cfg = AiFaceDetectionConfig()

    with session_scope() as session:
        first = AiFaceDetectionService(
            session, cfg, detector=_FakeAiDetector()
        ).detect_images([image_id])
    with session_scope() as session:
        second = AiFaceDetectionService(
            session, cfg, detector=_FakeAiDetector()
        ).detect_images([image_id])

    assert first.run_id != second.run_id
    with session_scope() as session:
        rows = session.query(AiFaceDetection).filter_by(image_id=image_id).all()
        # Replaced, not accumulated.
        assert len(rows) == 2
        assert {r.run_id for r in rows} == {second.run_id}


def test_config_thresholds_are_forwarded_to_detector(tmp_path):
    init_db(tmp_path / "faces.db")
    image_id = _seed_image(tmp_path)
    detector = _FakeAiDetector()
    cfg = AiFaceDetectionConfig(confidence_threshold=0.42, min_face_size=33)

    with session_scope() as session:
        AiFaceDetectionService(session, cfg, detector=detector).detect_images(
            [image_id]
        )

    assert detector.calls == [{"confidence_threshold": 0.42, "min_face_size": 33}]


def test_missing_model_reports_error_without_raising(tmp_path, monkeypatch):
    """Requirement: a missing AI model must never break the calling flow."""
    init_db(tmp_path / "faces.db")
    image_id = _seed_image(tmp_path)

    class _BrokenYuNet:
        def __init__(self, model_path=None):
            raise FileNotFoundError("YuNet model not found")

    monkeypatch.setattr(
        "app.detectors.yunet_detector.YuNetDetector", _BrokenYuNet
    )

    with session_scope() as session:
        stats = AiFaceDetectionService(
            session, AiFaceDetectionConfig()
        ).detect_images([image_id])

    assert not stats.available
    assert "YuNet model not found" in (stats.error or "")
    assert stats.images_processed == 0
    with session_scope() as session:
        assert session.query(AiFaceDetection).count() == 0


def test_unreadable_image_is_counted_failed_and_batch_continues(tmp_path):
    init_db(tmp_path / "faces.db")
    missing_id = _seed_image(tmp_path, name="gone.jpg", write_file=False)
    good_id = _seed_image(tmp_path, name="ok.jpg")

    with session_scope() as session:
        stats = AiFaceDetectionService(
            session, AiFaceDetectionConfig(), detector=_FakeAiDetector()
        ).detect_images([missing_id, good_id])

    assert stats.images_failed == 1
    assert stats.images_processed == 1
    assert stats.faces_found == 2


def test_disabled_config_is_a_noop(tmp_path):
    init_db(tmp_path / "faces.db")
    image_id = _seed_image(tmp_path)

    with session_scope() as session:
        stats = AiFaceDetectionService(
            session,
            AiFaceDetectionConfig(enabled=False),
            detector=_FakeAiDetector(),
        ).detect_images([image_id])

    assert not stats.available
    with session_scope() as session:
        assert session.query(AiFaceDetection).count() == 0


def test_classic_face_rows_and_flags_stay_untouched(tmp_path):
    """The AI pass is analysis-only: Face rows and image flags never change."""
    init_db(tmp_path / "faces.db")
    image_id = _seed_image(tmp_path)

    with session_scope() as session:
        session.add(
            Face(
                image_id=image_id,
                bbox_x=5, bbox_y=5, bbox_w=40, bbox_h=40,
                confidence=0.88,
                detector_backend="cpu",
            )
        )
        image = session.get(Image, image_id)
        image.detection_done = True

    with session_scope() as session:
        AiFaceDetectionService(
            session, AiFaceDetectionConfig(), detector=_FakeAiDetector()
        ).detect_images([image_id])

    with session_scope() as session:
        faces = session.query(Face).filter_by(image_id=image_id).all()
        assert len(faces) == 1
        assert faces[0].detector_backend == "cpu"
        assert session.get(Image, image_id).detection_done is True
        assert session.query(AiFaceDetection).count() == 2
