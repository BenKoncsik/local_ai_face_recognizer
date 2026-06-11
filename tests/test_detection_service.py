"""Unit tests for face detection service."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.config import AppConfig
from app.db.database import init_db, session_scope
from app.db.models import Face, Image
from app.detectors.base import Detection, FaceDetector
from app.services.detection_service import DetectionService


class _DummyDetector(FaceDetector):
    @property
    def backend_name(self) -> str:
        return "dummy"

    def detect(
        self,
        image_bgr: np.ndarray,
        confidence_threshold: float = 0.5,
        min_face_size: int = 50,
    ):
        return [
            Detection(x=10, y=10, w=70, h=70, confidence=0.95),
            Detection(x=120, y=15, w=65, h=65, confidence=0.92),
        ]


class _LandmarkDetector(FaceDetector):
    """Detector that returns one face carrying 5 landmarks."""

    LANDMARKS = [[20, 30], [50, 31], [35, 45], [22, 60], [48, 61]]

    @property
    def backend_name(self) -> str:
        return "yunet"

    def detect(
        self,
        image_bgr: np.ndarray,
        confidence_threshold: float = 0.5,
        min_face_size: int = 50,
    ):
        return [
            Detection(
                x=10, y=10, w=70, h=70, confidence=0.95, landmarks=self.LANDMARKS
            ),
        ]


def test_detection_service_persists_landmarks(monkeypatch, tmp_path):
    """Landmarks from the detector must be stored on the Face row and forwarded
    to save_face_crop so the aligned crop mode can use them."""
    db_path = tmp_path / "faces.db"
    init_db(db_path)

    image_path = tmp_path / "p.jpg"
    assert cv2.imwrite(str(image_path), np.full((240, 320, 3), 255, np.uint8))

    with session_scope() as session:
        image = Image(
            file_path=str(image_path),
            file_hash="hash",
            file_mtime=image_path.stat().st_mtime,
        )
        session.add(image)
        session.flush()
        image_id = image.id

    captured_landmarks: list = []

    def fake_save_face_crop(*args, landmarks=None, **kwargs):
        captured_landmarks.append(landmarks)
        return Path(kwargs.get("crops_dir", tmp_path)) / "crop.jpg"

    monkeypatch.setattr(
        "app.services.detection_service.save_face_crop", fake_save_face_crop
    )

    cfg = AppConfig(base_dir=str(tmp_path))
    cfg.storage.db_path = str(db_path)
    cfg.storage.crops_dir = "crops"
    cfg.embedding.crop_mode = "aligned"

    with session_scope() as session:
        service = DetectionService(
            session=session, detector=_LandmarkDetector(), config=cfg
        )
        assert service.process([image_id]) == 1

    # Landmarks were forwarded to the crop function.
    assert captured_landmarks == [_LandmarkDetector.LANDMARKS]

    # And persisted on the Face row, round-tripping through float32 bytes.
    with session_scope() as session:
        face = session.query(Face).one()
        stored = face.get_landmarks()
    assert stored is not None
    assert stored.shape == (5, 2)
    assert np.allclose(stored, np.array(_LandmarkDetector.LANDMARKS, dtype=float))


def test_detection_service_uses_face_id_for_crop_naming(monkeypatch, tmp_path):
    """Each crop filename must be keyed by the face's DB primary key.

    Using face.id (rather than a per-image sequential counter) guarantees that
    later bbox-update operations can overwrite the exact same file without
    accidentally colliding with another face's crop stored in the same image.
    """
    db_path = tmp_path / "faces.db"
    init_db(db_path)

    image_path = tmp_path / "family.jpg"
    img = np.full((240, 320, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), img)

    with session_scope() as session:
        image = Image(
            file_path=str(image_path),
            file_hash="hash",
            file_mtime=image_path.stat().st_mtime,
        )
        session.add(image)
        session.flush()
        image_id = image.id

    captured_indexes: list[int] = []

    def fake_save_face_crop(
        img_bgr,
        detection,
        crops_dir,
        image_id,
        thumbnail_size,
        face_index=0,
        dest_path=None,
        crop_mode="legacy",
        landmarks=None,
    ):
        captured_indexes.append(face_index)
        dest = dest_path or (Path(crops_dir) / f"img{image_id:06d}_face{face_index:06d}.jpg")
        return dest

    monkeypatch.setattr(
        "app.services.detection_service.save_face_crop",
        fake_save_face_crop,
    )

    cfg = AppConfig(base_dir=str(tmp_path))
    cfg.storage.db_path = str(db_path)
    cfg.storage.crops_dir = "crops"

    with session_scope() as session:
        service = DetectionService(
            session=session,
            detector=_DummyDetector(),
            config=cfg,
        )
        detected = service.process([image_id])

    assert detected == 2

    with session_scope() as session:
        faces = session.query(Face).order_by(Face.id).all()

    assert len(faces) == 2
    face_ids = [f.id for f in faces]

    # face_index passed to save_face_crop must be the face's actual DB id,
    # not a per-image sequential counter (0, 1, 2, ...).
    assert captured_indexes == face_ids, (
        "crop filename index must equal face.id to prevent collisions "
        "with future bbox-update operations"
    )

    # Every face must have a distinct, non-None crop path.
    assert faces[0].crop_path is not None
    assert faces[1].crop_path is not None
    assert faces[0].crop_path != faces[1].crop_path


def test_redetection_does_not_duplicate_manual_face(monkeypatch, tmp_path):
    """A manually drawn face must protect its region from a re-detection.

    Re-running detection over an image that already has a *manual* face box
    should skip any new detection that overlaps it, instead of creating a
    duplicate '?' box for a face the user already marked.
    """
    db_path = tmp_path / "faces.db"
    init_db(db_path)

    image_path = tmp_path / "p.jpg"
    assert cv2.imwrite(str(image_path), np.full((240, 320, 3), 255, np.uint8))

    with session_scope() as session:
        image = Image(
            file_path=str(image_path),
            file_hash="hash",
            file_mtime=image_path.stat().st_mtime,
        )
        session.add(image)
        session.flush()
        image_id = image.id

        # A manually drawn face overlapping the dummy detector's first box
        # (10,10,70,70).  detector_backend="manual" and unassigned.
        manual = Face(
            image_id=image_id,
            bbox_x=15, bbox_y=15, bbox_w=70, bbox_h=70,
            confidence=1.0,
            detector_backend="manual",
        )
        session.add(manual)
        session.flush()
        manual_id = manual.id

    monkeypatch.setattr(
        "app.services.detection_service.save_face_crop",
        lambda *a, **k: Path(tmp_path) / "crop.jpg",
    )

    cfg = AppConfig(base_dir=str(tmp_path))
    cfg.storage.db_path = str(db_path)
    cfg.storage.crops_dir = "crops"

    with session_scope() as session:
        service = DetectionService(
            session=session, detector=_DummyDetector(), config=cfg
        )
        service.process([image_id])

    with session_scope() as session:
        faces = session.query(Face).order_by(Face.id).all()
        backends = {f.id: f.detector_backend for f in faces}

    # The manual face survives, and only the non-overlapping second detection
    # (120,15,65,65) becomes a new auto face — no duplicate over the manual box.
    assert manual_id in backends
    assert backends[manual_id] == "manual"
    auto = [f for f in faces if f.detector_backend == "dummy"]
    assert len(auto) == 1, "overlapping detection must be deduped against the manual face"
    assert len(faces) == 2


class _ThresholdSensitiveDetector(FaceDetector):
    """Simulates a faded photo: faces only surface below a confidence floor.

    Returns two low-confidence (0.45) faces when the caller's threshold is
    <= 0.45, otherwise nothing — mirroring how the YuNet/SSD models score
    real faces in old group photos far below the strict 0.65 default.
    """

    @property
    def backend_name(self) -> str:
        return "yunet"

    def detect(
        self,
        image_bgr: np.ndarray,
        confidence_threshold: float = 0.5,
        min_face_size: int = 50,
    ):
        if confidence_threshold > 0.45:
            return []
        return [
            Detection(x=10, y=10, w=40, h=40, confidence=0.45),
            Detection(x=200, y=20, w=40, h=40, confidence=0.45),
        ]


def test_adaptive_escalation_rescues_faded_photo(tmp_path):
    """Strict pass finds nothing; adaptive escalation recovers the faces."""
    cfg = AppConfig(base_dir=str(tmp_path))
    cfg.storage.crops_dir = "crops"
    # Escalation mechanics tested in isolation — the verification gate has
    # its own tests below (the synthetic image holds no verifiable face).
    cfg.detection.verification_enabled = False
    assert cfg.detection.adaptive_escalation is True

    img = np.full((659, 1056, 3), 200, np.uint8)

    with session_scope() as session:
        service = DetectionService(
            session=session, detector=_ThresholdSensitiveDetector(), config=cfg
        )
        # Strict single pass at 0.65 → nothing; escalation kicks in.
        dets = service._run_detection(img)

    assert len(dets) == 2, "adaptive ladder should recover the low-confidence faces"


def test_adaptive_escalation_disabled_stays_strict(tmp_path):
    """With escalation off, a strict miss stays a miss — no relaxation."""
    cfg = AppConfig(base_dir=str(tmp_path))
    cfg.storage.crops_dir = "crops"
    cfg.detection.verification_enabled = False
    cfg.detection.adaptive_escalation = False

    img = np.full((659, 1056, 3), 200, np.uint8)

    with session_scope() as session:
        service = DetectionService(
            session=session, detector=_ThresholdSensitiveDetector(), config=cfg
        )
        dets = service._run_detection(img)

    assert dets == []


def test_adaptive_escalation_skipped_when_strict_succeeds(tmp_path):
    """A detector that finds faces strictly must never trigger escalation."""
    cfg = AppConfig(base_dir=str(tmp_path))
    cfg.storage.crops_dir = "crops"

    img = np.full((659, 1056, 3), 200, np.uint8)

    with session_scope() as session:
        service = DetectionService(
            session=session, detector=_DummyDetector(), config=cfg
        )
        dets = service._run_detection(img)

    # _DummyDetector returns 2 faces at any threshold → strict pass already wins.
    assert len(dets) == 2
