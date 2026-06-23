"""Unit tests for the crop-level detection verification gate."""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from app.config import AppConfig, load_config
from app.db.database import init_db, session_scope
from app.db.models import Face, Image
from app.detectors.base import Detection, FaceDetector
from app.services.detection_service import DetectionService
from app.services.face_verification_service import FaceVerifier


class _CropEchoDetector(FaceDetector):
    """Fake verifier backend: optionally re-finds a face centered in any crop.

    ``shrink`` scales the returned box relative to half the crop size, so the
    oversized-box refinement path can be exercised.
    """

    def __init__(
        self,
        find: bool = True,
        confidence: float = 0.9,
        shrink: float = 1.0,
        landmarks: bool = False,
    ) -> None:
        self._find = find
        self._confidence = confidence
        self._shrink = shrink
        self._landmarks = landmarks

    @property
    def backend_name(self) -> str:
        return "fake_verifier"

    def detect(
        self,
        image_bgr: np.ndarray,
        confidence_threshold: float = 0.5,
        min_face_size: int = 50,
    ) -> List[Detection]:
        if not self._find or self._confidence < confidence_threshold:
            return []
        h, w = image_bgr.shape[:2]
        fw = max(2, int(w * 0.5 * self._shrink))
        fh = max(2, int(h * 0.5 * self._shrink))
        x, y = (w - fw) // 2, (h - fh) // 2
        lm = None
        if self._landmarks:
            lm = [
                [x + fw * 0.3, y + fh * 0.4],
                [x + fw * 0.7, y + fh * 0.4],
                [x + fw * 0.5, y + fh * 0.6],
                [x + fw * 0.35, y + fh * 0.8],
                [x + fw * 0.65, y + fh * 0.8],
            ]
        return [Detection(x=x, y=y, w=fw, h=fh, confidence=self._confidence, landmarks=lm)]


def _img(w: int = 640, h: int = 480) -> np.ndarray:
    return np.full((h, w, 3), 128, np.uint8)


def test_verify_rejects_when_no_face_refound():
    cfg = AppConfig().detection
    verifier = FaceVerifier(cfg, detector=_CropEchoDetector(find=False))
    det = Detection(x=100, y=100, w=60, h=60, confidence=0.4)
    assert verifier.verify(_img(), det) is None


def test_verify_confirms_overlapping_face():
    cfg = AppConfig().detection
    verifier = FaceVerifier(cfg, detector=_CropEchoDetector(find=True))
    det = Detection(x=100, y=100, w=60, h=60, confidence=0.4)
    out = verifier.verify(_img(), det)
    assert out is not None
    # Box well-fitted → geometry kept, confidence lifted to the verified score.
    assert (out.x, out.y, out.w, out.h) == (100, 100, 60, 60)
    assert out.confidence == 0.9


def test_verify_refines_oversized_box():
    cfg = AppConfig().detection
    verifier = FaceVerifier(cfg, detector=_CropEchoDetector(find=True, shrink=0.6))
    det = Detection(x=100, y=100, w=100, h=100, confidence=0.4)
    out = verifier.verify(_img(), det)
    assert out is not None
    # Verified box area is well under half the original → tighter box adopted.
    assert out.w * out.h <= 0.5 * det.w * det.h
    # And it stays at the candidate location (center inside the original box).
    assert det.x <= out.x + out.w / 2 <= det.x + det.w
    assert det.y <= out.y + out.h / 2 <= det.y + det.h


def test_verify_adopts_landmarks_for_landmark_less_detection():
    cfg = AppConfig().detection
    verifier = FaceVerifier(
        cfg, detector=_CropEchoDetector(find=True, shrink=0.6, landmarks=True)
    )
    det = Detection(x=100, y=100, w=100, h=100, confidence=0.4)
    out = verifier.verify(_img(), det)
    assert out is not None
    assert out.landmarks is not None and len(out.landmarks) == 5
    # Landmarks are mapped back into image coordinates (inside the crop region).
    for px, py in out.landmarks:
        assert 0 <= px <= 640 and 0 <= py <= 480


class _BadLandmarkDetector(FaceDetector):
    """Re-finds a centered, overlapping box but with non-face (collinear) points."""

    @property
    def backend_name(self) -> str:
        return "bad_lm"

    def detect(
        self,
        image_bgr: np.ndarray,
        confidence_threshold: float = 0.5,
        min_face_size: int = 50,
    ) -> List[Detection]:
        h, w = image_bgr.shape[:2]
        fw, fh = int(w * 0.5), int(h * 0.5)
        x, y = (w - fw) // 2, (h - fh) // 2
        cx = x + fw / 2.0
        lm = [[cx, y + fh * f] for f in (0.1, 0.3, 0.5, 0.7, 0.9)]
        return [Detection(x=x, y=y, w=fw, h=fh, confidence=0.9, landmarks=lm)]


def test_verify_rejects_geometrically_implausible_crop():
    cfg = AppConfig().detection
    verifier = FaceVerifier(cfg, detector=_BadLandmarkDetector())
    det = Detection(x=100, y=100, w=60, h=60, confidence=0.4)
    # The crop re-detection overlaps the candidate but its landmarks are not a
    # face — the geometry gate must reject it despite the box overlap.
    assert verifier.verify(_img(), det) is None


def test_verify_geometry_gate_honours_disable_flag():
    """landmark_geometry_enabled=False must skip the verifier's geometry check,
    so the same implausible-but-overlapping crop is now CONFIRMED."""
    cfg = AppConfig().detection
    cfg.landmark_geometry_enabled = False
    verifier = FaceVerifier(cfg, detector=_BadLandmarkDetector())
    det = Detection(x=100, y=100, w=60, h=60, confidence=0.4)
    assert verifier.verify(_img(), det) is not None


def test_verifier_unavailable_passes_through(monkeypatch):
    from app.detectors import yunet_detector

    monkeypatch.setattr(
        yunet_detector.YuNetDetector,
        "_resolve_model_path",
        staticmethod(lambda _path: None),
    )
    cfg = AppConfig().detection
    verifier = FaceVerifier(cfg, detector=None)
    assert verifier.available is False
    det = Detection(x=10, y=10, w=50, h=50, confidence=0.4)
    assert verifier.verify(_img(), det) is det


class _JunkDetector(FaceDetector):
    """Primary detector that always reports one low-confidence 'face'."""

    @property
    def backend_name(self) -> str:
        return "junk"

    def detect(
        self,
        image_bgr: np.ndarray,
        confidence_threshold: float = 0.5,
        min_face_size: int = 50,
    ) -> List[Detection]:
        return [Detection(x=50, y=50, w=60, h=60, confidence=0.45)]


class _ConfidentDetector(FaceDetector):
    @property
    def backend_name(self) -> str:
        return "confident"

    def detect(
        self,
        image_bgr: np.ndarray,
        confidence_threshold: float = 0.5,
        min_face_size: int = 50,
    ) -> List[Detection]:
        return [Detection(x=50, y=50, w=60, h=60, confidence=0.95)]


def _service(tmp_path, detector, find: bool) -> DetectionService:
    cfg = AppConfig(base_dir=str(tmp_path))
    cfg.storage.crops_dir = "crops"
    verifier = FaceVerifier(cfg.detection, detector=_CropEchoDetector(find=find))
    init_db(tmp_path / "faces.db")
    session = session_scope().__enter__()
    return DetectionService(
        session=session, detector=detector, config=cfg, verifier=verifier
    )


def test_detection_service_drops_unverified_low_confidence(tmp_path):
    svc = _service(tmp_path, _JunkDetector(), find=False)
    dets = svc._run_detection(_img())
    assert dets == [], "low-confidence detection must die when not re-found in its crop"


def test_detection_service_keeps_verified_low_confidence(tmp_path):
    svc = _service(tmp_path, _JunkDetector(), find=True)
    dets = svc._run_detection(_img())
    assert len(dets) == 1
    assert dets[0].confidence == 0.9  # lifted to the verified crop-level score


def test_detection_service_exempts_high_confidence(tmp_path):
    # Verifier finds nothing, but a 0.95 detection is trusted without it.
    svc = _service(tmp_path, _ConfidentDetector(), find=False)
    dets = svc._run_detection(_img())
    assert len(dets) == 1
    assert dets[0].confidence == 0.95


def test_config_loads_verification_fields(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "detection:\n"
        "  verification_enabled: false\n"
        "  verification_threshold: 0.7\n"
        "  verification_confidence_exempt: 0.9\n",
        encoding="utf-8",
    )
    cfg = load_config(str(p))
    assert cfg.detection.verification_enabled is False
    assert cfg.detection.verification_threshold == 0.7
    assert cfg.detection.verification_confidence_exempt == 0.9
    # Untouched fields keep their defaults.
    assert cfg.detection.verification_min_crop == 160
