"""Unit tests for detector interface and factory."""

from __future__ import annotations

import numpy as np
import pytest

from app.config import DetectionConfig
from app.detectors.base import Detection, FaceDetector


class TestDetectionDataclass:
    def test_clamp_within_bounds(self):
        det = Detection(x=-5, y=-5, w=200, h=200, confidence=0.9)
        clamped = det.clamp(100, 100)
        assert clamped.x == 0
        assert clamped.y == 0
        assert clamped.x2 == 100
        assert clamped.y2 == 100

    def test_as_tuple(self):
        det = Detection(x=10, y=20, w=30, h=40, confidence=0.8)
        assert det.as_tuple() == (10, 20, 30, 40)

    def test_x2_y2(self):
        det = Detection(x=5, y=10, w=50, h=60, confidence=0.7)
        assert det.x2 == 55
        assert det.y2 == 70


class TestFactoryFallback:
    """Verify the factory returns a CpuDetector when Coral is unavailable."""

    def test_cpu_fallback_when_no_coral_config(self, monkeypatch):
        """With no coral_model_path configured, factory must return CpuDetector.

        use_yunet is disabled so we exercise the Caffe/Haar CPU path explicitly.
        """
        from app.detectors.cpu_detector import CpuDetector
        from app.detectors.factory import create_detector

        config = DetectionConfig(
            coral_model_path=None, use_yunet=False, multistage_use_insightface=False
        )
        detector = create_detector(config)

        # Should be a CPU detector (possibly Haar if model files absent)
        assert isinstance(detector, CpuDetector)

    def test_cpu_fallback_when_coral_probe_fails(self, monkeypatch):
        """Even with coral_model_path set, if probe fails, use CpuDetector."""
        from app.detectors.cpu_detector import CpuDetector
        from app.detectors import factory

        # Monkeypatch probe_coral to return False
        monkeypatch.setattr(factory, "probe_coral", lambda *args, **kwargs: False)

        config = DetectionConfig(
            coral_model_path="/nonexistent/model.tflite", use_yunet=False,
            multistage_use_insightface=False,
        )
        detector = factory.create_detector(config)
        assert isinstance(detector, CpuDetector)

    def test_yunet_preferred_when_available(self, monkeypatch):
        """With use_yunet and a loadable model, the factory picks YuNet over the
        Caffe/Haar CPU detector."""
        from app.detectors import factory
        from app.detectors.yunet_detector import YuNetDetector

        config = DetectionConfig(
            coral_model_path=None, use_yunet=True, multistage_use_insightface=False
        )
        detector = factory.create_detector(config)
        # Falls back to CpuDetector only if the YuNet model is missing.
        from app.paths import resource_path
        from app.detectors.yunet_detector import _DEFAULT_MODEL

        if resource_path(_DEFAULT_MODEL).exists():
            assert isinstance(detector, YuNetDetector)
            assert detector.backend_name == "yunet"


class _DummyDetector(FaceDetector):
    """Concrete subclass for interface contract tests."""

    @property
    def backend_name(self) -> str:
        return "dummy"

    def detect(self, image_bgr, confidence_threshold=0.5):
        h, w = image_bgr.shape[:2]
        return [Detection(x=0, y=0, w=w // 2, h=h // 2, confidence=0.99)]


class TestFaceDetectorInterface:
    def test_detect_returns_list(self):
        detector = _DummyDetector()
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        results = detector.detect(img)
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0].confidence == 0.99

    def test_repr(self):
        detector = _DummyDetector()
        assert "dummy" in repr(detector)


class _FixedDetector(FaceDetector):
    """Returns a fixed detection list regardless of input."""

    def __init__(self, name, dets):
        self._name = name
        self._dets = dets

    @property
    def backend_name(self):
        return self._name

    def detect(self, image_bgr, confidence_threshold=0.5, min_face_size=50):
        return list(self._dets)


class TestCompositeDetector:
    """The composite keeps every primary box and adds only *new* secondary ones."""

    def _img(self):
        return np.zeros((400, 400, 3), dtype=np.uint8)

    def test_secondary_adds_non_overlapping_face(self):
        from app.detectors.composite_detector import CompositeDetector

        primary = _FixedDetector("yunet", [Detection(10, 10, 50, 50, 0.9)])
        secondary = _FixedDetector(  # one overlapping, one brand-new (profile)
            "insightface_scrfd",
            [Detection(12, 12, 48, 48, 0.8), Detection(300, 300, 60, 60, 0.7)],
        )
        comp = CompositeDetector(primary, [secondary])
        out = comp.detect(self._img())

        # Primary box kept; overlapping secondary dropped; new one added.
        assert len(out) == 2
        assert any(d.x == 10 for d in out)          # primary survives
        assert any(d.x == 300 for d in out)         # recovered profile face
        assert "yunet" in comp.backend_name and "insightface" in comp.backend_name

    def test_collapses_primary_duplicate_via_anchor(self):
        # The primary fires two offset boxes on one face (mutual IoU ~0.27); a
        # secondary box sees it as one → the lower-confidence duplicate is dropped.
        from app.detectors.composite_detector import CompositeDetector

        primary = _FixedDetector("yunet", [
            Detection(100, 100, 120, 120, 0.80),
            Detection(150, 150, 120, 120, 0.70),  # offset duplicate of the same face
        ])
        secondary = _FixedDetector("insightface_scrfd", [Detection(110, 110, 150, 150, 0.85)])
        out = CompositeDetector(primary, [secondary]).detect(self._img())

        # One face: the higher-confidence primary box survives, the duplicate and
        # the overlapping anchor are not added.
        assert len(out) == 1
        assert out[0].confidence == 0.80

    def test_secondary_with_center_inside_primary_not_added(self):
        # A secondary box much larger/looser than the primary (IoU below the add
        # threshold but its centre inside the primary) is the same face — not added.
        from app.detectors.composite_detector import CompositeDetector

        primary = _FixedDetector("yunet", [Detection(200, 200, 80, 80, 0.8)])
        secondary = _FixedDetector(
            "insightface_scrfd", [Detection(120, 120, 260, 260, 0.78)]  # center (250,250) in primary
        )
        out = CompositeDetector(primary, [secondary]).detect(self._img())
        assert len(out) == 1
        assert (out[0].x, out[0].y) == (200, 200)

    def test_get_reaches_member_detector(self):
        from app.detectors.composite_detector import CompositeDetector

        primary = _FixedDetector("yunet", [])
        secondary = _FixedDetector("insightface_scrfd", [])
        comp = CompositeDetector(primary, [secondary])
        assert comp.get("insightface") is secondary
        assert comp.primary is primary
        assert comp.get("nope") is None
