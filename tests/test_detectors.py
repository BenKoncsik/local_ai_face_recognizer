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

        config = DetectionConfig(coral_model_path=None, use_yunet=False)
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
            coral_model_path="/nonexistent/model.tflite", use_yunet=False
        )
        detector = factory.create_detector(config)
        assert isinstance(detector, CpuDetector)

    def test_yunet_preferred_when_available(self, monkeypatch):
        """With use_yunet and a loadable model, the factory picks YuNet over the
        Caffe/Haar CPU detector."""
        from app.detectors import factory
        from app.detectors.yunet_detector import YuNetDetector

        config = DetectionConfig(coral_model_path=None, use_yunet=True)
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
