"""Tests for the YuNet landmark-capable CPU detector.

The real ONNX model rarely fires on synthetic test images, so the detection
tests stub ``cv2.FaceDetectorYN``'s raw output and verify that the wrapper
parses bbox + 5 landmarks + confidence correctly.  A separate test exercises
the real model when it is present on disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.detectors.base import Detection
from app.detectors.yunet_detector import _DEFAULT_MODEL, YuNetDetector
from app.paths import resource_path


def _fake_row(x, y, w, h, landmarks, score):
    """Build one YuNet output row: bbox(4) + 5 landmarks(10) + score(1)."""
    flat = [x, y, w, h]
    for px, py in landmarks:
        flat.extend([px, py])
    flat.append(score)
    return np.array(flat, dtype=np.float32)


class _FakeNet:
    """Stand-in for cv2.FaceDetectorYN with a scripted detect() result."""

    def __init__(self, faces):
        self._faces = faces
        self.input_size = None
        self.score_threshold = None
        self.detect_shape = None

    def setInputSize(self, size):  # noqa: N802 (OpenCV API name)
        self.input_size = size

    def setScoreThreshold(self, t):  # noqa: N802
        self.score_threshold = t

    def detect(self, image):
        self.detect_shape = image.shape
        return 1, self._faces


def _make_detector_with_net(monkeypatch, faces):
    """Construct a YuNetDetector whose model load and net are stubbed out."""
    monkeypatch.setattr(
        YuNetDetector, "_resolve_model_path", staticmethod(lambda *_: Path("fake.onnx"))
    )
    monkeypatch.setattr(
        "cv2.FaceDetectorYN.create", lambda *a, **k: _FakeNet(faces)
    )
    return YuNetDetector()


class TestYuNetParsing:
    def test_missing_model_raises(self, monkeypatch):
        monkeypatch.setattr(
            YuNetDetector, "_resolve_model_path", staticmethod(lambda *_: None)
        )
        with pytest.raises(FileNotFoundError):
            YuNetDetector()

    def test_backend_name(self, monkeypatch):
        det = _make_detector_with_net(monkeypatch, None)
        assert det.backend_name == "yunet"

    def test_none_result_returns_empty(self, monkeypatch):
        det = _make_detector_with_net(monkeypatch, None)
        assert det.detect(np.zeros((100, 100, 3), np.uint8)) == []

    def test_landmarks_parsed_in_arcface_order(self, monkeypatch):
        lms = [[10, 20], [30, 22], [20, 35], [12, 50], [28, 51]]
        faces = np.stack([_fake_row(5, 6, 80, 80, lms, 0.9)])
        det = _make_detector_with_net(monkeypatch, faces)

        out = det.detect(np.zeros((200, 200, 3), np.uint8), min_face_size=10)
        assert len(out) == 1
        d = out[0]
        assert (d.x, d.y, d.w, d.h) == (5, 6, 80, 80)
        assert d.confidence == pytest.approx(0.9)
        assert d.landmarks is not None
        assert np.allclose(np.array(d.landmarks), np.array(lms, dtype=float))

    def test_input_size_and_threshold_pushed_to_net(self, monkeypatch):
        det = _make_detector_with_net(monkeypatch, None)
        det.detect(np.zeros((120, 160, 3), np.uint8), confidence_threshold=0.4)
        assert det._net.input_size == (160, 128)  # (w, h), padded to YuNet stride
        assert det._net.detect_shape == (128, 160, 3)
        assert det._net.score_threshold == pytest.approx(0.4)

    def test_small_faces_filtered(self, monkeypatch):
        lms = [[1, 1], [2, 1], [1, 2], [1, 3], [2, 3]]
        faces = np.stack([_fake_row(0, 0, 5, 5, lms, 0.99)])
        det = _make_detector_with_net(monkeypatch, faces)
        assert det.detect(np.zeros((50, 50, 3), np.uint8), min_face_size=50) == []


@pytest.mark.skipif(
    not resource_path(_DEFAULT_MODEL).exists(),
    reason="YuNet ONNX model not present in models/",
)
class TestYuNetRealModel:
    def test_loads_and_runs(self):
        det = YuNetDetector()
        assert det.backend_name == "yunet"
        # Blank image → no faces, but the call must succeed and return a list.
        out = det.detect(np.full((240, 320, 3), 127, np.uint8))
        assert isinstance(out, list)
        for d in out:
            assert isinstance(d, Detection)
            assert d.landmarks is not None
            assert np.array(d.landmarks).shape == (5, 2)
