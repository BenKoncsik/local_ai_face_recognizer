"""Tests for Coral Edge TPU detector helpers and optional hardware."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.detectors.coral_detector import (
    CoralDetector,
    _find_edgetpu_lib,
    _get_detections,
    _set_input,
)
from app.detectors.base import Detection


def _fake_interpreter(boxes, scores, count):
    interp = MagicMock()
    inp = {"index": 0, "shape": [1, 320, 320, 3]}
    out_boxes = {"index": 0}
    out_scores = {"index": 2}
    out_count = {"index": 3}
    interp.get_input_details.return_value = [inp]
    interp.get_output_details.return_value = [
        out_boxes,
        {"index": 1},
        out_scores,
        out_count,
    ]

    tensors = {
        0: np.expand_dims(boxes, axis=0),
        2: np.expand_dims(scores, axis=0),
        3: count,
    }

    def _get_tensor(idx):
        return tensors[idx]

    interp.get_tensor.side_effect = _get_tensor
    interp.set_tensor = MagicMock()
    return interp


class TestPureHelpers:
    def test_find_edgetpu_lib_returns_string(self):
        path = _find_edgetpu_lib()
        assert isinstance(path, str)
        assert path

    def test_set_input_expands_batch_dim(self):
        interp = MagicMock()
        interp.get_input_details.return_value = [{"index": 3}]
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        _set_input(interp, rgb)
        interp.set_tensor.assert_called_once()
        tensor = interp.set_tensor.call_args[0][1]
        assert tensor.shape == (1, 100, 100, 3)

    def test_get_detections_parses_ssd_output(self):
        boxes = np.array([[0.1, 0.2, 0.5, 0.6]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        count = np.array([1], dtype=np.int32)
        interp = _fake_interpreter(boxes, scores, count)
        dets = _get_detections(interp, img_w=1000, img_h=800, score_threshold=0.5)
        assert len(dets) == 1
        x, y, w, h, conf = dets[0]
        assert conf == pytest.approx(0.9)
        assert x == 200
        assert y == 80
        assert w == 400
        assert h == 320

    def test_get_detections_filters_low_scores(self):
        boxes = np.array([[0.0, 0.0, 0.5, 0.5]], dtype=np.float32)
        scores = np.array([0.2], dtype=np.float32)
        count = np.array([1], dtype=np.int32)
        interp = _fake_interpreter(boxes, scores, count)
        assert _get_detections(interp, 100, 100, score_threshold=0.5) == []


def _coral_available() -> bool:
    try:
        from app.detectors.coral_detector import _make_interpreter  # noqa: PLC0415
        from app.paths import resource_path  # noqa: PLC0415

        model = resource_path("models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite")
        if not model.exists():
            return False
        interp, _ = _make_interpreter(str(model))
        del interp
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _coral_available(), reason="Coral Edge TPU unavailable")
class TestCoralDetectorLive:
    def test_detect_on_blank_image(self):
        from app.paths import resource_path

        model = resource_path(
            "models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite"
        )
        detector = CoralDetector(str(model))
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect(img, confidence_threshold=0.99)
        assert isinstance(results, list)
        assert all(isinstance(d, Detection) for d in results)

    def test_missing_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Coral model file not found"):
            CoralDetector(str(tmp_path / "missing.tflite"))
