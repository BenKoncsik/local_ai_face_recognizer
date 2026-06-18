"""Tests for ThumbnailRunnable."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PySide6.QtGui import QImage

from app.workers.thumbnail_worker import ThumbnailRunnable


def _rgb_image(w: int = 200, h: int = 100) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestThumbnailRunnable:
    def test_emits_ready_with_scaled_image(self, qtbot, tmp_path):
        img_path = tmp_path / "photo.jpg"
        img_path.write_bytes(b"fake")
        bgr = _rgb_image(400, 200)

        with patch(
            "app.utils.image_utils.load_image_bgr_normalized",
            return_value=bgr,
        ):
            worker = ThumbnailRunnable(str(img_path), cache_key="k1", size=100)
            with qtbot.waitSignal(worker.signals.ready, timeout=3000) as blocker:
                worker.run()

        key, qimg = blocker.args
        assert key == "k1"
        assert isinstance(qimg, QImage)
        assert max(qimg.width(), qimg.height()) == 100

    def test_failed_when_image_missing(self, qtbot, tmp_path):
        with patch(
            "app.utils.image_utils.load_image_bgr_normalized",
            return_value=None,
        ):
            worker = ThumbnailRunnable(str(tmp_path / "missing.jpg"), "k2")
            with qtbot.waitSignal(worker.signals.failed, timeout=3000) as blocker:
                worker.run()
        assert list(blocker.args) == ["k2"]

    def test_failed_on_zero_dimensions(self, qtbot, tmp_path):
        bgr = np.zeros((0, 100, 3), dtype=np.uint8)
        with patch(
            "app.utils.image_utils.load_image_bgr_normalized",
            return_value=bgr,
        ):
            worker = ThumbnailRunnable("/x.jpg", "k3")
            with qtbot.waitSignal(worker.signals.failed, timeout=3000) as blocker:
                worker.run()
        assert list(blocker.args) == ["k3"]

    def test_failed_on_exception(self, qtbot):
        with patch(
            "app.utils.image_utils.load_image_bgr_normalized",
            side_effect=OSError("read error"),
        ):
            worker = ThumbnailRunnable("/bad.jpg", "k4")
            with qtbot.waitSignal(worker.signals.failed, timeout=3000) as blocker:
                worker.run()
        assert list(blocker.args) == ["k4"]
