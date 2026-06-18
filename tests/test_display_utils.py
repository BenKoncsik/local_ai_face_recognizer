"""Tests for display enumeration and active-window bounds."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QRect
from PySide6.QtWidgets import QApplication, QWidget

from app.services.screen_recorder_service import RecordingDisplayInfo
from app.ui.display_utils import active_window_bounds, enumerate_displays


class TestEnumerateDisplays:
    def test_returns_recording_display_info_list(self, qapp):
        displays = enumerate_displays()
        assert isinstance(displays, list)
        if displays:
            d = displays[0]
            assert isinstance(d, RecordingDisplayInfo)
            assert d.width > 0 and d.height > 0
            assert d.av_index is None  # filled later by probe_screen_indices

    def test_survives_broken_qgui(self, monkeypatch):
        def _boom():
            raise RuntimeError("no display")

        monkeypatch.setattr(
            "PySide6.QtGui.QGuiApplication.screens", staticmethod(_boom)
        )
        assert enumerate_displays() == []


class TestActiveWindowBounds:
    def test_returns_geometry_for_top_level_window(self, qapp, qtbot):
        win = QWidget()
        win.setGeometry(QRect(50, 40, 300, 200))
        win.show()
        qtbot.addWidget(win)
        qtbot.waitExposed(win)

        bounds = active_window_bounds(win)
        assert bounds is not None
        x, y, w, h = bounds
        assert w == 300
        assert h == 200
        assert x >= 0 and y >= 0

    def test_none_for_missing_widget(self):
        assert active_window_bounds(None) is None

    def test_none_on_failure(self, monkeypatch):
        class _BrokenWidget:
            def window(self):
                raise RuntimeError("no window")

        assert active_window_bounds(_BrokenWidget()) is None
