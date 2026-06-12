"""Tests for the content-aware shared thumbnail cache."""

from __future__ import annotations

import os
import time

import pytest
from PySide6.QtGui import QColor, QImage

from app.ui.helpers import thumbnail_cache as tc


@pytest.fixture(autouse=True)
def _clear():
    tc.clear_cache()
    yield
    tc.clear_cache()


def _write_png(path, color="#5b8def"):
    img = QImage(96, 96, QImage.Format_RGB32)
    img.fill(QColor(color))
    img.save(str(path))


def test_missing_path_returns_none(qtbot, tmp_path):
    assert tc.get_thumbnail(None, 96) is None
    assert tc.get_thumbnail(str(tmp_path / "nope.png"), 96) is None


def test_loads_and_caches(qtbot, tmp_path):
    p = tmp_path / "c.png"
    _write_png(p)
    first = tc.get_thumbnail(str(p), 96)
    assert first is not None
    assert tc.cache_size() == 1
    second = tc.get_thumbnail(str(p), 96)
    assert second is first              # served from cache (same instance)
    assert tc.cache_size() == 1


def test_different_size_is_separate_entry(qtbot, tmp_path):
    p = tmp_path / "c.png"
    _write_png(p)
    tc.get_thumbnail(str(p), 96)
    tc.get_thumbnail(str(p), 72)
    assert tc.cache_size() == 2


def test_mtime_change_invalidates(qtbot, tmp_path):
    p = tmp_path / "c.png"
    _write_png(p, "#ff0000")
    a = tc.get_thumbnail(str(p), 96)
    # Rewrite with new content + bump mtime well past the 1ms key precision.
    time.sleep(0.01)
    _write_png(p, "#00ff00")
    os.utime(p, (time.time() + 5, time.time() + 5))
    b = tc.get_thumbnail(str(p), 96)
    assert b is not a                   # reloaded, not the stale pixmap
    # Old entry for the same path+size is dropped, not accumulated.
    assert tc.cache_size() == 1
