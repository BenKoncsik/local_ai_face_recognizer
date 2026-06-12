"""Tests for lazy (viewport-windowed) crop loading in ClusterPanel."""

from __future__ import annotations

import pytest

from app.ui.panels.cluster_panel import ClusterPanel


class _Loader:
    """Stand-in tile that records whether its crop was decoded."""

    def __init__(self) -> None:
        self.loaded = False

    def ensure_loaded(self) -> None:
        self.loaded = True


@pytest.fixture
def panel(qtbot):
    p = ClusterPanel()
    qtbot.addWidget(p)
    p.resize(420, 300)
    p.show()
    qtbot.waitExposed(p)
    return p


def test_only_visible_rows_load(panel):
    panel._cur_cols = 4
    loaders = [_Loader() for _ in range(200)]
    panel._ordered_thumbs = loaders
    panel._scroll.verticalScrollBar().setValue(0)
    panel._load_visible_thumbs()

    # The very first tile is on screen → loaded.
    assert loaders[0].loaded is True
    # Far-down tiles are well outside the viewport + buffer → still lazy.
    assert loaders[-1].loaded is False
    assert not all(ldr.loaded for ldr in loaders)


def test_widgets_without_loader_are_skipped(panel):
    # A mix of loaders and plain objects must not raise.
    panel._cur_cols = 4
    panel._ordered_thumbs = [_Loader(), object(), _Loader()]
    panel._load_visible_thumbs()        # should not raise
    assert panel._ordered_thumbs[0].loaded is True
