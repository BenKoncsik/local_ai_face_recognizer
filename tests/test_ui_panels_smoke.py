"""Parametrized smoke tests for untested app.ui.panels.* modules."""

from __future__ import annotations

from typing import Callable

import pytest
from PySide6.QtWidgets import QWidget

PanelFactory = Callable[[], QWidget]

_PANEL_FACTORIES: dict[str, PanelFactory] = {}


def _register(module: str, factory: PanelFactory) -> None:
    _PANEL_FACTORIES[module] = factory


_register(
    "collage_panel",
    lambda: __import__(
        "app.ui.panels.collage_panel",
        fromlist=["CollagePanel"],
    ).CollagePanel(),
)

_register(
    "family_search_panel",
    lambda: __import__(
        "app.ui.panels.family_search_panel",
        fromlist=["FamilySearchPanel"],
    ).FamilySearchPanel(),
)

_register(
    "groups_panel",
    lambda: __import__(
        "app.ui.panels.groups_panel",
        fromlist=["GroupsPanel"],
    ).GroupsPanel(),
)

_register(
    "locations_panel",
    lambda: __import__(
        "app.ui.panels.locations_panel",
        fromlist=["LocationsPanel"],
    ).LocationsPanel(),
)

_register(
    "log_panel",
    lambda: __import__(
        "app.ui.panels.log_panel",
        fromlist=["LogPanel"],
    ).LogPanel(),
)

_register(
    "objects_panel",
    lambda: __import__(
        "app.ui.panels.objects_panel",
        fromlist=["ObjectsPanel"],
    ).ObjectsPanel(),
)

_register(
    "preview_panel",
    lambda: __import__(
        "app.ui.panels.preview_panel",
        fromlist=["PreviewPanel"],
    ).PreviewPanel(),
)

_register(
    "sidebar_panel",
    lambda: __import__(
        "app.ui.panels.sidebar_panel",
        fromlist=["SidebarPanel"],
    ).SidebarPanel(),
)


@pytest.mark.parametrize(
    "module_name",
    sorted(_PANEL_FACTORIES.keys()),
    ids=sorted(_PANEL_FACTORIES.keys()),
)
def test_panel_module_smoke(module_name: str, qtbot):
    panel = _PANEL_FACTORIES[module_name]()
    qtbot.addWidget(panel)
    assert panel is not None


def test_preview_panel_accepts_config(qtbot):
    panel = __import__(
        "app.ui.panels.preview_panel",
        fromlist=["PreviewPanel"],
    ).PreviewPanel()
    qtbot.addWidget(panel)
    assert panel._current_image_path is None


def test_objects_panel_imports_cleanly():
    mod = __import__("app.ui.panels.objects_panel", fromlist=["ObjectsPanel"])
    assert mod.ObjectsPanel is not None


def test_collage_panel_has_empty_state(qtbot):
    from app.ui.panels.collage_panel import CollagePanel

    panel = CollagePanel()
    qtbot.addWidget(panel)
    assert panel._current_collage_id is None
