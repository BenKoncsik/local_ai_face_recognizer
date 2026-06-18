"""Parametrized smoke tests for app.ui.widgets.* modules."""

from __future__ import annotations

from typing import Callable

import pytest
from PySide6.QtWidgets import QLabel, QWidget

from app.db.models import Place


WidgetFactory = Callable[[], QWidget]

_WIDGET_FACTORIES: dict[str, WidgetFactory] = {}


def _register(module: str, factory: WidgetFactory) -> None:
    _WIDGET_FACTORIES[module] = factory


_register(
    "address_autocomplete_edit",
    lambda: __import__(
        "app.ui.widgets.address_autocomplete_edit",
        fromlist=["AddressAutocompleteEdit"],
    ).AddressAutocompleteEdit(),
)

_register(
    "audio_level_meter",
    lambda: __import__(
        "app.ui.widgets.audio_level_meter",
        fromlist=["AudioLevelMeter"],
    ).AudioLevelMeter("Mic"),
)

_register(
    "collapsible_section",
    lambda: __import__(
        "app.ui.widgets.collapsible_section",
        fromlist=["CollapsibleSection"],
    ).CollapsibleSection("Section", QLabel("content")),
)

_register(
    "face_timeline_view",
    lambda: __import__(
        "app.ui.widgets.face_timeline_view",
        fromlist=["FaceTimelineView"],
    ).FaceTimelineView(),
)

_register(
    "flow_layout",
    lambda: __import__(
        "app.ui.widgets.flow_layout",
        fromlist=["FlowContainer"],
    ).FlowContainer(),
)

_register(
    "group_chip_select",
    lambda: __import__(
        "app.ui.widgets.group_chip_select",
        fromlist=["GroupChipSelect"],
    ).GroupChipSelect(),
)

_register(
    "object_gallery_widget",
    lambda: __import__(
        "app.ui.widgets.object_gallery_widget",
        fromlist=["ObjectGalleryWidget"],
    ).ObjectGalleryWidget(),
)

_register(
    "person_search_select",
    lambda: __import__(
        "app.ui.widgets.person_search_select",
        fromlist=["PersonSearchSelect"],
    ).PersonSearchSelect(),
)

_register(
    "place_gallery_widget",
    lambda: __import__(
        "app.ui.widgets.place_gallery_widget",
        fromlist=["PlaceGalleryWidget"],
    ).PlaceGalleryWidget(),
)

_register(
    "place_map_picker_widget",
    lambda: __import__(
        "app.ui.widgets.place_map_picker_widget",
        fromlist=["PlaceMapPickerWidget"],
    ).PlaceMapPickerWidget(),
)

_register(
    "place_map_widget",
    lambda: __import__(
        "app.ui.widgets.place_map_widget",
        fromlist=["PlaceMapWidget"],
    ).PlaceMapWidget(),
)

_register(
    "place_search_select",
    lambda: __import__(
        "app.ui.widgets.place_search_select",
        fromlist=["PlaceSearchSelect"],
    ).PlaceSearchSelect(),
)

_register(
    "recording_controls",
    lambda: __import__(
        "app.ui.widgets.recording_controls",
        fromlist=["RecordingControls"],
    ).RecordingControls(),
)

_register(
    "selectable_face_grid",
    lambda: __import__(
        "app.ui.widgets.selectable_face_grid",
        fromlist=["SelectableFaceGrid"],
    ).SelectableFaceGrid(),
)

_register(
    "universal_search_bar",
    lambda: __import__(
        "app.ui.widgets.universal_search_bar",
        fromlist=["UniversalSearchBar"],
    ).UniversalSearchBar(placeholder="search"),
)


@pytest.mark.parametrize(
    "module_name",
    sorted(_WIDGET_FACTORIES.keys()),
    ids=sorted(_WIDGET_FACTORIES.keys()),
)
def test_widget_module_smoke(module_name: str, qtbot):
    widget = _WIDGET_FACTORIES[module_name]()
    qtbot.addWidget(widget)
    assert widget is not None


def test_place_search_select_accepts_places(qtbot):
    from app.ui.widgets.place_search_select import PlaceSearchSelect

    widget = PlaceSearchSelect()
    qtbot.addWidget(widget)
    place = Place(name="Balaton")
    place.id = 1
    widget.set_places([place])
    assert widget._entries


def test_universal_search_bar_empty_tokens(qtbot):
    from app.ui.widgets.universal_search_bar import UniversalSearchBar

    bar = UniversalSearchBar()
    qtbot.addWidget(bar)
    assert bar.get_tokens() == []
