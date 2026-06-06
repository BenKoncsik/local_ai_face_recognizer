"""Create or edit a place with a structured address and an interactive map.

Left pane: a form (display name, settlement/street autocomplete, house number,
type, coordinate source, lat/lon, accuracy radius, parent). Right pane: a
draggable Leaflet map (PlaceMapPickerWidget). The two stay in sync — choosing a
settlement/street/address moves the marker, and dragging/clicking the marker
sets a manual coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from PySide6.QtCore import Qt, QThreadPool, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.db.models import Place
from app.services.place_service import (
    ANONYMOUS_GPS_PLACE_NAME,
    COORD_SOURCE_GEOCODED_ADDRESS,
    COORD_SOURCE_GEOCODED_SETTLEMENT,
    COORD_SOURCE_GEOCODED_STREET,
    COORD_SOURCE_MANUAL_MAP_PIN,
    DEFAULT_PLACE_TYPE,
    PLACE_TYPE_AREA,
    PLACE_TYPE_EXACT,
    PLACE_TYPE_REGION,
    build_display_name,
    classify_address,
    default_radius_for,
    normalize_place_type,
)
from app.ui.i18n import t
from app.ui.widgets.address_autocomplete_edit import AddressAutocompleteEdit
from app.ui.widgets.place_map_picker_widget import PlaceMapPickerWidget
from app.workers.geocoding_worker import GeocodeWorker

# (place_type, i18n-key) in display order.
_TYPE_CHOICES: Tuple[Tuple[str, str], ...] = (
    (PLACE_TYPE_EXACT, "places_type_exact"),
    (PLACE_TYPE_AREA, "places_type_area"),
    (PLACE_TYPE_REGION, "places_type_region"),
)

_HOUSE_DEBOUNCE_MS = 450


@dataclass(frozen=True)
class PlaceEditResult:
    name: str
    place_type: str
    latitude: Optional[float]
    longitude: Optional[float]
    accuracy_radius_meters: Optional[float]  # None → service uses the type default
    parent_id: Optional[int]
    settlement_name: Optional[str] = None
    street_name: Optional[str] = None
    house_number: Optional[str] = None
    display_name: Optional[str] = None
    coordinate_source: Optional[str] = None
    is_exact_coordinate: Optional[bool] = None


class PlaceEditDialog(QDialog):
    """Edit an existing place or define a new one (structured address + map)."""

    def __init__(
        self,
        place: Optional[Place] = None,
        parents: Optional[List[Place]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._place = place
        self._parents = parents or []
        self._coord_source: Optional[str] = None
        self._is_exact: Optional[bool] = None
        self._name_overridden = False
        self._loading = True
        self.setWindowTitle(
            t("place_edit_title_edit") if place else t("place_edit_title_new")
        )
        self.resize(900, 560)
        self._build_ui()
        self._load_from_place()
        self._loading = False

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)

        self._splitter = QSplitter(Qt.Horizontal)

        form_holder = QWidget()
        outer = QVBoxLayout(form_holder)
        form = QFormLayout()

        self._name = QLineEdit()
        self._name.textEdited.connect(self._on_name_edited)
        form.addRow(t("place_edit_name"), self._name)

        self._settlement = AddressAutocompleteEdit("settlement")
        self._settlement.suggestion_chosen.connect(self._on_settlement_chosen)
        self._settlement.edited_text.connect(self._on_settlement_text)
        form.addRow(t("place_edit_settlement"), self._settlement)

        self._street = AddressAutocompleteEdit("street")
        self._street.suggestion_chosen.connect(self._on_street_chosen)
        self._street.edited_text.connect(self._on_street_text)
        form.addRow(t("place_edit_street"), self._street)

        self._house = QLineEdit()
        self._house.textEdited.connect(self._on_house_edited)
        form.addRow(t("place_edit_house"), self._house)

        self._type = QComboBox()
        for value, key in _TYPE_CHOICES:
            self._type.addItem(t(key), value)
        self._type.currentIndexChanged.connect(self._on_type_changed)
        form.addRow(t("place_edit_type"), self._type)

        self._source_lbl = QLabel("—")
        form.addRow(t("place_edit_coord_source"), self._source_lbl)

        self._lat = QLineEdit()
        self._lat.setPlaceholderText("46.0727")
        self._lat.editingFinished.connect(self._on_coord_edited)
        form.addRow(t("place_edit_lat"), self._lat)
        self._lon = QLineEdit()
        self._lon.setPlaceholderText("18.2323")
        self._lon.editingFinished.connect(self._on_coord_edited)
        form.addRow(t("place_edit_lon"), self._lon)

        self._radius_auto = QCheckBox(t("place_edit_radius_auto"))
        self._radius_auto.setChecked(True)
        self._radius_auto.toggled.connect(self._on_radius_auto_toggled)
        form.addRow("", self._radius_auto)

        self._radius = QDoubleSpinBox()
        self._radius.setRange(1.0, 1_000_000.0)
        self._radius.setDecimals(0)
        self._radius.setSingleStep(50.0)
        self._radius.valueChanged.connect(self._on_radius_value_changed)
        form.addRow(t("place_edit_radius"), self._radius)

        self._parent = QComboBox()
        self._parent.addItem(t("places_parent_none"), None)
        for cand in self._parents:
            label = cand.display_name or cand.name
            if cand.is_anonymous and cand.source == "exif":
                label = f"{ANONYMOUS_GPS_PLACE_NAME} #{cand.id}"
            self._parent.addItem(label, cand.id)
        form.addRow(t("place_edit_parent"), self._parent)

        self._hint = QLabel("")
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet("color:#F9E2AF;font-size:0.85em;")

        outer.addLayout(form)
        outer.addWidget(self._hint)
        outer.addStretch()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        self._map = PlaceMapPickerWidget()
        self._map.coordinate_picked.connect(self._on_map_pick)
        self._map.setMinimumWidth(420)

        self._splitter.addWidget(form_holder)
        self._splitter.addWidget(self._map)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        # Start roughly half/half; the map keeps a usable minimum width.
        self._splitter.setSizes([440, 460])
        root.addWidget(self._splitter)

        self._settlement.set_placeholder(t("place_edit_settlement_ph"))
        self._street.set_placeholder(t("place_edit_street_ph"))

        # House-number geocode debounce.
        self._house_timer = QTimer(self)
        self._house_timer.setSingleShot(True)
        self._house_timer.setInterval(_HOUSE_DEBOUNCE_MS)
        self._house_timer.timeout.connect(self._geocode_full_address)
        self._geocode_req = 0

    def _load_from_place(self) -> None:
        place = self._place
        init_lat, init_lon, zoom = 47.0, 19.0, 7
        if place is None:
            idx = self._type.findData(DEFAULT_PLACE_TYPE)
            if idx >= 0:
                self._type.setCurrentIndex(idx)
            self._sync_radius_default()
        else:
            if place.settlement_name:
                self._settlement.set_text(place.settlement_name)
                self._street.set_settlement_context(place.settlement_name)
            if place.street_name:
                self._street.set_text(place.street_name)
            if place.house_number:
                self._house.setText(place.house_number)
            display = place.display_name or place.name
            if place.is_anonymous and place.source == "exif":
                display = ""
            self._name.setText(display)
            if place.display_name and place.display_name != place.name:
                self._name_overridden = place.name != place.display_name
            idx = self._type.findData(normalize_place_type(place.place_type))
            if idx >= 0:
                self._type.setCurrentIndex(idx)
            if place.latitude is not None:
                self._lat.setText(f"{place.latitude:.6f}")
                init_lat = place.latitude
            if place.longitude is not None:
                self._lon.setText(f"{place.longitude:.6f}")
                init_lon = place.longitude
            if place.latitude is not None and place.longitude is not None:
                zoom = 15 if place.is_exact_coordinate else 12
            if place.accuracy_radius_meters is not None:
                self._radius_auto.setChecked(False)
                self._radius.setValue(float(place.accuracy_radius_meters))
            else:
                self._sync_radius_default()
            self._coord_source = place.coordinate_source
            self._is_exact = bool(place.is_exact_coordinate)
            if place.parent_id is not None:
                pidx = self._parent.findData(place.parent_id)
                if pidx >= 0:
                    self._parent.setCurrentIndex(pidx)

        self._refresh_source_label()
        # Bring up the map and show the marker if we have a coordinate.
        self._map.prepare(init_lat, init_lon, zoom)
        lat, lon = self._current_coords()
        if lat is not None and lon is not None:
            self._map.set_marker(lat, lon, zoom)
            self._map.set_accuracy_circle(self._radius.value(), self._current_type())

    # ------------------------------------------------------------------
    # Address events
    # ------------------------------------------------------------------

    def _on_settlement_chosen(self, row: dict) -> None:
        name = row.get("name") or ""
        self._street.set_settlement_context(name)
        self._apply_classification(reset_street_house=False)
        lat, lon = _coord_from(row)
        if lat is not None and lon is not None:
            self._set_coords(lat, lon, COORD_SOURCE_GEOCODED_SETTLEMENT, is_exact=False, zoom=12)
        self._update_display_name()

    def _on_settlement_text(self, text: str) -> None:
        if self._loading:
            return
        self._street.set_settlement_context(text or None)
        self._update_display_name()
        self._apply_classification(reset_street_house=False)

    def _on_street_chosen(self, row: dict) -> None:
        self._apply_classification(reset_street_house=False)
        lat, lon = _coord_from(row)
        if lat is not None and lon is not None:
            self._set_coords(lat, lon, COORD_SOURCE_GEOCODED_STREET, is_exact=False, zoom=15)
        self._update_display_name()

    def _on_street_text(self, _text: str) -> None:
        if self._loading:
            return
        self._update_display_name()
        self._apply_classification(reset_street_house=False)
        self._house_timer.start()

    def _on_house_edited(self, _text: str) -> None:
        if self._loading:
            return
        self._update_display_name()
        self._apply_classification(reset_street_house=False)
        self._house_timer.start()

    def _on_name_edited(self, text: str) -> None:
        self._name_overridden = bool(text.strip())

    def _geocode_full_address(self) -> None:
        settlement = self._settlement.text()
        street = self._street.text()
        house = self._house.text().strip()
        if not settlement or not street or not house:
            return
        self._geocode_req += 1
        req = self._geocode_req
        worker = GeocodeWorker(settlement, street, house, req)
        worker.signals.geocode_ready.connect(self._on_geocode_ready)
        QThreadPool.globalInstance().start(worker)

    def _on_geocode_ready(self, request_id: int, payload) -> None:
        if request_id != self._geocode_req or not payload:
            return
        lat = payload.get("latitude")
        lon = payload.get("longitude")
        if lat is None or lon is None:
            return
        self._set_coords(lat, lon, COORD_SOURCE_GEOCODED_ADDRESS, is_exact=True, zoom=17)

    # ------------------------------------------------------------------
    # Type / radius / coordinates
    # ------------------------------------------------------------------

    def _apply_classification(self, *, reset_street_house: bool) -> None:
        """Derive type/source/exact from the current address parts unless the
        user has pinned a manual coordinate."""
        if self._coord_source == COORD_SOURCE_MANUAL_MAP_PIN:
            self._refresh_source_label()
            return
        auto = classify_address(
            self._settlement.text(), self._street.text(), self._house.text()
        )
        idx = self._type.findData(auto.place_type)
        if idx >= 0 and not self._type_user_locked():
            self._type.setCurrentIndex(idx)
        self._coord_source = auto.coordinate_source
        self._is_exact = auto.is_exact_coordinate
        self._refresh_source_label()
        self._update_hint()

    def _type_user_locked(self) -> bool:
        # We keep it simple: the type combo always reflects auto-classification
        # unless a manual pin is active. (A future toggle could lock it.)
        return self._coord_source == COORD_SOURCE_MANUAL_MAP_PIN

    def _current_type(self) -> str:
        return self._type.currentData() or DEFAULT_PLACE_TYPE

    def _sync_radius_default(self) -> None:
        self._radius.setValue(default_radius_for(self._current_type()))

    def _on_type_changed(self, _idx: int) -> None:
        if self._radius_auto.isChecked():
            self._sync_radius_default()
        self._map.set_accuracy_circle(self._radius.value(), self._current_type())

    def _on_radius_auto_toggled(self, checked: bool) -> None:
        self._radius.setEnabled(not checked)
        if checked:
            self._sync_radius_default()

    def _on_radius_value_changed(self, _value: float) -> None:
        if not self._loading:
            self._map.set_accuracy_circle(self._radius.value(), self._current_type())

    def _on_coord_edited(self) -> None:
        if self._loading:
            return
        lat, lon = self._current_coords()
        if lat is None or lon is None:
            return
        self._set_coords(lat, lon, COORD_SOURCE_MANUAL_MAP_PIN, is_exact=True, zoom=None)

    def _on_map_pick(self, lat: float, lon: float) -> None:
        # The JS marker already moved; sync the fields and mark it manual.
        self._set_coords(lat, lon, COORD_SOURCE_MANUAL_MAP_PIN, is_exact=True, zoom=None, move_marker=False)

    def _set_coords(
        self,
        lat: float,
        lon: float,
        source: str,
        *,
        is_exact: bool,
        zoom: Optional[int],
        move_marker: bool = True,
    ) -> None:
        self._lat.setText(f"{lat:.6f}")
        self._lon.setText(f"{lon:.6f}")
        self._coord_source = source
        self._is_exact = is_exact
        self._refresh_source_label()
        self._update_hint()
        if move_marker:
            self._map.set_marker(lat, lon, zoom or 15)
        self._map.set_accuracy_circle(self._radius.value(), self._current_type())

    # ------------------------------------------------------------------
    # Display name / hints
    # ------------------------------------------------------------------

    def _update_display_name(self) -> None:
        if self._name_overridden:
            return
        display = build_display_name(
            self._settlement.text(), self._street.text(), self._house.text()
        )
        if display:
            self._name.setText(display)

    def _refresh_source_label(self) -> None:
        self._source_lbl.setText(self._coord_source or "—")

    def _update_hint(self) -> None:
        if self._coord_source == COORD_SOURCE_GEOCODED_SETTLEMENT:
            self._hint.setText(t("place_edit_hint_area"))
        elif self._coord_source == COORD_SOURCE_MANUAL_MAP_PIN:
            self._hint.setText(t("place_edit_hint_manual"))
        else:
            self._hint.setText("")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_coord(text_value: str) -> Optional[float]:
        text_value = (text_value or "").strip()
        if not text_value:
            return None
        return float(text_value.replace(",", "."))

    def _current_coords(self) -> Tuple[Optional[float], Optional[float]]:
        try:
            return self._parse_coord(self._lat.text()), self._parse_coord(self._lon.text())
        except ValueError:
            return None, None

    def _on_accept(self) -> None:
        settlement = self._settlement.text()
        name = self._name.text().strip()
        if not settlement and not name:
            QMessageBox.warning(self, self.windowTitle(), t("place_edit_need_settlement"))
            return
        try:
            self._parse_coord(self._lat.text())
            self._parse_coord(self._lon.text())
        except ValueError:
            QMessageBox.warning(
                self, self.windowTitle(), t("place_edit_save_error", error="lat/lon")
            )
            return
        self.accept()

    def result_value(self) -> PlaceEditResult:
        radius = None if self._radius_auto.isChecked() else float(self._radius.value())
        settlement = self._settlement.text() or None
        street = self._street.text() or None
        house = self._house.text().strip() or None
        display = build_display_name(settlement, street, house) if settlement else None
        lat, lon = self._current_coords()
        return PlaceEditResult(
            name=self._name.text().strip() or (display or ""),
            place_type=self._current_type(),
            latitude=lat,
            longitude=lon,
            accuracy_radius_meters=radius,
            parent_id=self._parent.currentData(),
            settlement_name=settlement,
            street_name=street,
            house_number=house,
            display_name=display,
            coordinate_source=self._coord_source,
            is_exact_coordinate=self._is_exact,
        )


def _coord_from(row: dict) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(row, dict):
        return None, None
    return row.get("latitude"), row.get("longitude")
