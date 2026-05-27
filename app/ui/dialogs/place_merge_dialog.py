"""Conflict-choice dialog for merging places."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QVBoxLayout

from app.db.models import Place
from app.services.place_service import ANONYMOUS_GPS_PLACE_NAME
from app.ui.i18n import t


@dataclass(frozen=True)
class PlaceMergeChoice:
    name: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    thumbnail_path: Optional[str]


class PlaceMergeDialog(QDialog):
    def __init__(self, places: List[Place], target_id: int, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("places_merge_title"))
        self._places = places
        self._target_id = target_id
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._name_combo = QComboBox()
        self._coord_combo = QComboBox()
        self._thumb_combo = QComboBox()

        for place in self._places:
            label = place.name
            if place.is_anonymous and place.source == "exif":
                label = f"{ANONYMOUS_GPS_PLACE_NAME} #{place.id}"
            self._name_combo.addItem(label, place.id)
            if place.latitude is not None and place.longitude is not None:
                self._coord_combo.addItem(
                    f"{label} ({place.latitude:.6f}, {place.longitude:.6f})",
                    place.id,
                )
            if place.thumbnail_path:
                self._thumb_combo.addItem(label, place.id)

        self._set_combo_to_target(self._name_combo)
        self._set_combo_to_target(self._coord_combo)
        self._set_combo_to_target(self._thumb_combo)

        form.addRow(t("places_merge_keep_name"), self._name_combo)
        form.addRow(t("places_merge_keep_coords"), self._coord_combo)
        form.addRow(t("places_merge_keep_thumbnail"), self._thumb_combo)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _set_combo_to_target(self, combo: QComboBox) -> None:
        for idx in range(combo.count()):
            if combo.itemData(idx) == self._target_id:
                combo.setCurrentIndex(idx)
                return

    def choice(self) -> PlaceMergeChoice:
        by_id = {p.id: p for p in self._places}
        name_place = by_id.get(self._name_combo.currentData())
        coord_place = by_id.get(self._coord_combo.currentData())
        thumb_place = by_id.get(self._thumb_combo.currentData())
        return PlaceMergeChoice(
            name=name_place.name if name_place else None,
            latitude=coord_place.latitude if coord_place else None,
            longitude=coord_place.longitude if coord_place else None,
            thumbnail_path=thumb_place.thumbnail_path if thumb_place else None,
        )
