"""Reusable searchable place selector."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QLabel,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.db.models import Place
from app.services.place_service import ANONYMOUS_GPS_PLACE_NAME
from app.ui.i18n import t
from app.utils.person_search import normalize

_ROLE_ID = Qt.UserRole
_MAX_VISIBLE_ITEMS = 8
_ITEM_HEIGHT = 24


@dataclass
class PlaceEntry:
    place_id: int
    name: str
    display_text: str = ""
    _normalized: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.display_text:
            self.display_text = self.name
        self._normalized = normalize(self.name)


def search_places(query: str, entries: List[PlaceEntry], max_results: int = 50) -> List[PlaceEntry]:
    if not query.strip():
        return entries[:max_results]
    q = normalize(query.strip())
    matched = [e for e in entries if q in e._normalized]
    matched.sort(key=lambda e: (not e._normalized.startswith(q), e._normalized))
    return matched[:max_results]


class PlaceSearchSelect(QWidget):
    place_selected: Signal = Signal(int)
    place_double_clicked: Signal = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._entries: List[PlaceEntry] = []
        self._selected_id: Optional[int] = None

        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._search = QLineEdit()
        self._search.setPlaceholderText(t("place_search_placeholder"))
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._on_text_changed)
        self._search.installEventFilter(self)
        layout.addWidget(self._search)

        self._list = QListWidget()
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._list.setMaximumHeight(_ITEM_HEIGHT * _MAX_VISIBLE_ITEMS + 4)
        self._list.setMinimumWidth(0)
        self._list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._list.itemClicked.connect(self._on_item_clicked)
        self._list.itemDoubleClicked.connect(self._on_item_double_clicked)
        self._list.installEventFilter(self)
        layout.addWidget(self._list)

        self._no_results = QLabel(t("pss_no_results"))
        self._no_results.setStyleSheet("color: #888; font-style: italic; font-size: 11px;")
        self._no_results.setAlignment(Qt.AlignCenter)
        self._no_results.setVisible(False)
        layout.addWidget(self._no_results)

    def set_places(self, places: List[Place]) -> None:
        self._entries = []
        for place in sorted(places, key=lambda p: (not p.is_anonymous, p.name.casefold(), p.id)):
            label = place.name
            if place.is_anonymous and place.source == "exif":
                label = f"{ANONYMOUS_GPS_PLACE_NAME} #{place.id}"
            if place.latitude is not None and place.longitude is not None:
                label = f"{label}  ({place.latitude:.5f}, {place.longitude:.5f})"
            self._entries.append(PlaceEntry(place.id, place.name, label))
        self._refresh_list()

    def current_place_id(self) -> Optional[int]:
        item = self._list.currentItem()
        if item is None:
            return None
        data = item.data(_ROLE_ID)
        return int(data) if data is not None else None

    def current_query(self) -> str:
        return self._search.text().strip()

    def set_current_by_id(self, place_id: Optional[int]) -> None:
        self._selected_id = place_id
        self._sync_selection()

    def clear_selection(self) -> None:
        self._selected_id = None
        self._list.clearSelection()
        self._list.setCurrentItem(None)

    def retranslate(self) -> None:
        self._search.setPlaceholderText(t("place_search_placeholder"))
        self._no_results.setText(t("pss_no_results"))

    def _refresh_list(self) -> None:
        visible = search_places(self._search.text(), self._entries)
        self._list.blockSignals(True)
        self._list.clear()
        for entry in visible:
            item = QListWidgetItem(entry.display_text)
            item.setData(_ROLE_ID, entry.place_id)
            self._list.addItem(item)
        self._list.blockSignals(False)
        self._sync_selection()
        has_items = self._list.count() > 0
        self._list.setVisible(has_items)
        self._no_results.setVisible(not has_items and bool(self._search.text().strip()))

    def _sync_selection(self) -> None:
        if self._selected_id is None:
            return
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item and item.data(_ROLE_ID) == self._selected_id:
                self._list.setCurrentItem(item)
                return

    def _commit_current(self) -> None:
        place_id = self.current_place_id()
        if place_id is not None:
            self._selected_id = place_id
            self.place_selected.emit(place_id)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QKeyEvent

        if event.type() == QEvent.KeyPress:
            key_event: QKeyEvent = event
            key = key_event.key()
            if watched is self._search:
                if key == Qt.Key_Down:
                    self._list.setFocus()
                    if self._list.currentRow() < 0 and self._list.count() > 0:
                        self._list.setCurrentRow(0)
                    return True
                if key in (Qt.Key_Return, Qt.Key_Enter):
                    if self._list.count() > 0 and self._list.currentRow() < 0:
                        self._list.setCurrentRow(0)
                    self._commit_current()
                    return True
                if key == Qt.Key_Escape:
                    self._search.clear()
                    return True
            if watched is self._list:
                if key in (Qt.Key_Return, Qt.Key_Enter):
                    self._commit_current()
                    return True
                if key == Qt.Key_Escape:
                    self._search.clear()
                    self._search.setFocus()
                    return True
        return super().eventFilter(watched, event)

    def _on_text_changed(self, _text: str) -> None:
        self._refresh_list()

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        self._commit_current()

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        place_id = self.current_place_id()
        if place_id is not None:
            self.place_double_clicked.emit(place_id)
