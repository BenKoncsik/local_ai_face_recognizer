"""Reusable searchable person-selector widget.

Usage
-----
::

    widget = PersonSearchSelect(parent=self)
    widget.set_persons(persons, priority_ids=recent_ids)
    widget.person_selected.connect(self._on_person_chosen)

    # read back the selection
    person_id = widget.current_person_id()   # None if nothing selected

    # pre-select a person (e.g. after loading a face)
    widget.set_current_by_id(42)
"""

from __future__ import annotations

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

from app.db.models import Person
from app.ui.i18n import t
from app.utils.person_search import PersonEntry, search_persons

_ROLE_ID = Qt.UserRole
_MAX_VISIBLE_ITEMS = 8
_ITEM_HEIGHT = 24  # px — approximate row height


class PersonSearchSelect(QWidget):
    """A compact, always-visible searchable list of persons.

    Signals
    -------
    person_selected(int)
        Emitted when the user commits a selection (single-click or Enter).
        Carries the selected *person_id*.
    person_double_clicked(int)
        Emitted when the user double-clicks a person in the list.
        Carries the selected *person_id*.  Qt fires itemClicked first, so
        the selection is already committed before this signal is emitted.
    """

    person_selected: Signal = Signal(int)
    person_double_clicked: Signal = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._entries: List[PersonEntry] = []
        self._selected_id: Optional[int] = None

        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._search = QLineEdit()
        self._search.setPlaceholderText(t("pss_search_placeholder"))
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

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def set_persons(
        self,
        persons: List[Person],
        priority_ids: Optional[List[int]] = None,
    ) -> None:
        """Populate the selector with *persons*.

        *priority_ids* — if given, those persons appear at the top of the
        unfiltered list (e.g. recently used).  The ordering of *priority_ids*
        is preserved.
        """
        priority_ids = priority_ids or []

        def _rank(p: Person) -> tuple:
            try:
                idx = priority_ids.index(p.id)
            except ValueError:
                idx = len(priority_ids) + 1
            return (idx, p.name.casefold())

        sorted_persons = sorted(persons, key=_rank)
        self._entries = [PersonEntry(p.id, p.name) for p in sorted_persons]
        self._refresh_list()

    def set_entries(self, entries: List[PersonEntry]) -> None:
        """Populate with pre-built :class:`PersonEntry` objects (e.g. with custom display_text)."""
        self._entries = list(entries)
        self._refresh_list()

    def current_person_id(self) -> Optional[int]:
        """Return the currently selected person_id, or ``None``."""
        item = self._list.currentItem()
        if item is None:
            return None
        data = item.data(_ROLE_ID)
        return int(data) if data is not None else None

    def set_current_by_id(self, person_id: Optional[int]) -> None:
        """Highlight the list row whose person_id matches *person_id*."""
        self._selected_id = person_id
        self._sync_selection()

    def clear_selection(self) -> None:
        """Deselect all rows and forget the stored selection."""
        self._selected_id = None
        self._list.clearSelection()
        self._list.setCurrentItem(None)

    def retranslate(self) -> None:
        self._search.setPlaceholderText(t("pss_search_placeholder"))
        self._no_results.setText(t("pss_no_results"))

    # ──────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────

    def _refresh_list(self) -> None:
        query = self._search.text()
        visible = search_persons(query, self._entries)
        self._list.blockSignals(True)
        self._list.clear()
        for entry in visible:
            item = QListWidgetItem(entry.display_text)
            item.setData(_ROLE_ID, entry.person_id)
            self._list.addItem(item)
        self._list.blockSignals(False)
        self._sync_selection()
        has_items = self._list.count() > 0
        self._list.setVisible(has_items)
        self._no_results.setVisible(not has_items and bool(query.strip()))

    def _sync_selection(self) -> None:
        """Re-highlight the row for ``self._selected_id`` after a list rebuild."""
        if self._selected_id is None:
            return
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item and item.data(_ROLE_ID) == self._selected_id:
                self._list.setCurrentItem(item)
                return

    def _commit_current(self) -> None:
        person_id = self.current_person_id()
        if person_id is not None:
            self._selected_id = person_id
            self.person_selected.emit(person_id)

    # ──────────────────────────────────────────────────────────────────
    # Qt event filter — keyboard navigation
    # ──────────────────────────────────────────────────────────────────

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
                if key == Qt.Key_Return or key == Qt.Key_Enter:
                    if self._list.count() > 0 and self._list.currentRow() < 0:
                        self._list.setCurrentRow(0)
                    self._commit_current()
                    return True
                if key == Qt.Key_Escape:
                    self._search.clear()
                    return True

            if watched is self._list:
                if key == Qt.Key_Return or key == Qt.Key_Enter:
                    self._commit_current()
                    return True
                if key == Qt.Key_Escape:
                    self._search.clear()
                    self._search.setFocus()
                    return True
                # Let Up/Down pass through to QListWidget's own handler
                if key in (Qt.Key_Up, Qt.Key_Down):
                    return False

        return super().eventFilter(watched, event)

    # ──────────────────────────────────────────────────────────────────
    # Slots
    # ──────────────────────────────────────────────────────────────────

    def _on_text_changed(self, _text: str) -> None:
        self._refresh_list()

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        self._commit_current()

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        # itemClicked already ran (_commit_current), so current_person_id() is set.
        person_id = self.current_person_id()
        if person_id is not None:
            self.person_double_clicked.emit(person_id)
