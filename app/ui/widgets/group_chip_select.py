"""Searchable multi-select chip widget for person groups.

Usage::

    widget = GroupChipSelect(parent=self)
    widget.set_available_groups([(1, "Kórus"), (2, "Munkahely")])
    widget.set_selected_groups([(1, "Kórus")])
    widget.group_created.connect(self._on_new_group)
    widget.groups_changed.connect(self._on_groups_changed)

    selected_ids = widget.selected_group_ids()
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.ui.widgets.flow_layout import FlowContainer as _FlowContainer

log = logging.getLogger(__name__)

_ROLE_ID = Qt.UserRole
_ROLE_IS_CREATE = Qt.UserRole + 1


# ---------------------------------------------------------------------------
# Internal: single removable chip (selected group)
# ---------------------------------------------------------------------------

class _SelectedChip(QFrame):
    """Blue badge with × for an already-selected group."""

    removed = Signal(int)  # emits group_id

    def __init__(self, group_id: int, name: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._group_id = group_id
        self.setObjectName("selChip")
        self.setToolTip(name)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 2, 2, 2)
        lay.setSpacing(2)

        lbl = QLabel(name)
        lbl.setObjectName("selChipLabel")

        btn = QPushButton("×")
        btn.setObjectName("selChipRemove")
        btn.setFixedSize(16, 16)
        btn.setFlat(True)
        btn.setToolTip("Eltávolítás")
        btn.clicked.connect(lambda: self.removed.emit(self._group_id))

        lay.addWidget(lbl)
        lay.addWidget(btn)

        self.setStyleSheet("""
            #selChip {
                background-color: #4a90d9;
                border-radius: 4px;
                border: none;
            }
            #selChipLabel { color: white; font-size: 12px; }
            #selChipRemove {
                color: rgba(255,255,255,180);
                border: none;
                font-weight: bold;
                font-size: 14px;
                padding: 0;
                background: transparent;
            }
            #selChipRemove:hover { color: white; }
        """)


# ---------------------------------------------------------------------------
# Internal: clickable chip for an available (not yet selected) group
# ---------------------------------------------------------------------------

class _AvailableChip(QPushButton):
    """Grey clickable chip for an available group — click to select."""

    add_requested = Signal(int, str)  # (group_id, name)

    def __init__(self, group_id: int, name: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(f"+ {name}", parent)
        self._group_id = group_id
        self._name = name
        self.setObjectName("availChip")
        self.setToolTip(f"Hozzáadás: {name}")
        self.setFlat(True)
        self.clicked.connect(lambda: self.add_requested.emit(self._group_id, self._name))
        self.setStyleSheet("""
            #availChip {
                background-color: #e8e8e8;
                border: 1px solid #ccc;
                border-radius: 4px;
                color: #444;
                font-size: 12px;
                padding: 2px 8px;
            }
            #availChip:hover {
                background-color: #d0e8ff;
                border-color: #4a90d9;
                color: #1a5fa8;
            }
        """)


# ---------------------------------------------------------------------------
# Public widget
# ---------------------------------------------------------------------------

class GroupChipSelect(QWidget):
    """Multi-select chip input for person groups.

    Layout (top to bottom):
    1. QLineEdit — search / create
    2. QListWidget — inline suggestion dropdown (shown while typing)
    3. Selected chips in a flow layout
    4. Available (unselected) groups as clickable chips

    Signals
    -------
    groups_changed(list[int])
        Emitted whenever the selection changes.
    group_created(int, str)
        Emitted when the user creates a new group via the widget.
    """

    groups_changed = Signal(list)     # list[int]
    group_created = Signal(int, str)  # (group_id, name)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._selected: dict[int, str] = {}        # id → name
        self._all_groups: List[Tuple[int, str]] = []

        self._build_ui()

    # ------------------------------------------------------------------
    # Size / layout overrides (needed so QScrollArea allocates full height)
    # ------------------------------------------------------------------

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        lay = self.layout()
        return lay.heightForWidth(width) if lay else -1

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if event.size().width() != event.oldSize().width():
            self.updateGeometry()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_available_groups(self, groups: List[Tuple[int, str]]) -> None:
        """Provide the full list of existing groups."""
        self._all_groups = sorted(groups, key=lambda g: g[1].lower())
        self._refresh_available()

    def set_selected_groups(self, groups: List[Tuple[int, str]]) -> None:
        """Pre-select groups (replaces current selection)."""
        self._selected = {gid: name for gid, name in groups}
        self._refresh_selected()
        self._refresh_available()

    def selected_group_ids(self) -> List[int]:
        return list(self._selected.keys())

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        # --- 1. Search input ---
        self._search = QLineEdit()
        self._search.setPlaceholderText("Keresés vagy új kategória …")
        self._search.textChanged.connect(self._on_text_changed)
        self._search.installEventFilter(self)
        outer.addWidget(self._search)

        # --- 2. Inline suggestion list (hidden until typing) ---
        self._suggestion_list = QListWidget()
        self._suggestion_list.setFrameStyle(QFrame.Box)
        self._suggestion_list.setMaximumHeight(160)
        self._suggestion_list.hide()
        self._suggestion_list.itemClicked.connect(self._on_suggestion_clicked)
        outer.addWidget(self._suggestion_list)

        # --- 3. Selected chips ---
        self._selected_container = _FlowContainer(h_spacing=4, v_spacing=4)
        self._selected_layout = self._selected_container.layout()

        self._selected_placeholder = QLabel(
            "Add meg, milyen társaságokhoz vagy közösségekhez tartozik ez a személy."
        )
        self._selected_placeholder.setWordWrap(True)
        self._selected_placeholder.setStyleSheet(
            "color: #888; font-style: italic; font-size: 11px;"
        )
        outer.addWidget(self._selected_placeholder)
        outer.addWidget(self._selected_container)

        # --- 4. Available groups section ---
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        outer.addWidget(sep)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        self._available_label = QLabel("Meglévő csoportok:")
        self._available_label.setStyleSheet("color: #666; font-size: 11px;")
        header_row.addWidget(self._available_label)
        header_row.addStretch()

        from app.ui.i18n import t as _t
        self._manage_btn = QPushButton(_t("manage_groups"))
        self._manage_btn.setFlat(True)
        self._manage_btn.setCursor(Qt.PointingHandCursor)
        self._manage_btn.setStyleSheet(
            "QPushButton { color: #4a90d9; border: none; font-size: 11px; }"
            "QPushButton:hover { text-decoration: underline; }"
        )
        self._manage_btn.clicked.connect(self._open_group_manager)
        header_row.addWidget(self._manage_btn)
        outer.addLayout(header_row)

        self._available_container = _FlowContainer(h_spacing=4, v_spacing=4)
        self._available_layout = self._available_container.layout()
        outer.addWidget(self._available_container)

        self._available_empty_label = QLabel("(még nincs létrehozva csoport)")
        self._available_empty_label.setStyleSheet("color: #aaa; font-size: 11px; font-style: italic;")
        outer.addWidget(self._available_empty_label)

        self._refresh_selected()
        self._refresh_available()

    # ------------------------------------------------------------------
    # Refresh helpers
    # ------------------------------------------------------------------

    def _refresh_selected(self) -> None:
        """Redraw the selected-chips flow area."""
        while self._selected_layout.count():
            item = self._selected_layout.takeAt(0)
            if item and item.widget():
                w = item.widget()
                w.hide()
                w.deleteLater()

        has_selected = bool(self._selected)
        self._selected_placeholder.setVisible(not has_selected)
        self._selected_container.setVisible(has_selected)

        for gid, name in sorted(self._selected.items(), key=lambda kv: kv[1].lower()):
            chip = _SelectedChip(gid, name, self._selected_container)
            chip.removed.connect(self._on_chip_removed)
            self._selected_layout.addWidget(chip)

        self._selected_container.updateGeometry()
        self.updateGeometry()

    def _refresh_available(self) -> None:
        """Redraw the available-groups area (unselected groups as clickable chips)."""
        while self._available_layout.count():
            item = self._available_layout.takeAt(0)
            if item and item.widget():
                w = item.widget()
                w.hide()
                w.deleteLater()

        unselected = [
            (gid, name)
            for gid, name in self._all_groups
            if gid not in self._selected
        ]

        has_unselected = bool(unselected)
        self._available_empty_label.setVisible(not has_unselected)
        self._available_container.setVisible(has_unselected)

        for gid, name in unselected:
            chip = _AvailableChip(gid, name, self._available_container)
            chip.add_requested.connect(self._select_group)
            self._available_layout.addWidget(chip)

        self._available_container.updateGeometry()
        self.updateGeometry()

    def _refresh_suggestion_list(self, search: str) -> None:
        """Populate and show/hide the inline suggestion list."""
        self._suggestion_list.clear()
        if not search.strip():
            self._suggestion_list.hide()
            return

        lower = search.lower()
        matches = [
            (gid, name)
            for gid, name in self._all_groups
            if lower in name.lower() and gid not in self._selected
        ]
        exact = any(name.lower() == lower for _, name in matches)

        for gid, name in matches:
            item = QListWidgetItem(name)
            item.setData(_ROLE_ID, gid)
            item.setData(_ROLE_IS_CREATE, False)
            self._suggestion_list.addItem(item)

        trimmed = search.strip()
        if trimmed and not exact:
            item = QListWidgetItem(f'+ Létrehozás: "{trimmed}"')
            item.setData(_ROLE_ID, -1)
            item.setData(_ROLE_IS_CREATE, True)
            self._suggestion_list.addItem(item)

        if self._suggestion_list.count():
            self._suggestion_list.show()
        else:
            self._suggestion_list.hide()

    # ------------------------------------------------------------------
    # Event filter — keyboard navigation in the search field
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QKeyEvent

        if obj is not self._search:
            return super().eventFilter(obj, event)

        if event.type() == QEvent.KeyPress:
            key = event.key()
            if key == Qt.Key_Down and self._suggestion_list.isVisible():
                self._suggestion_list.setFocus()
                if self._suggestion_list.count():
                    self._suggestion_list.setCurrentRow(0)
                return True
            if key in (Qt.Key_Return, Qt.Key_Enter):
                self._on_enter_pressed()
                return True
            if key == Qt.Key_Escape:
                self._suggestion_list.hide()
                return True

        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_text_changed(self, text: str) -> None:
        self._refresh_suggestion_list(text)

    def _on_enter_pressed(self) -> None:
        text = self._search.text().strip()
        if not text:
            return

        # If suggestion list is visible and has a selection, use it
        current = self._suggestion_list.currentItem()
        if current is not None and self._suggestion_list.isVisible():
            self._on_suggestion_clicked(current)
            return

        # If exactly one match, use it
        lower = text.lower()
        matches = [
            (gid, name)
            for gid, name in self._all_groups
            if lower in name.lower() and gid not in self._selected
        ]
        exact = [(gid, name) for gid, name in matches if name.lower() == lower]
        if exact:
            self._select_group(exact[0][0], exact[0][1])
        else:
            self._create_group(text)

    def _on_suggestion_clicked(self, item: QListWidgetItem) -> None:
        if item.data(_ROLE_IS_CREATE):
            # strip '+ Létrehozás: "..."' wrapper
            text = item.text()
            name = text[len('+ Létrehozás: "'):-1]
            self._create_group(name)
        else:
            self._select_group(item.data(_ROLE_ID), item.text())

    def _on_chip_removed(self, gid: int) -> None:
        self._selected.pop(gid, None)
        self._refresh_selected()
        self._refresh_available()
        self.groups_changed.emit(self.selected_group_ids())

    def _select_group(self, gid: int, name: str) -> None:
        self._search.clear()
        self._suggestion_list.hide()
        if gid in self._selected:
            return
        self._selected[gid] = name
        self._refresh_selected()
        self._refresh_available()
        self.groups_changed.emit(self.selected_group_ids())

    def _create_group(self, name: str) -> None:
        from app.db.database import session_scope
        from app.services.person_group_service import PersonGroupService

        name = name.strip()
        if not name:
            return
        try:
            with session_scope() as session:
                svc = PersonGroupService(session)
                group = svc.get_or_create(name)
                gid, gname = group.id, group.name
        except Exception:
            log.exception("Failed to create group '%s'", name)
            return

        if not any(g[0] == gid for g in self._all_groups):
            self._all_groups.append((gid, gname))
            self._all_groups.sort(key=lambda g: g[1].lower())
            self.group_created.emit(gid, gname)

        self._select_group(gid, gname)

    # ------------------------------------------------------------------
    # Group manager
    # ------------------------------------------------------------------

    def _open_group_manager(self) -> None:
        """Open the group manager dialog, then refresh from the DB if changed."""
        from app.ui.dialogs.group_manager_dialog import GroupManagerDialog

        dlg = GroupManagerDialog(self)
        dlg.exec()
        if dlg.groups_changed:
            self._reload_from_db()

    def _reload_from_db(self) -> None:
        """Re-read the full group list from the DB after external edits.

        Renames are reflected in the selected chips; groups deleted elsewhere
        are dropped from the selection.
        """
        from app.db.database import session_scope
        from app.db.models import PersonGroup as _PG

        try:
            with session_scope() as session:
                groups = [
                    (g.id, g.name)
                    for g in session.query(_PG).order_by(_PG.name).all()
                ]
        except Exception:
            log.exception("Failed to reload groups from DB")
            return

        names = {gid: name for gid, name in groups}
        # Keep only still-existing selections, updating their (possibly renamed) names.
        self._selected = {
            gid: names[gid] for gid in self._selected if gid in names
        }
        self._all_groups = sorted(groups, key=lambda g: g[1].lower())
        self._refresh_selected()
        self._refresh_available()
        self.groups_changed.emit(self.selected_group_ids())
