"""Object picker dialog — choose an existing object or create a new one.

Shown when the user clicks a point on an image in object-tagging mode.  Returns
the chosen/created object id plus an optional per-image note for the occurrence.
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.db.database import session_scope
from app.services.object_service import ObjectService
from app.ui.i18n import t

log = logging.getLogger(__name__)

_ROLE_ID = Qt.UserRole


class ObjectPickerDialog(QDialog):
    """Pick an existing :class:`TaggedObject` or create a new one."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("object_picker_title"))
        self.setMinimumWidth(420)
        self.resize(440, 540)

        self._chosen_object_id: Optional[int] = None
        self._occurrence_note: str = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # ── Existing objects ────────────────────────────────────────────────
        existing_box = QGroupBox(t("object_picker_existing"))
        existing_layout = QVBoxLayout(existing_box)

        self._search = QLineEdit()
        self._search.setPlaceholderText(t("object_picker_search"))
        # Debounce the search: each keystroke otherwise fires a fresh DB query,
        # which is costly on a large object database.  Coalesce rapid typing
        # into a single query 200 ms after the user pauses.
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(200)
        self._search_timer.timeout.connect(self._refresh_list)
        self._search.textChanged.connect(lambda _t: self._search_timer.start())
        existing_layout.addWidget(self._search)

        self._list = QListWidget()
        self._list.itemSelectionChanged.connect(self._on_list_selection)
        self._list.itemDoubleClicked.connect(lambda _it: self._accept_existing())
        existing_layout.addWidget(self._list)
        layout.addWidget(existing_box, stretch=1)

        # ── New object ──────────────────────────────────────────────────────
        new_box = QGroupBox(t("object_picker_new"))
        new_layout = QVBoxLayout(new_box)

        new_layout.addWidget(QLabel(t("object_name")))
        self._new_name = QLineEdit()
        self._new_name.setPlaceholderText(t("object_example_name"))
        self._new_name.textChanged.connect(self._on_new_name_changed)
        new_layout.addWidget(self._new_name)

        new_layout.addWidget(QLabel(t("object_description")))
        self._new_desc = QLineEdit()
        self._new_desc.setPlaceholderText(t("object_example_desc"))
        new_layout.addWidget(self._new_desc)
        layout.addWidget(new_box)

        # ── Per-image note ──────────────────────────────────────────────────
        layout.addWidget(QLabel(t("object_picker_occ_note")))
        self._note = QTextEdit()
        self._note.setFixedHeight(56)
        layout.addWidget(self._note)

        # ── Buttons ─────────────────────────────────────────────────────────
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._refresh_list()
        self._update_ok_enabled()

    # ------------------------------------------------------------------

    def _refresh_list(self) -> None:
        query = self._search.text().strip()
        self._list.clear()
        try:
            with session_scope() as session:
                results = ObjectService(session).search_objects(query, max_results=200)
        except Exception:
            log.exception("Failed to load objects for picker")
            results = []
        for s in results:
            label = s.name
            if s.image_count:
                label = f"{s.name}  ({s.image_count})"
            item = QListWidgetItem(label)
            item.setData(_ROLE_ID, s.object_id)
            self._list.addItem(item)

    def _on_list_selection(self) -> None:
        if self._list.selectedItems():
            self._new_name.clear()  # existing selection wins over new-name
        self._update_ok_enabled()

    def _on_new_name_changed(self, text: str) -> None:
        if text.strip():
            self._list.clearSelection()
        self._update_ok_enabled()

    def _update_ok_enabled(self) -> None:
        has_choice = bool(self._list.selectedItems()) or bool(
            self._new_name.text().strip()
        )
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(has_choice)

    def _accept_existing(self) -> None:
        items = self._list.selectedItems()
        if items:
            self._on_accept()

    def _on_accept(self) -> None:
        self._occurrence_note = self._note.toPlainText().strip()
        items = self._list.selectedItems()
        new_name = self._new_name.text().strip()

        try:
            with session_scope() as session:
                svc = ObjectService(session)
                if new_name:
                    obj = svc.create_object(new_name, self._new_desc.text().strip())
                    self._chosen_object_id = obj.id
                elif items:
                    self._chosen_object_id = items[0].data(_ROLE_ID)
                else:
                    return
        except ValueError as exc:
            QMessageBox.warning(self, t("object_picker_title"), str(exc))
            return
        except Exception:
            log.exception("Failed to create/select object")
            return
        self.accept()

    # ------------------------------------------------------------------

    @property
    def chosen_object_id(self) -> Optional[int]:
        return self._chosen_object_id

    @property
    def occurrence_note(self) -> str:
        return self._occurrence_note
