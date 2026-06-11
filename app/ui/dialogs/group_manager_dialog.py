"""Group manager — edit the selectable person groups (communities).

Lets the user list every group, rename it (e.g. fix a typo) and attach an
optional free-text note. Groups can also be created and deleted here.

The reusable :class:`GroupManagerWidget` holds all the logic; it is embedded
both in the standalone :class:`GroupManagerDialog` (reached from the group
chip selector) and in the *Groups* main-window tab.
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.db.database import session_scope
from app.services.person_group_service import PersonGroupService
from app.ui.i18n import t

log = logging.getLogger(__name__)

_ROLE_ID = Qt.UserRole


class GroupManagerWidget(QWidget):
    """List and edit selectable person groups (rename + note + create/delete).

    Signals
    -------
    groups_changed()
        Emitted after any group is created, renamed, edited or deleted.
    """

    groups_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._current_id: Optional[int] = None
        self._build_ui()
        self._reload_list()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)

        hint = QLabel(t("group_manager_hint"))
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888; font-size: 11px;")
        outer.addWidget(hint)

        body = QHBoxLayout()
        outer.addLayout(body, stretch=1)

        # --- Left: list + create/delete ---
        left = QVBoxLayout()
        body.addLayout(left, stretch=1)

        self._list = QListWidget()
        self._list.currentItemChanged.connect(self._on_selection_changed)
        left.addWidget(self._list, stretch=1)

        self._empty_label = QLabel(t("group_none_yet"))
        self._empty_label.setStyleSheet(
            "color: #aaa; font-style: italic; font-size: 11px;"
        )
        self._empty_label.hide()
        left.addWidget(self._empty_label)

        list_btns = QHBoxLayout()
        self._new_btn = QPushButton(t("group_new"))
        self._new_btn.clicked.connect(self._on_new)
        self._delete_btn = QPushButton(t("group_delete"))
        self._delete_btn.clicked.connect(self._on_delete)
        self._delete_btn.setEnabled(False)
        list_btns.addWidget(self._new_btn)
        list_btns.addWidget(self._delete_btn)
        list_btns.addStretch()
        left.addLayout(list_btns)

        # --- Right: editor ---
        right = QVBoxLayout()
        body.addLayout(right, stretch=1)

        self._editor = QWidget()
        ed = QVBoxLayout(self._editor)
        ed.setContentsMargins(0, 0, 0, 0)

        ed.addWidget(QLabel(t("group_name")))
        self._name_edit = QLineEdit()
        ed.addWidget(self._name_edit)

        self._members_label = QLabel("")
        self._members_label.setStyleSheet("color: #888; font-size: 11px;")
        ed.addWidget(self._members_label)

        ed.addWidget(QLabel(t("group_note")))
        self._note_edit = QTextEdit()
        self._note_edit.setPlaceholderText(t("group_note_placeholder"))
        ed.addWidget(self._note_edit, stretch=1)

        self._save_btn = QPushButton(t("group_save"))
        self._save_btn.clicked.connect(self._on_save)
        ed.addWidget(self._save_btn, alignment=Qt.AlignRight)

        self._prompt_label = QLabel(t("group_select_prompt"))
        self._prompt_label.setWordWrap(True)
        self._prompt_label.setAlignment(Qt.AlignCenter)
        self._prompt_label.setStyleSheet("color: #888;")

        right.addWidget(self._prompt_label)
        right.addWidget(self._editor)
        self._editor.hide()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _reload_list(self, select_id: Optional[int] = None) -> None:
        """Reload all groups from the DB, optionally re-selecting *select_id*."""
        try:
            with session_scope() as session:
                infos = PersonGroupService(session).list_groups()
        except Exception:
            log.exception("Failed to list groups")
            return

        self._list.blockSignals(True)
        self._list.clear()
        for info in infos:
            label = f"{info.name}  ({info.member_count})"
            item = QListWidgetItem(label)
            item.setData(_ROLE_ID, info.id)
            self._list.addItem(item)
        self._list.blockSignals(False)

        self._empty_label.setVisible(not infos)

        if select_id is not None:
            self._select_by_id(select_id)
        elif infos:
            self._list.setCurrentRow(0)
        else:
            self._show_editor(None)

    def _select_by_id(self, group_id: int) -> None:
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item.data(_ROLE_ID) == group_id:
                self._list.setCurrentRow(row)
                return
        self._show_editor(None)

    def _show_editor(self, group_id: Optional[int]) -> None:
        self._current_id = group_id
        has = group_id is not None
        self._editor.setVisible(has)
        self._prompt_label.setVisible(not has)
        self._delete_btn.setEnabled(has)
        if not has:
            return

        try:
            with session_scope() as session:
                info = next(
                    (g for g in PersonGroupService(session).list_groups()
                     if g.id == group_id),
                    None,
                )
        except Exception:
            log.exception("Failed to load group id=%s", group_id)
            return
        if info is None:
            self._show_editor(None)
            return

        self._name_edit.setText(info.name)
        self._note_edit.setPlainText(info.description or "")
        self._members_label.setText(t("group_members_count", n=info.member_count))

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_selection_changed(self, current: Optional[QListWidgetItem], _prev) -> None:
        if current is None:
            self._show_editor(None)
        else:
            self._show_editor(current.data(_ROLE_ID))

    def _on_save(self) -> None:
        if self._current_id is None:
            return
        name = self._name_edit.text().strip()
        note = self._note_edit.toPlainText()
        try:
            with session_scope() as session:
                PersonGroupService(session).update_group(
                    self._current_id, name=name, description=note
                )
        except ValueError as exc:
            QMessageBox.warning(self, t("group_save_error_title"), str(exc))
            return
        except Exception:
            log.exception("Failed to update group id=%s", self._current_id)
            QMessageBox.warning(
                self, t("group_save_error_title"), t("group_save_error_title")
            )
            return
        self._reload_list(select_id=self._current_id)
        self.groups_changed.emit()

    def _on_new(self) -> None:
        name, ok = QInputDialog.getText(
            self, t("group_new_title"), t("group_new_name")
        )
        if not ok or not name.strip():
            return
        try:
            with session_scope() as session:
                group = PersonGroupService(session).create_group(name)
                new_id = group.id
        except ValueError as exc:
            QMessageBox.warning(self, t("group_save_error_title"), str(exc))
            return
        except Exception:
            log.exception("Failed to create group %r", name)
            return
        self._reload_list(select_id=new_id)
        self.groups_changed.emit()

    def _on_delete(self) -> None:
        if self._current_id is None:
            return
        item = self._list.currentItem()
        name = self._name_edit.text().strip() or (item.text() if item else "")
        try:
            with session_scope() as session:
                svc = PersonGroupService(session)
                member_count = len(svc.get_persons_in_group(self._current_id))
        except Exception:
            log.exception("Failed to count members for group id=%s", self._current_id)
            member_count = 0

        reply = QMessageBox.question(
            self,
            t("group_delete_confirm_title"),
            t("group_delete_confirm", name=name, n=member_count),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        try:
            with session_scope() as session:
                PersonGroupService(session).delete_group(self._current_id)
        except Exception:
            log.exception("Failed to delete group id=%s", self._current_id)
            return
        self._current_id = None
        self._reload_list()
        self.groups_changed.emit()


class GroupManagerDialog(QDialog):
    """Standalone dialog wrapper around :class:`GroupManagerWidget`."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("manage_groups_title"))
        self.setMinimumSize(560, 420)
        self.resize(620, 460)

        self._changed = False

        layout = QVBoxLayout(self)
        self._widget = GroupManagerWidget()
        self._widget.groups_changed.connect(self._on_changed)
        layout.addWidget(self._widget, stretch=1)

        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(self.accept)
        layout.addWidget(btns)

    def _on_changed(self) -> None:
        self._changed = True

    @property
    def groups_changed(self) -> bool:
        """True if any group was created, renamed, edited or deleted."""
        return self._changed
