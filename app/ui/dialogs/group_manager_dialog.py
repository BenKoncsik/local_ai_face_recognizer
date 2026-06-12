"""Group manager — edit the selectable person groups (communities).

Lets the user list every group, rename it (e.g. fix a typo) and attach an
optional free-text note. Groups can also be created and deleted here.

The reusable :class:`GroupManagerWidget` holds all the logic; it is embedded
both in the standalone :class:`GroupManagerDialog` (reached from the group
chip selector) and in the *Groups* main-window tab.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Optional

from PySide6.QtCore import QEvent, Qt, Signal
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
from app.db.models import Person
from app.services.person_group_service import PersonGroupService
from app.ui.i18n import t
from app.ui.widgets.person_search_select import PersonSearchSelect

log = logging.getLogger(__name__)

_ROLE_ID = Qt.UserRole
_ROLE_COUNT = Qt.UserRole + 1


class AddMemberDialog(QDialog):
    """Pick a person to add to a group (searchable list)."""

    def __init__(self, persons, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("group_member_add_title"))
        self.setMinimumWidth(340)
        self._selected_person_id: Optional[int] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        layout.addWidget(QLabel(t("group_member_add_prompt")))

        self._selector = PersonSearchSelect(self)
        self._selector.set_persons(persons)
        self._selector.person_selected.connect(self._on_person_selected)
        self._selector.person_double_clicked.connect(self._on_person_double_clicked)
        layout.addWidget(self._selector)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        layout.addWidget(self._buttons)

    def _on_person_selected(self, person_id: int) -> None:
        self._selected_person_id = person_id
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(True)

    def _on_person_double_clicked(self, person_id: int) -> None:
        self._selected_person_id = person_id
        self.accept()

    def selected_person_id(self) -> Optional[int]:
        return self._selected_person_id


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
        # Last values loaded into the editor; auto-save compares against these
        # so it only writes (and emits) when the user actually changed something.
        self._loaded_name: str = ""
        self._loaded_note: str = ""
        # Guards re-entrancy while the editor is being populated programmatically
        # (e.g. during _reload_list), so those field changes don't trigger a save.
        self._loading: bool = False
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
        # Auto-save when the user finishes editing (Enter or focus leaves).
        self._name_edit.editingFinished.connect(self._auto_save)
        ed.addWidget(self._name_edit)

        ed.addWidget(QLabel(t("group_note")))
        self._note_edit = QTextEdit()
        self._note_edit.setMaximumHeight(70)
        self._note_edit.setPlaceholderText(t("group_note_placeholder"))
        # QTextEdit has no editingFinished; auto-save when it loses focus.
        self._note_edit.installEventFilter(self)
        ed.addWidget(self._note_edit)

        self._save_btn = QPushButton(t("group_save"))
        self._save_btn.clicked.connect(self._on_save)
        ed.addWidget(self._save_btn, alignment=Qt.AlignRight)

        # --- Members ---
        self._members_label = QLabel("")
        self._members_label.setStyleSheet(
            "color: #888; font-size: 11px; font-weight: bold;"
        )
        ed.addWidget(self._members_label)

        self._members_list = QListWidget()
        self._members_list.currentItemChanged.connect(
            self._on_member_selection_changed
        )
        ed.addWidget(self._members_list, stretch=1)

        self._members_empty = QLabel(t("group_members_none"))
        self._members_empty.setStyleSheet(
            "color: #aaa; font-style: italic; font-size: 11px;"
        )
        self._members_empty.hide()
        ed.addWidget(self._members_empty)

        member_btns = QHBoxLayout()
        self._add_member_btn = QPushButton(t("group_member_add"))
        self._add_member_btn.clicked.connect(self._on_add_member)
        self._remove_member_btn = QPushButton(t("group_member_remove"))
        self._remove_member_btn.clicked.connect(self._on_remove_member)
        self._remove_member_btn.setEnabled(False)
        member_btns.addWidget(self._add_member_btn)
        member_btns.addWidget(self._remove_member_btn)
        member_btns.addStretch()
        ed.addLayout(member_btns)

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

        # Re-selecting a row below fires currentItemChanged; guard so it does
        # not auto-save the editor mid-rebuild (the values are about to be
        # replaced by the freshly loaded group anyway).
        self._loading = True
        try:
            self._list.blockSignals(True)
            self._list.clear()
            for info in infos:
                item = QListWidgetItem(self._item_label(info.name, info.member_count))
                item.setData(_ROLE_ID, info.id)
                item.setData(_ROLE_COUNT, info.member_count)
                self._list.addItem(item)
            self._list.blockSignals(False)

            self._empty_label.setVisible(not infos)

            if select_id is not None:
                self._select_by_id(select_id)
            elif infos:
                self._list.setCurrentRow(0)
            else:
                self._show_editor(None)
        finally:
            self._loading = False

    @staticmethod
    def _item_label(name: str, count: int) -> str:
        return f"{name}  ({count})"

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

        self._loaded_name = info.name
        self._loaded_note = info.description or ""
        self._name_edit.setText(self._loaded_name)
        self._note_edit.setPlainText(self._loaded_note)
        self._members_label.setText(t("group_members_count", n=info.member_count))
        self._reload_members(group_id)

    def _reload_members(self, group_id: int) -> None:
        """Refresh the member list for *group_id*."""
        try:
            with session_scope() as session:
                persons = PersonGroupService(session).get_persons_in_group(group_id)
                rows = [(p.id, p.name) for p in persons]
        except Exception:
            log.exception("Failed to load members for group id=%s", group_id)
            return

        self._members_list.blockSignals(True)
        self._members_list.clear()
        for pid, name in rows:
            item = QListWidgetItem(name)
            item.setData(_ROLE_ID, pid)
            self._members_list.addItem(item)
        self._members_list.blockSignals(False)

        self._members_list.setVisible(bool(rows))
        self._members_empty.setVisible(not rows)
        self._remove_member_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_selection_changed(
        self, current: Optional[QListWidgetItem], prev: Optional[QListWidgetItem]
    ) -> None:
        # Persist any pending name/note edits to the group we're leaving before
        # loading the newly selected one (skipped during programmatic reloads).
        if not self._loading and self._persist_current(show_errors=False):
            self._relabel_item(prev)
            self.groups_changed.emit()
        self._show_editor(current.data(_ROLE_ID) if current is not None else None)

    def _persist_current(self, *, show_errors: bool) -> bool:
        """Write the editor's name/note to the DB if they changed.

        Returns ``True`` only when an update was actually written. Shared by the
        Save button and the auto-save triggers. When *show_errors* is true an
        invalid name (empty/duplicate) raises a warning dialog; otherwise the
        save is silently skipped so background triggers never interrupt the user.
        """
        if self._current_id is None or self._loading:
            return False
        name = self._name_edit.text().strip()
        note = self._note_edit.toPlainText()
        if name == self._loaded_name and note == self._loaded_note:
            return False
        try:
            with session_scope() as session:
                PersonGroupService(session).update_group(
                    self._current_id, name=name, description=note
                )
        except ValueError as exc:
            if show_errors:
                QMessageBox.warning(self, t("group_save_error_title"), str(exc))
            return False
        except Exception:
            log.exception("Failed to update group id=%s", self._current_id)
            if show_errors:
                QMessageBox.warning(
                    self, t("group_save_error_title"), t("group_save_error_title")
                )
            return False
        self._loaded_name = name
        self._loaded_note = note
        return True

    def _relabel_item(self, item: Optional[QListWidgetItem]) -> None:
        """Re-render *item*'s label after a name change (member count unchanged)."""
        if item is None:
            return
        count = item.data(_ROLE_COUNT) or 0
        item.setText(self._item_label(self._loaded_name, count))

    def _auto_save(self) -> None:
        """Persist pending name/note edits triggered by losing focus / Enter."""
        if self._persist_current(show_errors=True):
            self._relabel_item(self._list.currentItem())
            self.groups_changed.emit()

    def _on_save(self) -> None:
        if self._current_id is None:
            return
        if self._persist_current(show_errors=True):
            self._relabel_item(self._list.currentItem())
            self.groups_changed.emit()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        # Auto-save the note when the QTextEdit loses focus.
        if watched is self._note_edit and event.type() == QEvent.FocusOut:
            self._auto_save()
        return super().eventFilter(watched, event)

    def hideEvent(self, event) -> None:  # noqa: N802
        # Persist quietly when the tab/dialog goes away so edits are never lost.
        if self._persist_current(show_errors=False):
            self.groups_changed.emit()
        super().hideEvent(event)

    def _on_member_selection_changed(
        self, current: Optional[QListWidgetItem], _prev
    ) -> None:
        self._remove_member_btn.setEnabled(current is not None)

    def _on_add_member(self) -> None:
        if self._current_id is None:
            return
        # Save any pending name/note edit first; _after_membership_change reloads
        # the editor and would otherwise discard it.
        self._persist_current(show_errors=True)
        try:
            with session_scope() as session:
                svc = PersonGroupService(session)
                members = {p.id for p in svc.get_persons_in_group(self._current_id)}
                persons = (
                    session.query(Person)
                    .filter(Person.is_protected.is_(False))
                    .order_by(Person.name)
                    .all()
                )
                # Snapshot the data the selector needs (id, name, unknown flag)
                # into detached stand-ins before the session closes.  Carrying
                # is_auto_named keeps unnamed "Unknown N" clusters out of the
                # base list (they still surface on search).
                candidates = [
                    SimpleNamespace(
                        id=p.id,
                        name=p.name,
                        is_auto_named=p.is_auto_named,
                        is_protected=False,
                    )
                    for p in persons
                    if p.id not in members
                ]
        except Exception:
            log.exception("Failed to load persons for group id=%s", self._current_id)
            return

        if not candidates:
            QMessageBox.information(
                self,
                t("group_member_add_title"),
                t("group_member_none_available"),
            )
            return

        dialog = AddMemberDialog(candidates, self)
        if dialog.exec() != QDialog.Accepted:
            return
        person_id = dialog.selected_person_id()
        if person_id is None:
            return
        try:
            with session_scope() as session:
                PersonGroupService(session).add_person_to_group(
                    person_id, self._current_id
                )
        except ValueError as exc:
            QMessageBox.warning(self, t("group_member_add_title"), str(exc))
            return
        except Exception:
            log.exception(
                "Failed to add person=%s to group=%s", person_id, self._current_id
            )
            return
        self._after_membership_change()

    def _on_remove_member(self) -> None:
        if self._current_id is None:
            return
        item = self._members_list.currentItem()
        if item is None:
            return
        # Save any pending name/note edit first (the reload below would lose it).
        self._persist_current(show_errors=True)
        person_id = item.data(_ROLE_ID)
        try:
            with session_scope() as session:
                PersonGroupService(session).remove_person_from_group(
                    person_id, self._current_id
                )
        except Exception:
            log.exception(
                "Failed to remove person=%s from group=%s",
                person_id,
                self._current_id,
            )
            return
        self._after_membership_change()

    def _after_membership_change(self) -> None:
        """Refresh member list + counts and notify listeners after a change."""
        gid = self._current_id
        if gid is None:
            return
        # Re-selecting the group re-triggers _show_editor, which reloads both
        # the member list and the member-count label; the left list's "(N)"
        # counts are rebuilt too.
        self._reload_list(select_id=gid)
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
