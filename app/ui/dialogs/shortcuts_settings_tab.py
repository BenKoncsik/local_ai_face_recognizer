"""Keyboard shortcuts settings tab — part of the Settings dialog."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.services.shortcut_service import ShortcutDef, get_shortcut_service, normalize_key
from app.ui.i18n import t


class _ShortcutRow(QWidget):
    """A single row in the shortcut table.

    Emits signals so the parent tab can coordinate capture-mode
    across all rows (only one row can be in capture mode at a time).
    """

    modify_requested = Signal(str)       # sc_id
    delete_requested = Signal(str)       # sc_id
    save_requested   = Signal(str, str)  # sc_id, new_key
    cancel_requested = Signal(str)       # sc_id

    _STYLE_NORMAL  = "font-family: monospace; color: #ccc; background: #2a2a2a; padding: 2px 6px; border-radius: 3px;"
    _STYLE_CAPTURE = "font-family: monospace; color: #ffcc00; background: #2a2a2a; padding: 2px 6px; border-radius: 3px;"
    _STYLE_CONFLICT= "font-family: monospace; color: #f44336; background: #2a2a2a; padding: 2px 6px; border-radius: 3px;"
    _STYLE_EMPTY   = "font-family: monospace; color: #555; font-style: italic; background: #2a2a2a; padding: 2px 6px; border-radius: 3px;"

    def __init__(self, sc: ShortcutDef, parent=None) -> None:
        super().__init__(parent)
        self._sc = sc
        self._pending_key = ""
        self._in_capture = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 3, 4, 3)
        layout.setSpacing(6)

        # Function name
        self._name_lbl = QLabel(t(self._sc.name_key))
        self._name_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(self._name_lbl, stretch=1)

        # Current / pending key display
        self._key_lbl = QLabel()
        self._key_lbl.setFixedWidth(130)
        self._key_lbl.setAlignment(Qt.AlignCenter)
        self._update_key_display()
        layout.addWidget(self._key_lbl)

        # Modify button (normal mode)
        self._mod_btn = QPushButton(t("sc_modify_btn"))
        self._mod_btn.setFixedWidth(68)
        self._mod_btn.clicked.connect(lambda: self.modify_requested.emit(self._sc.id))
        layout.addWidget(self._mod_btn)

        # Save button (capture mode only)
        self._save_btn = QPushButton(t("sc_save_btn"))
        self._save_btn.setFixedWidth(58)
        self._save_btn.clicked.connect(self._on_save)
        self._save_btn.setVisible(False)
        layout.addWidget(self._save_btn)

        # Cancel button (capture mode only)
        self._cancel_btn = QPushButton(t("sc_cancel_btn"))
        self._cancel_btn.setFixedWidth(58)
        self._cancel_btn.clicked.connect(lambda: self.cancel_requested.emit(self._sc.id))
        self._cancel_btn.setVisible(False)
        layout.addWidget(self._cancel_btn)

        # Delete button (normal mode, disabled for non-deletable shortcuts)
        self._del_btn = QPushButton(t("sc_delete_btn"))
        self._del_btn.setFixedWidth(65)
        self._del_btn.setEnabled(self._sc.deletable)
        self._del_btn.clicked.connect(lambda: self.delete_requested.emit(self._sc.id))
        layout.addWidget(self._del_btn)

    # ── Display helpers ───────────────────────────────────────────────────

    def _update_key_display(self) -> None:
        key = self._sc.current_key
        if not key:
            self._key_lbl.setText(t("sc_not_set"))
            self._key_lbl.setStyleSheet(self._STYLE_EMPTY)
        else:
            self._key_lbl.setText(key)
            self._key_lbl.setStyleSheet(self._STYLE_NORMAL)

    # ── Capture mode ──────────────────────────────────────────────────────

    def enter_capture_mode(self) -> None:
        self._in_capture = True
        self._pending_key = ""
        self._key_lbl.setText(t("sc_press_key"))
        self._key_lbl.setStyleSheet(self._STYLE_CAPTURE)
        self._mod_btn.setVisible(False)
        self._del_btn.setVisible(False)
        self._save_btn.setVisible(True)
        self._cancel_btn.setVisible(True)

    def leave_capture_mode(self) -> None:
        self._in_capture = False
        self._pending_key = ""
        self._update_key_display()
        self._mod_btn.setVisible(True)
        self._del_btn.setVisible(True)
        self._save_btn.setVisible(False)
        self._cancel_btn.setVisible(False)

    def show_pending_key(self, key_str: str, conflict: bool = False) -> None:
        self._pending_key = key_str
        self._key_lbl.setText(key_str)
        self._key_lbl.setStyleSheet(self._STYLE_CONFLICT if conflict else self._STYLE_CAPTURE)

    def update_from_service(self) -> None:
        """Refresh from the service after an external change."""
        sc = get_shortcut_service().get(self._sc.id)
        if sc:
            self._sc = sc
        if not self._in_capture:
            self._update_key_display()

    # ── Actions ───────────────────────────────────────────────────────────

    def _on_save(self) -> None:
        self.save_requested.emit(self._sc.id, self._pending_key)


# ── Main tab widget ───────────────────────────────────────────────────────────

class ShortcutsSettingsTab(QWidget):
    """Settings tab for configuring keyboard shortcuts."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._svc = get_shortcut_service()
        self._rows: dict[str, _ShortcutRow] = {}
        self._capturing_id: Optional[str] = None
        self.setFocusPolicy(Qt.StrongFocus)
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 10, 8, 8)
        outer.setSpacing(8)

        # Global on/off toggle
        self._enable_chk = QCheckBox(t("sc_enable_all"))
        self._enable_chk.setChecked(self._svc.is_enabled())
        self._enable_chk.toggled.connect(self._on_global_toggle)
        outer.addWidget(self._enable_chk)

        # Conflict warning (hidden by default)
        self._conflict_lbl = QLabel()
        self._conflict_lbl.setStyleSheet("color: #f44336; font-size: 11px;")
        self._conflict_lbl.setWordWrap(True)
        self._conflict_lbl.setVisible(False)
        outer.addWidget(self._conflict_lbl)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        self._list_layout = QVBoxLayout(content)
        self._list_layout.setSpacing(2)
        self._list_layout.setContentsMargins(0, 0, 0, 0)

        self._populate_rows()
        self._list_layout.addStretch()

        scroll.setWidget(content)
        outer.addWidget(scroll, stretch=1)

    def _populate_rows(self) -> None:
        shortcuts = self._svc.all_shortcuts()
        current_cat = None

        header_font = QFont()
        header_font.setBold(True)

        for sc in shortcuts:
            cat = t(sc.category_key)
            if cat != current_cat:
                current_cat = cat
                if self._list_layout.count() > 0:
                    spacer = QWidget()
                    spacer.setFixedHeight(6)
                    self._list_layout.addWidget(spacer)
                hdr = QLabel(cat)
                hdr.setFont(header_font)
                hdr.setStyleSheet("color: #aaa; padding-top: 4px;")
                self._list_layout.addWidget(hdr)
                sep = QWidget()
                sep.setFixedHeight(1)
                sep.setStyleSheet("background: #383838;")
                self._list_layout.addWidget(sep)

            row = _ShortcutRow(sc, self)
            row.modify_requested.connect(self._on_modify)
            row.delete_requested.connect(self._on_delete)
            row.save_requested.connect(self._on_save)
            row.cancel_requested.connect(self._on_cancel)
            self._rows[sc.id] = row
            self._list_layout.addWidget(row)

    # ── Slot handlers ─────────────────────────────────────────────────────

    def _on_global_toggle(self, enabled: bool) -> None:
        self._svc.set_enabled(enabled)

    def _on_modify(self, sc_id: str) -> None:
        # Cancel any previous capture
        if self._capturing_id and self._capturing_id != sc_id:
            self._finish_capture(self._capturing_id)

        self._capturing_id = sc_id
        self._conflict_lbl.setVisible(False)
        row = self._rows.get(sc_id)
        if row:
            row.enter_capture_mode()

        self._svc.set_capturing(True)
        self.grabKeyboard()

    def _on_delete(self, sc_id: str) -> None:
        self._svc.set_key(sc_id, "")
        row = self._rows.get(sc_id)
        if row:
            row.update_from_service()
        self._conflict_lbl.setVisible(False)

    def _on_save(self, sc_id: str, new_key: str) -> None:
        if not new_key:
            self._finish_capture(sc_id)
            return

        sc = self._svc.get(sc_id)
        conflict = self._svc.find_conflict(new_key, exclude_id=sc_id)
        if conflict:
            self._conflict_lbl.setText(t("sc_conflict_msg", func=t(conflict.name_key)))
            self._conflict_lbl.setVisible(True)
            # Don't save — keep in capture mode so user can try again
            return

        self._svc.set_key(sc_id, new_key)
        self._finish_capture(sc_id)

    def _on_cancel(self, sc_id: str) -> None:
        self._finish_capture(sc_id)

    def _finish_capture(self, sc_id: str) -> None:
        row = self._rows.get(sc_id)
        if row:
            row.update_from_service()
            row.leave_capture_mode()
        if self._capturing_id == sc_id:
            self._capturing_id = None
        self._conflict_lbl.setVisible(False)
        self._svc.set_capturing(False)
        self.releaseKeyboard()

    # ── Key capture ───────────────────────────────────────────────────────

    def keyPressEvent(self, event) -> None:
        if not self._capturing_id:
            super().keyPressEvent(event)
            return

        key_str = normalize_key(event)

        # Escape cancels capture mode
        if key_str == "Esc":
            self._on_cancel(self._capturing_id)
            event.accept()
            return

        if not key_str:
            event.accept()
            return

        # Show the pending key and check for conflicts
        sc = self._svc.get(self._capturing_id)
        conflict = self._svc.find_conflict(key_str, exclude_id=self._capturing_id)
        if conflict:
            self._conflict_lbl.setText(t("sc_conflict_msg", func=t(conflict.name_key)))
            self._conflict_lbl.setVisible(True)
        else:
            self._conflict_lbl.setVisible(False)

        row = self._rows.get(self._capturing_id)
        if row:
            row.show_pending_key(key_str, conflict=conflict is not None)

        event.accept()
