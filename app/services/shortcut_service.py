"""Central keyboard shortcut service.

All configurable key bindings live here.  Other modules register handlers;
the MainWindow's app-level event filter dispatches them.
"""

from __future__ import annotations

import copy
import logging
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from PySide6.QtCore import QKeyCombination, Qt
from PySide6.QtGui import QKeyEvent, QKeySequence

# On macOS F11 is grabbed by Mission Control ("Show Desktop") before Qt
# sees it, so we use Ctrl+Shift+F (⌘⇧F) as the fullscreen default instead.
_FULLSCREEN_DEFAULT = "Ctrl+Shift+F" if sys.platform == "darwin" else "F11"

log = logging.getLogger(__name__)

_SETTINGS_NS = "shortcuts"


@dataclass
class ShortcutDef:
    id: str
    name_key: str       # i18n key
    category_key: str   # i18n key
    default_key: str    # canonical portable string, e.g. "Ctrl+,"
    current_key: str = ""
    deletable: bool = True  # False → key field shown but cannot be cleared

    def __post_init__(self) -> None:
        if not self.current_key:
            self.current_key = self.default_key


_DEFAULTS: List[ShortcutDef] = [
    # ── General ──────────────────────────────────────────────────────────
    ShortcutDef("general.settings",      "sc_fn_settings",        "sc_cat_general", "Ctrl+,"),
    ShortcutDef("general.search_focus",  "sc_fn_search_focus",    "sc_cat_general", "Ctrl+F"),
    ShortcutDef("general.fullscreen",    "sc_fn_fullscreen",      "sc_cat_general", _FULLSCREEN_DEFAULT),
    ShortcutDef("general.log_panel",     "sc_fn_log_panel",       "sc_cat_general", "Ctrl+L"),
    # ── Image browser ─────────────────────────────────────────────────────
    ShortcutDef("image.previous",        "sc_fn_image_prev",      "sc_cat_image",   "Left"),
    ShortcutDef("image.next",            "sc_fn_image_next",      "sc_cat_image",   "Right"),
    ShortcutDef("image.manual_sel",      "sc_fn_manual_sel",      "sc_cat_image",   "B"),
    ShortcutDef("face.assign",           "sc_fn_face_assign",     "sc_cat_image",   "A"),
    ShortcutDef("face.confirm",          "sc_fn_face_confirm",    "sc_cat_image",   "Return"),
    ShortcutDef("image.deselect",        "sc_fn_deselect",        "sc_cat_image",   "Esc",     deletable=False),
    ShortcutDef("bbox.delete",           "sc_fn_bbox_delete",     "sc_cat_image",   "Del"),
    ShortcutDef("bbox.edit",             "sc_fn_bbox_edit",       "sc_cat_image",   "E"),
    ShortcutDef("bbox.undo",             "sc_fn_bbox_undo",       "sc_cat_image",   "Ctrl+Z"),
    ShortcutDef("bbox.redo",             "sc_fn_bbox_redo",       "sc_cat_image",   "Ctrl+Y"),
    ShortcutDef("bbox.next",             "sc_fn_bbox_next",       "sc_cat_image",   "]"),
    ShortcutDef("bbox.prev",             "sc_fn_bbox_prev",       "sc_cat_image",   "["),
    ShortcutDef("image.zoom_in",         "sc_fn_zoom_in",         "sc_cat_image",   "+"),
    ShortcutDef("image.zoom_out",        "sc_fn_zoom_out",        "sc_cat_image",   "-"),
    ShortcutDef("image.fit",             "sc_fn_fit",             "sc_cat_image",   "0"),
    ShortcutDef("image.info",            "sc_fn_info",            "sc_cat_image",   "I"),
    # ── Faces / persons ───────────────────────────────────────────────────
    ShortcutDef("person.new",            "sc_fn_person_new",      "sc_cat_faces",   "Ctrl+N"),
    ShortcutDef("person.rename",         "sc_fn_person_rename",   "sc_cat_faces",   "F2"),
    ShortcutDef("person.merge",          "sc_fn_person_merge",    "sc_cat_faces",   "Ctrl+M"),
    ShortcutDef("person.reassign",       "sc_fn_person_reassign", "sc_cat_faces",   "R"),
    ShortcutDef("person.exclude",        "sc_fn_person_exclude",  "sc_cat_faces",   "X"),
    # ── Collage ───────────────────────────────────────────────────────────
    ShortcutDef("collage.import",        "sc_fn_collage_import",  "sc_cat_collage", "Ctrl+I"),
    ShortcutDef("collage.face_overlay",  "sc_fn_face_overlay",    "sc_cat_collage", "F"),
    ShortcutDef("collage.node_delete",   "sc_fn_node_delete",     "sc_cat_collage", "Del"),
    ShortcutDef("collage.html_export",   "sc_fn_html_export",     "sc_cat_collage", "Ctrl+H"),
]


# ── Key normalisation ─────────────────────────────────────────────────────────

def normalize_key(event: QKeyEvent) -> str:
    """Convert a QKeyEvent to a portable shortcut string, e.g. 'Ctrl+A'."""
    key = event.key()
    if key in (
        Qt.Key_Control, Qt.Key_Shift, Qt.Key_Alt, Qt.Key_Meta,
        Qt.Key_unknown, 0,
    ):
        return ""
    # QKeyCombination is the correct API for PySide6 6.4+ where
    # KeyboardModifiers is no longer directly int()-convertible.
    combo = QKeyCombination(event.modifiers(), Qt.Key(key))
    seq = QKeySequence(combo)
    return seq.toString(QKeySequence.PortableText)


def is_input_widget(widget) -> bool:
    """Return True if *widget* is an editable text field that should suppress shortcuts."""
    if widget is None:
        return False
    from PySide6.QtWidgets import (
        QAbstractSpinBox, QLineEdit, QPlainTextEdit, QTextEdit,
    )
    return isinstance(widget, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox))


# ── Service ───────────────────────────────────────────────────────────────────

class ShortcutService:
    """Manages shortcut definitions, persistence and event dispatch."""

    def __init__(self) -> None:
        self._shortcuts: Dict[str, ShortcutDef] = {}
        self._handlers: Dict[str, Callable[[], None]] = {}
        self._enabled: bool = True
        self._capturing: bool = False   # True while settings capture mode is active
        self._init_defaults()
        self._load()

    # ── Init ─────────────────────────────────────────────────────────────

    def _init_defaults(self) -> None:
        for d in _DEFAULTS:
            self._shortcuts[d.id] = copy.copy(d)

    # Older builds stored "Escape"/"Delete"; normalize_key now returns "Esc"/"Del".
    _KEY_MIGRATE = {"Escape": "Esc", "Delete": "Del"}

    def _load(self) -> None:
        from app.app_settings import app_qsettings
        qs = app_qsettings()
        self._enabled = qs.value(f"{_SETTINGS_NS}/enabled", True, type=bool)
        for sc in self._shortcuts.values():
            stored = qs.value(f"{_SETTINGS_NS}/{sc.id}", None)
            if stored is not None:
                sc.current_key = self._KEY_MIGRATE.get(stored, stored)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self) -> None:
        from app.app_settings import app_qsettings
        qs = app_qsettings()
        qs.setValue(f"{_SETTINGS_NS}/enabled", self._enabled)
        for sc in self._shortcuts.values():
            qs.setValue(f"{_SETTINGS_NS}/{sc.id}", sc.current_key)

    # ── Accessors ─────────────────────────────────────────────────────────

    def is_enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self.save()

    def all_shortcuts(self) -> List[ShortcutDef]:
        return list(self._shortcuts.values())

    def get(self, sc_id: str) -> Optional[ShortcutDef]:
        return self._shortcuts.get(sc_id)

    def set_key(self, sc_id: str, key: str) -> None:
        """Assign a new key string to a shortcut and persist."""
        if sc_id not in self._shortcuts:
            return
        self._shortcuts[sc_id].current_key = key
        self.save()

    def find_conflict(self, key: str, exclude_id: str = "") -> Optional[ShortcutDef]:
        """Return the first shortcut already mapped to *key*, or None."""
        if not key:
            return None
        for sc in self._shortcuts.values():
            if sc.id != exclude_id and sc.current_key == key:
                return sc
        return None

    # ── Capture-mode flag (used by ShortcutsSettingsTab) ─────────────────

    def set_capturing(self, capturing: bool) -> None:
        self._capturing = capturing

    def is_capturing(self) -> bool:
        return self._capturing

    # ── Handler registry ──────────────────────────────────────────────────

    def register(self, sc_id: str, handler: Callable[[], None]) -> None:
        self._handlers[sc_id] = handler

    def unregister(self, sc_id: str) -> None:
        self._handlers.pop(sc_id, None)

    # ── Dispatch ──────────────────────────────────────────────────────────

    def dispatch(self, key_str: str) -> bool:
        """Try to dispatch *key_str*. Returns True if a registered handler ran."""
        if not self._enabled or not key_str or self._capturing:
            return False
        for sc in self._shortcuts.values():
            if sc.current_key != key_str:
                continue
            handler = self._handlers.get(sc.id)
            if handler is None:
                continue
            try:
                handler()
            except Exception:
                log.exception("Shortcut handler %r raised", sc.id)
            return True
        return False


# ── Module-level singleton ────────────────────────────────────────────────────

_service: Optional[ShortcutService] = None


def get_shortcut_service() -> ShortcutService:
    global _service
    if _service is None:
        _service = ShortcutService()
    return _service
