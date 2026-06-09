"""Central keyboard shortcut service.

All configurable key bindings live here.  Other modules register handlers;
the service binds each registered shortcut to a QShortcut on a host widget
(the MainWindow) so Qt dispatches it natively.

Historically a single application-wide event filter installed on the
QApplication dispatched these shortcuts.  That crashes PySide6 with a SIGSEGV
(in PySide::getWrapperForQObject, via the QMainWindow event-filter wrapper)
as soon as a QWebEngineView is alive — WebEngine hands the filter native
QObjects whose Python wrappers fail to marshal.  QShortcut avoids the global
filter entirely, and its native ShortcutOverride handling preserves the old
behaviour of letting text fields keep editing keys (Ctrl+Z, etc.).
"""

from __future__ import annotations

import copy
import logging
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from PySide6.QtCore import QKeyCombination, Qt
from PySide6.QtGui import QKeyEvent, QKeySequence, QShortcut

# On macOS F11 is grabbed by Mission Control ("Show Desktop") before Qt
# sees it, so we use Ctrl+Shift+F (⌘⇧F) as the fullscreen default instead.
_FULLSCREEN_DEFAULT = "Ctrl+Shift+F" if sys.platform == "darwin" else "F11"

log = logging.getLogger(__name__)

_SETTINGS_NS = "shortcuts"


# ── Contexts (pages) ────────────────────────────────────────────────────────
#
# A shortcut belongs to exactly one *context* (a page / functional namespace).
# The same key combination may be reused freely across different contexts —
# conflicts are only checked within the same context.  Contexts listed in
# ``_GLOBAL_CONTEXTS`` are active on every page, so their keys must stay unique
# against *all* other contexts (they would otherwise shadow a page binding).
_GLOBAL_CONTEXTS = frozenset({"general"})


@dataclass
class ShortcutDef:
    id: str
    name_key: str       # i18n key
    category_key: str   # i18n key — also used as the page/context label
    default_key: str    # canonical portable string, e.g. "Ctrl+,"
    current_key: str = ""
    deletable: bool = True  # False → key field shown but cannot be cleared
    context: str = ""   # page / functional namespace; derived from category if empty

    def __post_init__(self) -> None:
        if not self.current_key:
            self.current_key = self.default_key
        if not self.context:
            # Derive the context from the category (e.g. "sc_cat_image" → "image").
            self.context = self.category_key.replace("sc_cat_", "") or "general"

    @property
    def is_global(self) -> bool:
        """True if this shortcut is active on every page."""
        return self.context in _GLOBAL_CONTEXTS


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
    ShortcutDef("face.cycle_next",       "sc_fn_face_cycle_next", "sc_cat_image",   "Ctrl+A"),
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
        self._qshortcuts: Dict[str, QShortcut] = {}
        self._host = None  # QWidget that owns the QShortcut objects
        self._enabled: bool = True
        self._capturing: bool = False   # True while settings capture mode is active
        # The currently active page/context. ``None`` means "no scoping" — every
        # shortcut fires (backward-compatible fallback used before any page is
        # selected). Once set, only global shortcuts and those belonging to the
        # active context are live.
        self._active_context: Optional[str] = None
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
        # Flush to disk explicitly. The transient QSettings object would
        # otherwise only persist on destruction, which is unreliable on
        # Windows (writes can be silently dropped) — so modified shortcuts
        # were not being saved there.
        qs.sync()

    # ── Accessors ─────────────────────────────────────────────────────────

    def is_enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self.save()
        self._refresh_enabled()

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
        self._sync_shortcut(sc_id)

    @staticmethod
    def _contexts_overlap(a: str, b: str) -> bool:
        """True if two contexts can be active at the same time.

        Same context → overlap. A global context (active on every page) overlaps
        with every other context, so its keys must stay unique everywhere.
        """
        if a == b:
            return True
        return a in _GLOBAL_CONTEXTS or b in _GLOBAL_CONTEXTS

    def find_conflict(
        self, key: str, exclude_id: str = "", context: Optional[str] = None
    ) -> Optional[ShortcutDef]:
        """Return a conflicting shortcut for *key*, scoped to its context.

        A conflict only exists between shortcuts whose contexts overlap (same
        page, or one of them is global). The same key on two different pages is
        therefore allowed. ``context`` defaults to the context of *exclude_id*
        (the shortcut being edited); when neither is known the check falls back
        to a global, context-agnostic search.
        """
        if not key:
            return None
        if context is None:
            editing = self._shortcuts.get(exclude_id)
            context = editing.context if editing else None
        for sc in self._shortcuts.values():
            if sc.id == exclude_id or sc.current_key != key:
                continue
            if context is None or self._contexts_overlap(context, sc.context):
                return sc
        return None

    # ── Capture-mode flag (used by ShortcutsSettingsTab) ─────────────────

    def set_capturing(self, capturing: bool) -> None:
        self._capturing = capturing
        # While the settings dialog is capturing a new key combination, our own
        # QShortcuts must not swallow it — disable them all for the duration.
        self._refresh_enabled()

    def is_capturing(self) -> bool:
        return self._capturing

    # ── Active context (page) ─────────────────────────────────────────────

    def set_active_context(self, context: Optional[str]) -> None:
        """Select the active page. Only its shortcuts (plus global ones) fire.

        Passing ``None`` disables context scoping (every shortcut is live).
        """
        if context == self._active_context:
            return
        self._active_context = context
        self._refresh_enabled()

    def active_context(self) -> Optional[str]:
        return self._active_context

    # ── Host / QShortcut wiring ───────────────────────────────────────────

    def set_host(self, host) -> None:
        """Set the QWidget that owns the QShortcut objects (the MainWindow).

        Creates QShortcuts for every already-registered handler; later
        register() calls create theirs on demand.
        """
        self._host = host
        for sc_id in list(self._handlers):
            self._sync_shortcut(sc_id)

    # ── Handler registry ──────────────────────────────────────────────────

    def register(self, sc_id: str, handler: Callable[[], None]) -> None:
        self._handlers[sc_id] = handler
        self._sync_shortcut(sc_id)

    def unregister(self, sc_id: str) -> None:
        self._handlers.pop(sc_id, None)
        qsc = self._qshortcuts.pop(sc_id, None)
        if qsc is not None:
            qsc.setParent(None)
            qsc.deleteLater()

    # ── QShortcut management ──────────────────────────────────────────────

    def _sync_shortcut(self, sc_id: str) -> None:
        """Create / update / drop the QShortcut backing *sc_id*."""
        if self._host is None:
            return
        sc = self._shortcuts.get(sc_id)
        handler = self._handlers.get(sc_id)
        qsc = self._qshortcuts.get(sc_id)

        # No handler or no key → no live shortcut.
        if sc is None or handler is None or not sc.current_key:
            if qsc is not None:
                qsc.setParent(None)
                qsc.deleteLater()
                self._qshortcuts.pop(sc_id, None)
            return

        seq = QKeySequence(sc.current_key)
        if qsc is None:
            qsc = QShortcut(seq, self._host)
            qsc.setContext(Qt.WindowShortcut)
            qsc.activated.connect(lambda _id=sc_id: self._fire(_id))
            self._qshortcuts[sc_id] = qsc
        else:
            qsc.setKey(seq)
        qsc.setEnabled(self._shortcut_active(sc_id))

    def _active(self) -> bool:
        """Global gate: shortcuts are enabled and not in settings-capture mode."""
        return self._enabled and not self._capturing

    def _context_active(self, context: str) -> bool:
        """True if *context* is live under the current active page."""
        if self._active_context is None:
            return True  # no scoping selected yet → everything is live
        return context in _GLOBAL_CONTEXTS or context == self._active_context

    def _shortcut_active(self, sc_id: str) -> bool:
        """True if the shortcut may fire right now (gate + context)."""
        if not self._active():
            return False
        sc = self._shortcuts.get(sc_id)
        return sc is not None and self._context_active(sc.context)

    def _refresh_enabled(self) -> None:
        for sc_id, qsc in self._qshortcuts.items():
            qsc.setEnabled(self._shortcut_active(sc_id))

    def _fire(self, sc_id: str) -> None:
        if not self._shortcut_active(sc_id):
            return
        handler = self._handlers.get(sc_id)
        if handler is None:
            return
        try:
            handler()
        except Exception:
            log.exception("Shortcut handler %r raised", sc_id)

    # ── Dispatch ──────────────────────────────────────────────────────────

    def dispatch(self, key_str: str) -> bool:
        """Try to dispatch *key_str*. Returns True if a registered handler ran."""
        if not self._enabled or not key_str or self._capturing:
            return False
        for sc in self._shortcuts.values():
            if sc.current_key != key_str:
                continue
            if not self._context_active(sc.context):
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
