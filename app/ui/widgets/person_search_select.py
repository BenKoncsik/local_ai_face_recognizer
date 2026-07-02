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

from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
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
from app.utils.person_search import (
    PersonEntry,
    filter_entries,
    person_is_unknown,
    search_persons,
)

_ROLE_ID = Qt.UserRole
_MAX_VISIBLE_ITEMS = 8
_MIN_VISIBLE_ITEMS = 5  # keep the scroll area usable in tight popups
_ITEM_HEIGHT = 24  # px — approximate row height
_MATCH_SORT_QSETTINGS_KEY = "person_search/match_sort"


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
        self._match_scores: Dict[int, float] = {}
        # True while background match scoring is still running for the current
        # face — shows the "computing…" hint until set_match_scores() arrives.
        self._scores_pending: bool = False
        # Hide placeholder/unknown identities from the base (empty-query) list.
        # On by default so every selector follows the shared rule; navigation
        # surfaces that must list unknown clusters (e.g. the persons sidebar)
        # opt out via :meth:`set_hide_unknown_in_base`.
        self._hide_unknown_in_base: bool = True

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

        # Optional "order by face-match similarity" toggle.  Hidden until the
        # caller supplies match scores via :meth:`set_match_scores`, so the
        # default behaviour (and every selector that has no face context) is
        # unchanged.
        # Gap above the checkbox so it is not crammed against the search box.
        # Shown/hidden together with the checkbox (see set_match_scores).
        self._match_spacer_top = QWidget()
        self._match_spacer_top.setFixedHeight(8)
        self._match_spacer_top.setVisible(False)
        layout.addWidget(self._match_spacer_top)

        # Shown while background match scoring runs, so the user knows the
        # percentages are still being computed rather than missing.
        self._computing_label = QLabel(t("pss_match_computing"))
        self._computing_label.setStyleSheet(
            "color: #888; font-style: italic; font-size: 11px;"
        )
        self._computing_label.setVisible(False)
        layout.addWidget(self._computing_label)

        self._match_checkbox = QCheckBox(t("pss_match_sort"))
        self._match_checkbox.setChecked(self._load_match_pref())
        self._match_checkbox.setVisible(False)
        # Shrink only the *font* (via QFont, not a stylesheet) so the long label
        # fits inside a narrow popup such as the 230px inline assign editor.
        # A widget-level QCheckBox stylesheet would drop the theme's styled
        # ::indicator and let a default indicator overlap the text, so we leave
        # the indicator and spacing to the global theme.
        _cb_font = self._match_checkbox.font()
        if _cb_font.pointSize() > 0:
            _cb_font.setPointSize(max(1, _cb_font.pointSize() - 2))
        self._match_checkbox.setFont(_cb_font)
        self._match_checkbox.setMinimumWidth(0)
        self._match_checkbox.toggled.connect(self._on_match_toggled)
        layout.addWidget(self._match_checkbox)

        # Dedicated gap below the checkbox so it is not crammed against the
        # list.  A QCheckBox ignores setContentsMargins (it has no inner
        # layout), so we use a real spacer widget that is shown/hidden together
        # with the checkbox — no extra gap when the checkbox is absent.
        self._match_spacer = QWidget()
        self._match_spacer.setFixedHeight(8)
        self._match_spacer.setVisible(False)
        layout.addWidget(self._match_spacer)

        self._list = QListWidget()
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._list.setMaximumHeight(_ITEM_HEIGHT * _MAX_VISIBLE_ITEMS + 4)
        # Guarantee a usable scrollable height: without a minimum the list
        # collapses to ~1 row when the host popup (e.g. the inline assign
        # editor) sizes itself to the content's size hint.
        self._list.setMinimumHeight(_ITEM_HEIGHT * _MIN_VISIBLE_ITEMS)
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
        self._entries = [
            PersonEntry(p.id, p.name, is_unknown=person_is_unknown(p))
            for p in sorted_persons
        ]
        self._refresh_list()

    def set_entries(self, entries: List[PersonEntry]) -> None:
        """Populate with pre-built :class:`PersonEntry` objects (e.g. with custom display_text)."""
        self._entries = list(entries)
        self._refresh_list()

    def set_hide_unknown_in_base(self, hide: bool) -> None:
        """Toggle hiding of unknown/placeholder persons in the base list.

        When ``True`` (the default) entries flagged
        :attr:`~app.utils.person_search.PersonEntry.is_unknown` are absent from
        the unfiltered list but still appear when the user searches.  Set
        ``False`` for navigation surfaces that must always show every person
        (e.g. the persons sidebar, where unknown clusters need to be clickable).
        """
        self._hide_unknown_in_base = bool(hide)
        self._refresh_list()

    def set_match_scores(
        self,
        scores: Optional[Dict[int, float]],
        *,
        default_sort: bool = False,
    ) -> None:
        """Provide per-person face-match similarity scores.

        *scores* maps ``person_id`` → similarity (higher is a closer match).
        When non-empty, the "order by face-match similarity" checkbox becomes
        visible; if it is checked the list is reordered best-match-first and a
        percentage is shown next to each scored person.  Passing ``None`` or an
        empty mapping hides the checkbox and restores the default ordering, so
        a missing embedding never breaks the selector.

        *default_sort* — when ``True`` and scores are present, the match-sort
        checkbox is turned on for this selector regardless of the saved global
        preference, so the list opens best-match-first (e.g. the merge dialog).
        The toggle is set without persisting, so the user's global preference is
        untouched; they can still uncheck it to fall back to name order.
        """
        self._match_scores = dict(scores or {})
        self._scores_pending = False
        self._computing_label.setVisible(False)
        has_scores = bool(self._match_scores)
        self._match_checkbox.setVisible(has_scores)
        self._match_spacer_top.setVisible(has_scores)
        self._match_spacer.setVisible(has_scores)
        if has_scores and default_sort and not self._match_checkbox.isChecked():
            # Force match-sort on for this selector without writing the global
            # QSettings preference (block the toggled signal that persists it).
            self._match_checkbox.blockSignals(True)
            self._match_checkbox.setChecked(True)
            self._match_checkbox.blockSignals(False)
        self._refresh_list()

    def set_scores_pending(self) -> None:
        """Signal that match scores are being computed in the background.

        Shows a small "computing similarity…" hint (and hides any stale scores)
        until :meth:`set_match_scores` delivers the result, so the user knows
        the percentages are on their way rather than missing.
        """
        self._match_scores = {}
        self._scores_pending = True
        self._computing_label.setVisible(True)
        self._match_checkbox.setVisible(False)
        self._match_spacer_top.setVisible(False)
        self._match_spacer.setVisible(False)
        self._refresh_list()

    def current_person_id(self) -> Optional[int]:
        """Return the currently selected person_id, or ``None``."""
        item = self._list.currentItem()
        if item is None:
            return None
        data = item.data(_ROLE_ID)
        return int(data) if data is not None else None

    def set_current_by_id(self, person_id: Optional[int]) -> None:
        """Highlight the list row whose person_id matches *person_id* and
        scroll it into view.

        Scrolling is deterministic so a freshly populated selector never
        inherits the previous target's scroll position: if the row is found it
        is revealed (centred); otherwise the list snaps back to the top.
        """
        self._selected_id = person_id
        self._sync_selection()
        item = self._list.currentItem()
        if item is not None:
            self._list.scrollToItem(item, QListWidget.PositionAtCenter)
        else:
            self._list.scrollToTop()

    def clear_selection(self) -> None:
        """Deselect all rows and forget the stored selection."""
        self._selected_id = None
        self._list.clearSelection()
        self._list.blockSignals(True)
        self._list.setCurrentItem(None)
        self._list.blockSignals(False)

    def reset_filter(self) -> None:
        """Clear the search text and show the full (unfiltered) person list."""
        self._search.clear()  # triggers textChanged → _on_text_changed → _refresh_list

    def scroll_to_top(self) -> None:
        """Scroll the result list back to its first row."""
        self._list.scrollToTop()

    def preselect_best_match(self) -> bool:
        """Reset the scroll position and, when ordered by face-match similarity,
        select and reveal the top (best-match) row.

        Always scrolls the list to the top so a freshly populated selector never
        inherits a previous target's scroll position.  Returns ``True`` if a row
        was selected (only happens while match-sorting is active, so an
        alphabetical list is never auto-selected and the Assign button stays
        safe).
        """
        self._list.scrollToTop()
        if self._match_active() and self._list.count() > 0:
            item = self._list.item(0)
            self._list.blockSignals(True)
            self._list.setCurrentItem(item)
            self._list.blockSignals(False)
            data = item.data(_ROLE_ID)
            self._selected_id = int(data) if data is not None else None
            return True
        return False

    def focus_search(self) -> None:
        """Set keyboard focus to the search input."""
        self._search.setFocus()

    def retranslate(self) -> None:
        self._search.setPlaceholderText(t("pss_search_placeholder"))
        self._no_results.setText(t("pss_no_results"))
        self._match_checkbox.setText(t("pss_match_sort"))
        self._computing_label.setText(t("pss_match_computing"))

    # ──────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────

    def _match_active(self) -> bool:
        """True when results should be ordered by face-match similarity."""
        return self._match_checkbox.isChecked() and bool(self._match_scores)

    def _entries_by_match(self) -> List[PersonEntry]:
        """Entries ordered best-match-first; unscored persons keep their order."""
        scored = [e for e in self._entries if e.person_id in self._match_scores]
        scored.sort(key=lambda e: self._match_scores[e.person_id], reverse=True)
        rest = [e for e in self._entries if e.person_id not in self._match_scores]
        return scored + rest

    def _display_text_for(self, entry: PersonEntry) -> str:
        """Append the match percentage whenever a score is known.

        The percentage is informative on its own, so it is shown regardless of
        whether the match-sort checkbox is checked — the checkbox only controls
        the *ordering* of the list.
        """
        if entry.person_id in self._match_scores:
            pct = max(0, min(100, round(self._match_scores[entry.person_id] * 100)))
            return t("pss_match_percent", name=entry.display_text, pct=pct)
        return entry.display_text

    def _refresh_list(self) -> None:
        query = self._search.text()
        if self._match_active():
            # Match-sorting is itself a relevance ranking, so unknown clusters
            # stay visible (the "search logic justifies it" case).
            visible = filter_entries(query, self._entries_by_match())
        else:
            visible = search_persons(
                query, self._entries, hide_unknown=self._hide_unknown_in_base
            )
        self._list.blockSignals(True)
        self._list.clear()
        for entry in visible:
            item = QListWidgetItem(self._display_text_for(entry))
            item.setData(_ROLE_ID, entry.person_id)
            self._list.addItem(item)
        self._list.blockSignals(False)
        self._sync_selection()
        has_items = self._list.count() > 0
        self._list.setVisible(has_items)
        self._no_results.setVisible(not has_items and bool(query.strip()))

    @staticmethod
    def _load_match_pref() -> bool:
        try:
            from app.app_settings import app_qsettings

            return bool(
                app_qsettings().value(_MATCH_SORT_QSETTINGS_KEY, False, type=bool)
            )
        except Exception:  # noqa: BLE001 — settings are best-effort, never fatal
            return False

    def _on_match_toggled(self, checked: bool) -> None:
        try:
            from app.app_settings import app_qsettings

            app_qsettings().setValue(_MATCH_SORT_QSETTINGS_KEY, bool(checked))
        except Exception:  # noqa: BLE001
            pass
        self._refresh_list()

    def _sync_selection(self) -> None:
        """Re-highlight the row for ``self._selected_id`` after a list rebuild."""
        if self._selected_id is None:
            return
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item and item.data(_ROLE_ID) == self._selected_id:
                self._list.blockSignals(True)
                self._list.setCurrentItem(item)
                self._list.blockSignals(False)
                return

    def _commit_current(self) -> None:
        person_id = self.current_person_id()
        if person_id is not None:
            self._selected_id = person_id
            self.person_selected.emit(person_id)

    def set_navigate_selects(self, enabled: bool) -> None:
        """When True, Up/Down keyboard navigation immediately emits person_selected.

        Use on always-visible navigation surfaces (e.g. the sidebar). Keep
        disabled for inline editors where the user picks before committing.
        """
        if enabled:
            self._list.currentItemChanged.connect(self._on_navigate_select)
        else:
            try:
                self._list.currentItemChanged.disconnect(self._on_navigate_select)
            except RuntimeError:
                pass

    def _on_navigate_select(self, current: QListWidgetItem, _previous: QListWidgetItem) -> None:
        if current is None:
            return
        person_id = current.data(_ROLE_ID)
        if person_id is not None and person_id != self._selected_id:
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
                    if self._search.text():
                        self._search.clear()
                        return True
                    return False  # propagate so parent dialog/popup can close

            if watched is self._list:
                if key == Qt.Key_Return or key == Qt.Key_Enter:
                    self._commit_current()
                    return True
                if key == Qt.Key_Escape:
                    if self._search.text():
                        self._search.clear()
                        self._search.setFocus()
                        return True
                    self._search.setFocus()
                    return False  # propagate so parent dialog/popup can close
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
