"""Universal multi-value search bar with autocomplete and chip/token display.

Usage
-----
::

    def my_suggest_fn(query: str) -> list[tuple[str, str, str]]:
        # Returns list of (value, token_type, display_text).
        # token_type: "person" | "nickname" | "family_code" | "place" | "date" | "image"
        ...

    bar = UniversalSearchBar(suggest_fn=my_suggest_fn, parent=self)
    bar.set_placeholder("Search…")
    bar.search_requested.connect(self._do_search)

    tokens = bar.get_tokens()  # list[SearchToken]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from PySide6.QtCore import QEvent, QPoint, QRect, QSize, Qt, Signal, QTimer
from PySide6.QtGui import QFocusEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.ui.i18n import t

log = logging.getLogger(__name__)

# Token type constants — used as strings so callers can compare without import
TOKEN_PERSON = "person"
TOKEN_NICKNAME = "nickname"
TOKEN_FAMILY_CODE = "family_code"
TOKEN_PLACE = "place"
TOKEN_DATE = "date"
TOKEN_IMAGE = "image"
TOKEN_OBJECT = "object"
TOKEN_ANY = "any"

# i18n key for each token type
_TOKEN_TYPE_I18N = {
    TOKEN_PERSON: "search_token_person",
    TOKEN_NICKNAME: "search_token_nickname",
    TOKEN_FAMILY_CODE: "search_token_family_code",
    TOKEN_PLACE: "search_token_place",
    TOKEN_DATE: "search_token_date",
    TOKEN_IMAGE: "search_token_image",
    TOKEN_OBJECT: "search_token_object",
    TOKEN_ANY: "",
}

# Visual accent colours per token type  (foreground, background)
_TYPE_COLOURS: dict[str, tuple[str, str]] = {
    TOKEN_PERSON:      ("#1a6b3c", "#d4f0e0"),
    TOKEN_NICKNAME:    ("#5b3e8a", "#ede0f7"),
    TOKEN_FAMILY_CODE: ("#0d5a8c", "#d0e8f8"),
    TOKEN_PLACE:       ("#7a4500", "#fce8cb"),
    TOKEN_DATE:        ("#5a5a00", "#f5f5c0"),
    TOKEN_IMAGE:       ("#444444", "#e8e8e8"),
    TOKEN_OBJECT:      ("#0a5a66", "#cdeef2"),
    TOKEN_ANY:         ("#333333", "#e0e0e0"),
}

SuggestionFn = Callable[[str], List[Tuple[str, str, str]]]


@dataclass
class SearchToken:
    value: str
    token_type: str = TOKEN_ANY


# ---------------------------------------------------------------------------
# Internal: FlowLayout
# ---------------------------------------------------------------------------

class _FlowLayout(QLayout):
    """Items flow left-to-right and wrap to the next row automatically."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        h_spacing: int = 4,
        v_spacing: int = 4,
    ) -> None:
        super().__init__(parent)
        self._items: list = []
        self._h = h_spacing
        self._v = v_spacing

    def addItem(self, item) -> None:  # noqa: N802
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):  # noqa: N802
        return Qt.Orientations()

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._do_layout(QRect(0, 0, width, 0), test=True)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._do_layout(rect, test=False)

    def sizeHint(self) -> QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _do_layout(self, rect: QRect, test: bool) -> int:
        m = self.contentsMargins()
        left = rect.x() + m.left()
        right = rect.right() - m.right()
        x, y = left, rect.y() + m.top()
        row_h = 0

        for item in self._items:
            sh = item.sizeHint()
            next_x = x + sh.width()
            if next_x > right and row_h > 0:
                x, y = left, y + row_h + self._v
                next_x = x + sh.width()
                row_h = 0
            if not test:
                item.setGeometry(QRect(QPoint(x, y), sh))
            x = next_x + self._h
            row_h = max(row_h, sh.height())

        return y + row_h - rect.y() + m.bottom()


# ---------------------------------------------------------------------------
# Internal: Chip widget
# ---------------------------------------------------------------------------

class _Chip(QWidget):
    """A single removable token chip shown inside the search bar."""

    remove_me = Signal()

    def __init__(self, token: SearchToken, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._token = token
        fg, bg = _TYPE_COLOURS.get(token.token_type, ("#333", "#e0e0e0"))
        type_key = _TOKEN_TYPE_I18N.get(token.token_type, "")
        type_label = t(type_key) if type_key else ""

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 2, 2)
        layout.setSpacing(2)

        lbl = QLabel()
        if type_label:
            lbl.setText(
                f"<span style='color:{fg}'><b>{token.value}</b></span>"
                f"<span style='color:{fg}99; font-size:10px'> {type_label}</span>"
            )
        else:
            lbl.setText(f"<span style='color:{fg}'><b>{token.value}</b></span>")
        layout.addWidget(lbl)

        btn = QPushButton("×")
        btn.setFlat(True)
        btn.setFixedSize(18, 18)
        btn.setStyleSheet(
            f"QPushButton {{ color: {fg}; font-size: 14px; padding: 0; border: none; }}"
            f"QPushButton:hover {{ color: #cc0000; }}"
        )
        btn.clicked.connect(self.remove_me)
        layout.addWidget(btn)

        self.setStyleSheet(
            f"_Chip {{ background: {bg}; border: 1px solid {fg}55;"
            f" border-radius: 4px; }}"
        )
        self.setMaximumHeight(28)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)

    @property
    def token(self) -> SearchToken:
        return self._token


# ---------------------------------------------------------------------------
# Public: UniversalSearchBar
# ---------------------------------------------------------------------------

class UniversalSearchBar(QWidget):
    """Multi-value search bar with autocomplete and removable chip/token display.

    Signals
    -------
    tokens_changed()
        Emitted when the token list changes (chip added or removed).
    search_requested()
        Emitted when the user presses Enter, selects an autocomplete item,
        or removes a chip (triggers auto-refresh).
    """

    tokens_changed = Signal()
    search_requested = Signal()

    def __init__(
        self,
        suggest_fn: Optional[SuggestionFn] = None,
        placeholder: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._suggest_fn = suggest_fn
        self._tokens: list[SearchToken] = []
        self._chips: list[_Chip] = []
        self._placeholder = placeholder

        self._ac_timer = QTimer(self)
        self._ac_timer.setSingleShot(True)
        self._ac_timer.setInterval(180)
        self._ac_timer.timeout.connect(self._populate_suggestions)

        self._build_ui()

    # ──────────────────────────────────────────────────────────────────
    # Build
    # ──────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)

        # Outer container styled as an input field
        self._container = QFrame()
        self._container.setFrameShape(QFrame.StyledPanel)
        self._container.setStyleSheet(
            "QFrame {"
            "  background: #1e1e1e;"
            "  border: 1px solid #444;"
            "  border-radius: 4px;"
            "}"
            "QFrame:focus-within {"
            "  border-color: #5588cc;"
            "}"
        )
        c_layout = QVBoxLayout(self._container)
        c_layout.setContentsMargins(4, 4, 4, 4)
        c_layout.setSpacing(2)

        # Scrollable chip area (hidden when empty)
        self._chip_scroll = QScrollArea()
        self._chip_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._chip_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._chip_scroll.setWidgetResizable(True)
        self._chip_scroll.setMaximumHeight(72)
        self._chip_scroll.setFrameShape(QFrame.NoFrame)
        self._chip_scroll.setStyleSheet("background: transparent;")
        self._chip_scroll.hide()

        self._chip_area = QWidget()
        self._chip_area.setStyleSheet("background: transparent;")
        self._chip_layout = _FlowLayout(self._chip_area, h_spacing=4, v_spacing=4)
        self._chip_layout.setContentsMargins(0, 0, 0, 0)
        self._chip_scroll.setWidget(self._chip_area)
        c_layout.addWidget(self._chip_scroll)

        # Text input
        self._input = QLineEdit()
        self._input.setPlaceholderText(self._placeholder)
        self._input.setStyleSheet(
            "QLineEdit {"
            "  background: transparent;"
            "  border: none;"
            "  color: #cdd6f4;"
            "}"
        )
        self._input.textChanged.connect(self._on_text_changed)
        self._input.returnPressed.connect(self._on_return_pressed)
        self._input.installEventFilter(self)
        c_layout.addWidget(self._input)

        root.addWidget(self._container)

        # Autocomplete popup (separate top-level tool window)
        self._popup = QFrame(None)
        self._popup.setWindowFlags(
            Qt.Tool | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint
        )
        self._popup.setAttribute(Qt.WA_ShowWithoutActivating)
        self._popup.setStyleSheet(
            "QFrame {"
            "  background: #2a2a2a;"
            "  border: 1px solid #555;"
            "  border-radius: 4px;"
            "}"
        )
        pop_layout = QVBoxLayout(self._popup)
        pop_layout.setContentsMargins(0, 0, 0, 0)

        self._popup_list = QListWidget()
        self._popup_list.setStyleSheet(
            "QListWidget {"
            "  background: transparent;"
            "  border: none;"
            "  color: #cdd6f4;"
            "  outline: 0;"
            "}"
            "QListWidget::item { padding: 3px 8px; }"
            "QListWidget::item:selected { background: #3a6e9e; }"
            "QListWidget::item:hover { background: #333355; }"
        )
        self._popup_list.setMaximumHeight(220)
        self._popup_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._popup_list.itemClicked.connect(self._on_suggestion_clicked)
        self._popup_list.installEventFilter(self)
        pop_layout.addWidget(self._popup_list)

        self._popup.hide()

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def get_tokens(self) -> list[SearchToken]:
        """Return the current list of selected search tokens."""
        return list(self._tokens)

    def clear(self) -> None:
        """Remove all chips and clear the input field."""
        for chip in list(self._chips):
            chip.setParent(None)
            chip.deleteLater()
        self._chips.clear()
        self._tokens.clear()
        self._chip_scroll.hide()
        self._input.clear()
        self._popup.hide()

    def set_placeholder(self, text: str) -> None:
        self._placeholder = text
        self._input.setPlaceholderText(text)

    def set_suggest_fn(self, fn: Optional[SuggestionFn]) -> None:
        self._suggest_fn = fn

    def retranslate(self) -> None:
        """Re-render all chip type labels after a language change."""
        self._input.setPlaceholderText(self._placeholder)
        # Rebuild chips with updated labels
        tokens = list(self._tokens)
        self.clear()
        for tok in tokens:
            self._add_token(tok)

    # ──────────────────────────────────────────────────────────────────
    # Chip management
    # ──────────────────────────────────────────────────────────────────

    def _add_token(self, token: SearchToken) -> None:
        if any(t.value == token.value and t.token_type == token.token_type
               for t in self._tokens):
            return
        self._tokens.append(token)
        chip = _Chip(token, parent=self._chip_area)
        chip.remove_me.connect(lambda tok=token, c=chip: self._on_chip_remove(tok, c))
        self._chips.append(chip)
        self._chip_layout.addWidget(chip)
        self._chip_scroll.show()
        self._chip_area.adjustSize()
        log.debug("UniversalSearchBar: added token %r (%s)", token.value, token.token_type)
        self.tokens_changed.emit()

    def _on_chip_remove(self, token: SearchToken, chip: _Chip) -> None:
        if token in self._tokens:
            self._tokens.remove(token)
        if chip in self._chips:
            self._chips.remove(chip)
        chip.setParent(None)
        chip.deleteLater()
        if not self._chips:
            self._chip_scroll.hide()
        self._chip_area.adjustSize()
        log.debug(
            "UniversalSearchBar: removed token %r (%s)", token.value, token.token_type
        )
        self.tokens_changed.emit()
        self.search_requested.emit()

    # ──────────────────────────────────────────────────────────────────
    # Input handling
    # ──────────────────────────────────────────────────────────────────

    def _on_text_changed(self, text: str) -> None:
        if text.strip():
            self._ac_timer.start()
        else:
            self._ac_timer.stop()
            self._popup.hide()

    def _on_return_pressed(self) -> None:
        raw = self._input.text().strip()
        if not raw:
            self._popup.hide()
            self.search_requested.emit()
            return
        for part in (p.strip() for p in raw.split(",") if p.strip()):
            self._add_token(SearchToken(value=part, token_type=TOKEN_ANY))
        self._input.clear()
        self._popup.hide()
        self.search_requested.emit()

    # ──────────────────────────────────────────────────────────────────
    # Autocomplete
    # ──────────────────────────────────────────────────────────────────

    def _populate_suggestions(self) -> None:
        query = self._input.text().strip()
        if not query or self._suggest_fn is None:
            self._popup.hide()
            return
        try:
            suggestions = self._suggest_fn(query)
        except Exception:
            log.exception("UniversalSearchBar: suggestion error")
            self._popup.hide()
            return
        if not suggestions:
            self._popup.hide()
            return

        self._popup_list.blockSignals(True)
        self._popup_list.clear()
        for value, token_type, display in suggestions:
            type_key = _TOKEN_TYPE_I18N.get(token_type, "")
            type_label = t(type_key) if type_key else ""
            text = f"{display}  —  {type_label}" if type_label else display
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, (value, token_type))
            self._popup_list.addItem(item)
        self._popup_list.blockSignals(False)

        self._reposition_popup()
        self._popup.show()

    def _reposition_popup(self) -> None:
        global_pos = self._input.mapToGlobal(QPoint(0, self._input.height() + 2))
        row_h = self._popup_list.sizeHintForRow(0) if self._popup_list.count() else 24
        rows = min(self._popup_list.count(), 8)
        popup_h = min(row_h * rows + 8, 220)
        self._popup.resize(max(self._container.width(), 240), popup_h)
        self._popup.move(global_pos)

    def _on_suggestion_clicked(self, item: QListWidgetItem) -> None:
        data = item.data(Qt.UserRole)
        if data:
            value, token_type = data
            self._add_token(SearchToken(value=value, token_type=token_type))
        self._input.clear()
        self._popup.hide()
        self._input.setFocus()
        self.search_requested.emit()

    # ──────────────────────────────────────────────────────────────────
    # Event filter — keyboard navigation
    # ──────────────────────────────────────────────────────────────────

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if event.type() == QEvent.KeyPress:
            key = event.key()

            if watched is self._input:
                if key == Qt.Key_Down and self._popup.isVisible():
                    self._popup_list.setFocus()
                    if self._popup_list.currentRow() < 0 and self._popup_list.count() > 0:
                        self._popup_list.setCurrentRow(0)
                    return True
                if key == Qt.Key_Escape:
                    self._popup.hide()
                    return True
                # Backspace on empty input removes the last chip
                if key == Qt.Key_Backspace and not self._input.text() and self._chips:
                    last_chip = self._chips[-1]
                    self._on_chip_remove(last_chip.token, last_chip)
                    return True

            if watched is self._popup_list:
                if key in (Qt.Key_Return, Qt.Key_Enter):
                    item = self._popup_list.currentItem()
                    if item:
                        self._on_suggestion_clicked(item)
                    return True
                if key == Qt.Key_Escape:
                    self._popup.hide()
                    self._input.setFocus()
                    return True
                if key == Qt.Key_Up and self._popup_list.currentRow() == 0:
                    self._input.setFocus()
                    return True

        return super().eventFilter(watched, event)

    def hideEvent(self, event) -> None:  # noqa: N802
        self._popup.hide()
        super().hideEvent(event)
