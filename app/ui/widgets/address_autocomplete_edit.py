"""Autocomplete line edit for settlements and streets.

A ``QLineEdit`` with a floating suggestion popup. Typing is debounced (~300 ms)
and dispatched to a background worker (see app.workers.geocoding_worker), so the
GUI never blocks and the network — when enabled at all — is hit sparingly.

Two modes:
    kind="settlement"  → suggests settlements from the typed prefix.
    kind="street"      → suggests streets within ``settlement_context``; the
                         edit is disabled until a settlement is set.

Signals:
    suggestion_chosen(dict)  emitted when the user picks a suggestion; the dict
                             carries at least {"name", "latitude", "longitude"}.
    edited_text(str)         emitted (debounced) on free typing without a pick.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.workers.geocoding_worker import (
    SettlementSuggestWorker,
    StreetSuggestWorker,
)

_DEBOUNCE_MS = 300
_ROLE_DATA = Qt.UserRole


class AddressAutocompleteEdit(QWidget):
    suggestion_chosen = Signal(dict)
    edited_text = Signal(str)

    def __init__(
        self,
        kind: str = "settlement",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._kind = kind
        self._settlement_context: Optional[str] = None
        self._request_id = 0
        self._suppress = False  # set while we programmatically change the text

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._edit = QLineEdit()
        self._edit.textEdited.connect(self._on_text_edited)
        layout.addWidget(self._edit)

        self._popup = QListWidget()
        self._popup.setMaximumHeight(160)
        self._popup.setVisible(False)
        self._popup.itemClicked.connect(self._on_item_chosen)
        layout.addWidget(self._popup)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(_DEBOUNCE_MS)
        self._timer.timeout.connect(self._dispatch)

        if kind == "street":
            self._edit.setEnabled(False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def text(self) -> str:
        return self._edit.text().strip()

    def set_text(self, value: str) -> None:
        self._suppress = True
        self._edit.setText(value or "")
        self._suppress = False
        self._hide_popup()

    def set_placeholder(self, text: str) -> None:
        self._edit.setPlaceholderText(text)

    def set_settlement_context(self, settlement: Optional[str]) -> None:
        """For street mode: enable only when a settlement is selected."""
        self._settlement_context = (settlement or "").strip() or None
        if self._kind == "street":
            self._edit.setEnabled(self._settlement_context is not None)
            if self._settlement_context is None:
                self.set_text("")

    def clear(self) -> None:
        self.set_text("")

    # ------------------------------------------------------------------
    # Typing / debounce
    # ------------------------------------------------------------------

    def _on_text_edited(self, text: str) -> None:
        if self._suppress:
            return
        self.edited_text.emit(text.strip())
        if len(text.strip()) < 2:
            self._hide_popup()
            self._timer.stop()
            return
        self._timer.start()

    def _dispatch(self) -> None:
        prefix = self._edit.text().strip()
        if len(prefix) < 2:
            return
        self._request_id += 1
        req = self._request_id
        if self._kind == "settlement":
            worker = SettlementSuggestWorker(prefix, req)
            worker.signals.settlements_ready.connect(self._on_results)
            worker.signals.failed.connect(self._on_failed)
        else:
            if not self._settlement_context:
                return
            worker = StreetSuggestWorker(self._settlement_context, prefix, req)
            worker.signals.streets_ready.connect(self._on_results)
            worker.signals.failed.connect(self._on_failed)
        from PySide6.QtCore import QThreadPool

        QThreadPool.globalInstance().start(worker)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def _on_results(self, request_id: int, rows: list) -> None:
        if request_id != self._request_id:
            return  # stale response from an earlier keystroke
        self._popup.clear()
        for row in rows:
            name = row.get("name") if isinstance(row, dict) else None
            if not name:
                continue
            item = QListWidgetItem(name)
            item.setData(_ROLE_DATA, row)
            self._popup.addItem(item)
        self._popup.setVisible(self._popup.count() > 0)

    def _on_failed(self, request_id: int, _message: str) -> None:
        if request_id == self._request_id:
            self._hide_popup()

    def _on_item_chosen(self, item: QListWidgetItem) -> None:
        row = item.data(_ROLE_DATA) or {}
        self.set_text(item.text())
        self._hide_popup()
        if isinstance(row, dict):
            self.suggestion_chosen.emit(row)

    def _hide_popup(self) -> None:
        self._popup.clear()
        self._popup.setVisible(False)
