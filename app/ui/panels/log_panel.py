"""Log panel — scrollable activity log with coloured levels."""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from app.ui.i18n import t

log = logging.getLogger(__name__)

_LEVEL_COLORS = {
    logging.DEBUG:    "#888888",
    logging.INFO:     "#cccccc",
    logging.WARNING:  "#ffcc00",
    logging.ERROR:    "#ff6666",
    logging.CRITICAL: "#ff4444",
}


class LogPanel(QWidget):
    """Scrolling log widget that accepts records via a Qt slot.

    Connect :meth:`append_log` to the :class:`~app.logging_setup.QLogHandler`
    signal.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(2000)
        self._text.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self._text.setStyleSheet(
            "QPlainTextEdit { background: #1a1a1a; color: #cccccc; "
            "font-family: monospace; font-size: 11px; }"
        )
        layout.addWidget(self._text)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        clear_btn = QPushButton(t("clear"))
        clear_btn.setFixedWidth(60)
        clear_btn.clicked.connect(self._text.clear)
        btn_row.addWidget(clear_btn)
        layout.addLayout(btn_row)

    @Slot(str, int)
    def append_log(self, message: str, level: int = logging.INFO) -> None:
        """Append a formatted log line to the text area.

        Args:
            message: Formatted log string.
            level:   :mod:`logging` level integer for colour selection.
        """
        color_hex = _LEVEL_COLORS.get(level, "#cccccc")
        color = QColor(color_hex)

        cursor = self._text.textCursor()
        cursor.movePosition(QTextCursor.End)

        fmt = QTextCharFormat()
        fmt.setForeground(color)
        cursor.setCharFormat(fmt)
        cursor.insertText(message + "\n")

        self._text.setTextCursor(cursor)
        self._text.ensureCursorVisible()

    @Slot(str)
    def append_plain(self, message: str) -> None:
        """Append a plain (uncoloured) message."""
        self.append_log(message, logging.INFO)
