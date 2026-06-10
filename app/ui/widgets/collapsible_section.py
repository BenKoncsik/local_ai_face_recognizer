"""A simple collapsible section: arrow header button + hideable content.

Used for the "already decided" history lists so they don't steal space from
the actionable items above them.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QToolButton, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    """Header button that expands/collapses its content widget.

    Signals:
        toggled: ``(expanded: bool)`` — emitted on every header click; useful
                 for lazy-loading the content the first time it opens.
    """

    toggled = Signal(bool)

    def __init__(
        self,
        title: str,
        content: QWidget,
        parent: Optional[QWidget] = None,
        expanded: bool = False,
    ) -> None:
        super().__init__(parent)
        self._content = content

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._header = QToolButton()
        self._header.setText(title)
        self._header.setCheckable(True)
        self._header.setChecked(expanded)
        self._header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._header.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._header.setStyleSheet(
            "QToolButton { border: none; color: #A6ADC8; font-weight: bold; "
            "padding: 4px 2px; }"
            "QToolButton:hover { color: #cdd6f4; }"
        )
        self._header.toggled.connect(self._on_toggled)
        layout.addWidget(self._header)

        content.setVisible(expanded)
        layout.addWidget(content)

    def set_title(self, title: str) -> None:
        self._header.setText(title)

    def is_expanded(self) -> bool:
        return self._header.isChecked()

    def _on_toggled(self, checked: bool) -> None:
        self._header.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self._content.setVisible(checked)
        self.toggled.emit(checked)
