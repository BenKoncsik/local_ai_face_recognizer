"""Reusable flow layout — wraps child widgets onto new rows like words in text.

Qt ships no wrapping box layout, so this is the classic ``FlowLayout`` port:
items flow left-to-right and break to the next line when they run out of
horizontal room.  ``FlowContainer`` is a ready-made host widget that reports
the correct height-for-width so parent layouts / scroll areas reserve the full
multi-row height instead of clipping the overflow.
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QLayout, QLayoutItem, QSizePolicy, QWidget


class FlowLayout(QLayout):
    """A layout that wraps its child widgets like words in a paragraph."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        h_spacing: int = 4,
        v_spacing: int = 4,
    ) -> None:
        super().__init__(parent)
        self._items: List[QLayoutItem] = []
        self._h_spacing = h_spacing
        self._v_spacing = v_spacing

    def addItem(self, item: QLayoutItem) -> None:  # noqa: N802
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> Optional[QLayoutItem]:  # noqa: N802
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int) -> Optional[QLayoutItem]:  # noqa: N802
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self) -> Qt.Orientations:  # noqa: N802
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _do_layout(self, rect: QRect, *, test_only: bool) -> int:
        m = self.contentsMargins()
        eff = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x = eff.x()
        y = eff.y()
        line_height = 0

        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + self._h_spacing
            if next_x - self._h_spacing > eff.right() and line_height > 0:
                x = eff.x()
                y = y + line_height + self._v_spacing
                next_x = x + hint.width() + self._h_spacing
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())

        return y + line_height - rect.y() + m.bottom()


class FlowContainer(QWidget):
    """QWidget that hosts a :class:`FlowLayout` and propagates height-for-width.

    Plain ``QWidget.sizeHint()`` delegates to ``QLayout.sizeHint()`` which for a
    flow layout returns only one row's height (its ``minimumSize``).  This
    container overrides the height-for-width machinery at the *widget* level so
    parent ``QVBoxLayout`` / ``QScrollArea`` allocate the full multi-row height
    instead of clipping items that wrapped onto later rows.

    Add children with ``container.layout().addWidget(...)``.
    """

    def __init__(
        self,
        h_spacing: int = 4,
        v_spacing: int = 4,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        fl = FlowLayout(self, h_spacing=h_spacing, v_spacing=v_spacing)
        fl.setContentsMargins(0, 0, 0, 0)
        sp = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sp.setHeightForWidth(True)
        self.setSizePolicy(sp)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        h = self.layout().heightForWidth(width)
        return max(h, 0)

    def sizeHint(self) -> QSize:  # noqa: N802
        w = self.width()
        if w > 0:
            return QSize(w, self.heightForWidth(w))
        return QSize(0, 0)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(0, 0)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if event.size().width() != event.oldSize().width():
            self.updateGeometry()
