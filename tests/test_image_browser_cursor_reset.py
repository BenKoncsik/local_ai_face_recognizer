"""Cursor must reset to default after interactive bbox edit ends.

Regression: after dragging/resizing/moving a face frame the resize/move
cursor stayed frozen on screen because mouseReleaseEvent / edit-exit paths
never restored it. _reset_cursor_state() now centralises the cleanup.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication

from app.ui.panels.image_browser_panel import _DrawableImageLabel


def _make_label(qtbot) -> _DrawableImageLabel:
    label = _DrawableImageLabel()
    qtbot.addWidget(label)
    label.resize(200, 200)
    label.set_source_pixmap(QPixmap(100, 100))
    return label


def test_reset_after_resize_drag_restores_cross(qtbot):
    label = _make_label(qtbot)
    label.set_interactive_edit((10, 10, 40, 40), 100, 100)

    # Simulate an in-progress corner resize: the drag cursor is a resize arrow.
    label._update_edit_cursor("br")
    assert label.cursor().shape() == Qt.SizeFDiagCursor

    # Drag handle released → no handle under a far-away pointer → back to cross.
    label._imode_handle = "br"
    label._reset_cursor_state()

    assert label._imode_handle is None
    assert label.cursor().shape() == Qt.CrossCursor


def test_exit_edit_resets_cursor(qtbot):
    label = _make_label(qtbot)
    label.set_interactive_edit((10, 10, 40, 40), 100, 100)
    label._update_edit_cursor("mc")          # move cursor
    assert label.cursor().shape() == Qt.SizeAllCursor

    label.commit_interactive_edit()          # save AND exit

    assert label._imode is False
    assert label.cursor().shape() == Qt.CrossCursor


def test_cancel_resets_cursor(qtbot):
    label = _make_label(qtbot)
    label.set_interactive_edit((10, 10, 40, 40), 100, 100)
    label._update_edit_cursor("tl")
    assert label.cursor().shape() == Qt.SizeFDiagCursor

    label.cancel_interactive_edit()          # ESC / cancel

    assert label._imode is False
    assert label.cursor().shape() == Qt.CrossCursor


def test_reset_drains_application_override_cursor(qtbot):
    label = _make_label(qtbot)
    # Two stacked override cursors (e.g. nested busy-cursor pushes) must all
    # be drained, otherwise the OS cursor stays stuck after the operation.
    QApplication.setOverrideCursor(Qt.WaitCursor)
    QApplication.setOverrideCursor(Qt.WaitCursor)

    label._reset_cursor_state()

    assert QApplication.overrideCursor() is None
