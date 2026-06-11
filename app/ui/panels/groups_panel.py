"""Groups panel — main-window tab for managing person groups (communities).

Thin wrapper around :class:`GroupManagerWidget`; sits between the Objects and
Collage tabs. Re-emits :attr:`groups_changed` so the main window can refresh
anything that displays group labels.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from app.ui.dialogs.group_manager_dialog import GroupManagerWidget


class GroupsPanel(QWidget):
    """Tab panel exposing the group manager."""

    groups_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self._manager = GroupManagerWidget()
        self._manager.groups_changed.connect(self.groups_changed)
        layout.addWidget(self._manager)

    def reload(self) -> None:
        """Re-read groups from the DB (e.g. when the tab becomes active)."""
        self._manager._reload_list()
