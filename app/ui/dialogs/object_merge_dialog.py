"""Object merge dialog — pick the target object to merge the others into."""

from __future__ import annotations

from typing import List, Optional, Tuple

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from app.ui.i18n import t


class ObjectMergeDialog(QDialog):
    """Choose which of the selected objects to keep (the merge target)."""

    def __init__(
        self,
        candidates: List[Tuple[int, str]],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("object_merge_title"))
        self.setMinimumWidth(360)
        self._candidates = candidates

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        layout.addWidget(QLabel(t("object_merge_target")))
        self._combo = QComboBox()
        for oid, name in candidates:
            self._combo.addItem(name, oid)
        layout.addWidget(self._combo)

        hint = QLabel(t("object_merge_hint"))
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(hint)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def target_id(self) -> Optional[int]:
        data = self._combo.currentData()
        return int(data) if data is not None else None

    @property
    def source_ids(self) -> List[int]:
        target = self.target_id
        return [oid for oid, _ in self._candidates if oid != target]
