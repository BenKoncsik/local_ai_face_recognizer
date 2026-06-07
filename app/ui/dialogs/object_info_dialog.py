"""Object info dialog — edit a tagged object's name, description and notes."""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.db.database import session_scope
from app.services.object_service import ObjectService
from app.ui.i18n import t

log = logging.getLogger(__name__)


class ObjectInfoDialog(QDialog):
    """Edit the editable fields of a :class:`TaggedObject`."""

    def __init__(self, object_id: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._object_id = object_id
        self.setWindowTitle(t("objects_edit"))
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self._name = QLineEdit()
        self._name.setPlaceholderText(t("object_example_name"))
        form.addRow(t("object_name"), self._name)

        self._description = QLineEdit()
        self._description.setPlaceholderText(t("object_example_desc"))
        form.addRow(t("object_description"), self._description)
        layout.addLayout(form)

        layout.addWidget(QLabel(t("object_notes")))
        self._notes = QTextEdit()
        self._notes.setFixedHeight(100)
        layout.addWidget(self._notes)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._load()

    def _load(self) -> None:
        from app.db.models import TaggedObject
        try:
            with session_scope() as session:
                obj = session.get(TaggedObject, self._object_id)
                if obj is None:
                    return
                self._name.setText(obj.name or "")
                self._description.setText(obj.description or "")
                self._notes.setPlainText(obj.notes or "")
        except Exception:
            log.exception("Failed to load object %d", self._object_id)

    def accept(self) -> None:
        try:
            with session_scope() as session:
                ObjectService(session).update_object(
                    self._object_id,
                    name=self._name.text(),
                    description=self._description.text(),
                    notes=self._notes.toPlainText(),
                )
        except ValueError as exc:
            QMessageBox.warning(self, t("objects_edit"), str(exc))
            return
        except Exception:
            log.exception("Failed to save object %d", self._object_id)
            return
        super().accept()
