"""Collage node detail / metadata editor dialog.

Shows node information (filename, path, persons) and allows the user
to edit year, location, event and notes fields.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCompleter,
    QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit,
    QTextEdit, QVBoxLayout, QWidget,
)

from app.db.database import session_scope
from app.services.collage_service import CollageService

log = logging.getLogger(__name__)


class CollageNodeDialog(QDialog):
    """Detail + editor dialog for a single collage node.

    Args:
        node_id:    Database id of the CollageNode.
        info_lines: List of HTML strings shown as read-only info.
        parent:     Parent QWidget.
    """

    metadata_saved = Signal()   # emitted after successful save

    def __init__(
        self,
        node_id: int,
        info_lines: List[str],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._node_id = node_id
        self.setWindowTitle("Kollázs elem részletei")
        self.setMinimumWidth(520)
        self._build_ui(info_lines)
        self._setup_completers()
        self._load_existing()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self, info_lines: List[str]) -> None:
        layout = QVBoxLayout(self)

        # Read-only info section
        info_html = "<br>".join(info_lines)
        info_label = QLabel(info_html)
        info_label.setWordWrap(True)
        info_label.setTextFormat(1)  # Qt.RichText
        info_label.setStyleSheet(
            "background:#1c1c1c; color:#ddd; border:1px solid #333; "
            "border-radius:4px; padding:8px;"
        )
        layout.addWidget(info_label)

        # Editable fields
        form = QFormLayout()
        form.setContentsMargins(0, 12, 0, 0)

        self._year_edit = QLineEdit()
        self._year_edit.setPlaceholderText("pl. 1969")
        self._year_edit.setMaxLength(16)
        form.addRow("Év:", self._year_edit)

        self._location_edit = QLineEdit()
        self._location_edit.setPlaceholderText("pl. Budapest, Balaton")
        form.addRow("Helyszín:", self._location_edit)

        self._event_edit = QLineEdit()
        self._event_edit.setPlaceholderText("pl. Nyaralás, Esküvő")
        form.addRow("Esemény:", self._event_edit)

        self._notes_edit = QTextEdit()
        self._notes_edit.setPlaceholderText("Szabad megjegyzés…")
        self._notes_edit.setFixedHeight(80)
        form.addRow("Megjegyzés:", self._notes_edit)

        layout.addLayout(form)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    # Autocomplete
    # ------------------------------------------------------------------

    def _fetch_collage_values(self, field: str) -> List[str]:
        from sqlalchemy import select, distinct
        from app.db.models import CollageNode
        try:
            with session_scope() as session:
                col = getattr(CollageNode, field)
                rows = session.execute(
                    select(distinct(col)).where(col.isnot(None)).where(col != "")
                ).scalars().all()
                return sorted(rows)
        except Exception:
            log.exception("Failed to load autocomplete values for %s", field)
            return []

    def _make_completer(self, values: List[str]) -> QCompleter:
        completer = QCompleter(values, self)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        return completer

    def _setup_completers(self) -> None:
        self._location_edit.setCompleter(
            self._make_completer(self._fetch_collage_values("location"))
        )
        self._event_edit.setCompleter(
            self._make_completer(self._fetch_collage_values("event_name"))
        )

    # ------------------------------------------------------------------
    # Load existing data
    # ------------------------------------------------------------------

    def _load_existing(self) -> None:
        try:
            with session_scope() as session:
                node = session.get(
                    __import__(
                        "app.db.models", fromlist=["CollageNode"]
                    ).CollageNode,
                    self._node_id,
                )
                if node is None:
                    return
                self._year_edit.setText(node.year or "")
                self._location_edit.setText(node.location or "")
                self._event_edit.setText(node.event_name or "")
                self._notes_edit.setPlainText(node.notes or "")
        except Exception:
            log.exception("Failed to load node metadata")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _on_save(self) -> None:
        try:
            with session_scope() as session:
                svc = CollageService(session)
                svc.update_node_metadata(
                    self._node_id,
                    year=self._year_edit.text().strip(),
                    location=self._location_edit.text().strip(),
                    event_name=self._event_edit.text().strip(),
                    notes=self._notes_edit.toPlainText().strip(),
                )
            self.metadata_saved.emit()
            self.accept()
        except Exception as exc:
            log.exception("Failed to save node metadata")
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Mentési hiba", str(exc))
