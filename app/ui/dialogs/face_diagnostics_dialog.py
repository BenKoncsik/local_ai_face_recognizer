"""Developer dialog showing why a face has its current identity.

Renders a :class:`app.services.face_diagnostics_service.FaceDiagnostics` record:
the current assignment, the adaptive threshold, ranked named-person and Unknown
matches with similarity scores, and a plain-language verdict.
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.services.face_diagnostics_service import CandidateScore, FaceDiagnostics
from app.ui.i18n import t


class FaceDiagnosticsDialog(QDialog):
    """Read-only diagnostics view for one face."""

    def __init__(self, diag: FaceDiagnostics, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._diag = diag
        self.setWindowTitle(t("diag_title", id=diag.face_id))
        self.setMinimumWidth(560)
        self.resize(640, 640)
        self._build_ui()

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        d = self._diag
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        current = d.current_person_name or t("diag_none")
        layout.addWidget(self._kv(t("diag_current"), current))
        layout.addWidget(self._kv(
            t("diag_source"),
            f"{d.assignment_source or t('diag_none')}"
            + (f"  (conf={d.assignment_confidence:.3f})"
               if d.assignment_confidence is not None else ""),
        ))
        layout.addWidget(self._kv(
            t("diag_threshold"),
            f"{d.adaptive_threshold:.3f}  (base {d.base_threshold:.3f}, "
            f"margin ≥ {d.margin_required:.3f})",
        ))
        q = f"{d.quality_score:.2f}" if d.quality_score is not None else t("diag_none")
        layout.addWidget(self._kv(
            t("diag_quality"),
            f"{q}" + ("  ⚠ low" if d.is_low_quality else ""),
        ))

        layout.addWidget(self._section(t("diag_named_header"), d.top_named, d))
        layout.addWidget(self._section(t("diag_unknown_header"), d.top_unknown, d))

        verdict_lbl = QLabel(f"<b>{t('diag_verdict')}:</b> {d.verdict}")
        verdict_lbl.setWordWrap(True)
        verdict_lbl.setStyleSheet("background:#1E1E2E; padding:8px; border-radius:6px;")
        layout.addWidget(verdict_lbl)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton(t("scanModes.close"))
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    @staticmethod
    def _kv(key: str, value: str) -> QLabel:
        lbl = QLabel(f"<b>{key}:</b> {value}")
        lbl.setWordWrap(True)
        return lbl

    def _section(
        self, header: str, rows: List[CandidateScore], d: FaceDiagnostics
    ) -> QWidget:
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 4, 0, 0)
        v.setSpacing(2)
        v.addWidget(QLabel(f"<b>{header}</b>"))

        if not rows:
            v.addWidget(QLabel(t("diag_none")))
            return box

        table = QTableWidget(len(rows), 3)
        table.setHorizontalHeaderLabels(["Person", "Similarity", "Faces"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for i, c in enumerate(rows):
            mark = "  ✓" if c.similarity >= d.adaptive_threshold else ""
            table.setItem(i, 0, QTableWidgetItem(c.name))
            table.setItem(i, 1, QTableWidgetItem(f"{c.similarity:.3f}{mark}"))
            table.setItem(i, 2, QTableWidgetItem(str(c.face_count)))
        table.setMaximumHeight(40 + 24 * len(rows))
        v.addWidget(table)
        return box
