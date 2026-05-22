"""Name-suggestion review dialog.

Shows proposed Unknown → Known matches.  Each row pairs the unknown
person's representative face with the suggested named person's face and
lets the user approve (merge) or reject (record a 'different person'
correction) the suggestion.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.config import SuggestionConfig
from app.db.database import session_scope
from app.services.suggestion_service import Suggestion, SuggestionService
from app.ui.i18n import t

log = logging.getLogger(__name__)

_CROP_SIZE = 72


def _crop_pixmap(path: Optional[str]) -> Optional[QPixmap]:
    """Load a face crop scaled to the row thumbnail size, or None."""
    if path and Path(path).exists():
        return QPixmap(path).scaled(
            _CROP_SIZE, _CROP_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
    return None


class _SuggestionRow(QFrame):
    """One suggestion: unknown face → suggested named face, with actions.

    Signals:
        approved: ``(candidate_person_id, target_person_id)``
        rejected: ``(candidate_person_id, target_person_id)``
    """

    approved = Signal(int, int)
    rejected = Signal(int, int)

    def __init__(self, suggestion: Suggestion, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._s = suggestion
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("QFrame { background: #232323; border-radius: 6px; }")

        row = QHBoxLayout(self)
        row.setContentsMargins(8, 8, 8, 8)
        row.setSpacing(10)

        row.addWidget(self._crop_label(suggestion.candidate_crop_path))
        row.addWidget(
            self._name_block(suggestion.candidate_name, suggestion.candidate_face_count)
        )

        arrow = QLabel("→")
        arrow.setStyleSheet("font-size: 22px; color: #88aaff; border: none;")
        row.addWidget(arrow)

        row.addWidget(self._crop_label(suggestion.target_crop_path))
        row.addWidget(
            self._name_block(suggestion.target_name, suggestion.target_face_count)
        )

        pct = int(round(max(0.0, min(1.0, suggestion.similarity)) * 100))
        sim_label = QLabel(t("suggestions_similarity", pct=pct))
        sim_label.setAlignment(Qt.AlignCenter)
        sim_label.setFixedWidth(110)
        sim_label.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #88ee88; border: none;"
        )
        row.addWidget(sim_label)

        row.addStretch()

        approve_btn = QPushButton(t("suggestions_approve"))
        approve_btn.setStyleSheet("QPushButton { color: #88ee88; }")
        approve_btn.clicked.connect(
            lambda: self.approved.emit(
                self._s.candidate_person_id, self._s.target_person_id
            )
        )
        row.addWidget(approve_btn)

        reject_btn = QPushButton(t("suggestions_reject"))
        reject_btn.setStyleSheet("QPushButton { color: #ff8888; }")
        reject_btn.clicked.connect(
            lambda: self.rejected.emit(
                self._s.candidate_person_id, self._s.target_person_id
            )
        )
        row.addWidget(reject_btn)

    def _crop_label(self, path: Optional[str]) -> QLabel:
        label = QLabel()
        label.setFixedSize(_CROP_SIZE, _CROP_SIZE)
        label.setAlignment(Qt.AlignCenter)
        pixmap = _crop_pixmap(path)
        if pixmap is not None:
            label.setPixmap(pixmap)
            label.setStyleSheet("border: 1px solid #555; border-radius: 4px;")
        else:
            label.setText("?")
            label.setStyleSheet(
                "border: 1px solid #555; border-radius: 4px; "
                "color: #888; font-size: 20px;"
            )
        return label

    @staticmethod
    def _name_block(name: str, face_count: int) -> QWidget:
        widget = QWidget()
        widget.setFixedWidth(150)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        name_label = QLabel(name)
        name_label.setWordWrap(True)
        name_label.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #eee; border: none;"
        )
        count_label = QLabel(t("suggestions_faces", n=face_count))
        count_label.setStyleSheet("font-size: 10px; color: #888; border: none;")

        layout.addWidget(name_label)
        layout.addWidget(count_label)
        return widget


class SuggestionDialog(QDialog):
    """Reviewable list of Unknown → Known name suggestions.

    Signals:
        data_changed: emitted after an approval changes person assignments.
    """

    data_changed = Signal()

    def __init__(
        self, config: SuggestionConfig, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._suggestions: List[Suggestion] = []

        self.setWindowTitle(t("suggestions_title"))
        self.setMinimumSize(760, 540)
        self._build_ui()
        self._reload()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        intro = QLabel(t("suggestions_intro"))
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa;")
        root.addWidget(intro)

        controls = QHBoxLayout()
        controls.addWidget(QLabel(t("suggestions_threshold")))

        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(0.0, 1.0)
        self._threshold_spin.setSingleStep(0.05)
        self._threshold_spin.setDecimals(2)
        self._threshold_spin.setValue(float(self._config.similarity_threshold))
        self._threshold_spin.valueChanged.connect(lambda *_: self._reload())
        controls.addWidget(self._threshold_spin)

        controls.addStretch()

        self._count_label = QLabel("")
        self._count_label.setStyleSheet("color: #88aaff; font-weight: bold;")
        controls.addWidget(self._count_label)
        root.addLayout(controls)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._list_widget = QWidget()
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setSpacing(6)
        self._list_layout.addStretch()
        scroll.setWidget(self._list_widget)
        root.addWidget(scroll, stretch=1)

        self._empty_label = QLabel(t("suggestions_empty"))
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet(
            "color: #888; font-style: italic; padding: 20px;"
        )
        root.addWidget(self._empty_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _reload(self) -> None:
        """Regenerate suggestions at the current threshold and rebuild rows."""
        threshold = self._threshold_spin.value()
        try:
            with session_scope() as session:
                self._suggestions = SuggestionService(
                    session, self._config
                ).generate_suggestions(threshold=threshold)
        except Exception as exc:  # noqa: BLE001
            log.exception("Failed to generate suggestions")
            QMessageBox.critical(self, t("error"), str(exc))
            self._suggestions = []
        self._rebuild_rows()

    def _rebuild_rows(self) -> None:
        # Remove existing rows, keeping the trailing stretch item.
        while self._list_layout.count() > 1:
            item = self._list_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        for suggestion in self._suggestions:
            row = _SuggestionRow(suggestion)
            row.approved.connect(self._on_approve)
            row.rejected.connect(self._on_reject)
            self._list_layout.insertWidget(self._list_layout.count() - 1, row)

        has_rows = bool(self._suggestions)
        self._empty_label.setVisible(not has_rows)
        self._count_label.setText(t("suggestions_count", n=len(self._suggestions)))

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    @Slot(int, int)
    def _on_approve(self, candidate_id: int, target_id: int) -> None:
        cand_name, target_name = self._pair_names(candidate_id, target_id)
        reply = QMessageBox.question(
            self,
            t("suggestions_approve"),
            t("suggestions_approve_confirm", cand=cand_name, target=target_name),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            with session_scope() as session:
                SuggestionService(session, self._config).approve(
                    candidate_id, target_id
                )
        except Exception as exc:  # noqa: BLE001
            log.exception("Suggestion approval failed")
            QMessageBox.warning(self, t("error"), str(exc))
            return

        log.info("Suggestion approved: person %d → %d", candidate_id, target_id)
        self.data_changed.emit()
        self._reload()

    @Slot(int, int)
    def _on_reject(self, candidate_id: int, target_id: int) -> None:
        try:
            with session_scope() as session:
                SuggestionService(session, self._config).reject(
                    candidate_id, target_id
                )
        except Exception as exc:  # noqa: BLE001
            log.exception("Recording suggestion rejection failed")
            QMessageBox.warning(self, t("error"), str(exc))
            return

        log.info("Suggestion rejected: person %d ✗ %d", candidate_id, target_id)
        self._reload()

    def _pair_names(self, candidate_id: int, target_id: int) -> tuple[str, str]:
        cand = next(
            (s.candidate_name for s in self._suggestions
             if s.candidate_person_id == candidate_id),
            str(candidate_id),
        )
        target = next(
            (s.target_name for s in self._suggestions
             if s.target_person_id == target_id),
            str(target_id),
        )
        return cand, target
