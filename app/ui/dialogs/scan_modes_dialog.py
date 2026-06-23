"""Scan and maintenance chooser dialog — simplified 3-workflow version."""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.ui.i18n import t

log = logging.getLogger(__name__)


class ScanModesDialog(QDialog):
    """Modal dialog for choosing a scan or maintenance workflow."""

    scan_workflow_started = Signal(str)  # "face_detection" | "full_rescan" | "train_model"

    def __init__(self, parent: Optional[QWidget] = None, config=None) -> None:
        super().__init__(parent)
        self._config = config
        self.setWindowTitle(t("scanModes.title"))
        self.setMinimumWidth(420)
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        container = QWidget()
        cards = QVBoxLayout(container)
        cards.setContentsMargins(4, 4, 4, 4)
        cards.setSpacing(10)
        scroll.setWidget(container)

        cards.addWidget(self._make_card(
            title=t("workflow_face_detection_title"),
            desc=t("workflow_face_detection_desc"),
            workflow="face_detection",
            danger=False,
        ))
        cards.addWidget(self._make_card(
            title=t("workflow_train_model_title"),
            desc=t("workflow_train_model_desc"),
            workflow="train_model",
            danger=False,
        ))
        cards.addWidget(self._make_card(
            title=t("workflow_full_rescan_title"),
            desc=t("workflow_full_rescan_desc"),
            workflow="full_rescan",
            danger=True,
        ))
        cards.addStretch()
        outer.addWidget(scroll)

        # Multi-technology "verify every face" toggle (no confidence exemption).
        verify_box = QVBoxLayout()
        verify_box.setSpacing(2)
        self._verify_all_chk = QCheckBox(t("scanModes.verifyAll.label"))
        det = getattr(self._config, "detection", None)
        self._verify_all_chk.setChecked(bool(getattr(det, "verification_verify_all", False)))
        self._verify_all_chk.setEnabled(self._config is not None)
        self._verify_all_chk.toggled.connect(self._on_verify_all_toggled)
        verify_box.addWidget(self._verify_all_chk)

        verify_tip = QLabel(t("scanModes.verifyAll.tip"))
        verify_tip.setWordWrap(True)
        verify_tip.setStyleSheet("color: #A6ADC8; font-size: 11px;")
        verify_tip.setContentsMargins(22, 0, 0, 0)
        verify_box.addWidget(verify_tip)
        outer.addLayout(verify_box)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton(t("scanModes.close"))
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        outer.addLayout(btn_row)

    def _make_card(self, title: str, desc: str, workflow: str, danger: bool) -> QFrame:
        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        card.setFrameShadow(QFrame.Raised)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        title_lbl = QLabel(f"<b>{title}</b>")
        title_lbl.setWordWrap(True)
        layout.addWidget(title_lbl)

        desc_lbl = QLabel(desc)
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet("color: #A6ADC8;")
        desc_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(desc_lbl)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn = QPushButton(t("button_start"))
        btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        if danger:
            btn.setStyleSheet(
                "QPushButton { color: #F38BA8; border-color: #6B3040; }"
                "QPushButton:hover { background-color: #3D2030; border-color: #F38BA8; }"
            )
        btn.clicked.connect(lambda: self._launch(workflow))
        btn_row.addWidget(btn)
        layout.addLayout(btn_row)

        return card

    def _on_verify_all_toggled(self, checked: bool) -> None:
        """Update the live config and persist the choice to config.yaml."""
        if self._config is None:
            return
        self._config.detection.verification_verify_all = bool(checked)
        try:
            from app.config import save_detection_values

            save_detection_values({"verification_verify_all": bool(checked)})
            log.info("verification_verify_all set to %s (persisted)", checked)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not persist verification_verify_all: %s", exc)

    def _launch(self, workflow: str) -> None:
        self.accept()
        self.scan_workflow_started.emit(workflow)
