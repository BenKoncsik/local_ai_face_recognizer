"""Scan-modes chooser dialog.

Opens from the toolbar 'Scan Modes …' button and presents all four
scanning operations as cards with a title, description, and launch button.
The dialog is scrollable so it works on small screens too.
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
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


class ScanModesDialog(QDialog):
    """Modal dialog that lets the user choose and start a scanning mode."""

    def __init__(
        self,
        on_incremental: Callable[[], None],
        on_full_rescan: Callable[[], None],
        on_face_rescan_fast: Callable[[], None],
        on_face_rescan_accurate: Callable[[], None],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._on_incremental = on_incremental
        self._on_full_rescan = on_full_rescan
        self._on_face_rescan_fast = on_face_rescan_fast
        self._on_face_rescan_accurate = on_face_rescan_accurate

        self.setWindowTitle(t("scanModes.title"))
        self.setMinimumWidth(480)
        self.setMinimumHeight(300)
        self.resize(560, 680)
        self._build_ui()

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # ── Scrollable card area ─────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        container = QWidget()
        cards_layout = QVBoxLayout(container)
        cards_layout.setContentsMargins(4, 4, 4, 4)
        cards_layout.setSpacing(10)

        cards_layout.addWidget(self._make_card(
            title=t("scanModes.incremental.title"),
            description=t("scanModes.incremental.description"),
            button_label=t("scanModes.incremental.startButton"),
            warning=None,
            callback=self._launch_incremental,
            danger=False,
        ))
        cards_layout.addWidget(self._make_card(
            title=t("scanModes.faceRescan.title"),
            description=t("scanModes.faceRescan.description"),
            button_label=t("scanModes.faceRescan.startButton"),
            warning=None,
            callback=self._launch_face_rescan_fast,
            danger=False,
        ))
        cards_layout.addWidget(self._make_card(
            title=t("scanModes.preciseRescan.title"),
            description=t("scanModes.preciseRescan.description"),
            button_label=t("scanModes.preciseRescan.startButton"),
            warning=t("scanModes.preciseRescan.warning"),
            callback=self._launch_face_rescan_accurate,
            danger=False,
        ))
        cards_layout.addWidget(self._make_card(
            title=t("scanModes.fullRescan.title"),
            description=t("scanModes.fullRescan.description"),
            button_label=t("scanModes.fullRescan.startButton"),
            warning=t("scanModes.fullRescan.warning"),
            callback=self._launch_full_rescan,
            danger=True,
        ))
        cards_layout.addStretch()

        scroll.setWidget(container)
        outer.addWidget(scroll)

        # ── Sticky close button ──────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton(t("scanModes.close"))
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        outer.addLayout(btn_row)

    # ------------------------------------------------------------------

    def _make_card(
        self,
        title: str,
        description: str,
        button_label: str,
        warning: Optional[str],
        callback: Callable[[], None],
        danger: bool,
    ) -> QFrame:
        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        card.setFrameShadow(QFrame.Raised)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        # Title
        title_lbl = QLabel(f"<b>{title}</b>")
        title_lbl.setWordWrap(True)
        layout.addWidget(title_lbl)

        # Description
        desc_lbl = QLabel(description)
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet("color: #A6ADC8;")
        desc_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(desc_lbl)

        # Optional warning line
        if warning:
            warn_lbl = QLabel(f"⚠ {warning}")
            warn_lbl.setWordWrap(True)
            if danger:
                warn_lbl.setStyleSheet("color: #F38BA8;")
            else:
                warn_lbl.setStyleSheet("color: #F9E2AF;")
            layout.addWidget(warn_lbl)

        # Button row (right-aligned)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn = QPushButton(button_label)
        btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        if danger:
            btn.setStyleSheet(
                "QPushButton { color: #F38BA8; border-color: #6B3040; }"
                "QPushButton:hover { background-color: #3D2030; border-color: #F38BA8; }"
                "QPushButton:disabled { color: #6C7086; border-color: #313244; }"
            )
        btn.clicked.connect(callback)
        btn_row.addWidget(btn)
        layout.addLayout(btn_row)

        return card

    # ------------------------------------------------------------------
    # Callbacks — close dialog then trigger the action so the main window
    # guard checks (busy, no folder) can still run normally.

    def _launch_incremental(self) -> None:
        self.accept()
        self._on_incremental()

    def _launch_full_rescan(self) -> None:
        self.accept()
        self._on_full_rescan()

    def _launch_face_rescan_fast(self) -> None:
        self.accept()
        self._on_face_rescan_fast()

    def _launch_face_rescan_accurate(self) -> None:
        self.accept()
        self._on_face_rescan_accurate()
