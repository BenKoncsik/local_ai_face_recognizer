"""Scan and maintenance chooser dialog.

Opens from the toolbar.  Two tabs:

* **AI recognition (new)** — the simplified deep-learning path with exactly
  two operations: *Re-scan* (place unknown faces with known people, clean up
  overlapping boxes) and *Rebuild from scratch*.
* **Classic** — every legacy scan / re-detection / cleanup operation.

The dialog is scrollable so it works on small screens too.
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.ui.i18n import t


class ScanModesDialog(QDialog):
    """Modal dialog that lets the user choose scan and cleanup actions."""

    def __init__(
        self,
        on_incremental: Callable[[], None],
        on_full_rescan: Callable[[], None],
        on_face_rescan_fast: Callable[[], None],
        on_face_rescan_accurate: Callable[[], None],
        on_reset_unknown_persons: Callable[[], None],
        on_find_overlapping_unknown_faces: Callable[[], None],
        on_identity_repair_scan: Optional[Callable[[], None]] = None,
        on_cleanup_empty_unknown_persons: Optional[Callable[[], None]] = None,
        on_manage_ignored_faces: Optional[Callable[[], None]] = None,
        on_deep_rescan: Optional[Callable[[], None]] = None,
        on_deep_rebuild: Optional[Callable[[], None]] = None,
        on_deep_train: Optional[Callable[[], None]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._on_incremental = on_incremental
        self._on_full_rescan = on_full_rescan
        self._on_face_rescan_fast = on_face_rescan_fast
        self._on_face_rescan_accurate = on_face_rescan_accurate
        self._on_reset_unknown_persons = on_reset_unknown_persons
        self._on_find_overlapping_unknown_faces = on_find_overlapping_unknown_faces
        self._on_identity_repair_scan = on_identity_repair_scan
        self._on_cleanup_empty_unknown_persons = on_cleanup_empty_unknown_persons
        self._on_manage_ignored_faces = on_manage_ignored_faces
        self._on_deep_rescan = on_deep_rescan
        self._on_deep_rebuild = on_deep_rebuild
        self._on_deep_train = on_deep_train

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

        tabs = QTabWidget()
        tabs.addTab(self._build_deep_tab(), t("scanModes.tab.deep"))
        tabs.addTab(self._build_classic_tab(), t("scanModes.tab.classic"))
        outer.addWidget(tabs)

        # ── Sticky close button ──────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton(t("scanModes.close"))
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        outer.addLayout(btn_row)

    def _make_scroll_container(self) -> tuple[QScrollArea, QVBoxLayout]:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        container = QWidget()
        cards_layout = QVBoxLayout(container)
        cards_layout.setContentsMargins(4, 4, 4, 4)
        cards_layout.setSpacing(10)
        scroll.setWidget(container)
        return scroll, cards_layout

    # ------------------------------------------------------------------
    # AI (deep learning) tab — three operations, kept simple.
    # ------------------------------------------------------------------

    def _build_deep_tab(self) -> QWidget:
        scroll, cards_layout = self._make_scroll_container()

        intro = QLabel(t("scanModes.deep.intro"))
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #A6ADC8; font-style: italic;")
        cards_layout.addWidget(intro)

        cards_layout.addWidget(self._make_card(
            title=t("scanModes.deepRescan.title"),
            description=t("scanModes.deepRescan.description"),
            button_label=t("scanModes.deepRescan.startButton"),
            warning=t("scanModes.deepRescan.warning"),
            callback=self._launch_deep_rescan,
            danger=False,
        ))
        cards_layout.addWidget(self._make_card(
            title=t("scanModes.deepTrain.title"),
            description=t("scanModes.deepTrain.description"),
            button_label=t("scanModes.deepTrain.startButton"),
            warning=t("scanModes.deepTrain.warning"),
            callback=self._launch_deep_train,
            danger=False,
        ))
        cards_layout.addWidget(self._make_card(
            title=t("scanModes.deepRebuild.title"),
            description=t("scanModes.deepRebuild.description"),
            button_label=t("scanModes.deepRebuild.startButton"),
            warning=t("scanModes.deepRebuild.warning"),
            callback=self._launch_deep_rebuild,
            danger=True,
        ))
        cards_layout.addStretch()
        return scroll

    # ------------------------------------------------------------------
    # Classic tab — the legacy operations, unchanged.
    # ------------------------------------------------------------------

    def _build_classic_tab(self) -> QWidget:
        scroll, cards_layout = self._make_scroll_container()

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
            title=t("scanModes.resetUnknowns.title"),
            description=t("scanModes.resetUnknowns.description"),
            button_label=t("scanModes.resetUnknowns.startButton"),
            warning=t("scanModes.resetUnknowns.warning"),
            callback=self._launch_reset_unknown_persons,
            danger=False,
        ))
        cards_layout.addWidget(self._make_card(
            title=t("scanModes.overlapCleanup.title"),
            description=t("scanModes.overlapCleanup.description"),
            button_label=t("scanModes.overlapCleanup.startButton"),
            warning=t("scanModes.overlapCleanup.warning"),
            callback=self._launch_overlapping_unknown_cleanup,
            danger=False,
        ))
        if self._on_identity_repair_scan is not None:
            cards_layout.addWidget(self._make_card(
                title=t("scanModes.identityRepair.title"),
                description=t("scanModes.identityRepair.description"),
                button_label=t("scanModes.identityRepair.startButton"),
                warning=t("scanModes.identityRepair.warning"),
                callback=self._launch_identity_repair_scan,
                danger=False,
            ))
        if self._on_cleanup_empty_unknown_persons is not None:
            cards_layout.addWidget(self._make_card(
                title=t("scanModes.cleanupEmptyUnknowns.title"),
                description=t("scanModes.cleanupEmptyUnknowns.description"),
                button_label=t("scanModes.cleanupEmptyUnknowns.startButton"),
                warning=t("scanModes.cleanupEmptyUnknowns.warning"),
                callback=self._launch_cleanup_empty_unknown_persons,
                danger=False,
            ))
        if self._on_manage_ignored_faces is not None:
            cards_layout.addWidget(self._make_card(
                title=t("scanModes.ignoredFaces.title"),
                description=t("scanModes.ignoredFaces.description"),
                button_label=t("scanModes.ignoredFaces.startButton"),
                warning=None,
                callback=self._launch_manage_ignored_faces,
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
        return scroll

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

    def _launch_deep_rescan(self) -> None:
        self.accept()
        if self._on_deep_rescan is not None:
            self._on_deep_rescan()

    def _launch_deep_rebuild(self) -> None:
        self.accept()
        if self._on_deep_rebuild is not None:
            self._on_deep_rebuild()

    def _launch_deep_train(self) -> None:
        self.accept()
        if self._on_deep_train is not None:
            self._on_deep_train()

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

    def _launch_overlapping_unknown_cleanup(self) -> None:
        self.accept()
        self._on_find_overlapping_unknown_faces()

    def _launch_reset_unknown_persons(self) -> None:
        self.accept()
        self._on_reset_unknown_persons()

    def _launch_identity_repair_scan(self) -> None:
        self.accept()
        if self._on_identity_repair_scan is not None:
            self._on_identity_repair_scan()

    def _launch_cleanup_empty_unknown_persons(self) -> None:
        self.accept()
        if self._on_cleanup_empty_unknown_persons is not None:
            self._on_cleanup_empty_unknown_persons()

    def _launch_manage_ignored_faces(self) -> None:
        self.accept()
        if self._on_manage_ignored_faces is not None:
            self._on_manage_ignored_faces()
