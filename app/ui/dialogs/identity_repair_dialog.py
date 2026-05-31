"""Identity Repair results dialog.

Shows the merge candidates returned by
:class:`app.services.identity_repair_service.IdentityRepairService` as a
checklist of "Unknown A ≈ Unknown B" pairs with face thumbnails.  The user keeps
the pairs they want and confirms; the caller then applies the approved pairs.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
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

from app.services.identity_repair_service import RepairCandidate
from app.ui.i18n import t

_THUMB_PX = 56


class IdentityRepairDialog(QDialog):
    """Modal checklist of fragmented-identity merge candidates."""

    def __init__(
        self,
        candidates: List[RepairCandidate],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._candidates = candidates
        self._merge_requested = False
        self._checkboxes: List[Tuple[QCheckBox, RepairCandidate]] = []

        self.setWindowTitle(t("repair_title"))
        self.setMinimumWidth(560)
        self.resize(640, 720)
        self._build_ui()

    # ------------------------------------------------------------------

    def merge_requested(self) -> bool:
        return self._merge_requested

    def selected_pairs(self) -> List[Tuple[int, int]]:
        return [
            (c.person_a_id, c.person_b_id)
            for cb, c in self._checkboxes
            if cb.isChecked()
        ]

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        intro = QLabel(t("repair_intro"))
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #A6ADC8;")
        outer.addWidget(intro)

        count = QLabel(t("repair_count", n=len(self._candidates)))
        count.setStyleSheet("font-weight: bold;")
        outer.addWidget(count)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        container = QWidget()
        rows = QVBoxLayout(container)
        rows.setContentsMargins(4, 4, 4, 4)
        rows.setSpacing(8)

        for cand in self._candidates:
            rows.addWidget(self._make_row(cand))
        rows.addStretch()

        scroll.setWidget(container)
        outer.addWidget(scroll)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        merge_btn = QPushButton(t("repair_merge_btn"))
        merge_btn.clicked.connect(self._on_merge)
        close_btn = QPushButton(t("scanModes.close"))
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(merge_btn)
        btn_row.addWidget(close_btn)
        outer.addLayout(btn_row)

    def _make_row(self, cand: RepairCandidate) -> QFrame:
        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout = QHBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(10)

        checkbox = QCheckBox()
        checkbox.setChecked(True)
        layout.addWidget(checkbox)
        self._checkboxes.append((checkbox, cand))

        layout.addWidget(self._thumb(cand.rep_crop_a))
        layout.addWidget(self._thumb(cand.rep_crop_b))

        pct = int(round(cand.confidence * 100))
        label = QLabel(
            t(
                "repair_pair_label",
                a=cand.name_a, b=cand.name_b, pct=pct,
                fa=cand.face_count_a, fb=cand.face_count_b,
            )
        )
        label.setWordWrap(True)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(label, stretch=1)

        return card

    @staticmethod
    def _thumb(crop_path: Optional[str]) -> QLabel:
        lbl = QLabel()
        lbl.setFixedSize(_THUMB_PX, _THUMB_PX)
        lbl.setAlignment(Qt.AlignCenter)
        if crop_path:
            pix = QPixmap(crop_path)
            if not pix.isNull():
                lbl.setPixmap(
                    pix.scaled(
                        _THUMB_PX, _THUMB_PX,
                        Qt.KeepAspectRatio, Qt.SmoothTransformation,
                    )
                )
                return lbl
        lbl.setText("?")
        lbl.setStyleSheet("color: #6C7086; border: 1px solid #313244;")
        return lbl

    def _on_merge(self) -> None:
        if not self.selected_pairs():
            self.reject()
            return
        self._merge_requested = True
        self.accept()
