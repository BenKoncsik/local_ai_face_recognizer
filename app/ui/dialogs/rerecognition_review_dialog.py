"""Review dialog for uncertain re-recognition matches.

Walks the user through the SUGGEST items one face at a time (matching the
spec's card layout): the Unknown face crop on the left, its ranked candidate
people as selectable rows on the right, and three actions — **Merge** (into the
selected candidate), **Skip**, or **Choose another person…** (a searchable
picker).  Each merge is applied immediately through
:class:`ReRecognitionService.apply_user_decision`, tagged with the same
``batch_id`` as the run so it shows up — and can be undone — together with the
automatic merges.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from app.config import RecognitionConfig
from app.db.database import session_scope
from app.db.models import Person
from app.services.rerecognition_service import (
    Candidate,
    ReRecognitionService,
    SuggestItem,
)
from app.ui.i18n import t
from app.ui.widgets.person_search_select import PersonSearchSelect

log = logging.getLogger(__name__)

_CROP_SIZE = 160


class _PersonPickerDialog(QDialog):
    """Tiny modal wrapper around :class:`PersonSearchSelect`."""

    def __init__(self, persons: List[Person], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("rerec_pick_person_title"))
        self.setMinimumWidth(340)
        self.selected_id: Optional[int] = None
        self.selected_name: Optional[str] = None

        layout = QVBoxLayout(self)
        self._selector = PersonSearchSelect(self)
        self._selector.set_persons(persons)
        self._selector.person_selected.connect(self._on_selected)
        self._selector.person_double_clicked.connect(self._on_double)
        layout.addWidget(self._selector)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._names = {p.id: p.name for p in persons}

    def _on_selected(self, person_id: int) -> None:
        self.selected_id = person_id
        self.selected_name = self._names.get(person_id)

    def _on_double(self, person_id: int) -> None:
        self._on_selected(person_id)
        self.accept()


class ReRecognitionReviewDialog(QDialog):
    """Step through uncertain matches and merge/skip each one."""

    # Emitted after every applied merge so the host can refresh its view.
    applied = Signal()

    def __init__(
        self,
        suggest_items: List[SuggestItem],
        batch_id: str,
        persons: List[Person],
        recognition_config: Optional[RecognitionConfig] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("rerec_review_title"))
        self.setMinimumWidth(520)
        self._items = list(suggest_items)
        self._batch_id = batch_id
        self._persons = list(persons)
        self._rec_cfg = recognition_config
        self._index = 0
        self._radio_group: Optional[QButtonGroup] = None
        self._candidates: List[Candidate] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        self._remaining_lbl = QLabel()
        self._remaining_lbl.setStyleSheet("color: #888;")
        root.addWidget(self._remaining_lbl)

        body = QHBoxLayout()
        body.setSpacing(14)
        root.addLayout(body, stretch=1)

        # Left: Unknown face crop + previous label
        left = QVBoxLayout()
        self._crop_lbl = QLabel()
        self._crop_lbl.setFixedSize(_CROP_SIZE, _CROP_SIZE)
        self._crop_lbl.setAlignment(Qt.AlignCenter)
        self._crop_lbl.setStyleSheet(
            "background:#1a1a1a; border:1px solid #333; border-radius:4px;"
        )
        left.addWidget(self._crop_lbl)
        self._unknown_lbl = QLabel()
        self._unknown_lbl.setAlignment(Qt.AlignCenter)
        self._unknown_lbl.setStyleSheet("font-weight:bold;")
        left.addWidget(self._unknown_lbl)
        left.addStretch()
        body.addLayout(left)

        # Right: candidate radio list
        right = QVBoxLayout()
        right.addWidget(QLabel(t("rerec_review_candidates")))
        self._candidates_box = QWidget()
        self._candidates_layout = QVBoxLayout(self._candidates_box)
        self._candidates_layout.setContentsMargins(0, 0, 0, 0)
        self._candidates_layout.setSpacing(4)
        right.addWidget(self._candidates_box)
        right.addStretch()
        body.addLayout(right, stretch=1)

        # Action buttons
        actions = QHBoxLayout()
        self._merge_btn = QPushButton(t("rerec_review_merge"))
        self._merge_btn.clicked.connect(self._on_merge)
        self._skip_btn = QPushButton(t("rerec_review_skip"))
        self._skip_btn.clicked.connect(self._on_skip)
        self._other_btn = QPushButton(t("rerec_review_other"))
        self._other_btn.clicked.connect(self._on_choose_other)
        actions.addWidget(self._merge_btn)
        actions.addWidget(self._skip_btn)
        actions.addWidget(self._other_btn)
        actions.addStretch()
        self._close_btn = QPushButton(t("rerec_review_done"))
        self._close_btn.clicked.connect(self.accept)
        actions.addWidget(self._close_btn)
        root.addLayout(actions)

        self._show_current()

    # ------------------------------------------------------------------

    def _show_current(self) -> None:
        if self._index >= len(self._items):
            self._remaining_lbl.setText(t("rerec_review_all_done"))
            self._crop_lbl.clear()
            self._unknown_lbl.clear()
            self._clear_candidates()
            self._merge_btn.setEnabled(False)
            self._skip_btn.setEnabled(False)
            self._other_btn.setEnabled(False)
            return

        item = self._items[self._index]
        remaining = len(self._items) - self._index
        self._remaining_lbl.setText(t("rerec_review_remaining", n=remaining))

        # Crop
        pix = self._load_crop(item.face.crop_path)
        if pix is not None:
            self._crop_lbl.setPixmap(pix)
        else:
            self._crop_lbl.setText("?")
        prev = item.face.prev_person_name or t("rerec_review_unknown")
        self._unknown_lbl.setText(prev)

        # Candidates
        self._clear_candidates()
        self._candidates = item.candidates
        self._radio_group = QButtonGroup(self)
        for i, cand in enumerate(item.candidates):
            pct = int(round(cand.score * 100))
            radio = QRadioButton(f"{cand.name} ({pct}%)")
            if i == 0:
                radio.setChecked(True)
            self._radio_group.addButton(radio, i)
            self._candidates_layout.addWidget(radio)

        self._merge_btn.setEnabled(True)
        self._skip_btn.setEnabled(True)
        self._other_btn.setEnabled(True)

    def _clear_candidates(self) -> None:
        while self._candidates_layout.count():
            child = self._candidates_layout.takeAt(0)
            w = child.widget()
            if w is not None:
                w.deleteLater()
        self._radio_group = None

    @staticmethod
    def _load_crop(path: Optional[str]) -> Optional[QPixmap]:
        if not path or not Path(path).exists():
            return None
        pix = QPixmap(path)
        if pix.isNull():
            return None
        return pix.scaled(
            _CROP_SIZE, _CROP_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

    # ------------------------------------------------------------------

    def _on_merge(self) -> None:
        if self._radio_group is None or not self._candidates:
            return
        idx = self._radio_group.checkedId()
        if idx < 0 or idx >= len(self._candidates):
            return
        cand = self._candidates[idx]
        self._apply(cand.person_id, cand.name, cand.score)

    def _on_choose_other(self) -> None:
        picker = _PersonPickerDialog(self._persons, self)
        if picker.exec() != QDialog.Accepted or picker.selected_id is None:
            return
        self._apply(picker.selected_id, picker.selected_name or "", 0.0)

    def _on_skip(self) -> None:
        self._index += 1
        self._show_current()

    def _apply(self, person_id: int, name: str, score: float) -> None:
        item = self._items[self._index]
        try:
            with session_scope() as session:
                svc = ReRecognitionService(session, self._rec_cfg)
                svc.apply_user_decision(
                    item.face, person_id, name, score, self._batch_id
                )
        except Exception:  # noqa: BLE001
            log.exception("Applying reviewed re-recognition merge failed")
            return
        self.applied.emit()
        self._index += 1
        self._show_current()
