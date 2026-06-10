"""Review list for faces auto-merged from an Unknown cluster.

Lists every face currently flagged ``pending`` by
:class:`app.services.unknown_merge_service.UnknownMergeService` as a card:
the face crop, its current person, the original Unknown it came from, the source
image path and date, plus per-row **Accept**, **Move…** and **Delete** actions.
An **Accept all** button clears the whole queue at once.  Each action goes
through the same service the in-canvas review uses, so behaviour is identical
everywhere.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.config import RecognitionConfig
from app.db.database import session_scope
from app.db.models import Person
from app.services.unknown_merge_service import PendingAutoMerge, UnknownMergeService
from app.ui.i18n import t
from app.ui.widgets.person_search_select import PersonSearchSelect

log = logging.getLogger(__name__)

_CROP_SIZE = 96


class _PersonPickerDialog(QDialog):
    """Modal person picker reused for the per-row "Move…" action."""

    def __init__(self, persons: List[Person], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("amerge_move"))
        self.setMinimumWidth(340)
        self.selected_id: Optional[int] = None

        layout = QVBoxLayout(self)
        self._selector = PersonSearchSelect(self)
        self._selector.set_persons(persons)
        self._selector.person_selected.connect(self._on_selected)
        self._selector.person_double_clicked.connect(self._on_double)
        layout.addWidget(self._selector)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_selected(self, person_id: int) -> None:
        self.selected_id = person_id

    def _on_double(self, person_id: int) -> None:
        self.selected_id = person_id
        self.accept()


class AutoMergeReviewDialog(QDialog):
    """List of pending auto-merged faces with per-face review actions."""

    # Emitted whenever something changed so the host can refresh its views.
    applied = Signal()

    def __init__(
        self,
        recognition_config: Optional[RecognitionConfig] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("amerge_review_title"))
        self.setMinimumSize(560, 480)
        self._rec_cfg = recognition_config
        self._dirty = False

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        header = QHBoxLayout()
        self._count_lbl = QLabel()
        self._count_lbl.setStyleSheet("color:#bbb; font-weight:bold;")
        header.addWidget(self._count_lbl, stretch=1)
        self._accept_all_btn = QPushButton(t("amerge_accept_all"))
        self._accept_all_btn.clicked.connect(self._on_accept_all)
        header.addWidget(self._accept_all_btn)
        root.addLayout(header)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._list_widget = QWidget()
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(6)
        self._list_layout.addStretch()
        self._scroll.setWidget(self._list_widget)
        root.addWidget(self._scroll, stretch=1)

        self._empty_lbl = QLabel(t("amerge_review_empty"))
        self._empty_lbl.setAlignment(Qt.AlignCenter)
        self._empty_lbl.setStyleSheet("color:#888; font-style:italic;")
        root.addWidget(self._empty_lbl)

        close_btn = QPushButton(t("amerge_close"))
        close_btn.clicked.connect(self.accept)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

        self._reload()

    # ------------------------------------------------------------------

    def _reload(self) -> None:
        # Clear existing cards (keep the trailing stretch at the end).
        while self._list_layout.count() > 1:
            child = self._list_layout.takeAt(0)
            w = child.widget()
            if w is not None:
                w.deleteLater()

        with session_scope() as session:
            rows = UnknownMergeService(session, self._rec_cfg).list_pending()

        self._count_lbl.setText(t("amerge_review_count", n=len(rows)))
        self._accept_all_btn.setEnabled(bool(rows))
        self._empty_lbl.setVisible(not rows)
        self._scroll.setVisible(bool(rows))

        for row in rows:
            self._list_layout.insertWidget(
                self._list_layout.count() - 1, self._build_card(row)
            )

    def _build_card(self, row: PendingAutoMerge) -> QFrame:
        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        card.setStyleSheet(
            "QFrame { background:#222; border:1px solid #3a3a3a; border-radius:5px; }"
        )
        lay = QHBoxLayout(card)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(10)

        crop = QLabel()
        crop.setFixedSize(_CROP_SIZE, _CROP_SIZE)
        crop.setAlignment(Qt.AlignCenter)
        crop.setStyleSheet("background:#1a1a1a; border:1px solid #333;")
        pix = self._load_crop(row.crop_path)
        crop.setPixmap(pix) if pix is not None else crop.setText("?")
        lay.addWidget(crop)

        info = QVBoxLayout()
        info.setSpacing(2)
        name = QLabel(row.person_name or "?")
        name.setStyleSheet("font-weight:bold; color:#fff;")
        info.addWidget(name)
        info.addWidget(self._dim_label(t("amerge_col_source", name=row.source_person_name)))
        if row.image_path:
            info.addWidget(self._dim_label(Path(row.image_path).name))
        if row.assigned_at is not None:
            info.addWidget(self._dim_label(row.assigned_at.strftime("%Y-%m-%d %H:%M")))
        info.addStretch()
        lay.addLayout(info, stretch=1)

        actions = QVBoxLayout()
        actions.setSpacing(4)
        accept_btn = QPushButton("✓ " + t("amerge_accept"))
        accept_btn.clicked.connect(lambda: self._on_accept(row.face_id))
        move_btn = QPushButton("⤴ " + t("amerge_move"))
        move_btn.clicked.connect(lambda: self._on_move(row.face_id))
        del_btn = QPushButton("🗑 " + t("amerge_delete"))
        del_btn.clicked.connect(lambda: self._on_delete(row.face_id))
        for b in (accept_btn, move_btn, del_btn):
            b.setFixedWidth(190)
            actions.addWidget(b)
        actions.addStretch()
        lay.addLayout(actions)
        return card

    @staticmethod
    def _dim_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color:#999; font-size:11px;")
        return lbl

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

    def _on_accept(self, face_id: int) -> None:
        with session_scope() as session:
            UnknownMergeService(session).confirm_auto_merge(face_id)
        self._after_change()

    def _on_move(self, face_id: int) -> None:
        with session_scope() as session:
            persons = (
                session.query(Person).order_by(Person.name).all()
            )
            # Detach the data the picker needs while the session is open.
            picker = _PersonPickerDialog(persons, self)
        if picker.exec() != QDialog.Accepted or picker.selected_id is None:
            return
        with session_scope() as session:
            UnknownMergeService(session).move_auto_merge(face_id, picker.selected_id)
        self._after_change()

    def _on_delete(self, face_id: int) -> None:
        reply = QMessageBox.question(
            self,
            t("remove_face_title"),
            t("remove_face_msg"),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        with session_scope() as session:
            UnknownMergeService(session).delete_face(face_id)
        self._after_change()

    def _on_accept_all(self) -> None:
        with session_scope() as session:
            svc = UnknownMergeService(session)
            for fid in svc.pending_face_ids():
                svc.confirm_auto_merge(fid)
        self._after_change()

    def _after_change(self) -> None:
        self._dirty = True
        self.applied.emit()
        self._reload()
