"""History + undo dialog for re-recognition merge batches.

Lists every re-recognition run (newest first) with its date, face count and the
people faces were merged into, and lets the user undo a whole batch.  Selecting a
batch shows its per-face audit rows (image id, previous Unknown label, matched
person, score).  Undo restores every still-applied face in the batch via
:meth:`ReRecognitionService.undo_batch`.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.db.database import session_scope
from app.services.rerecognition_service import BatchSummary, ReRecognitionService
from app.ui.i18n import t

log = logging.getLogger(__name__)

_ROLE_BATCH = Qt.UserRole + 1


class ReRecognitionHistoryDialog(QDialog):
    """Browse and undo past re-recognition runs."""

    # Emitted whenever a batch is undone, so the host can refresh its view.
    changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("rerec_hist_title"))
        self.setMinimumSize(620, 420)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        body = QHBoxLayout()
        root.addLayout(body, stretch=1)

        self._batch_list = QListWidget()
        self._batch_list.setMaximumWidth(260)
        self._batch_list.currentItemChanged.connect(self._on_batch_selected)
        body.addWidget(self._batch_list)

        self._detail_list = QListWidget()
        body.addWidget(self._detail_list, stretch=1)

        actions = QHBoxLayout()
        self._undo_btn = QPushButton(t("rerec_hist_undo"))
        self._undo_btn.clicked.connect(self._on_undo)
        self._undo_btn.setEnabled(False)
        actions.addWidget(self._undo_btn)
        actions.addStretch()
        self._close_btn = QPushButton(t("rerec_hist_close"))
        self._close_btn.clicked.connect(self.accept)
        actions.addWidget(self._close_btn)
        root.addLayout(actions)

        self._reload()

    # ------------------------------------------------------------------

    def _reload(self) -> None:
        self._batch_list.clear()
        self._detail_list.clear()
        batches = self._load_batches()
        if not batches:
            placeholder = QListWidgetItem(t("rerec_hist_empty"))
            placeholder.setFlags(Qt.NoItemFlags)
            self._batch_list.addItem(placeholder)
            self._undo_btn.setEnabled(False)
            return
        for batch in batches:
            self._batch_list.addItem(self._make_batch_item(batch))
        self._batch_list.setCurrentRow(0)

    def _make_batch_item(self, batch: BatchSummary) -> QListWidgetItem:
        when = batch.created_at.strftime("%Y-%m-%d %H:%M")
        people = ", ".join(batch.target_names[:3])
        if len(batch.target_names) > 3:
            people += " …"
        status = (
            t("rerec_hist_status_undone")
            if batch.is_undone
            else t("rerec_hist_status_active")
        )
        label = f"{when}  ·  {batch.n_faces} 🙂  ·  {status}\n{people}"
        item = QListWidgetItem(label)
        item.setData(_ROLE_BATCH, batch.batch_id)
        if batch.is_undone:
            item.setForeground(Qt.gray)
        return item

    def _on_batch_selected(
        self, current: Optional[QListWidgetItem], _prev: Optional[QListWidgetItem]
    ) -> None:
        self._detail_list.clear()
        if current is None:
            self._undo_btn.setEnabled(False)
            return
        batch_id = current.data(_ROLE_BATCH)
        if not batch_id:
            self._undo_btn.setEnabled(False)
            return
        rows, any_active = self._load_rows(batch_id)
        for line in rows:
            self._detail_list.addItem(line)
        self._undo_btn.setEnabled(any_active)

    def _on_undo(self) -> None:
        current = self._batch_list.currentItem()
        if current is None:
            return
        batch_id = current.data(_ROLE_BATCH)
        if not batch_id:
            return
        try:
            with session_scope() as session:
                n = ReRecognitionService(session).undo_batch(batch_id)
        except Exception as exc:  # noqa: BLE001
            log.exception("Undo of re-recognition batch failed")
            QMessageBox.critical(
                self, t("rerec_error_title"), t("rerec_undo_error", error=str(exc))
            )
            return
        QMessageBox.information(
            self, t("rerec_hist_title"), t("rerec_undo_done", n=n)
        )
        self.changed.emit()
        self._reload()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_batches(self) -> List[BatchSummary]:
        with session_scope() as session:
            return ReRecognitionService(session).list_batches()

    def _load_rows(self, batch_id: str) -> "tuple[List[str], bool]":
        lines: List[str] = []
        any_active = False
        with session_scope() as session:
            rows = ReRecognitionService(session).get_batch_rows(batch_id)
            for r in rows:
                if r.undone_at is None:
                    any_active = True
                lines.append(
                    t(
                        "rerec_hist_detail",
                        image=r.image_id if r.image_id is not None else "?",
                        prev=r.prev_person_name or t("rerec_review_unknown"),
                        matched=r.matched_person_name or "?",
                        score=int(round((r.score or 0.0) * 100)),
                    )
                )
        return lines, any_active
