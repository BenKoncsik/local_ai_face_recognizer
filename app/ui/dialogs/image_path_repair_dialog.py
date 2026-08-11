"""Review UI for re-attaching database records to moved/renamed image files.

Companion to :mod:`app.services.image_path_matcher`.  The user picks the folders
to search, starts the scan (which runs as a **background task** — visible,
pausable and stoppable in the Task Manager), then walks the result list row by
row: every missing image shows its ranked candidates in a drop-down, with a
preview of the highlighted file.  Accepted rows are written back in a second
background task.

The dialog is non-modal on purpose: the scan can take minutes on a large photo
tree and the rest of the app stays usable meanwhile.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.db.database import session_scope
from app.services.image_path_matcher import (
    ImagePathMatcher,
    MatchProposal,
    MatchReport,
)
from app.ui.i18n import t

_PREVIEW_PX = 260

# Table columns
_COL_USE = 0
_COL_FILE = 1
_COL_EXPECTED = 2
_COL_MATCH = 3
_COL_CONF = 4

#: Item data role holding the candidate path of a combo entry.
_PATH_ROLE = Qt.UserRole


class ImagePathRepairDialog(QDialog):
    """Scan folders for missing originals and re-attach them one by one."""

    def __init__(
        self,
        *,
        search_roots: Optional[List[str]] = None,
        extensions: Optional[List[str]] = None,
        on_applied: Optional[Callable[[int], None]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._extensions = extensions
        self._on_applied = on_applied
        self._proposals: List[MatchProposal] = []
        self._task = None
        self._alive = True

        self.setWindowTitle(t("pathfix_title"))
        self.setMinimumWidth(860)
        self.resize(1000, 700)
        self.setModal(False)
        self._build_ui()
        for root in search_roots or []:
            self._add_folder(root)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        intro = QLabel(t("pathfix_intro"))
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #A6ADC8;")
        outer.addWidget(intro)

        # --- search folders -------------------------------------------------
        folders_box = QGroupBox(t("pathfix_folders_label"))
        folders_layout = QHBoxLayout(folders_box)
        self._folders = QListWidget()
        self._folders.setMaximumHeight(90)
        folders_layout.addWidget(self._folders, stretch=1)

        folder_btns = QVBoxLayout()
        add_btn = QPushButton(t("pathfix_add_folder"))
        add_btn.clicked.connect(self._on_add_folder)
        remove_btn = QPushButton(t("pathfix_remove_folder"))
        remove_btn.clicked.connect(self._on_remove_folder)
        folder_btns.addWidget(add_btn)
        folder_btns.addWidget(remove_btn)
        folder_btns.addStretch()
        folders_layout.addLayout(folder_btns)
        outer.addWidget(folders_box)

        # --- controls -------------------------------------------------------
        controls = QHBoxLayout()
        self._verify_chk = QCheckBox(t("pathfix_verify_hash"))
        self._verify_chk.setChecked(True)
        self._verify_chk.setToolTip(t("pathfix_verify_hash_tip"))
        controls.addWidget(self._verify_chk)
        controls.addStretch()
        self._start_btn = QPushButton(f"🔍  {t('pathfix_start_btn')}")
        self._start_btn.clicked.connect(self._on_start)
        controls.addWidget(self._start_btn)
        outer.addLayout(controls)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        outer.addWidget(self._progress)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: #A6ADC8;")
        outer.addWidget(self._status)

        # --- results table + preview ---------------------------------------
        splitter = QSplitter(Qt.Horizontal)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels([
            "",
            t("pathfix_col_file"),
            t("pathfix_col_expected"),
            t("pathfix_col_match"),
            t("pathfix_col_confidence"),
        ])
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(_COL_USE, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(_COL_FILE, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(_COL_EXPECTED, QHeaderView.Stretch)
        header.setSectionResizeMode(_COL_MATCH, QHeaderView.Stretch)
        header.setSectionResizeMode(_COL_CONF, QHeaderView.ResizeToContents)
        self._table.itemSelectionChanged.connect(self._update_preview)
        splitter.addWidget(self._table)

        preview_box = QWidget()
        preview_layout = QVBoxLayout(preview_box)
        preview_layout.setContentsMargins(6, 0, 0, 0)
        self._preview = QLabel(t("pathfix_preview_none"))
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setMinimumSize(_PREVIEW_PX, _PREVIEW_PX)
        self._preview.setStyleSheet("border: 1px solid #45475A; color: #A6ADC8;")
        preview_layout.addWidget(self._preview)
        self._preview_path = QLabel("")
        self._preview_path.setWordWrap(True)
        self._preview_path.setStyleSheet("color: #A6ADC8; font-size: 11px;")
        preview_layout.addWidget(self._preview_path)
        preview_layout.addStretch()
        splitter.addWidget(preview_box)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        outer.addWidget(splitter, stretch=1)

        # --- selection helpers + apply --------------------------------------
        btn_row = QHBoxLayout()
        confident_btn = QPushButton(t("pathfix_select_confident"))
        confident_btn.clicked.connect(lambda: self._set_all_checked(confident_only=True))
        all_btn = QPushButton(t("pathfix_select_all"))
        all_btn.clicked.connect(lambda: self._set_all_checked(confident_only=False))
        none_btn = QPushButton(t("pathfix_select_none"))
        none_btn.clicked.connect(self._clear_all_checked)
        btn_row.addWidget(confident_btn)
        btn_row.addWidget(all_btn)
        btn_row.addWidget(none_btn)
        btn_row.addStretch()
        self._apply_btn = QPushButton(f"✔  {t('pathfix_apply_btn')}")
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._on_apply)
        close_btn = QPushButton(t("close"))
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(self._apply_btn)
        btn_row.addWidget(close_btn)
        outer.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Search folders
    # ------------------------------------------------------------------

    def _add_folder(self, folder: str) -> None:
        if not folder:
            return
        existing = {
            self._folders.item(i).text() for i in range(self._folders.count())
        }
        if folder in existing:
            return
        self._folders.addItem(folder)

    def _on_add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, t("pathfix_add_folder"), str(Path.home())
        )
        if folder:
            self._add_folder(folder)

    def _on_remove_folder(self) -> None:
        for item in self._folders.selectedItems():
            self._folders.takeItem(self._folders.row(item))

    def _search_roots(self) -> List[str]:
        return [self._folders.item(i).text() for i in range(self._folders.count())]

    # ------------------------------------------------------------------
    # Scan (background task)
    # ------------------------------------------------------------------

    def _on_start(self) -> None:
        roots = self._search_roots()
        if not roots:
            QMessageBox.warning(
                self, t("pathfix_title"), t("pathfix_need_folder")
            )
            return

        verify = self._verify_chk.isChecked()
        extensions = self._extensions

        def work(ctx):  # noqa: ANN001 — runs on the task thread
            matcher = ImagePathMatcher(
                roots, extensions=extensions, verify_hash=verify
            )
            with session_scope() as session:
                return matcher.run(
                    session,
                    progress_cb=lambda pct, msg: ctx.report(max(pct, 0), msg),
                    checkpoint=ctx.checkpoint,
                )

        from app.tasks import TaskPriority, get_task_manager

        self._start_btn.setEnabled(False)
        self._apply_btn.setEnabled(False)
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._status.setText(t("pathfix_running"))

        self._task = get_task_manager().submit(
            t("task_path_match"),
            work,
            supports_pause=True,
            priority=TaskPriority.LOW,
            on_done=self._on_scan_done,
            on_error=self._on_task_error,
            on_cancelled=self._on_task_cancelled,
        )
        self._task.progress_changed.connect(self._on_progress)

    def _on_progress(self, percent: int, message: str) -> None:
        if not self._alive:
            return
        self._progress.setValue(max(0, min(percent, 100)))
        if message:
            self._status.setText(message)

    def _on_scan_done(self, report: object) -> None:
        if not self._alive:
            return
        self._task = None
        self._start_btn.setEnabled(True)
        self._progress.setVisible(False)
        if not isinstance(report, MatchReport):
            return

        self._proposals = report.proposals
        self._populate(report)
        if report.missing_total == 0:
            self._status.setText(t("pathfix_no_missing"))
            return
        self._status.setText(
            t(
                "pathfix_summary",
                missing=report.missing_total,
                matched=report.matched_count,
                confident=report.confident_count,
                scanned=report.scanned_files,
            )
        )
        self._apply_btn.setEnabled(bool(report.matched_count))

    def _on_task_error(self, message: str) -> None:
        if not self._alive:
            return
        self._task = None
        self._start_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._status.setText("")
        QMessageBox.critical(self, t("pathfix_title"), message)

    def _on_task_cancelled(self) -> None:
        if not self._alive:
            return
        self._task = None
        self._start_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._status.setText(t("pathfix_cancelled"))

    # ------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------

    def _populate(self, report: MatchReport) -> None:
        self._table.setRowCount(0)
        for proposal in report.proposals:
            self._add_row(proposal)

    def _add_row(self, proposal: MatchProposal) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)

        check = QCheckBox()
        check.setEnabled(bool(proposal.candidates))
        check.setChecked(proposal.is_confident)
        holder = QWidget()
        holder_layout = QHBoxLayout(holder)
        holder_layout.setContentsMargins(0, 0, 0, 0)
        holder_layout.setAlignment(Qt.AlignCenter)
        holder_layout.addWidget(check)
        self._table.setCellWidget(row, _COL_USE, holder)

        name_item = QTableWidgetItem(proposal.missing.name)
        # The image id travels with the row so applying never depends on order.
        name_item.setData(Qt.UserRole, proposal.missing.image_id)
        self._table.setItem(row, _COL_FILE, name_item)

        expected = QTableWidgetItem(proposal.missing.resolved_path)
        expected.setToolTip(proposal.missing.resolved_path)
        self._table.setItem(row, _COL_EXPECTED, expected)

        combo = QComboBox()
        combo.addItem(t("pathfix_skip_option"), None)
        for cand in proposal.candidates:
            label = cand.path
            if cand.is_proof:
                label = f"✔ {label}"
            combo.addItem(label, cand.path)
        combo.addItem(t("pathfix_browse_option"), "__browse__")
        if proposal.candidates:
            combo.setCurrentIndex(1)
        combo.currentIndexChanged.connect(
            lambda _idx, r=row: self._on_combo_changed(r)
        )
        self._table.setCellWidget(row, _COL_MATCH, combo)

        conf_item = QTableWidgetItem(self._confidence_label(proposal))
        self._table.setItem(row, _COL_CONF, conf_item)

    @staticmethod
    def _confidence_label(proposal: MatchProposal) -> str:
        best = proposal.best
        if best is None:
            return t("pathfix_conf_none")
        if best.is_proof:
            return t("pathfix_conf_proof")
        if best.hash_match is False:
            return t("pathfix_conf_mismatch")
        return t("pathfix_conf_guess", score=int(best.score * 100))

    def _on_combo_changed(self, row: int) -> None:
        combo = self._table.cellWidget(row, _COL_MATCH)
        if not isinstance(combo, QComboBox):
            return
        if combo.currentData() == "__browse__":
            path, _ = QFileDialog.getOpenFileName(
                self, t("pathfix_browse_option"), str(Path.home())
            )
            if path:
                # Insert the hand-picked file just above the "Browse…" entry and
                # select it, so a manual choice behaves like any other candidate.
                combo.blockSignals(True)
                combo.insertItem(combo.count() - 1, path, path)
                combo.setCurrentIndex(combo.count() - 2)
                combo.blockSignals(False)
                self._set_row_checked(row, True)
            else:
                combo.blockSignals(True)
                combo.setCurrentIndex(0)
                combo.blockSignals(False)
        self._update_preview()

    def _set_row_checked(self, row: int, checked: bool) -> None:
        check = self._row_checkbox(row)
        if check is not None and check.isEnabled():
            check.setChecked(checked)

    def _row_checkbox(self, row: int) -> Optional[QCheckBox]:
        holder = self._table.cellWidget(row, _COL_USE)
        if holder is None:
            return None
        return holder.findChild(QCheckBox)

    def _set_all_checked(self, *, confident_only: bool) -> None:
        for row in range(self._table.rowCount()):
            proposal = self._proposal_for_row(row)
            if proposal is None or not proposal.candidates:
                continue
            want = proposal.is_confident if confident_only else True
            self._set_row_checked(row, want)

    def _clear_all_checked(self) -> None:
        for row in range(self._table.rowCount()):
            self._set_row_checked(row, False)

    def _proposal_for_row(self, row: int) -> Optional[MatchProposal]:
        item = self._table.item(row, _COL_FILE)
        if item is None:
            return None
        image_id = item.data(Qt.UserRole)
        for proposal in self._proposals:
            if proposal.missing.image_id == image_id:
                return proposal
        return None

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _update_preview(self) -> None:
        rows = self._table.selectionModel().selectedRows() if self._table.selectionModel() else []
        if not rows:
            self._preview.setText(t("pathfix_preview_none"))
            self._preview.setPixmap(QPixmap())
            self._preview_path.setText("")
            return
        row = rows[0].row()
        combo = self._table.cellWidget(row, _COL_MATCH)
        path = combo.currentData() if isinstance(combo, QComboBox) else None
        if not path or path == "__browse__":
            self._preview.setPixmap(QPixmap())
            self._preview.setText(t("pathfix_preview_none"))
            self._preview_path.setText("")
            return
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self._preview.setPixmap(QPixmap())
            self._preview.setText(t("pathfix_preview_none"))
        else:
            self._preview.setPixmap(
                pixmap.scaled(
                    _PREVIEW_PX, _PREVIEW_PX,
                    Qt.KeepAspectRatio, Qt.SmoothTransformation,
                )
            )
        self._preview_path.setText(str(path))

    # ------------------------------------------------------------------
    # Apply (background task)
    # ------------------------------------------------------------------

    def _decisions(self) -> Dict[int, str]:
        decisions: Dict[int, str] = {}
        for row in range(self._table.rowCount()):
            check = self._row_checkbox(row)
            if check is None or not check.isChecked():
                continue
            item = self._table.item(row, _COL_FILE)
            combo = self._table.cellWidget(row, _COL_MATCH)
            if item is None or not isinstance(combo, QComboBox):
                continue
            path = combo.currentData()
            if not path or path == "__browse__":
                continue
            decisions[int(item.data(Qt.UserRole))] = str(path)
        return decisions

    def _on_apply(self) -> None:
        decisions = self._decisions()
        if not decisions:
            QMessageBox.information(
                self, t("pathfix_title"), t("pathfix_nothing_selected")
            )
            return

        def work(ctx):  # noqa: ANN001 — runs on the task thread
            with session_scope() as session:
                return ImagePathMatcher.apply(
                    session,
                    decisions,
                    progress_cb=lambda pct, msg: ctx.report(pct, msg),
                    checkpoint=ctx.checkpoint,
                )

        from app.tasks import TaskPriority, get_task_manager

        self._apply_btn.setEnabled(False)
        self._progress.setValue(0)
        self._progress.setVisible(True)

        task = get_task_manager().submit(
            t("task_path_apply"),
            work,
            supports_pause=True,
            priority=TaskPriority.LOW,
            on_done=self._on_apply_done,
            on_error=self._on_task_error,
        )
        task.progress_changed.connect(self._on_progress)

    def _on_apply_done(self, result: object) -> None:
        if not self._alive:
            return
        self._progress.setVisible(False)
        self._apply_btn.setEnabled(True)
        updated = getattr(result, "updated", 0)
        skipped = getattr(result, "skipped", 0)
        errors = list(getattr(result, "errors", []))
        message = t("pathfix_applied", n=updated, skipped=skipped)
        if errors:
            message += "\n\n" + "\n".join(errors[:10])
            if len(errors) > 10:
                message += f"\n… (+{len(errors) - 10})"
        QMessageBox.information(self, t("pathfix_title"), message)
        self._status.setText(t("pathfix_applied", n=updated, skipped=skipped))
        if updated and self._on_applied is not None:
            self._on_applied(updated)

    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 — Qt signature
        """Stop routing task callbacks into a window that is going away."""
        self._alive = False
        if self._task is not None:
            try:
                self._task.progress_changed.disconnect(self._on_progress)
            except (RuntimeError, TypeError):
                pass
        super().closeEvent(event)
