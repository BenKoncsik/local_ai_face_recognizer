"""Dialog listing unknown face boxes that overlap known faces."""

from __future__ import annotations

from typing import Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.services.duplicate_unknown_face_finder import OverlappingUnknownFaceMatch
from app.ui.i18n import t


class OverlappingUnknownFacesDialog(QDialog):
    """Review and select overlapping question-mark faces for deletion."""

    open_face_requested = Signal(int)

    def __init__(
        self,
        matches: Sequence[OverlappingUnknownFaceMatch],
        images_examined: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._matches = list(matches)
        self._delete_requested = False
        self._checkboxes: dict[int, QCheckBox] = {}
        self.setWindowTitle(t("overlap_dialog_title"))
        self.resize(980, 520)
        self._build_ui(images_examined)

    def delete_requested(self) -> bool:
        return self._delete_requested

    def selected_unknown_face_ids(self) -> list[int]:
        return [
            match.unknown_face_id
            for match in self._matches
            if self._checkboxes[match.unknown_face_id].isChecked()
        ]

    def _build_ui(self, images_examined: int) -> None:
        layout = QVBoxLayout(self)

        summary = QLabel(
            t(
                "overlap_summary",
                images=images_examined,
                matches=len(self._matches),
            )
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)

        self._table = QTableWidget(len(self._matches), 6)
        self._table.setHorizontalHeaderLabels(
            [
                t("overlap_col_image"),
                t("overlap_col_person"),
                t("overlap_col_unknown_id"),
                t("overlap_col_known_id"),
                t("overlap_col_iou"),
                t("overlap_col_delete"),
            ]
        )
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.itemDoubleClicked.connect(self._open_selected_row)

        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, 6):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)

        for row, match in enumerate(self._matches):
            self._add_row(row, match)
        if self._matches:
            self._table.selectRow(0)
        layout.addWidget(self._table)

        quick_row = QHBoxLayout()
        self._select_all_btn = QPushButton(t("overlap_select_all"))
        self._select_all_btn.clicked.connect(lambda: self._set_all_checked(True))
        quick_row.addWidget(self._select_all_btn)

        self._select_none_btn = QPushButton(t("overlap_select_none"))
        self._select_none_btn.clicked.connect(lambda: self._set_all_checked(False))
        quick_row.addWidget(self._select_none_btn)

        self._open_btn = QPushButton(t("overlap_open_image"))
        self._open_btn.clicked.connect(self._open_selected_row)
        quick_row.addWidget(self._open_btn)

        quick_row.addStretch()
        layout.addLayout(quick_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        self._delete_btn = QPushButton(t("overlap_delete_selected"))
        self._delete_btn.setEnabled(bool(self._matches))
        self._delete_btn.clicked.connect(self._request_delete)
        buttons.addButton(self._delete_btn, QDialogButtonBox.AcceptRole)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _add_row(self, row: int, match: OverlappingUnknownFaceMatch) -> None:
        path_item = QTableWidgetItem(match.display_path)
        path_item.setToolTip(match.image_path)
        self._table.setItem(row, 0, path_item)
        self._table.setItem(row, 1, QTableWidgetItem(match.known_person_name))
        self._table.setItem(row, 2, QTableWidgetItem(str(match.unknown_face_id)))
        self._table.setItem(row, 3, QTableWidgetItem(str(match.known_face_id)))
        self._table.setItem(row, 4, QTableWidgetItem(f"{match.iou:.2f}"))

        checkbox = QCheckBox()
        checkbox.setChecked(True)
        checkbox.setToolTip(t("overlap_delete_checkbox_tip"))
        wrapper = QWidget()
        wrapper_layout = QHBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.addWidget(checkbox, alignment=Qt.AlignCenter)
        self._table.setCellWidget(row, 5, wrapper)
        self._checkboxes[match.unknown_face_id] = checkbox

    def _selected_match(self) -> OverlappingUnknownFaceMatch | None:
        row = self._table.currentRow()
        if row < 0 or row >= len(self._matches):
            return None
        return self._matches[row]

    def _open_selected_row(self, *_args) -> None:
        match = self._selected_match()
        if match is not None:
            self.open_face_requested.emit(match.known_face_id)

    def _set_all_checked(self, checked: bool) -> None:
        for checkbox in self._checkboxes.values():
            checkbox.setChecked(checked)

    def _request_delete(self) -> None:
        if not self.selected_unknown_face_ids():
            QMessageBox.information(
                self,
                t("overlap_no_selection_title"),
                t("overlap_no_selection_msg"),
            )
            return
        self._delete_requested = True
        self.accept()
