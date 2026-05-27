"""Family relationship based person and image search panel."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.db.database import session_scope
from app.db.models import Person
from app.services.family_service import FamilyImageResult, FamilyService
from app.ui.i18n import t
from app.ui.widgets.person_search_select import PersonSearchSelect

log = logging.getLogger(__name__)

_ROLE_IMAGE_ID = Qt.UserRole
_PAGE_SIZE = 50
_THUMB_SIZE = 72


def _hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet("color: #3a3a3a;")
    return line


class FamilySearchPanel(QWidget):
    """Search shared photos by two selected persons and their relationship."""

    image_open_requested = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._persons: list[Person] = []
        self._offset = 0
        self._total = 0
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        selectors = QWidget()
        selectors_layout = QGridLayout(selectors)
        selectors_layout.setContentsMargins(0, 0, 0, 0)
        selectors_layout.setHorizontalSpacing(8)
        selectors_layout.setVerticalSpacing(6)

        self._person_a_lbl = QLabel()
        self._person_a = PersonSearchSelect()
        self._person_a.person_selected.connect(self._on_people_changed)
        selectors_layout.addWidget(self._person_a_lbl, 0, 0)
        selectors_layout.addWidget(self._person_a, 1, 0)

        self._person_b_lbl = QLabel()
        self._person_b = PersonSearchSelect()
        self._person_b.person_selected.connect(self._on_people_changed)
        selectors_layout.addWidget(self._person_b_lbl, 0, 1)
        selectors_layout.addWidget(self._person_b, 1, 1)

        self._relationship_lbl = QLabel()
        self._relationship_filter = QComboBox()
        selectors_layout.addWidget(self._relationship_lbl, 0, 2)
        selectors_layout.addWidget(self._relationship_filter, 1, 2)

        self._search_btn = QPushButton()
        self._search_btn.clicked.connect(self._search_first_page)
        selectors_layout.addWidget(self._search_btn, 1, 3)

        selectors_layout.setColumnStretch(0, 2)
        selectors_layout.setColumnStretch(1, 2)
        selectors_layout.setColumnStretch(2, 1)
        root.addWidget(selectors)

        mode_row = QHBoxLayout()
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.setSpacing(10)
        self._mode_both = QRadioButton()
        self._mode_exact = QRadioButton()
        self._mode_both.setChecked(True)
        mode_row.addWidget(self._mode_both)
        mode_row.addWidget(self._mode_exact)
        mode_row.addStretch()
        root.addLayout(mode_row)

        self._relationship_summary = QLabel()
        self._relationship_summary.setWordWrap(True)
        self._relationship_summary.setMinimumWidth(0)
        self._relationship_summary.setStyleSheet("color: #a6e3a1; font-size: 12px;")
        root.addWidget(self._relationship_summary)

        relation_row = QHBoxLayout()
        relation_row.setContentsMargins(0, 0, 0, 0)
        relation_row.setSpacing(6)
        self._relationship_add_lbl = QLabel()
        self._relationship_add = QComboBox()
        self._relationship_save_btn = QPushButton()
        self._relationship_save_btn.clicked.connect(self._save_relationship)
        relation_row.addWidget(self._relationship_add_lbl)
        relation_row.addWidget(self._relationship_add, 1)
        relation_row.addWidget(self._relationship_save_btn)
        root.addLayout(relation_row)

        root.addWidget(_hline())

        result_hdr = QHBoxLayout()
        self._result_label = QLabel()
        self._result_label.setStyleSheet("font-weight: bold; color: #888; font-size: 11px;")
        result_hdr.addWidget(self._result_label)
        result_hdr.addStretch()
        self._prev_btn = QPushButton()
        self._prev_btn.clicked.connect(self._prev_page)
        result_hdr.addWidget(self._prev_btn)
        self._next_btn = QPushButton()
        self._next_btn.clicked.connect(self._next_page)
        result_hdr.addWidget(self._next_btn)
        root.addLayout(result_hdr)

        self._results = QListWidget()
        self._results.setIconSize(QSize(_THUMB_SIZE, _THUMB_SIZE))
        self._results.itemDoubleClicked.connect(self._open_selected_item)
        self._results.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self._results, stretch=1)

        self.retranslate()

    def retranslate(self) -> None:
        self._person_a_lbl.setText(t("family_person_a"))
        self._person_b_lbl.setText(t("family_person_b"))
        self._relationship_lbl.setText(t("family_relationship_filter"))
        self._search_btn.setText(t("family_search_btn"))
        self._mode_both.setText(t("family_mode_both"))
        self._mode_exact.setText(t("family_mode_exact"))
        self._relationship_add_lbl.setText(t("family_add_relationship"))
        self._relationship_save_btn.setText(t("family_save_relationship"))
        self._prev_btn.setText(t("family_prev"))
        self._next_btn.setText(t("family_next"))
        self._person_a.retranslate()
        self._person_b.retranslate()
        self._populate_relation_combos()
        self._update_result_label()

    def refresh(self) -> None:
        with session_scope() as session:
            self._persons = session.query(Person).order_by(Person.name).all()
        self._person_a.set_persons(self._persons)
        self._person_b.set_persons(self._persons)
        self._on_people_changed()

    def _populate_relation_combos(self) -> None:
        filter_data = self._relationship_filter.currentData() or "any"
        self._relationship_filter.blockSignals(True)
        self._relationship_filter.clear()
        self._relationship_filter.addItem(t("family_filter_any"), "any")
        self._relationship_filter.addItem(t("family_filter_spouse"), "spouse")
        self._relationship_filter.addItem(t("family_filter_parent"), "parent")
        self._relationship_filter.addItem(t("family_filter_child"), "child")
        self._relationship_filter.addItem(t("family_filter_sibling"), "sibling")
        idx = self._relationship_filter.findData(filter_data)
        self._relationship_filter.setCurrentIndex(idx if idx >= 0 else 0)
        self._relationship_filter.blockSignals(False)

        add_data = self._relationship_add.currentData() or "spouse"
        self._relationship_add.clear()
        self._relationship_add.addItem(t("family_add_spouse"), "spouse")
        self._relationship_add.addItem(t("family_add_a_parent_b"), "a_parent_b")
        self._relationship_add.addItem(t("family_add_b_parent_a"), "b_parent_a")
        idx = self._relationship_add.findData(add_data)
        self._relationship_add.setCurrentIndex(idx if idx >= 0 else 0)

    def _selected_people(self) -> tuple[Optional[int], Optional[int]]:
        return self._person_a.current_person_id(), self._person_b.current_person_id()

    def _on_people_changed(self, *_args) -> None:
        person_a, person_b = self._selected_people()
        enabled = person_a is not None and person_b is not None and person_a != person_b
        self._search_btn.setEnabled(enabled)
        self._relationship_save_btn.setEnabled(enabled)
        if not enabled:
            self._relationship_summary.setText(t("family_select_two"))
            return
        with session_scope() as session:
            summary = FamilyService(session).describe_relationship(person_a, person_b)
        self._relationship_summary.setText(summary or t("family_no_relationship"))

    def _save_relationship(self) -> None:
        person_a, person_b = self._selected_people()
        if person_a is None or person_b is None or person_a == person_b:
            return
        rel = self._relationship_add.currentData()
        try:
            with session_scope() as session:
                svc = FamilyService(session)
                if rel == "spouse":
                    svc.add_spouse(person_a, person_b)
                elif rel == "a_parent_b":
                    svc.add_parent_child(person_a, person_b)
                elif rel == "b_parent_a":
                    svc.add_parent_child(person_b, person_a)
        except ValueError as exc:
            QMessageBox.warning(self, t("family_validation_title"), str(exc))
            return
        self._on_people_changed()

    def _search_first_page(self) -> None:
        self._offset = 0
        self._run_search()

    def _prev_page(self) -> None:
        self._offset = max(0, self._offset - _PAGE_SIZE)
        self._run_search()

    def _next_page(self) -> None:
        if self._offset + _PAGE_SIZE < self._total:
            self._offset += _PAGE_SIZE
            self._run_search()

    def _run_search(self) -> None:
        person_a, person_b = self._selected_people()
        self._results.clear()
        if person_a is None or person_b is None or person_a == person_b:
            self._total = 0
            self._update_result_label()
            return

        mode = "exact" if self._mode_exact.isChecked() else "both"
        relation = self._relationship_filter.currentData() or "any"
        with session_scope() as session:
            results, total = FamilyService(session).search_images_for_people(
                person_a,
                person_b,
                mode=mode,
                relationship_filter=relation,
                limit=_PAGE_SIZE,
                offset=self._offset,
            )
        self._total = total
        for result in results:
            self._add_result_item(result)
        if not results:
            item = QListWidgetItem(t("family_no_results"))
            item.setFlags(Qt.NoItemFlags)
            self._results.addItem(item)
        self._update_result_label()

    def _add_result_item(self, result: FamilyImageResult) -> None:
        label = t(
            "family_result_item",
            name=result.filename,
            persons=result.person_count,
            faces=result.face_count,
        )
        item = QListWidgetItem(label)
        item.setData(_ROLE_IMAGE_ID, result.image_id)
        item.setToolTip(result.file_path)
        pixmap = self._load_thumb(result.file_path)
        if pixmap is not None:
            item.setIcon(QIcon(pixmap))
        self._results.addItem(item)

    def _load_thumb(self, file_path: str) -> Optional[QPixmap]:
        if not Path(file_path).exists():
            return None
        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            return None
        return pixmap.scaled(_THUMB_SIZE, _THUMB_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def _open_selected_item(self, item: QListWidgetItem) -> None:
        image_id = item.data(_ROLE_IMAGE_ID)
        if image_id is not None:
            self.image_open_requested.emit(int(image_id))

    def _update_result_label(self) -> None:
        if self._total <= 0:
            self._result_label.setText(t("family_results_empty"))
        else:
            start = self._offset + 1
            end = min(self._offset + _PAGE_SIZE, self._total)
            self._result_label.setText(
                t("family_results_range", start=start, end=end, total=self._total)
            )
        self._prev_btn.setEnabled(self._offset > 0)
        self._next_btn.setEnabled(self._offset + _PAGE_SIZE < self._total)
