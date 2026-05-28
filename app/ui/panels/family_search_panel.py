"""Family/person based image search panel."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.db.database import session_scope
from app.services.family_service import (
    FamilyImageResult,
    FamilyImageSearchCriteria,
    FamilyService,
)
from app.ui.i18n import t

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
    """Search images by comma-separated person names and optional metadata."""

    image_open_requested = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._offset = 0
        self._total = 0
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        search_row = QHBoxLayout()
        search_row.setContentsMargins(0, 0, 0, 0)
        search_row.setSpacing(6)

        self._names_edit = QLineEdit()
        self._names_edit.returnPressed.connect(self._search_first_page)
        search_row.addWidget(self._names_edit, 1)

        self._allow_others_btn = QPushButton()
        self._allow_others_btn.setCheckable(True)
        self._allow_others_btn.setChecked(True)
        self._allow_others_btn.toggled.connect(self._update_allow_others_text)
        search_row.addWidget(self._allow_others_btn)

        self._search_btn = QPushButton()
        self._search_btn.clicked.connect(self._search_first_page)
        search_row.addWidget(self._search_btn)
        root.addLayout(search_row)

        self._details_toggle = QToolButton()
        self._details_toggle.setCheckable(True)
        self._details_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._details_toggle.toggled.connect(self._on_details_toggled)
        root.addWidget(self._details_toggle, alignment=Qt.AlignLeft)

        self._details = QWidget()
        details_layout = QFormLayout(self._details)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setHorizontalSpacing(8)
        details_layout.setVerticalSpacing(6)
        details_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self._relationship_filter = QComboBox()
        details_layout.addRow(t("family_relationship_filter"), self._relationship_filter)

        self._person_text = QLineEdit()
        details_layout.addRow(t("family_detail_person_text"), self._person_text)

        self._gender = QComboBox()
        details_layout.addRow(t("gender"), self._gender)

        self._family_code = QLineEdit()
        details_layout.addRow(t("family_code"), self._family_code)

        self._image_text = QLineEdit()
        details_layout.addRow(t("family_detail_image_text"), self._image_text)

        self._photo_date = QLineEdit()
        details_layout.addRow(t("family_detail_photo_date"), self._photo_date)

        self._place_text = QLineEdit()
        details_layout.addRow(t("family_detail_place_text"), self._place_text)

        self._details.setVisible(False)
        root.addWidget(self._details)

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
        self._names_edit.setPlaceholderText(t("family_names_placeholder"))
        self._search_btn.setText(t("family_search_btn"))
        self._details_toggle.setText(t("family_details_closed"))
        self._person_text.setPlaceholderText(t("family_detail_person_placeholder"))
        self._family_code.setPlaceholderText(t("example_family_code"))
        self._image_text.setPlaceholderText(t("family_detail_image_placeholder"))
        self._photo_date.setPlaceholderText(t("ibp_date_placeholder"))
        self._place_text.setPlaceholderText(t("family_detail_place_placeholder"))
        self._prev_btn.setText(t("family_prev"))
        self._next_btn.setText(t("family_next"))
        self._populate_relation_filter()
        self._populate_gender_filter()
        self._update_allow_others_text()
        self._update_result_label()

    def refresh(self) -> None:
        pass

    def _populate_relation_filter(self) -> None:
        data = self._relationship_filter.currentData() or "any"
        self._relationship_filter.blockSignals(True)
        self._relationship_filter.clear()
        self._relationship_filter.addItem(t("family_filter_any"), "any")
        self._relationship_filter.addItem(t("family_filter_spouse"), "spouse")
        self._relationship_filter.addItem(t("family_filter_parent"), "parent")
        self._relationship_filter.addItem(t("family_filter_child"), "child")
        self._relationship_filter.addItem(t("family_filter_sibling"), "sibling")
        idx = self._relationship_filter.findData(data)
        self._relationship_filter.setCurrentIndex(idx if idx >= 0 else 0)
        self._relationship_filter.blockSignals(False)

    def _populate_gender_filter(self) -> None:
        data = self._gender.currentData()
        self._gender.blockSignals(True)
        self._gender.clear()
        self._gender.addItem(t("family_gender_any"), None)
        self._gender.addItem(t("gender_male"), "male")
        self._gender.addItem(t("gender_female"), "female")
        idx = self._gender.findData(data)
        self._gender.setCurrentIndex(idx if idx >= 0 else 0)
        self._gender.blockSignals(False)

    def _on_details_toggled(self, expanded: bool) -> None:
        self._details.setVisible(expanded)
        self._details_toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._details_toggle.setText(
            t("family_details_open") if expanded else t("family_details_closed")
        )

    def _update_allow_others_text(self) -> None:
        self._allow_others_btn.setText(
            t("family_allow_others_yes")
            if self._allow_others_btn.isChecked()
            else t("family_allow_others_no")
        )

    def _criteria(self) -> FamilyImageSearchCriteria:
        terms = tuple(
            part.strip()
            for part in self._names_edit.text().split(",")
            if part.strip()
        )
        return FamilyImageSearchCriteria(
            name_terms=terms,
            allow_other_people=self._allow_others_btn.isChecked(),
            person_text=self._person_text.text().strip(),
            gender=self._gender.currentData(),
            family_code=self._family_code.text().strip(),
            image_text=self._image_text.text().strip(),
            photo_date=self._photo_date.text().strip(),
            place_text=self._place_text.text().strip(),
            relationship_filter=self._relationship_filter.currentData() or "any",
        )

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
        self._results.clear()
        criteria = self._criteria()
        with session_scope() as session:
            results, total = FamilyService(session).search_images_by_criteria(
                criteria,
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
        return pixmap.scaled(
            _THUMB_SIZE,
            _THUMB_SIZE,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

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
