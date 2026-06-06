"""Family/person based image search panel — redesigned with UniversalSearchBar."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
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
from app.ui.widgets.universal_search_bar import (
    TOKEN_ANY,
    TOKEN_DATE,
    TOKEN_FAMILY_CODE,
    TOKEN_IMAGE,
    TOKEN_NICKNAME,
    TOKEN_PERSON,
    TOKEN_PLACE,
    UniversalSearchBar,
)

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
    """Search images by universal query and optional detailed metadata filters."""

    image_open_requested = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._offset = 0
        self._total = 0
        self._build_ui()

    # ──────────────────────────────────────────────────────────────────
    # Build
    # ──────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Universal search bar ──────────────────────────────────────
        self._universal_bar = UniversalSearchBar(
            suggest_fn=self._suggest,
            parent=self,
        )
        self._universal_bar.search_requested.connect(self._search_first_page)
        root.addWidget(self._universal_bar)

        # ── Action row: checkbox + search button ──────────────────────
        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(8)

        self._only_person_cb = QCheckBox()
        self._only_person_cb.setChecked(False)
        action_row.addWidget(self._only_person_cb)
        action_row.addStretch(1)

        self._search_btn = QPushButton()
        self._search_btn.clicked.connect(self._search_first_page)
        action_row.addWidget(self._search_btn)
        root.addLayout(action_row)

        # ── Details toggle ────────────────────────────────────────────
        self._details_toggle = QToolButton()
        self._details_toggle.setCheckable(True)
        self._details_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._details_toggle.toggled.connect(self._on_details_toggled)
        root.addWidget(self._details_toggle, alignment=Qt.AlignLeft)

        # ── Details section ───────────────────────────────────────────
        self._details = QWidget()
        details_layout = QFormLayout(self._details)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setHorizontalSpacing(8)
        details_layout.setVerticalSpacing(6)
        details_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self._relationship_filter = QComboBox()
        details_layout.addRow(t("family_relationship_filter"), self._relationship_filter)

        # Names field — moved here from the old top position
        self._names_edit = QLineEdit()
        details_layout.addRow(t("search_names_detail_label"), self._names_edit)

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

        # ── Result header ─────────────────────────────────────────────
        result_hdr = QHBoxLayout()
        self._result_label = QLabel()
        self._result_label.setStyleSheet(
            "font-weight: bold; color: #888; font-size: 11px;"
        )
        result_hdr.addWidget(self._result_label)
        result_hdr.addStretch()
        self._prev_btn = QPushButton()
        self._prev_btn.clicked.connect(self._prev_page)
        result_hdr.addWidget(self._prev_btn)
        self._next_btn = QPushButton()
        self._next_btn.clicked.connect(self._next_page)
        result_hdr.addWidget(self._next_btn)
        root.addLayout(result_hdr)

        # ── Results list ──────────────────────────────────────────────
        self._results = QListWidget()
        self._results.setIconSize(QSize(_THUMB_SIZE, _THUMB_SIZE))
        self._results.itemDoubleClicked.connect(self._open_selected_item)
        self._results.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self._results, stretch=1)

        self.retranslate()

    # ──────────────────────────────────────────────────────────────────
    # Translations
    # ──────────────────────────────────────────────────────────────────

    def retranslate(self) -> None:
        self._universal_bar.set_placeholder(t("search_universal_placeholder"))
        self._universal_bar.retranslate()
        self._only_person_cb.setText(t("search_only_person_cb"))
        self._search_btn.setText(t("family_search_btn"))
        self._details_toggle.setText(t("family_details_closed"))
        self._names_edit.setPlaceholderText(t("search_names_detail_placeholder"))
        self._person_text.setPlaceholderText(t("family_detail_person_placeholder"))
        self._family_code.setPlaceholderText(t("example_family_code"))
        self._image_text.setPlaceholderText(t("family_detail_image_placeholder"))
        self._photo_date.setPlaceholderText(t("ibp_date_placeholder"))
        self._place_text.setPlaceholderText(t("family_detail_place_placeholder"))
        self._prev_btn.setText(t("family_prev"))
        self._next_btn.setText(t("family_next"))
        self._populate_relation_filter()
        self._populate_gender_filter()
        self._update_result_label()

    def refresh(self) -> None:
        pass

    # ──────────────────────────────────────────────────────────────────
    # Combo population
    # ──────────────────────────────────────────────────────────────────

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

    # ──────────────────────────────────────────────────────────────────
    # Details toggle
    # ──────────────────────────────────────────────────────────────────

    def _on_details_toggled(self, expanded: bool) -> None:
        self._details.setVisible(expanded)
        self._details_toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._details_toggle.setText(
            t("family_details_open") if expanded else t("family_details_closed")
        )

    # ──────────────────────────────────────────────────────────────────
    # Autocomplete suggestion provider
    # ──────────────────────────────────────────────────────────────────

    def _suggest(self, query: str) -> list[tuple[str, str, str]]:
        try:
            with session_scope() as session:
                return FamilyService(session).get_search_suggestions(query)
        except Exception:
            log.exception("FamilySearchPanel: autocomplete error")
            return []

    # ──────────────────────────────────────────────────────────────────
    # Criteria building
    # ──────────────────────────────────────────────────────────────────

    def _criteria(self) -> FamilyImageSearchCriteria:
        tokens = self._universal_bar.get_tokens()

        name_terms_bar: list[str] = []
        place_text = self._place_text.text().strip()
        photo_date = self._photo_date.text().strip()
        image_text = self._image_text.text().strip()
        any_terms: list[str] = []

        for tok in tokens:
            if tok.token_type in (TOKEN_PERSON, TOKEN_NICKNAME, TOKEN_FAMILY_CODE):
                name_terms_bar.append(tok.value)
            elif tok.token_type == TOKEN_PLACE:
                if not place_text:
                    place_text = tok.value
                else:
                    any_terms.append(tok.value)
            elif tok.token_type == TOKEN_DATE:
                if not photo_date:
                    photo_date = tok.value
                else:
                    any_terms.append(tok.value)
            elif tok.token_type == TOKEN_IMAGE:
                if not image_text:
                    image_text = tok.value
                else:
                    any_terms.append(tok.value)
            else:
                any_terms.append(tok.value)

        detail_names = tuple(
            p.strip()
            for p in self._names_edit.text().split(",")
            if p.strip()
        )
        all_name_terms = tuple(name_terms_bar) + detail_names

        log.debug(
            "FamilySearchPanel criteria: name_terms=%r any_terms=%r place=%r date=%r image=%r",
            all_name_terms,
            any_terms,
            place_text,
            photo_date,
            image_text,
        )
        return FamilyImageSearchCriteria(
            name_terms=all_name_terms,
            allow_other_people=not self._only_person_cb.isChecked(),
            person_text=self._person_text.text().strip(),
            gender=self._gender.currentData(),
            family_code=self._family_code.text().strip(),
            image_text=image_text,
            photo_date=photo_date,
            place_text=place_text,
            relationship_filter=self._relationship_filter.currentData() or "any",
            any_terms=tuple(any_terms),
        )

    # ──────────────────────────────────────────────────────────────────
    # Search execution
    # ──────────────────────────────────────────────────────────────────

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
        from app.utils.image_utils import load_pixmap_exif
        pixmap = load_pixmap_exif(file_path)
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
