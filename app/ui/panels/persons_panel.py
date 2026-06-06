"""Standalone Persons maintenance page.

A self-contained tab — modelled on :class:`app.ui.panels.locations_panel.LocationsPanel`
— that lists every PERSONS record in a sortable table, lets the user edit a
selected person's structured data, swap their thumbnail, and browse the faces
and images linked to them.  All business logic lives in
:class:`app.services.person_service.PersonService`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, QSize, QThreadPool, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.db.database import session_scope
from app.db.models import Person
from app.services.person_service import (
    PersonFaceCrop,
    PersonFilters,
    PersonService,
    PersonSummary,
)
from app.ui.dialogs.person_info_dialog import PersonInfoDialog
from app.ui.i18n import t
from app.ui.widgets.place_gallery_widget import PlaceGalleryWidget
from app.workers.thumbnail_worker import ThumbnailRunnable

log = logging.getLogger(__name__)

_ROLE_ID = Qt.UserRole
_ROLE_PROTECTED = Qt.UserRole + 1
_TABLE_THUMB = 40

# Column layout for the persons table.
COL_ID = 0
COL_THUMB = 1
COL_NAME = 2
COL_FAMILY_CODE = 3
COL_LAST_NAME = 4
COL_FIRST_NAME = 5
COL_SECOND_NAME = 6
COL_NICKNAME = 7
COL_MARRIED_NAME = 8
COL_GENDER = 9
COL_BIRTH_PLACE = 10
COL_BIRTH_DATE = 11
COL_DEATH_PLACE = 12
COL_DEATH_DATE = 13
COL_NOTES = 14
COL_AUTO_NAMED = 15
COL_PROTECTED = 16
_COL_COUNT = 17


class _NumericItem(QTableWidgetItem):
    """Table item that sorts numerically by its stored integer value."""

    def __init__(self, value: int) -> None:
        super().__init__(str(value))
        self._value = value
        self.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

    def __lt__(self, other: "QTableWidgetItem") -> bool:  # noqa: D401
        if isinstance(other, _NumericItem):
            return self._value < other._value
        return super().__lt__(other)


class _FaceThumb(QLabel):
    """Clickable face-crop thumbnail used in the thumbnail picker dialog."""

    clicked = Signal(int)  # face_id

    def __init__(self, face_id: int, crop_path: Optional[str], size: int = 96) -> None:
        super().__init__()
        self._face_id = face_id
        self.setFixedSize(size, size)
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(
            "QLabel { border: 1px solid #45475A; border-radius: 4px; "
            "background: #181825; color: #6C7086; }"
            "QLabel:hover { border: 2px solid #89B4FA; }"
        )
        pix = None
        if crop_path and Path(crop_path).exists():
            from app.utils.image_utils import load_pixmap_exif
            loaded = load_pixmap_exif(crop_path)
            if not loaded.isNull():
                pix = loaded.scaled(
                    size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
        if pix is not None:
            self.setPixmap(pix)
        else:
            self.setText("✗")  # missing / unloadable crop → placeholder

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self._face_id)
        super().mousePressEvent(event)


class ThumbnailPickerDialog(QDialog):
    """Grid of a person's face crops; clicking one selects it as thumbnail."""

    def __init__(
        self,
        person_name: str,
        crops: List[PersonFaceCrop],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("persons_thumb_picker_title", name=person_name))
        self.resize(560, 480)
        self._selected_face_id: Optional[int] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        hint = QLabel(t("persons_thumb_picker_hint"))
        hint.setStyleSheet("color: #888;")
        layout.addWidget(hint)

        usable = [c for c in crops if c.crop_path]
        if not usable:
            empty = QLabel(t("persons_thumb_no_faces"))
            empty.setStyleSheet("color: #888; font-style: italic;")
            empty.setAlignment(Qt.AlignCenter)
            layout.addWidget(empty, stretch=1)
            return

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        grid.setSpacing(6)
        cols = 5
        for i, crop in enumerate(usable):
            row, col = divmod(i, cols)
            thumb = _FaceThumb(crop.face_id, crop.crop_path)
            thumb.clicked.connect(self._on_pick)
            grid.addWidget(thumb, row, col)
        scroll.setWidget(grid_widget)
        layout.addWidget(scroll, stretch=1)

    def _on_pick(self, face_id: int) -> None:
        self._selected_face_id = face_id
        self.accept()

    def selected_face_id(self) -> Optional[int]:
        return self._selected_face_id


class PersonsPanel(QWidget):
    """List, filter, edit, and maintain Person records on a dedicated tab."""

    # Emitted after a change that other tabs may want to react to.
    person_data_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._current_person_id: Optional[int] = None
        self._thumb_keys: Dict[str, int] = {}  # thumb cache key → row
        self._build_ui()
        self.refresh()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Filter row
        filters = QHBoxLayout()
        self._name_filter = QLineEdit()
        self._name_filter.returnPressed.connect(self.refresh)
        filters.addWidget(self._name_filter, 2)
        self._code_filter = QLineEdit()
        self._code_filter.returnPressed.connect(self.refresh)
        filters.addWidget(self._code_filter, 1)
        self._filter_btn = QPushButton()
        self._filter_btn.clicked.connect(self.refresh)
        filters.addWidget(self._filter_btn)
        self._count_lbl = QLabel()
        self._count_lbl.setStyleSheet("color: #888;")
        filters.addWidget(self._count_lbl)
        filters.addStretch()
        root.addLayout(filters)

        splitter = QSplitter(Qt.Horizontal)

        # --- Table ---
        self._table = QTableWidget(0, _COL_COUNT)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        self._table.setSortingEnabled(True)
        self._table.setIconSize(QSize(_TABLE_THUMB, _TABLE_THUMB))
        self._table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self._table.verticalHeader().setDefaultSectionSize(_TABLE_THUMB + 8)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        self._table.itemChanged.connect(self._on_item_changed)
        self._table.itemDoubleClicked.connect(self._on_item_double_clicked)
        splitter.addWidget(self._table)

        # --- Detail pane ---
        detail_inner = QWidget()
        detail_layout = QVBoxLayout(detail_inner)
        detail_layout.setContentsMargins(8, 4, 4, 4)
        detail_layout.setSpacing(8)

        self._thumb = QLabel()
        self._thumb.setFixedSize(180, 180)
        self._thumb.setAlignment(Qt.AlignCenter)
        self._thumb.setStyleSheet("background:#181825;color:#888;border-radius:4px;")
        detail_layout.addWidget(self._thumb, alignment=Qt.AlignHCenter)

        form = QFormLayout()
        form.setSpacing(4)
        self._name_lbl = QLabel()
        self._name_lbl.setWordWrap(True)
        self._name_lbl.setStyleSheet("font-size: 14px; font-weight: bold;")
        self._face_count_lbl = QLabel()
        self._image_count_lbl = QLabel()
        form.addRow(self._name_lbl)
        self._face_count_caption = QLabel()
        self._image_count_caption = QLabel()
        form.addRow(self._face_count_caption, self._face_count_lbl)
        form.addRow(self._image_count_caption, self._image_count_lbl)
        detail_layout.addLayout(form)

        # Action buttons
        actions = QHBoxLayout()
        self._edit_btn = QPushButton()
        self._edit_btn.setEnabled(False)
        self._edit_btn.clicked.connect(self._on_edit)
        actions.addWidget(self._edit_btn)
        self._rename_btn = QPushButton()
        self._rename_btn.setEnabled(False)
        self._rename_btn.clicked.connect(self._on_rename)
        actions.addWidget(self._rename_btn)
        self._thumb_btn = QPushButton()
        self._thumb_btn.setEnabled(False)
        self._thumb_btn.clicked.connect(self._on_change_thumbnail)
        actions.addWidget(self._thumb_btn)
        detail_layout.addLayout(actions)

        # Faces gallery (crop paths, lazy-loaded)
        self._faces_hdr = QLabel()
        detail_layout.addWidget(self._faces_hdr)
        self._faces_gallery = PlaceGalleryWidget()
        detail_layout.addWidget(self._faces_gallery)

        # Related images gallery (lazy-loaded)
        self._images_hdr = QLabel()
        detail_layout.addWidget(self._images_hdr)
        self._images_gallery = PlaceGalleryWidget()
        detail_layout.addWidget(self._images_gallery)

        self._refresh_btn = QPushButton()
        self._refresh_btn.clicked.connect(self.refresh)
        detail_layout.addWidget(self._refresh_btn)

        detail_layout.addStretch()

        detail_scroll = QScrollArea()
        detail_scroll.setWidget(detail_inner)
        detail_scroll.setWidgetResizable(True)
        detail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        detail_scroll.setFrameShape(QScrollArea.NoFrame)

        splitter.addWidget(detail_scroll)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)

        self.retranslate()

    def retranslate(self) -> None:
        self._name_filter.setPlaceholderText(t("persons_filter_name"))
        self._code_filter.setPlaceholderText(t("persons_filter_code"))
        self._filter_btn.setText(t("persons_filter_apply"))
        self._table.setHorizontalHeaderLabels([
            t("persons_col_id"),
            t("persons_col_thumbnail"),
            t("persons_col_name"),
            t("persons_col_family_code"),
            t("persons_col_last_name"),
            t("persons_col_first_name"),
            t("persons_col_second_name"),
            t("persons_col_nickname"),
            t("persons_col_married_name"),
            t("persons_col_gender"),
            t("persons_col_birth_place"),
            t("persons_col_birth_date"),
            t("persons_col_death_place"),
            t("persons_col_death_date"),
            t("persons_col_notes"),
            t("persons_col_auto_named"),
            t("persons_col_protected"),
        ])
        self._edit_btn.setText(t("persons_edit_btn"))
        self._rename_btn.setText(t("persons_rename_btn"))
        self._thumb_btn.setText(t("persons_thumbnail_btn"))
        self._refresh_btn.setText(t("persons_refresh_btn"))
        self._faces_hdr.setText(t("persons_detail_faces"))
        self._images_hdr.setText(t("persons_detail_images"))
        self._face_count_caption.setText(t("persons_face_count"))
        self._image_count_caption.setText(t("persons_image_count"))
        if self._current_person_id is None:
            self._name_lbl.setText(t("persons_no_selection"))

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        filters = PersonFilters(
            name=self._name_filter.text(),
            family_code=self._code_filter.text(),
        )
        with session_scope() as session:
            summaries = PersonService(session).list_persons(filters)

        # Repopulate without firing itemChanged / re-sorting mid-build.
        self._table.blockSignals(True)
        self._table.setSortingEnabled(False)
        self._table.setRowCount(0)
        self._thumb_keys.clear()
        for row, s in enumerate(summaries):
            self._table.insertRow(row)
            self._populate_row(row, s)
        self._table.setSortingEnabled(True)
        self._table.blockSignals(False)

        self._count_lbl.setText(t("persons_count", n=len(summaries)))
        self._start_thumbnail_loads(summaries)

        # Preserve selection if the person still exists, else clear detail.
        if self._current_person_id is not None and self._select_person_row(
            self._current_person_id
        ):
            self._load_detail(self._current_person_id)
        else:
            self._current_person_id = None
            self._clear_detail()

    def _populate_row(self, row: int, s: PersonSummary) -> None:
        id_item = _NumericItem(s.person_id)
        id_item.setData(_ROLE_ID, s.person_id)
        id_item.setData(_ROLE_PROTECTED, s.is_protected)
        self._table.setItem(row, COL_ID, id_item)

        thumb_item = QTableWidgetItem()
        thumb_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        thumb_item.setData(_ROLE_ID, s.person_id)
        self._table.setItem(row, COL_THUMB, thumb_item)

        gender = ""
        if s.gender == "male":
            gender = t("gender_male")
        elif s.gender == "female":
            gender = t("gender_female")

        text_cols = {
            COL_NAME: s.name,
            COL_FAMILY_CODE: s.family_code or "",
            COL_LAST_NAME: s.last_name or "",
            COL_FIRST_NAME: s.first_name or "",
            COL_SECOND_NAME: s.second_name or "",
            COL_NICKNAME: s.nickname or "",
            COL_MARRIED_NAME: s.married_name or "",
            COL_GENDER: gender,
            COL_BIRTH_PLACE: s.birth_place or "",
            COL_BIRTH_DATE: s.birth_date or "",
            COL_DEATH_PLACE: s.death_place or "",
            COL_DEATH_DATE: s.death_date or "",
            COL_NOTES: s.notes or "",
            COL_AUTO_NAMED: t("persons_yes") if s.is_auto_named else t("persons_no"),
            COL_PROTECTED: t("persons_yes") if s.is_protected else t("persons_no"),
        }
        for col, value in text_cols.items():
            item = QTableWidgetItem(value)
            item.setData(_ROLE_ID, s.person_id)
            if col == COL_NAME and not s.is_protected:
                item.setFlags(
                    Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable
                )
                item.setToolTip(t("persons_name_edit_tip", default="Dupla kattintás az átnevezéshez"))
            else:
                item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            self._table.setItem(row, col, item)

    def _start_thumbnail_loads(self, summaries: List[PersonSummary]) -> None:
        """Kick off async loads for the small table thumbnails."""
        for row, s in enumerate(summaries):
            path = s.thumbnail_path
            if not path or not Path(path).exists():
                continue
            key = f"persontbl:{s.person_id}:{path}"
            self._thumb_keys[key] = s.person_id
            worker = ThumbnailRunnable(path, key, size=_TABLE_THUMB)
            worker.signals.ready.connect(self._on_table_thumb_ready)
            QThreadPool.globalInstance().start(worker)

    def _on_table_thumb_ready(self, key: str, img: QImage) -> None:
        person_id = self._thumb_keys.get(key)
        if person_id is None:
            return
        icon = QIcon(QPixmap.fromImage(img))
        for row in range(self._table.rowCount()):
            item = self._table.item(row, COL_THUMB)
            if item is not None and item.data(_ROLE_ID) == person_id:
                item.setIcon(icon)
                break

    def _select_person_row(self, person_id: int) -> bool:
        for row in range(self._table.rowCount()):
            item = self._table.item(row, COL_ID)
            if item is not None and item.data(_ROLE_ID) == person_id:
                self._table.blockSignals(True)
                self._table.selectRow(row)
                self._table.blockSignals(False)
                return True
        return False

    # ------------------------------------------------------------------
    # Selection / detail
    # ------------------------------------------------------------------

    def _on_selection_changed(self) -> None:
        items = self._table.selectedItems()
        if not items:
            self._current_person_id = None
            self._clear_detail()
            return
        person_id = items[0].data(_ROLE_ID)
        if person_id is not None:
            self._load_detail(int(person_id))

    def _load_detail(self, person_id: int) -> None:
        self._current_person_id = person_id
        with session_scope() as session:
            svc = PersonService(session)
            person = session.get(Person, person_id)
            if person is None:
                self._clear_detail()
                return
            name = person.name
            thumb = person.thumbnail_path
            is_protected = bool(person.is_protected)
            crops = svc.list_face_crops(person_id)
            image_paths = svc.list_images_for_person(person_id)

        crop_paths = [c.crop_path for c in crops if c.crop_path and Path(c.crop_path).exists()]

        # Fall back to the best available crop when no explicit thumbnail is set.
        if not thumb or not Path(thumb).exists():
            thumb = crop_paths[0] if crop_paths else None

        self._name_lbl.setText(name)
        self._face_count_lbl.setText(str(len(crops)))
        self._image_count_lbl.setText(str(len(image_paths)))
        self._set_thumbnail(thumb)
        self._faces_gallery.set_images(crop_paths)
        self._images_gallery.set_images(image_paths)

        self._edit_btn.setEnabled(True)
        self._rename_btn.setEnabled(not is_protected)
        self._thumb_btn.setEnabled(bool(crop_paths))

    def _clear_detail(self) -> None:
        self._name_lbl.setText(t("persons_no_selection"))
        self._face_count_lbl.clear()
        self._image_count_lbl.clear()
        self._thumb.setPixmap(QPixmap())
        self._thumb.setText(t("persons_no_thumbnail"))
        self._faces_gallery.clear()
        self._images_gallery.clear()
        self._edit_btn.setEnabled(False)
        self._rename_btn.setEnabled(False)
        self._thumb_btn.setEnabled(False)

    def _set_thumbnail(self, path: Optional[str]) -> None:
        self._thumb.setPixmap(QPixmap())
        if not path or not Path(path).exists():
            self._thumb.setText(t("persons_no_thumbnail"))
            return
        from app.utils.image_utils import load_pixmap_exif
        pixmap = load_pixmap_exif(path)
        if pixmap.isNull():
            self._thumb.setText(t("persons_no_thumbnail"))
            return
        self._thumb.setText("")
        self._thumb.setPixmap(
            pixmap.scaled(self._thumb.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """Handle inline name edits (column COL_NAME)."""
        if item.column() != COL_NAME:
            return
        person_id = item.data(_ROLE_ID)
        if person_id is None:
            return
        new_name = item.text().strip()
        if not new_name:
            QMessageBox.warning(self, t("empty_name_title"), t("empty_name_msg"))
            self._restore_name_cell(item, int(person_id))
            return
        try:
            with session_scope() as session:
                PersonService(session).rename_person(int(person_id), new_name)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self,
                t("persons_rename_title"),
                t("persons_save_error", error=str(exc)),
            )
            self._restore_name_cell(item, int(person_id))
            return
        self.person_data_changed.emit()
        if self._current_person_id == person_id:
            self._load_detail(int(person_id))

    def _restore_name_cell(self, item: QTableWidgetItem, person_id: int) -> None:
        with session_scope() as session:
            person = session.get(Person, person_id)
            name = person.name if person is not None else ""
        self._table.blockSignals(True)
        item.setText(name)
        self._table.blockSignals(False)

    def _on_item_double_clicked(self, item: QTableWidgetItem) -> None:
        # Double-clicking any read-only cell (other than the editable name)
        # opens the structured-data editor for convenience.
        if item.column() == COL_NAME:
            return
        person_id = item.data(_ROLE_ID)
        if person_id is not None:
            self._open_edit_dialog(int(person_id))

    def _on_edit(self) -> None:
        if self._current_person_id is not None:
            self._open_edit_dialog(self._current_person_id)

    def _open_edit_dialog(self, person_id: int) -> None:
        with session_scope() as session:
            person = session.get(Person, person_id)
            if person is None:
                return
            dlg = PersonInfoDialog(person, parent=self)
        if dlg.exec() != PersonInfoDialog.Accepted:
            return
        try:
            with session_scope() as session:
                PersonService(session).update_person(
                    person_id,
                    gender=dlg.gender(),
                    family_code=dlg.family_code(),
                    last_name=dlg.last_name(),
                    first_name=dlg.first_name(),
                    second_name=dlg.second_name(),
                    nickname=dlg.nickname(),
                    married_name=dlg.married_name(),
                    birth_place=dlg.birth_place(),
                    birth_date=dlg.birth_date(),
                    death_place=dlg.death_place(),
                    death_date=dlg.death_date(),
                    notes=dlg.notes(),
                )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self, t("persons_edit_btn"), t("persons_save_error", error=str(exc))
            )
            return
        self.person_data_changed.emit()
        self.refresh()

    def _on_rename(self) -> None:
        if self._current_person_id is None:
            return
        with session_scope() as session:
            person = session.get(Person, self._current_person_id)
            if person is None:
                return
            current_name = person.name
            is_protected = bool(person.is_protected)
        if is_protected:
            QMessageBox.warning(self, t("persons_rename_title"), t("persons_protected_msg"))
            return
        new_name, ok = QInputDialog.getText(
            self,
            t("persons_rename_title"),
            t("persons_rename_prompt"),
            QLineEdit.Normal,
            current_name,
        )
        if not ok:
            return
        new_name = new_name.strip()
        if not new_name:
            QMessageBox.warning(self, t("empty_name_title"), t("empty_name_msg"))
            return
        try:
            with session_scope() as session:
                PersonService(session).rename_person(self._current_person_id, new_name)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self, t("persons_rename_title"), t("persons_save_error", error=str(exc))
            )
            return
        self.person_data_changed.emit()
        self.refresh()

    # ------------------------------------------------------------------
    # Thumbnail change
    # ------------------------------------------------------------------

    def _on_change_thumbnail(self) -> None:
        if self._current_person_id is None:
            return
        with session_scope() as session:
            svc = PersonService(session)
            person = session.get(Person, self._current_person_id)
            if person is None:
                return
            name = person.name
            crops = svc.list_face_crops(self._current_person_id)

        dlg = ThumbnailPickerDialog(name, crops, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        face_id = dlg.selected_face_id()
        if face_id is None:
            return
        try:
            with session_scope() as session:
                PersonService(session).set_thumbnail_from_face(
                    self._current_person_id, face_id
                )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self, t("persons_thumbnail_btn"), t("persons_save_error", error=str(exc))
            )
            return
        self.person_data_changed.emit()
        self.refresh()
