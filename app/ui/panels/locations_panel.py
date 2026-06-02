"""Places / Locations panel."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.db.database import session_scope
from app.db.models import Place
from app.services.place_service import (
    ANONYMOUS_GPS_PLACE_NAME,
    PlaceFilters,
    PlaceService,
)
from app.ui.dialogs.place_merge_dialog import PlaceMergeDialog
from app.ui.i18n import t

_ROLE_ID = Qt.UserRole


class LocationsPanel(QWidget):
    """List, filter, inspect, and merge reusable places."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._current_place_id: Optional[int] = None
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        filters = QHBoxLayout()
        self._name_filter = QLineEdit()
        self._name_filter.returnPressed.connect(self.refresh)
        filters.addWidget(self._name_filter, 2)
        self._person_filter = QLineEdit()
        self._person_filter.returnPressed.connect(self.refresh)
        filters.addWidget(self._person_filter, 2)
        self._date_from = QLineEdit()
        self._date_from.returnPressed.connect(self.refresh)
        filters.addWidget(self._date_from, 1)
        self._date_to = QLineEdit()
        self._date_to.returnPressed.connect(self.refresh)
        filters.addWidget(self._date_to, 1)
        self._min_images = QSpinBox()
        self._min_images.setRange(0, 1_000_000)
        filters.addWidget(self._min_images)
        self._coord_filter = QComboBox()
        filters.addWidget(self._coord_filter)
        self._anon_only = QCheckBox()
        filters.addWidget(self._anon_only)
        self._filter_btn = QPushButton()
        self._filter_btn.clicked.connect(self.refresh)
        filters.addWidget(self._filter_btn)
        root.addLayout(filters)

        splitter = QSplitter(Qt.Horizontal)
        self._table = QTableWidget(0, 5)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.ExtendedSelection)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        splitter.addWidget(self._table)

        detail = QWidget()
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(8, 0, 0, 0)
        self._thumb = QLabel()
        self._thumb.setFixedSize(220, 150)
        self._thumb.setAlignment(Qt.AlignCenter)
        self._thumb.setStyleSheet("background: #181825; color: #888;")
        detail_layout.addWidget(self._thumb)

        form = QFormLayout()
        self._name = QLabel()
        self._coords = QLabel()
        self._image_count = QLabel()
        form.addRow(t("places_name"), self._name)
        form.addRow(t("places_coords"), self._coords)
        form.addRow(t("places_image_count"), self._image_count)
        detail_layout.addLayout(form)

        detail_layout.addWidget(QLabel(t("places_images")))
        self._images = QListWidget()
        detail_layout.addWidget(self._images, 2)

        detail_layout.addWidget(QLabel(t("places_persons")))
        self._persons = QListWidget()
        detail_layout.addWidget(self._persons, 1)

        actions = QHBoxLayout()
        self._merge_btn = QPushButton()
        self._merge_btn.clicked.connect(self._merge_selected)
        actions.addWidget(self._merge_btn)
        self._refresh_btn = QPushButton()
        self._refresh_btn.clicked.connect(self.refresh)
        actions.addWidget(self._refresh_btn)
        detail_layout.addLayout(actions)
        splitter.addWidget(detail)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)
        self.retranslate()

    def retranslate(self) -> None:
        self._name_filter.setPlaceholderText(t("places_filter_name"))
        self._person_filter.setPlaceholderText(t("places_filter_person_id"))
        self._date_from.setPlaceholderText(t("places_filter_date_from"))
        self._date_to.setPlaceholderText(t("places_filter_date_to"))
        self._min_images.setPrefix(t("places_filter_min_images"))
        self._coord_filter.clear()
        self._coord_filter.addItem(t("places_filter_coords_any"), None)
        self._coord_filter.addItem(t("places_filter_coords_yes"), True)
        self._coord_filter.addItem(t("places_filter_coords_no"), False)
        self._anon_only.setText(t("places_filter_anon"))
        self._filter_btn.setText(t("places_filter_apply"))
        self._table.setHorizontalHeaderLabels([
            t("places_name"),
            t("places_coords"),
            t("places_image_count"),
            t("places_person_count"),
            t("places_source"),
        ])
        self._merge_btn.setText(t("places_merge_btn"))
        self._refresh_btn.setText(t("places_refresh_btn"))

    def refresh(self) -> None:
        person_id = None
        person_text = self._person_filter.text().strip()
        if person_text:
            try:
                person_id = int(person_text)
            except ValueError:
                person_id = None
        filters = PlaceFilters(
            name=self._name_filter.text(),
            person_id=person_id,
            date_from=self._date_from.text(),
            date_to=self._date_to.text(),
            min_images=self._min_images.value() or None,
            has_coordinates=self._coord_filter.currentData(),
            anonymous_exif_only=self._anon_only.isChecked(),
        )
        with session_scope() as session:
            summaries = PlaceService(session).list_places(filters)

        self._table.setRowCount(0)
        for row, summary in enumerate(summaries):
            self._table.insertRow(row)
            name = summary.name
            if summary.is_anonymous and summary.source == "exif":
                name = f"{ANONYMOUS_GPS_PLACE_NAME} #{summary.place_id}"
            coords = ""
            if summary.latitude is not None and summary.longitude is not None:
                coords = f"{summary.latitude:.6f}, {summary.longitude:.6f}"
            values = [
                name,
                coords,
                str(summary.image_count),
                str(summary.person_count),
                summary.source or "",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(_ROLE_ID, summary.place_id)
                self._table.setItem(row, col, item)
        self._table.resizeColumnsToContents()

    def _on_selection_changed(self) -> None:
        rows = sorted({i.row() for i in self._table.selectedItems()})
        if not rows:
            self._current_place_id = None
            self._clear_detail()
            return
        item = self._table.item(rows[0], 0)
        place_id = item.data(_ROLE_ID) if item else None
        if place_id is not None:
            self._load_detail(int(place_id))

    def _load_detail(self, place_id: int) -> None:
        self._current_place_id = place_id
        with session_scope() as session:
            place = session.get(Place, place_id)
            if place is None:
                self._clear_detail()
                return
            images = PlaceService(session).list_images_for_place(place_id)
            persons = PlaceService(session).list_persons_for_place(place_id)
            name = place.name
            if place.is_anonymous and place.source == "exif":
                name = f"{ANONYMOUS_GPS_PLACE_NAME} #{place.id}"
            coords = (
                f"{place.latitude:.6f}, {place.longitude:.6f}"
                if place.latitude is not None and place.longitude is not None
                else t("places_no_coords")
            )
            thumb = place.thumbnail_path

        self._name.setText(name)
        self._coords.setText(coords)
        self._image_count.setText(str(len(images)))
        self._images.clear()
        for image in images:
            self._images.addItem(f"{Path(image.file_path).name}  {image.photo_date or ''}")
        self._persons.clear()
        for person in persons:
            self._persons.addItem(person.name)
        self._set_thumbnail(thumb)

    def _clear_detail(self) -> None:
        self._name.clear()
        self._coords.clear()
        self._image_count.clear()
        self._images.clear()
        self._persons.clear()
        self._thumb.setPixmap(QPixmap())
        self._thumb.setText(t("places_no_thumbnail"))

    def _set_thumbnail(self, path: Optional[str]) -> None:
        self._thumb.setPixmap(QPixmap())
        if not path or not Path(path).exists():
            self._thumb.setText(t("places_no_thumbnail"))
            return
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self._thumb.setText(t("places_no_thumbnail"))
            return
        self._thumb.setText("")
        self._thumb.setPixmap(
            pixmap.scaled(self._thumb.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _merge_selected(self) -> None:
        rows = sorted({i.row() for i in self._table.selectedItems()})
        if len(rows) < 2:
            QMessageBox.information(self, t("places_merge_title"), t("places_merge_need_two"))
            return
        ids = []
        for row in rows:
            item = self._table.item(row, 0)
            if item is not None:
                ids.append(int(item.data(_ROLE_ID)))

        # Prefer a named (non-anonymous) place as the merge target so that
        # the named place's ID survives and the anonymous one is absorbed.
        target_id = self._current_place_id or ids[0]
        with session_scope() as session:
            places = session.query(Place).filter(Place.id.in_(ids)).order_by(Place.id).all()
            for place in places:
                _ = list(place.aliases)
            named = [p for p in places if not p.is_anonymous]
            if named:
                named_id = named[0].id
                if target_id not in {p.id for p in named}:
                    target_id = named_id
            dlg = PlaceMergeDialog(places, target_id, self)
            if dlg.exec() != dlg.Accepted:
                return
            choice = dlg.choice()
        try:
            with session_scope() as session:
                PlaceService(session).merge_places(
                    [pid for pid in ids if pid != target_id],
                    target_id,
                    name=choice.name,
                    latitude=choice.latitude,
                    longitude=choice.longitude,
                    thumbnail_path=choice.thumbnail_path,
                )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self,
                t("places_merge_title"),
                t("places_merge_error", error=str(exc)),
            )
            return
        self.refresh()
        self._load_detail(target_id)
