"""Places / Locations panel."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, QSize, QRect
from PySide6.QtGui import QPixmap, QTextOption
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QHeaderView,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStyledItemDelegate,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.db.database import session_scope
from app.db.models import Place
from app.services.place_service import (
    ANONYMOUS_GPS_PLACE_NAME,
    PLACE_TYPE_AREA,
    PLACE_TYPE_EXACT,
    PLACE_TYPE_REGION,
    PlaceFilters,
    PlaceService,
)
from app.ui.dialogs.place_edit_dialog import PlaceEditDialog
from app.ui.dialogs.place_merge_dialog import PlaceMergeDialog
from app.ui.i18n import t
from app.ui.widgets.place_gallery_widget import PlaceGalleryWidget
from app.ui.widgets.place_map_widget import PlaceMapWidget

_ROLE_ID = Qt.UserRole

# Emoji prefix shown in the Type column / labels for each place type.
_TYPE_ICON = {
    PLACE_TYPE_EXACT: "📍",
    PLACE_TYPE_AREA: "🏘️",
    PLACE_TYPE_REGION: "🗺️",
}
_TYPE_I18N = {
    PLACE_TYPE_EXACT: "places_type_exact",
    PLACE_TYPE_AREA: "places_type_area",
    PLACE_TYPE_REGION: "places_type_region",
}


def _type_label(place_type: str) -> str:
    icon = _TYPE_ICON.get(place_type, "")
    label = t(_TYPE_I18N.get(place_type, "places_type_area"))
    return f"{icon} {label}".strip()


class _WrappingNameDelegate(QStyledItemDelegate):
    """Inline editor that wraps long names across multiple lines."""

    def createEditor(self, parent, option, index):
        editor = QTextEdit(parent)
        editor.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        editor.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        editor.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        editor.setFrameShape(QTextEdit.NoFrame)
        return editor

    def setEditorData(self, editor, index):
        editor.setPlainText(index.data(Qt.EditRole) or "")

    def setModelData(self, editor, model, index):
        model.setData(index, editor.toPlainText().strip(), Qt.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        doc = editor.document()
        doc.setTextWidth(option.rect.width())
        content_h = int(doc.size().height()) + 8
        height = max(option.rect.height(), content_h)
        editor.setGeometry(QRect(option.rect.x(), option.rect.y(), option.rect.width(), height))

    def sizeHint(self, option, index):
        text = index.data(Qt.DisplayRole) or ""
        doc = QTextEdit()
        doc.setPlainText(text)
        doc.document().setTextWidth(max(option.rect.width(), 120))
        h = int(doc.document().size().height()) + 6
        return QSize(option.rect.width(), max(option.rect.height(), h))

# Collect per-image GPS only if the coordinate differs from the place centre by
# more than this threshold (degrees) — avoids duplicating the main marker.
_MIN_GPS_DIFF = 5e-5


class LocationsPanel(QWidget):
    """List, filter, inspect, edit, and merge reusable places (hierarchical)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._current_place_id: Optional[int] = None
        self._build_ui()
        self.refresh()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

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
        self._type_filter = QComboBox()
        filters.addWidget(self._type_filter)
        self._coord_filter = QComboBox()
        filters.addWidget(self._coord_filter)
        self._anon_only = QCheckBox()
        filters.addWidget(self._anon_only)
        self._filter_btn = QPushButton()
        self._filter_btn.clicked.connect(self.refresh)
        filters.addWidget(self._filter_btn)
        root.addLayout(filters)

        splitter = QSplitter(Qt.Horizontal)

        self._tree = QTreeWidget()
        self._tree.setColumnCount(6)
        self._tree.setSelectionMode(QTreeWidget.ExtendedSelection)
        self._tree.setWordWrap(True)
        self._tree.setUniformRowHeights(False)
        self._tree.setExpandsOnDoubleClick(False)
        header = self._tree.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, 6):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self._tree.itemSelectionChanged.connect(self._on_selection_changed)
        self._tree.itemChanged.connect(self._on_tree_item_changed)
        self._name_delegate = _WrappingNameDelegate(self._tree)
        self._tree.setItemDelegateForColumn(0, self._name_delegate)
        splitter.addWidget(self._tree)

        detail_inner = QWidget()
        detail_layout = QVBoxLayout(detail_inner)
        detail_layout.setContentsMargins(8, 4, 4, 4)
        detail_layout.setSpacing(8)

        # Thumbnail
        self._thumb = QLabel()
        self._thumb.setFixedSize(220, 150)
        self._thumb.setAlignment(Qt.AlignCenter)
        self._thumb.setStyleSheet("background:#181825;color:#888;border-radius:4px;")
        detail_layout.addWidget(self._thumb)

        # Metadata form
        form = QFormLayout()
        form.setSpacing(4)
        self._name = QLabel()
        self._name.setWordWrap(True)
        name_row = QHBoxLayout()
        name_row.setSpacing(4)
        name_row.addWidget(self._name, 1)
        self._rename_btn = QPushButton()
        self._rename_btn.setToolTip(t("places_rename_tooltip"))
        self._rename_btn.setEnabled(False)
        self._rename_btn.clicked.connect(self._start_rename)
        name_row.addWidget(self._rename_btn)
        self._type_lbl = QLabel()
        self._coords = QLabel()
        self._image_count = QLabel()
        self._person_count_lbl = QLabel()
        form.addRow(t("places_name"), name_row)
        form.addRow(t("places_type"), self._type_lbl)
        form.addRow(t("places_coords"), self._coords)
        form.addRow(t("places_image_count"), self._image_count)
        form.addRow(t("places_person_count"), self._person_count_lbl)
        detail_layout.addLayout(form)

        # Interactive map
        self._map_widget = PlaceMapWidget()
        self._map_widget.setFixedHeight(240)
        detail_layout.addWidget(self._map_widget)

        # Visual gallery (replaces plain file-name list as primary view)
        gallery_hdr = QLabel()
        gallery_hdr.setObjectName("gallery_hdr")
        detail_layout.addWidget(gallery_hdr)

        self._gallery = PlaceGalleryWidget()
        self._gallery.set_place_thumbnail_requested.connect(
            self._on_set_place_thumbnail
        )
        detail_layout.addWidget(self._gallery)

        # Persons
        detail_layout.addWidget(QLabel(t("places_persons")))
        self._persons = QListWidget()
        self._persons.setMaximumHeight(120)
        detail_layout.addWidget(self._persons)

        # Action buttons
        actions = QHBoxLayout()
        self._new_btn = QPushButton()
        self._new_btn.clicked.connect(self._create_place)
        actions.addWidget(self._new_btn)
        self._edit_btn = QPushButton()
        self._edit_btn.setEnabled(False)
        self._edit_btn.clicked.connect(self._edit_selected)
        actions.addWidget(self._edit_btn)
        self._merge_btn = QPushButton()
        self._merge_btn.clicked.connect(self._merge_selected)
        actions.addWidget(self._merge_btn)
        self._refresh_btn = QPushButton()
        self._refresh_btn.clicked.connect(self.refresh)
        actions.addWidget(self._refresh_btn)
        detail_layout.addLayout(actions)

        detail_layout.addStretch()

        detail_scroll = QScrollArea()
        detail_scroll.setWidget(detail_inner)
        detail_scroll.setWidgetResizable(True)
        detail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        detail_scroll.setFrameShape(QScrollArea.NoFrame)

        splitter.addWidget(detail_scroll)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)

        self._gallery_hdr = gallery_hdr
        self.retranslate()

    def retranslate(self) -> None:
        self._name_filter.setPlaceholderText(t("places_filter_name"))
        self._person_filter.setPlaceholderText(t("places_filter_person_id"))
        self._date_from.setPlaceholderText(t("places_filter_date_from"))
        self._date_to.setPlaceholderText(t("places_filter_date_to"))
        self._min_images.setPrefix(t("places_filter_min_images"))
        self._type_filter.clear()
        self._type_filter.addItem(t("places_filter_type_any"), None)
        self._type_filter.addItem(_type_label(PLACE_TYPE_EXACT), PLACE_TYPE_EXACT)
        self._type_filter.addItem(_type_label(PLACE_TYPE_AREA), PLACE_TYPE_AREA)
        self._type_filter.addItem(_type_label(PLACE_TYPE_REGION), PLACE_TYPE_REGION)
        self._coord_filter.clear()
        self._coord_filter.addItem(t("places_filter_coords_any"), None)
        self._coord_filter.addItem(t("places_filter_coords_yes"), True)
        self._coord_filter.addItem(t("places_filter_coords_no"), False)
        self._anon_only.setText(t("places_filter_anon"))
        self._filter_btn.setText(t("places_filter_apply"))
        self._tree.setHeaderLabels([
            t("places_name"),
            t("places_type"),
            t("places_coords"),
            t("places_image_count"),
            t("places_person_count"),
            t("places_source"),
        ])
        self._gallery_hdr.setText(t("places_images"))
        self._rename_btn.setText(t("places_rename_btn"))
        self._rename_btn.setToolTip(t("places_rename_tooltip"))
        self._new_btn.setText(t("places_new_btn"))
        self._edit_btn.setText(t("places_edit_btn"))
        self._merge_btn.setText(t("places_merge_btn"))
        self._refresh_btn.setText(t("places_refresh_btn"))

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        person_id = None
        person_text = self._person_filter.text().strip()
        if person_text:
            try:
                person_id = int(person_text)
            except ValueError:
                person_id = None
        type_value = self._type_filter.currentData()
        filters = PlaceFilters(
            name=self._name_filter.text(),
            person_id=person_id,
            date_from=self._date_from.text(),
            date_to=self._date_to.text(),
            min_images=self._min_images.value() or None,
            has_coordinates=self._coord_filter.currentData(),
            anonymous_exif_only=self._anon_only.isChecked(),
            place_types=(type_value,) if type_value else None,
        )
        with session_scope() as session:
            summaries = PlaceService(session).list_places(filters)

        # Build the tree from the flat summaries using parent_id. A child whose
        # parent was filtered out is promoted to a root so nothing disappears.
        self._tree.blockSignals(True)
        self._tree.clear()
        present_ids = {s.place_id for s in summaries}
        items: Dict[int, QTreeWidgetItem] = {}
        # First pass: create every item.
        for summary in summaries:
            items[summary.place_id] = self._make_item(summary)
        # Second pass: attach to parent or to the root.
        for summary in summaries:
            item = items[summary.place_id]
            parent_id = summary.parent_id
            if parent_id is not None and parent_id in present_ids:
                items[parent_id].addChild(item)
            else:
                self._tree.addTopLevelItem(item)
        self._tree.expandAll()
        self._tree.blockSignals(False)

    def _make_item(self, summary) -> QTreeWidgetItem:
        name = summary.name
        if summary.is_anonymous and summary.source == "exif":
            name = f"{ANONYMOUS_GPS_PLACE_NAME} #{summary.place_id}"
        coords = ""
        if summary.latitude is not None and summary.longitude is not None:
            coords = f"{summary.latitude:.6f}, {summary.longitude:.6f}"
        values = [
            name,
            _type_label(summary.place_type),
            coords,
            str(summary.image_count),
            str(summary.person_count),
            summary.source or "",
        ]
        item = QTreeWidgetItem(values)
        for col in range(6):
            item.setData(col, _ROLE_ID, summary.place_id)
        # Name column editable inline (matches the legacy double-click rename).
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        item.setToolTip(0, t("places_name_edit_tip", default="Dupla kattintás a név szerkesztéséhez"))
        return item

    def _start_rename(self) -> None:
        """Open the inline editor for the selected place's name cell."""
        item = self._tree.currentItem()
        if item is None:
            items = self._tree.selectedItems()
            if not items:
                return
            item = items[0]
        self._tree.scrollToItem(item)
        self._tree.editItem(item, 0)

    def _on_selection_changed(self) -> None:
        items = self._tree.selectedItems()
        if not items:
            self._current_place_id = None
            self._clear_detail()
            return
        place_id = items[0].data(0, _ROLE_ID)
        if place_id is not None:
            self._load_detail(int(place_id))

    def _on_tree_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        """Handle inline editing of the name column."""
        if column != 0:
            return

        place_id = item.data(0, _ROLE_ID)
        if place_id is None:
            return

        new_name = item.text(0).strip()

        if not new_name:
            QMessageBox.warning(self, t("empty_name_title"), t("empty_name_msg"))
            with session_scope() as session:
                place = session.get(Place, place_id)
                if place is not None:
                    item.setText(0, place.name)
            return

        with session_scope() as session:
            place = session.get(Place, place_id)
            if place is not None and place.name == new_name:
                return

        try:
            with session_scope() as session:
                PlaceService(session).name_place(int(place_id), new_name)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self,
                t("places_rename_title", default="Rename Place"),
                t("places_rename_error", error=str(exc), default="Failed to rename place: {error}"),
            )
            with session_scope() as session:
                place = session.get(Place, place_id)
                if place is not None:
                    item.setText(0, place.name)
            return

        if self._current_place_id == place_id:
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

            place_type = place.place_type
            radius = place.accuracy_radius_meters
            lat = place.latitude
            lon = place.longitude
            coords = (
                f"{lat:.6f}, {lon:.6f}"
                if lat is not None and lon is not None
                else t("places_no_coords")
            )
            thumb = place.thumbnail_path

            image_paths = [img.file_path for img in images]
            person_names = [p.name for p in persons]
            image_count = len(images)
            person_count = len(persons)
            nearby = _collect_nearby_points(images, lat, lon)

        self._name.setText(name)
        self._rename_btn.setEnabled(True)
        self._edit_btn.setEnabled(True)
        type_text = _type_label(place_type)
        if radius is not None:
            type_text = f"{type_text}  (~{int(radius)} m)"
        self._type_lbl.setText(type_text)
        self._coords.setText(coords)
        self._image_count.setText(str(image_count))
        self._person_count_lbl.setText(str(person_count))

        self._persons.clear()
        for pname in person_names:
            self._persons.addItem(pname)

        self._set_thumbnail(thumb)
        self._map_widget.set_place(name, lat, lon, image_count, person_count, nearby)
        self._gallery.set_images(image_paths)

    def _clear_detail(self) -> None:
        self._name.clear()
        self._type_lbl.clear()
        self._coords.clear()
        self._image_count.clear()
        self._person_count_lbl.clear()
        self._persons.clear()
        self._thumb.setPixmap(QPixmap())
        self._thumb.setText(t("places_no_thumbnail"))
        self._map_widget.clear()
        self._gallery.clear()
        self._rename_btn.setEnabled(False)
        self._edit_btn.setEnabled(False)

    def _set_thumbnail(self, path: Optional[str]) -> None:
        self._thumb.setPixmap(QPixmap())
        if not path:
            self._thumb.setText(t("places_no_thumbnail"))
            return
        from pathlib import Path as _Path
        if not _Path(path).exists():
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

    # ------------------------------------------------------------------
    # Create / edit
    # ------------------------------------------------------------------

    def _candidate_parents(self, session, exclude_id: Optional[int]) -> List[Place]:
        """All places eligible as a parent of ``exclude_id`` (excludes itself
        and its descendants to prevent cycles)."""
        svc = PlaceService(session)
        forbidden = set()
        if exclude_id is not None:
            forbidden.add(exclude_id)
            forbidden.update(d.id for d in svc.list_descendants(exclude_id))
        places = session.query(Place).order_by(Place.name).all()
        return [p for p in places if p.id not in forbidden]

    def _create_place(self) -> None:
        with session_scope() as session:
            parents = self._candidate_parents(session, None)
            dlg = PlaceEditDialog(place=None, parents=parents, parent=self)
            if dlg.exec() != QDialog.Accepted:
                return
            res = dlg.result_value()
            try:
                svc = PlaceService(session)
                if res.settlement_name:
                    place = svc.create_place_from_address(
                        res.settlement_name,
                        res.street_name,
                        res.house_number,
                        latitude=res.latitude,
                        longitude=res.longitude,
                        place_type=res.place_type,
                        coordinate_source=res.coordinate_source,
                        is_exact_coordinate=res.is_exact_coordinate,
                        accuracy_radius_meters=res.accuracy_radius_meters,
                        parent_id=res.parent_id,
                        name=res.name,
                    )
                    self._record_address(session, res)
                else:
                    place = svc.create_place(
                        name=res.name,
                        latitude=res.latitude,
                        longitude=res.longitude,
                        source="manual",
                        place_type=res.place_type,
                        accuracy_radius_meters=res.accuracy_radius_meters,
                        parent_id=res.parent_id,
                    )
                new_id = place.id
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(self, t("place_edit_title_new"),
                                     t("place_edit_save_error", error=str(exc)))
                return
        self.refresh()
        self._load_detail(new_id)

    def _record_address(self, session, res) -> None:
        """Save the settlement/street into the offline suggestion store."""
        if not res.settlement_name:
            return
        try:
            from app.services.geocoding.factory import create_geocoding_service

            create_geocoding_service(session).record_address_use(
                res.settlement_name, res.street_name
            )
        except Exception:  # noqa: BLE001
            pass

    def _edit_selected(self) -> None:
        if self._current_place_id is None:
            return
        place_id = self._current_place_id
        with session_scope() as session:
            place = session.get(Place, place_id)
            if place is None:
                return
            parents = self._candidate_parents(session, place_id)
            dlg = PlaceEditDialog(place=place, parents=parents, parent=self)
            if dlg.exec() != QDialog.Accepted:
                return
            res = dlg.result_value()
            try:
                svc = PlaceService(session)
                if res.settlement_name:
                    svc.update_place_address(
                        place_id,
                        res.settlement_name,
                        res.street_name,
                        res.house_number,
                        latitude=res.latitude,
                        longitude=res.longitude,
                        place_type=res.place_type,
                        coordinate_source=res.coordinate_source,
                        is_exact_coordinate=res.is_exact_coordinate,
                        accuracy_radius_meters=res.accuracy_radius_meters,
                    )
                    self._record_address(session, res)
                else:
                    if res.name and res.name != place.name:
                        svc.name_place(place_id, res.name)
                    svc.set_place_type(
                        place_id,
                        res.place_type,
                        accuracy_radius_meters=res.accuracy_radius_meters,
                        reset_radius_to_default=res.accuracy_radius_meters is None,
                    )
                    place.latitude = res.latitude
                    place.longitude = res.longitude
                svc.set_parent(place_id, res.parent_id)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(self, t("place_edit_title_edit"),
                                     t("place_edit_save_error", error=str(exc)))
                return
        self.refresh()
        self._load_detail(place_id)

    # ------------------------------------------------------------------
    # Thumbnail management
    # ------------------------------------------------------------------

    def _on_set_place_thumbnail(self, image_path: str) -> None:
        """Called when the gallery requests a thumbnail change."""
        if self._current_place_id is None:
            return
        try:
            with session_scope() as session:
                svc = PlaceService(session)
                if image_path:
                    svc.set_place_thumbnail(self._current_place_id, image_path)
                else:
                    svc.clear_manual_place_thumbnail(self._current_place_id)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self,
                t("thumbnail_set_error", error="", default="Thumbnail error"),
                t("thumbnail_set_error", error=str(exc)),
            )
            return
        self._load_detail(self._current_place_id)

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def _merge_selected(self) -> None:
        ids: List[int] = []
        for item in self._tree.selectedItems():
            pid = item.data(0, _ROLE_ID)
            if pid is not None and int(pid) not in ids:
                ids.append(int(pid))
        if len(ids) < 2:
            QMessageBox.information(self, t("places_merge_title"), t("places_merge_need_two"))
            return

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
            if dlg.exec() != QDialog.Accepted:
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


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _collect_nearby_points(
    images: list,
    place_lat: Optional[float],
    place_lon: Optional[float],
) -> List[Tuple[float, float]]:
    """Return unique per-image GPS coordinates that differ from the place centre."""
    seen: set = set()
    result: List[Tuple[float, float]] = []
    for img in images:
        for lat, lon in (
            (img.image_latitude, img.image_longitude),
            (img.exif_latitude, img.exif_longitude),
        ):
            if lat is None or lon is None:
                continue
            if place_lat is not None and place_lon is not None:
                if (
                    abs(lat - place_lat) < _MIN_GPS_DIFF
                    and abs(lon - place_lon) < _MIN_GPS_DIFF
                ):
                    continue
            key = (round(lat, 5), round(lon, 5))
            if key not in seen:
                seen.add(key)
                result.append((lat, lon))
    return result
