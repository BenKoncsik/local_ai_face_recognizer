"""Objects panel — list, inspect, edit, merge tagged objects.

A domain entirely separate from face recognition.  Mirrors the structure of
:class:`app.ui.panels.locations_panel.LocationsPanel`: a left list/table with
filters and actions, and a right detail pane (data sheet + gallery + comments +
related persons).
"""

from __future__ import annotations

import logging
from typing import List, Optional

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.db.database import session_scope
from app.db.models import OBJECT_ROLES, Person
from app.services.object_service import (
    ObjectFilters,
    ObjectService,
)
from app.ui.dialogs.object_info_dialog import ObjectInfoDialog
from app.ui.dialogs.object_merge_dialog import ObjectMergeDialog
from app.ui.i18n import t
from app.ui.widgets.object_gallery_widget import (
    ObjectGalleryWidget,
    crop_pixmap,
)
from app.ui.widgets.person_search_select import PersonSearchSelect

log = logging.getLogger(__name__)

_ROLE_ID = Qt.UserRole

# Column indices in the objects table.
_COL_NAME = 0
_COL_IMAGES = 1
_COL_NOTES = 2
_COL_PERSONS = 3

# Role keys mapped to i18n string keys.
_ROLE_I18N = {role: f"object_role_{role}" for role in OBJECT_ROLES}


def _role_label(role: str) -> str:
    return t(_ROLE_I18N.get(role, "object_role_other"))


class _AddPersonDialog(QDialog):
    """Pick a person and a role to link to an object.

    Uses the shared :class:`PersonSearchSelect` so person selection here behaves
    exactly like everywhere else: relevance-ranked, double-click to confirm, and
    unknown clusters hidden from the base list but reachable by search.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("object_add_person"))
        self.setMinimumWidth(320)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._person = PersonSearchSelect()
        try:
            with session_scope() as session:
                persons = (
                    session.query(Person)
                    .filter(Person.is_protected == False)  # noqa: E712
                    .order_by(Person.name)
                    .all()
                )
                self._person.set_persons(persons)
        except Exception:
            log.exception("Failed to load persons for object link")
        # Double-click a person to confirm the dialog, matching the merge dialog.
        self._person.person_double_clicked.connect(self.accept)
        form.addRow(t("object_detail_persons"), self._person)

        self._role = QComboBox()
        for role in OBJECT_ROLES:
            self._role.addItem(_role_label(role), role)
        form.addRow(t("object_role"), self._role)

        self._note = QLineEdit()
        form.addRow(t("object_notes"), self._note)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def person_id(self) -> Optional[int]:
        return self._person.current_person_id()

    @property
    def role(self) -> str:
        return str(self._role.currentData())

    @property
    def note(self) -> str:
        return self._note.text().strip()


class ObjectsPanel(QWidget):
    """List, filter, inspect, edit and merge tagged objects."""

    object_data_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._current_object_id: Optional[int] = None
        self._current_occurrences: List = []
        self._build_ui()
        self.refresh()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal)

        # ── Left: filter + table + actions ──────────────────────────────────
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 4, 0)

        self._name_filter = QLineEdit()
        self._name_filter.setPlaceholderText(t("objects_filter_name"))
        self._name_filter.textChanged.connect(self.refresh)
        left_layout.addWidget(self._name_filter)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(
            [
                t("objects_col_name"),
                t("objects_col_images"),
                t("objects_col_notes"),
                t("objects_col_persons"),
            ]
        )
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(52)
        self._table.setIconSize(QSize(48, 48))
        self._table.horizontalHeader().setStretchLastSection(False)
        self._table.setColumnWidth(_COL_NAME, 240)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        left_layout.addWidget(self._table, stretch=1)

        actions = QHBoxLayout()
        self._new_btn = QPushButton(t("objects_new"))
        self._new_btn.clicked.connect(self._on_new)
        actions.addWidget(self._new_btn)
        self._delete_btn = QPushButton(t("objects_delete"))
        self._delete_btn.clicked.connect(self._on_delete)
        actions.addWidget(self._delete_btn)
        self._merge_btn = QPushButton(t("objects_merge"))
        self._merge_btn.clicked.connect(self._on_merge)
        actions.addWidget(self._merge_btn)
        left_layout.addLayout(actions)

        splitter.addWidget(left)

        # ── Right: detail ───────────────────────────────────────────────────
        detail_inner = QWidget()
        self._detail_layout = QVBoxLayout(detail_inner)
        self._detail_layout.setContentsMargins(8, 4, 4, 4)
        self._detail_layout.setSpacing(6)

        self._title = QLabel(t("objects_empty"))
        self._title.setStyleSheet("font-size: 15px; font-weight: bold;")
        self._title.setWordWrap(True)
        self._detail_layout.addWidget(self._title)

        self._desc_label = QLabel("")
        self._desc_label.setWordWrap(True)
        self._desc_label.setStyleSheet("color: #ccc;")
        self._detail_layout.addWidget(self._desc_label)

        self._meta_label = QLabel("")
        self._meta_label.setWordWrap(True)
        self._meta_label.setStyleSheet("color: #888; font-size: 11px;")
        self._detail_layout.addWidget(self._meta_label)

        self._edit_btn = QPushButton(t("objects_edit"))
        self._edit_btn.clicked.connect(self._on_edit)
        self._detail_layout.addWidget(self._edit_btn)

        # Related persons
        self._persons_hdr = QLabel(t("object_detail_persons"))
        self._persons_hdr.setStyleSheet("font-weight: bold; margin-top: 6px;")
        self._detail_layout.addWidget(self._persons_hdr)

        self._persons_list = QListWidget()
        self._persons_list.setMaximumHeight(120)
        self._detail_layout.addWidget(self._persons_list)

        persons_btns = QHBoxLayout()
        self._add_person_btn = QPushButton(t("object_add_person"))
        self._add_person_btn.clicked.connect(self._on_add_person)
        persons_btns.addWidget(self._add_person_btn)
        self._remove_person_btn = QPushButton(t("object_remove_person"))
        self._remove_person_btn.clicked.connect(self._on_remove_person)
        persons_btns.addWidget(self._remove_person_btn)
        self._detail_layout.addLayout(persons_btns)

        # Gallery
        self._gallery_hdr = QLabel(t("object_detail_gallery"))
        self._gallery_hdr.setStyleSheet("font-weight: bold; margin-top: 6px;")
        self._detail_layout.addWidget(self._gallery_hdr)
        self._gallery = ObjectGalleryWidget()
        self._gallery.set_thumbnail_requested.connect(self._on_set_thumbnail)
        self._gallery.clear_thumbnail_requested.connect(self._on_clear_thumbnail)
        self._detail_layout.addWidget(self._gallery)

        # All comments
        self._comments_hdr = QLabel(t("object_detail_comments"))
        self._comments_hdr.setStyleSheet("font-weight: bold; margin-top: 6px;")
        self._detail_layout.addWidget(self._comments_hdr)
        self._comments = QTextEdit()
        self._comments.setReadOnly(True)
        self._comments.setMinimumHeight(120)
        self._detail_layout.addWidget(self._comments)

        self._detail_layout.addStretch()

        detail_scroll = QScrollArea()
        detail_scroll.setWidget(detail_inner)
        detail_scroll.setWidgetResizable(True)
        detail_scroll.setFrameShape(QScrollArea.NoFrame)
        splitter.addWidget(detail_scroll)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

        self._clear_detail()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Reload the objects table, preserving the current selection."""
        filters = ObjectFilters(name=self._name_filter.text().strip())
        try:
            with session_scope() as session:
                svc = ObjectService(session)
                summaries = svc.list_objects(filters)
                thumb_specs = svc.get_thumbnail_specs()
        except Exception:
            log.exception("Failed to list objects")
            summaries = []
            thumb_specs = {}

        self._table.blockSignals(True)
        self._table.setRowCount(0)
        for s in summaries:
            row = self._table.rowCount()
            self._table.insertRow(row)
            name_item = QTableWidgetItem(s.name)
            name_item.setData(_ROLE_ID, s.object_id)
            spec = thumb_specs.get(s.object_id)
            if spec is not None:
                pix = crop_pixmap(spec[0], spec[1], size=48)
                if pix is not None:
                    name_item.setIcon(QIcon(pix))
            self._table.setItem(row, _COL_NAME, name_item)
            self._table.setItem(row, _COL_IMAGES, QTableWidgetItem(str(s.image_count)))
            self._table.setItem(row, _COL_NOTES, QTableWidgetItem(str(s.note_count)))
            self._table.setItem(row, _COL_PERSONS, QTableWidgetItem(str(s.person_count)))
        self._table.blockSignals(False)

        if self._current_object_id is not None and not self._select_row(
            self._current_object_id
        ):
            self._clear_detail()

    def mark_stale(self) -> None:
        """Allow external callers to request a refresh on next show."""
        self.refresh()

    def retranslate(self) -> None:
        self._table.setHorizontalHeaderLabels(
            [
                t("objects_col_name"),
                t("objects_col_images"),
                t("objects_col_notes"),
                t("objects_col_persons"),
            ]
        )
        self._name_filter.setPlaceholderText(t("objects_filter_name"))
        self._new_btn.setText(t("objects_new"))
        self._delete_btn.setText(t("objects_delete"))
        self._merge_btn.setText(t("objects_merge"))
        self._edit_btn.setText(t("objects_edit"))
        self._persons_hdr.setText(t("object_detail_persons"))
        self._gallery_hdr.setText(t("object_detail_gallery"))
        self._comments_hdr.setText(t("object_detail_comments"))
        self._add_person_btn.setText(t("object_add_person"))
        self._remove_person_btn.setText(t("object_remove_person"))

    def open_object(self, object_id: int) -> None:
        """Select, highlight and show an object (navigation from other panels)."""
        # Clear any active name filter so the target object is guaranteed visible.
        self._name_filter.blockSignals(True)
        self._name_filter.clear()
        self._name_filter.blockSignals(False)
        self._current_object_id = object_id
        self.refresh()
        if self._select_row(object_id):
            # Give the table focus so the row shows the active (bright) selection
            # colour rather than the faint inactive one.
            self._table.setFocus()
        else:
            self._load_detail(object_id)

    # ------------------------------------------------------------------
    # Selection / detail
    # ------------------------------------------------------------------

    def _select_row(self, object_id: int) -> bool:
        for row in range(self._table.rowCount()):
            item = self._table.item(row, _COL_NAME)
            if item is not None and item.data(_ROLE_ID) == object_id:
                self._table.setCurrentCell(row, _COL_NAME)
                self._table.selectRow(row)
                self._table.scrollToItem(item)
                return True
        return False

    def _selected_object_id(self) -> Optional[int]:
        items = self._table.selectedItems()
        if not items:
            return None
        name_item = self._table.item(items[0].row(), _COL_NAME)
        return name_item.data(_ROLE_ID) if name_item else None

    def _selected_object_ids(self) -> List[int]:
        ids: List[int] = []
        for idx in self._table.selectionModel().selectedRows():
            item = self._table.item(idx.row(), _COL_NAME)
            if item is not None:
                ids.append(item.data(_ROLE_ID))
        return ids

    def _on_selection_changed(self) -> None:
        oid = self._selected_object_id()
        if oid is None:
            self._clear_detail()
            return
        self._current_object_id = oid
        self._load_detail(oid)

    def _clear_detail(self) -> None:
        self._title.setText(t("objects_empty"))
        self._desc_label.setText("")
        self._meta_label.setText("")
        self._persons_list.clear()
        self._gallery.set_occurrences([])
        self._comments.setPlainText("")
        self._edit_btn.setEnabled(False)
        self._add_person_btn.setEnabled(False)
        self._remove_person_btn.setEnabled(False)

    def _load_detail(self, object_id: int) -> None:
        try:
            with session_scope() as session:
                svc = ObjectService(session)
                summary = svc.get_summary(object_id)
                occurrences = svc.get_occurrences(object_id)
                persons = svc.get_object_persons(object_id)
        except Exception:
            log.exception("Failed to load object detail %d", object_id)
            self._clear_detail()
            return

        self._current_occurrences = occurrences
        self._title.setText(summary.name)
        self._desc_label.setText(summary.description or "")
        self._meta_label.setText(
            f"{t('object_detail_images')}: {summary.image_count}   "
            f"{t('object_detail_notes')}: {summary.note_count}   "
            f"{t('object_detail_persons')}: {summary.person_count}\n"
            f"{t('object_detail_created')}: {(summary.created_at or '')[:19]}   "
            f"{t('object_detail_updated')}: {(summary.updated_at or '')[:19]}"
        )

        # Persons
        self._persons_list.clear()
        for p in persons:
            label = f"{p.name} — {_role_label(p.role)}"
            if p.note:
                label += f" ({p.note})"
            item = QListWidgetItem(label)
            item.setData(_ROLE_ID, (p.person_id, p.role))
            self._persons_list.addItem(item)

        # Gallery — each occurrence with its bbox frame.
        gallery_items = []
        for occ in occurrences:
            if not occ.image_path:
                continue
            bbox = None
            if None not in (occ.bbox_x, occ.bbox_y, occ.bbox_w, occ.bbox_h):
                bbox = (occ.bbox_x, occ.bbox_y, occ.bbox_w, occ.bbox_h)
            gallery_items.append((occ.occurrence_id, occ.image_path, bbox))
        self._gallery.set_occurrences(gallery_items)

        # All comments
        comment_blocks = []
        for occ in occurrences:
            if not occ.note:
                continue
            date = occ.photo_date or ""
            import os
            fname = os.path.basename(occ.image_path) if occ.image_path else ""
            comment_blocks.append(f"{date}\n{fname}\n{occ.note}")
        self._comments.setPlainText(
            "\n\n———\n\n".join(comment_blocks) if comment_blocks else t("object_no_comments")
        )

        self._edit_btn.setEnabled(True)
        self._add_person_btn.setEnabled(True)
        self._remove_person_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_new(self) -> None:
        name, ok = _prompt_text(self, t("objects_new"), t("object_name"))
        if not ok or not name.strip():
            return
        try:
            with session_scope() as session:
                obj = ObjectService(session).create_object(name)
                new_id = obj.id
        except ValueError as exc:
            QMessageBox.warning(self, t("objects_new"), str(exc))
            return
        except Exception:
            log.exception("Failed to create object")
            return
        self._current_object_id = new_id
        self.refresh()
        self.object_data_changed.emit()

    def _on_delete(self) -> None:
        oid = self._selected_object_id()
        if oid is None:
            return
        name = self._title.text()
        if QMessageBox.question(
            self,
            t("objects_delete"),
            t("objects_delete_confirm", name=name),
        ) != QMessageBox.Yes:
            return
        try:
            with session_scope() as session:
                ObjectService(session).delete_object(oid)
        except Exception:
            log.exception("Failed to delete object %d", oid)
            return
        self._current_object_id = None
        self.refresh()
        self.object_data_changed.emit()

    def _on_edit(self) -> None:
        oid = self._selected_object_id()
        if oid is None:
            return
        if ObjectInfoDialog(oid, self).exec() == QDialog.Accepted:
            self.refresh()
            self.object_data_changed.emit()

    def _on_merge(self) -> None:
        ids = self._selected_object_ids()
        if len(ids) < 2:
            QMessageBox.information(self, t("objects_merge"), t("object_merge_hint"))
            return
        candidates = []
        for row in range(self._table.rowCount()):
            item = self._table.item(row, _COL_NAME)
            if item is not None and item.data(_ROLE_ID) in ids:
                candidates.append((item.data(_ROLE_ID), item.text()))
        dlg = ObjectMergeDialog(candidates, self)
        if dlg.exec() != QDialog.Accepted or dlg.target_id is None:
            return
        try:
            with session_scope() as session:
                ObjectService(session).merge_objects(dlg.source_ids, dlg.target_id)
        except Exception:
            log.exception("Failed to merge objects")
            return
        self._current_object_id = dlg.target_id
        self.refresh()
        self.object_data_changed.emit()

    def _on_add_person(self) -> None:
        oid = self._selected_object_id()
        if oid is None:
            return
        dlg = _AddPersonDialog(self)
        if dlg.exec() != QDialog.Accepted or dlg.person_id is None:
            return
        try:
            with session_scope() as session:
                ObjectService(session).add_person_link(
                    oid, dlg.person_id, dlg.role, dlg.note
                )
        except ValueError as exc:
            QMessageBox.warning(self, t("object_add_person"), str(exc))
            return
        except Exception:
            log.exception("Failed to add person link")
            return
        self._load_detail(oid)
        self.refresh()
        self.object_data_changed.emit()

    def _on_remove_person(self) -> None:
        oid = self._selected_object_id()
        items = self._persons_list.selectedItems()
        if oid is None or not items:
            return
        person_id, role = items[0].data(_ROLE_ID)
        try:
            with session_scope() as session:
                ObjectService(session).remove_person_link(oid, person_id, role)
        except Exception:
            log.exception("Failed to remove person link")
            return
        self._load_detail(oid)
        self.refresh()
        self.object_data_changed.emit()

    # ------------------------------------------------------------------
    # Thumbnail (manual, like persons)
    # ------------------------------------------------------------------

    def _on_set_thumbnail(self, occurrence_id: int) -> None:
        """Use a chosen occurrence's bbox crop as the object's thumbnail."""
        if self._current_object_id is None:
            return
        occ = next(
            (o for o in self._current_occurrences if o.occurrence_id == occurrence_id),
            None,
        )
        if occ is None or not occ.image_path:
            return
        bbox = None
        if None not in (occ.bbox_x, occ.bbox_y, occ.bbox_w, occ.bbox_h):
            bbox = (occ.bbox_x, occ.bbox_y, occ.bbox_w, occ.bbox_h)
        pix = crop_pixmap(occ.image_path, bbox, size=256)
        if pix is None:
            return
        from app.paths import user_data_dir

        out_dir = user_data_dir() / "object_crops"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            log.exception("Cannot create object crops dir")
            return
        out_path = str(out_dir / f"obj_{self._current_object_id}.jpg")
        if not pix.save(out_path, "JPG"):
            log.warning("Failed to save object thumbnail to %s", out_path)
            return
        try:
            with session_scope() as session:
                ObjectService(session).set_thumbnail_path(
                    self._current_object_id, out_path, manual=True
                )
        except Exception:
            log.exception("Failed to set object thumbnail")
            return
        self.refresh()
        self.object_data_changed.emit()

    def _on_clear_thumbnail(self) -> None:
        """Clear a manually-set thumbnail (fall back to auto crop)."""
        if self._current_object_id is None:
            return
        try:
            with session_scope() as session:
                ObjectService(session).set_thumbnail_path(
                    self._current_object_id, None
                )
        except Exception:
            log.exception("Failed to clear object thumbnail")
            return
        self.refresh()
        self.object_data_changed.emit()


def _prompt_text(parent, title: str, label: str):
    from PySide6.QtWidgets import QInputDialog

    return QInputDialog.getText(parent, title, label)
