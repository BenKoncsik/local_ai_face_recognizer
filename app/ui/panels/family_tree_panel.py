"""Family-tree panel — main-window tab for viewing and editing the family tree.

Left: a graphical family tree (:class:`FamilyTreeView`) showing the whole family
forest by default, or a single person's tree when a focus is chosen. Right:
details for the selected person plus quick-add buttons and an "Edit
relationships" button that opens :class:`FamilyTreeEditorDialog`.

The panel is read-only over the data model except through services: it never
mutates relationships directly. Rendering uses the in-memory
:class:`FamilyTreeGraph`, so a single DB read per refresh drives the whole view.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.db.database import session_scope
from app.db.models import Person
from app.services.family_service import FamilyService
from app.services.family_tree_service import FamilyTreeGraph, FamilyTreeService
from app.ui.dialogs.family_tree_editor_dialog import (
    FamilyTreeEditorDialog,
    PersonPickerDialog,
    add_relative,
)
from app.ui.i18n import t
from app.ui.widgets.family_tree_view import FamilyTreeView


def _vline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.VLine)
    f.setFrameShadow(QFrame.Sunken)
    f.setFixedWidth(2)
    return f


class FamilyTreePanel(QWidget):
    """View the family tree and launch the relationship editor."""

    # Emitted when relationships/membership change so the main window can
    # refresh other person-aware views.
    person_data_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._root_id: Optional[int] = None
        self._selected_id: Optional[int] = None
        self._graph: Optional[FamilyTreeGraph] = None
        self._build_ui()
        self._load_default_root()
        self.refresh()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Controls are exposed as toolbar_widget so the main window can embed
        # them in its own toolbar row when this tab is active.
        self.toolbar_widget = QWidget()
        self.toolbar_widget.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Preferred
        )
        controls = QHBoxLayout(self.toolbar_widget)
        controls.setContentsMargins(4, 0, 4, 0)
        controls.setSpacing(6)
        controls.addStretch()
        self._root_label = QLabel()
        controls.addWidget(self._root_label)
        self._root_name = QLabel()
        self._root_name.setStyleSheet("font-weight: bold;")
        controls.addWidget(self._root_name)
        controls.addWidget(_vline())
        self._choose_root_btn = QPushButton()
        self._choose_root_btn.clicked.connect(self._on_choose_root)
        controls.addWidget(self._choose_root_btn)
        self._whole_tree_btn = QPushButton()
        self._whole_tree_btn.clicked.connect(self._on_show_whole_tree)
        controls.addWidget(self._whole_tree_btn)
        self._non_members_check = QCheckBox()
        self._non_members_check.setChecked(True)
        self._non_members_check.toggled.connect(self.refresh)
        controls.addWidget(self._non_members_check)
        controls.addWidget(_vline())
        self._regenerate_btn = QPushButton()
        self._regenerate_btn.clicked.connect(self._on_regenerate)
        controls.addWidget(self._regenerate_btn)
        self._refresh_btn = QPushButton()
        self._refresh_btn.clicked.connect(self.refresh)
        controls.addWidget(self._refresh_btn)
        controls.addWidget(_vline())
        # Zoom controls (also: mouse wheel zooms, drag pans, +/-/0 keys).
        self._zoom_out_btn = QPushButton("−")
        self._zoom_out_btn.setFixedWidth(36)
        controls.addWidget(self._zoom_out_btn)
        self._zoom_reset_btn = QPushButton("⊡")
        self._zoom_reset_btn.setFixedWidth(36)
        controls.addWidget(self._zoom_reset_btn)
        self._zoom_in_btn = QPushButton("+")
        self._zoom_in_btn.setFixedWidth(36)
        controls.addWidget(self._zoom_in_btn)
        root.addWidget(self.toolbar_widget)

        splitter = QSplitter(Qt.Horizontal)

        self._tree = FamilyTreeView()
        self._zoom_out_btn.clicked.connect(self._tree.zoom_out)
        self._zoom_in_btn.clicked.connect(self._tree.zoom_in)
        self._zoom_reset_btn.clicked.connect(self._tree.reset_zoom)
        self._tree.person_selected.connect(self._on_node_selected)
        self._tree.person_activated.connect(self._on_node_activated)
        splitter.addWidget(self._tree)

        # --- Detail panel ---
        detail_inner = QWidget()
        detail_layout = QVBoxLayout(detail_inner)
        detail_layout.setContentsMargins(8, 4, 4, 4)
        detail_layout.setSpacing(8)

        self._thumb = QLabel()
        self._thumb.setFixedSize(180, 180)
        self._thumb.setAlignment(Qt.AlignCenter)
        self._thumb.setStyleSheet(
            "background:#181825;color:#888;border-radius:4px;"
        )
        detail_layout.addWidget(self._thumb)

        self._name = QLabel()
        self._name.setWordWrap(True)
        self._name.setStyleSheet("font-size: 15px; font-weight: bold;")
        detail_layout.addWidget(self._name)

        self._not_member = QLabel()
        self._not_member.setStyleSheet("color:#888;")
        self._not_member.setVisible(False)
        detail_layout.addWidget(self._not_member)

        form = QFormLayout()
        form.setSpacing(4)
        self._relation_lbl = QLabel()
        self._gender_lbl = QLabel()
        self._nickname_lbl = QLabel()
        self._birth_lbl = QLabel()
        self._birth_place_lbl = QLabel()
        self._death_lbl = QLabel()
        self._death_place_lbl = QLabel()
        self._code_lbl = QLabel()
        for lbl in (
            self._nickname_lbl, self._birth_place_lbl,
            self._death_place_lbl, self._code_lbl,
        ):
            lbl.setWordWrap(True)
        self._relation_row = QLabel(t("ft_relation_to_root"))
        self._gender_row = QLabel(t("ft_gender"))
        self._nickname_row = QLabel(t("ft_nickname"))
        self._birth_row = QLabel(t("ft_birth"))
        self._birth_place_row = QLabel(t("ft_birth_place"))
        self._death_row = QLabel(t("ft_death"))
        self._death_place_row = QLabel(t("ft_death_place"))
        self._code_row = QLabel(t("ft_family_code"))
        form.addRow(self._relation_row, self._relation_lbl)
        form.addRow(self._gender_row, self._gender_lbl)
        form.addRow(self._nickname_row, self._nickname_lbl)
        form.addRow(self._birth_row, self._birth_lbl)
        form.addRow(self._birth_place_row, self._birth_place_lbl)
        form.addRow(self._death_row, self._death_lbl)
        form.addRow(self._death_place_row, self._death_place_lbl)
        form.addRow(self._code_row, self._code_lbl)
        detail_layout.addLayout(form)

        self._notes_lbl = QLabel()
        self._notes_lbl.setWordWrap(True)
        self._notes_lbl.setStyleSheet("color:#bbb; font-size: 12px;")
        detail_layout.addWidget(self._notes_lbl)

        # Person actions: full edit + image gallery.
        person_row = QHBoxLayout()
        self._edit_person_btn = QPushButton()
        self._edit_person_btn.setEnabled(False)
        self._edit_person_btn.clicked.connect(self._on_edit_person)
        person_row.addWidget(self._edit_person_btn)
        self._images_btn = QPushButton()
        self._images_btn.setEnabled(False)
        self._images_btn.clicked.connect(self._on_show_images)
        person_row.addWidget(self._images_btn)
        detail_layout.addLayout(person_row)

        # Quick-add buttons for the most common relationship edits.
        quick_row = QHBoxLayout()
        self._add_spouse_btn = QPushButton()
        self._add_spouse_btn.setEnabled(False)
        self._add_spouse_btn.clicked.connect(lambda: self._on_quick_add("spouse"))
        quick_row.addWidget(self._add_spouse_btn)
        self._add_child_btn = QPushButton()
        self._add_child_btn.setEnabled(False)
        self._add_child_btn.clicked.connect(lambda: self._on_quick_add("child"))
        quick_row.addWidget(self._add_child_btn)
        detail_layout.addLayout(quick_row)

        self._edit_btn = QPushButton()
        self._edit_btn.setEnabled(False)
        self._edit_btn.clicked.connect(self._on_edit_relationships)
        detail_layout.addWidget(self._edit_btn)

        self._hint = QLabel()
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet("color:#888;")
        detail_layout.addWidget(self._hint)

        detail_layout.addStretch()

        detail_scroll = QScrollArea()
        detail_scroll.setWidget(detail_inner)
        detail_scroll.setWidgetResizable(True)
        detail_scroll.setFrameShape(QScrollArea.NoFrame)
        splitter.addWidget(detail_scroll)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)

        self.retranslate()

    def retranslate(self) -> None:
        self._root_label.setText(t("ft_root"))
        self._choose_root_btn.setText(t("ft_choose_root"))
        self._whole_tree_btn.setText(t("ft_show_whole"))
        self._non_members_check.setText(t("ft_show_non_members"))
        self._regenerate_btn.setText(t("ft_regenerate"))
        self._regenerate_btn.setToolTip(t("ft_regenerate_tip"))
        self._refresh_btn.setText(t("ft_refresh"))
        self._zoom_in_btn.setToolTip(t("ft_zoom_in"))
        self._zoom_out_btn.setToolTip(t("ft_zoom_out"))
        self._zoom_reset_btn.setToolTip(t("ft_zoom_reset"))
        self._relation_row.setText(t("ft_relation_to_root"))
        self._gender_row.setText(t("ft_gender"))
        self._nickname_row.setText(t("ft_nickname"))
        self._birth_row.setText(t("ft_birth"))
        self._birth_place_row.setText(t("ft_birth_place"))
        self._death_row.setText(t("ft_death"))
        self._death_place_row.setText(t("ft_death_place"))
        self._code_row.setText(t("ft_family_code"))
        self._edit_person_btn.setText(t("ft_edit_person"))
        self._images_btn.setText(t("ft_show_images"))
        self._add_spouse_btn.setText(t("ft_add_spouse_btn"))
        self._add_child_btn.setText(t("ft_add_child_btn"))
        self._edit_btn.setText(t("ft_edit_relationships"))
        self._not_member.setText(t("ft_not_member_badge"))
        if self._root_id is None:
            self._root_name.setText(t("ft_whole_tree"))
        self._hint.setText(t("ft_select_hint"))

    # ------------------------------------------------------------------
    # Root selection
    # ------------------------------------------------------------------

    def _load_default_root(self) -> None:
        # Default to the whole family tree — no root has to be chosen. A focus
        # person can be selected later via "Focus on person …".
        self._root_id = None

    def _on_choose_root(self) -> None:
        with session_scope() as session:
            persons = session.query(Person).order_by(Person.name).all()
            session.expunge_all()
        dialog = PersonPickerDialog(persons, title=t("ft_choose_root"), parent=self)
        if dialog.exec() != QDialog.Accepted:
            return
        chosen = dialog.selected_id()
        if chosen is None:
            return
        self._root_id = chosen
        self._selected_id = chosen
        with session_scope() as session:
            FamilyTreeService(session).update_settings(
                default_root_person_id=chosen
            )
        self.refresh()

    def _on_show_whole_tree(self) -> None:
        """Clear the focus person and show the entire family forest."""
        self._root_id = None
        self.refresh()

    def _on_regenerate(self) -> None:
        """Re-derive relationships from every person's family code."""
        with session_scope() as session:
            count = FamilyService(session).regenerate_all_links()
        QMessageBox.information(
            self, t("ft_regenerate_title"), t("ft_regenerate_done", n=count)
        )
        self.refresh()
        self.person_data_changed.emit()

    # ------------------------------------------------------------------
    # Refresh / rendering
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        selected = self._selected_id
        include_non_members = self._non_members_check.isChecked()
        with session_scope() as session:
            tree = FamilyTreeService(session)
            if self._root_id is None:
                # Whole forest — no root needed.
                self._root_name.setText(t("ft_whole_tree"))
                graph = tree.build_graph(include_non_members=include_non_members)
            else:
                person = session.get(Person, self._root_id)
                if person is None:
                    self._root_id = None
                    self._root_name.setText(t("ft_whole_tree"))
                    graph = tree.build_graph(
                        include_non_members=include_non_members
                    )
                else:
                    self._root_name.setText(person.name)
                    # Focus mode: only the root's direct line (ancestors,
                    # descendants, spouses) — no siblings/cousins.
                    graph = tree.build_lineage(
                        self._root_id, include_non_members=include_non_members
                    )

        self._graph = graph
        self._tree.set_graph(graph, focus_id=self._root_id, selected_id=selected)

        # Keep the detail panel in sync with the (re)rendered selection.
        if selected is not None and selected in graph.nodes:
            self._selected_id = selected
            self._load_detail(selected)
        else:
            self._clear_detail()

    # ------------------------------------------------------------------
    # Selection / detail
    # ------------------------------------------------------------------

    def _on_node_selected(self, person_id: int) -> None:
        self._selected_id = int(person_id)
        # Recolour the tree to mark this person's ancestors/descendants.
        self._tree.set_selected(self._selected_id)
        self._load_detail(self._selected_id)

    def _on_node_activated(self, person_id: int) -> None:
        self._selected_id = int(person_id)
        self._on_edit_relationships()

    def _load_detail(self, person_id: int) -> None:
        gender_labels = {
            "male": t("gender_male"),
            "female": t("gender_female"),
        }
        with session_scope() as session:
            person = session.get(Person, person_id)
            if person is None:
                self._clear_detail()
                return
            name = person.name
            gender = gender_labels.get(person.gender or "", "")
            nickname = person.nickname or ""
            birth = person.birth_date or ""
            birth_place = person.birth_place or ""
            death = person.death_date or ""
            death_place = person.death_place or ""
            code = person.family_code or ""
            notes = person.notes or ""
            is_member = bool(person.is_family_tree_member)

            relation = ""
            if self._root_id is not None and self._root_id != person_id:
                kind = FamilyTreeService(session).relationship_path(
                    self._root_id, person_id
                ).kind
                relation = t(f"ft_kind_{kind}")
            elif self._root_id == person_id:
                relation = t("ft_kind_self")

        # Thumbnail comes from the rendered graph node (explicit thumbnail or
        # best face crop fallback).
        node = self._graph.node(person_id) if self._graph is not None else None
        thumb = node.thumbnail_path if node is not None else None

        self._name.setText(name)
        self._not_member.setVisible(not is_member)
        self._relation_lbl.setText(relation)
        self._gender_lbl.setText(gender)
        self._nickname_lbl.setText(nickname)
        self._birth_lbl.setText(birth)
        self._birth_place_lbl.setText(birth_place)
        self._death_lbl.setText(death)
        self._death_place_lbl.setText(death_place)
        self._code_lbl.setText(code)
        self._notes_lbl.setText(notes)
        # Relation-to-root only applies when a focus person is set.
        self._relation_row.setVisible(self._root_id is not None)
        self._relation_lbl.setVisible(self._root_id is not None)
        self._edit_btn.setEnabled(True)
        self._edit_person_btn.setEnabled(True)
        self._images_btn.setEnabled(True)
        self._add_spouse_btn.setEnabled(True)
        self._add_child_btn.setEnabled(True)
        self._hint.setVisible(False)
        self._set_thumbnail(thumb)

    def _clear_detail(self) -> None:
        self._selected_id = None
        for lbl in (
            self._name, self._relation_lbl, self._gender_lbl, self._nickname_lbl,
            self._birth_lbl, self._birth_place_lbl, self._death_lbl,
            self._death_place_lbl, self._code_lbl, self._notes_lbl,
        ):
            lbl.clear()
        self._not_member.setVisible(False)
        self._thumb.clear()
        self._edit_btn.setEnabled(False)
        self._edit_person_btn.setEnabled(False)
        self._images_btn.setEnabled(False)
        self._add_spouse_btn.setEnabled(False)
        self._add_child_btn.setEnabled(False)
        self._hint.setVisible(True)

    def _set_thumbnail(self, path: Optional[str]) -> None:
        if path and Path(path).exists():
            from app.utils.image_utils import load_pixmap_exif

            pixmap = load_pixmap_exif(path)
            if not pixmap.isNull():
                self._thumb.setPixmap(
                    pixmap.scaled(
                        self._thumb.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                )
                return
        self._thumb.clear()
        self._thumb.setText("—")

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------

    def _on_edit_relationships(self) -> None:
        if self._selected_id is None:
            return
        dialog = FamilyTreeEditorDialog(self._selected_id, parent=self)
        dialog.relationships_changed.connect(self._on_relationships_changed)
        dialog.exec()

    def _on_quick_add(self, kind: str) -> None:
        """Detail-page shortcut for adding a spouse / child to the selection."""
        if self._selected_id is None:
            return
        if add_relative(self._selected_id, kind, self):
            self._on_relationships_changed()

    def _on_edit_person(self) -> None:
        """Open the full person editor for the selected person and save edits."""
        if self._selected_id is None:
            return
        person_id = self._selected_id
        from app.services.person_service import PersonService
        from app.ui.dialogs.person_info_dialog import PersonInfoDialog

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
                    external_family_code=dlg.external_family_code(),
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
            QMessageBox.critical(self, t("ft_edit_person"), str(exc))
            return
        self._on_relationships_changed()

    def _on_show_images(self) -> None:
        """Show all images the selected person appears in."""
        if self._selected_id is None:
            return
        with session_scope() as session:
            person = session.get(Person, self._selected_id)
            name = person.name if person is not None else ""
        from app.ui.dialogs.suggestion_viewer import FaceGalleryDialog

        FaceGalleryDialog(self._selected_id, name, parent=self).exec()

    def _on_relationships_changed(self) -> None:
        # refresh() re-renders with the current selection highlighted and
        # reloads the detail panel for the still-selected person.
        keep = self._selected_id
        self.refresh()
        if keep is not None:
            self._selected_id = keep
            self._load_detail(keep)
        self.person_data_changed.emit()
