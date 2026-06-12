"""Graphical editor for user-defined family code schemes.

Lets a non-programmer describe their own family coding system: the root
persons (first letters), the marker letters for special relationships and the
optional extra notations.  Schemes can be created, duplicated, deleted,
exported/imported as JSON files, and exactly one of them is active at a time.
A live tester shows immediately what a typed code means under the scheme
being edited.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
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
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.services.family_code_interpreter import describe_family_code
from app.services.family_code_schemes import (
    BUILTIN_SCHEME_ID,
    FamilyCodeScheme,
    FamilyCodeSchemeStore,
    SchemeRoot,
    new_scheme_id,
    scheme_example_codes,
    scheme_problems,
)
from app.ui.i18n import t

log = logging.getLogger(__name__)

_ROLE_ID = Qt.UserRole + 1

# Marker rows of the editor: (kind, label key, hint key, guide key)
_MARKER_KINDS = (
    ("ancestor", "fcs_marker_ancestor", "fcs_marker_ancestor_hint", "fcs_guide_marker_anc"),
    ("sibling", "fcs_marker_sibling", "fcs_marker_sibling_hint", "fcs_guide_marker_sib"),
    ("spouse", "fcs_marker_spouse", "fcs_marker_spouse_hint", "fcs_guide_marker_spo"),
    ("friend", "fcs_marker_friend", "fcs_marker_friend_hint", "fcs_guide_marker_fri"),
)


class FamilyCodeSchemeDialog(QDialog):
    """List + form editor for family code schemes with a live code tester."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        store: Optional[FamilyCodeSchemeStore] = None,
    ) -> None:
        super().__init__(parent)
        self._store = store or FamilyCodeSchemeStore()
        self._current: Optional[FamilyCodeScheme] = None
        self._dirty = False
        self._loading = False

        self.setWindowTitle(t("fcs_title"))
        self.setMinimumSize(900, 640)
        self.resize(980, 720)
        self._build_ui()
        self._reload_list(select_id=self._store.active_scheme_id())

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        intro = QLabel(t("fcs_intro"))
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa;")
        outer.addWidget(intro)

        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter, stretch=1)

        # --- Left: scheme list + actions ---
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        left_layout.addWidget(QLabel(t("fcs_list_label")))
        self._list = QListWidget()
        self._list.currentItemChanged.connect(self._on_selection_changed)
        left_layout.addWidget(self._list, stretch=1)

        btn_grid = QGridLayout()
        btn_grid.setSpacing(4)
        self._new_btn = QPushButton(t("fcs_btn_new"))
        self._new_btn.clicked.connect(self._on_new)
        btn_grid.addWidget(self._new_btn, 0, 0)
        self._dup_btn = QPushButton(t("fcs_btn_duplicate"))
        self._dup_btn.clicked.connect(self._on_duplicate)
        btn_grid.addWidget(self._dup_btn, 0, 1)
        self._activate_btn = QPushButton(t("fcs_btn_activate"))
        self._activate_btn.clicked.connect(self._on_activate)
        btn_grid.addWidget(self._activate_btn, 1, 0)
        self._delete_btn = QPushButton(t("fcs_btn_delete"))
        self._delete_btn.clicked.connect(self._on_delete)
        btn_grid.addWidget(self._delete_btn, 1, 1)
        self._import_btn = QPushButton(t("fcs_btn_import"))
        self._import_btn.clicked.connect(self._on_import)
        btn_grid.addWidget(self._import_btn, 2, 0)
        self._export_btn = QPushButton(t("fcs_btn_export"))
        self._export_btn.clicked.connect(self._on_export)
        btn_grid.addWidget(self._export_btn, 2, 1)
        left_layout.addLayout(btn_grid)

        splitter.addWidget(left)

        # --- Right: editor form in a scroll area ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        content = QWidget()
        self._form_layout = QVBoxLayout(content)
        self._form_layout.setContentsMargins(8, 0, 8, 8)
        self._form_layout.setSpacing(8)
        scroll.setWidget(content)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 660])

        # Built-in notice
        self._builtin_note = QLabel(t("fcs_builtin_note"))
        self._builtin_note.setWordWrap(True)
        self._builtin_note.setStyleSheet(
            "color: #e0b050; border: 1px solid #806020; border-radius: 4px;"
            "padding: 6px; background: rgba(128, 96, 32, 0.15);"
        )
        self._form_layout.addWidget(self._builtin_note)

        # Name + description
        meta_form = QFormLayout()
        meta_form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self._name_edit = QLineEdit()
        self._name_edit.textEdited.connect(self._on_form_changed)
        meta_form.addRow(t("fcs_name_label"), self._name_edit)
        self._desc_edit = QTextEdit()
        self._desc_edit.setPlaceholderText(t("fcs_desc_placeholder"))
        self._desc_edit.setFixedHeight(64)
        self._desc_edit.textChanged.connect(self._on_form_changed)
        meta_form.addRow(t("fcs_desc_label"), self._desc_edit)
        self._form_layout.addLayout(meta_form)

        # Roots
        roots_group = QGroupBox(t("fcs_roots_group"))
        roots_layout = QVBoxLayout(roots_group)
        roots_hint = QLabel(t("fcs_roots_hint"))
        roots_hint.setWordWrap(True)
        roots_hint.setStyleSheet("color: #999; font-size: 12px;")
        roots_layout.addWidget(roots_hint)
        self._roots_table = QTableWidget(0, 3)
        self._roots_table.setHorizontalHeaderLabels(
            [t("fcs_root_col_letter"), t("fcs_root_col_name"), t("fcs_root_col_note")]
        )
        header = self._roots_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        self._roots_table.verticalHeader().setVisible(False)
        self._roots_table.setMinimumHeight(120)
        self._roots_table.setMaximumHeight(180)
        self._roots_table.itemChanged.connect(self._on_root_item_changed)
        roots_layout.addWidget(self._roots_table)
        roots_btns = QHBoxLayout()
        self._add_root_btn = QPushButton(t("fcs_btn_add_root"))
        self._add_root_btn.clicked.connect(self._on_add_root)
        roots_btns.addWidget(self._add_root_btn)
        self._remove_root_btn = QPushButton(t("fcs_btn_remove_root"))
        self._remove_root_btn.clicked.connect(self._on_remove_root)
        roots_btns.addWidget(self._remove_root_btn)
        roots_btns.addStretch()
        roots_layout.addLayout(roots_btns)
        self._form_layout.addWidget(roots_group)

        # Markers
        markers_group = QGroupBox(t("fcs_markers_group"))
        markers_layout = QVBoxLayout(markers_group)
        markers_hint = QLabel(t("fcs_markers_hint"))
        markers_hint.setWordWrap(True)
        markers_hint.setStyleSheet("color: #999; font-size: 12px;")
        markers_layout.addWidget(markers_hint)
        marker_grid = QGridLayout()
        marker_grid.setHorizontalSpacing(10)
        marker_grid.setVerticalSpacing(4)
        marker_grid.setColumnStretch(2, 1)
        self._marker_rows: dict[str, tuple[QCheckBox, QLineEdit, QLabel, QLabel]] = {}
        for row, (kind, label_key, hint_key, _guide_key) in enumerate(_MARKER_KINDS):
            check = QCheckBox(t(label_key))
            check.toggled.connect(self._on_marker_toggled)
            letter_edit = QLineEdit()
            letter_edit.setMaxLength(1)
            letter_edit.setFixedWidth(40)
            letter_edit.setAlignment(Qt.AlignCenter)
            letter_edit.textEdited.connect(self._on_marker_letter_edited)
            example_lbl = QLabel()
            example_lbl.setStyleSheet("color: #8fbf8f; font-size: 12px;")
            example_lbl.setWordWrap(True)
            hint_lbl = QLabel(t(hint_key))
            hint_lbl.setWordWrap(True)
            hint_lbl.setStyleSheet("color: #888; font-size: 11px;")
            marker_grid.addWidget(check, row * 2, 0)
            marker_grid.addWidget(letter_edit, row * 2, 1)
            marker_grid.addWidget(example_lbl, row * 2, 2)
            marker_grid.addWidget(hint_lbl, row * 2 + 1, 2)
            self._marker_rows[kind] = (check, letter_edit, example_lbl, hint_lbl)
        markers_layout.addLayout(marker_grid)
        self._form_layout.addWidget(markers_group)

        # Extra notations
        options_group = QGroupBox(t("fcs_options_group"))
        options_layout = QVBoxLayout(options_group)
        self._opt_unlisted = QCheckBox(t("fcs_opt_unlisted_roots"))
        self._opt_multi = QCheckBox(t("fcs_opt_multi"))
        self._opt_ranges = QCheckBox(t("fcs_opt_ranges"))
        self._opt_braces = QCheckBox(t("fcs_opt_braces"))
        self._opt_external = QCheckBox(t("fcs_opt_external"))
        for box in (
            self._opt_unlisted,
            self._opt_multi,
            self._opt_ranges,
            self._opt_braces,
            self._opt_external,
        ):
            box.toggled.connect(self._on_form_changed)
            options_layout.addWidget(box)
        self._form_layout.addWidget(options_group)

        # Step-by-step guide (collapsible)
        self._guide_btn = QToolButton()
        self._guide_btn.setText(t("fcs_guide_btn"))
        self._guide_btn.setCheckable(True)
        self._guide_btn.setChecked(False)
        self._guide_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._guide_btn.setArrowType(Qt.RightArrow)
        self._guide_btn.toggled.connect(self._on_guide_toggled)
        self._form_layout.addWidget(self._guide_btn)
        self._guide_lbl = QLabel()
        self._guide_lbl.setWordWrap(True)
        self._guide_lbl.setTextFormat(Qt.RichText)
        self._guide_lbl.setStyleSheet(
            "color: #bbb; font-size: 12px; border: 1px solid #444;"
            "border-radius: 4px; padding: 8px; background: rgba(64,64,64,0.2);"
        )
        self._guide_lbl.setVisible(False)
        self._form_layout.addWidget(self._guide_lbl)

        # Live tester
        tester_group = QGroupBox(t("fcs_tester_group"))
        tester_layout = QVBoxLayout(tester_group)
        tester_hint = QLabel(t("fcs_tester_hint"))
        tester_hint.setWordWrap(True)
        tester_hint.setStyleSheet("color: #999; font-size: 12px;")
        tester_layout.addWidget(tester_hint)
        self._tester_edit = QLineEdit()
        self._tester_edit.textChanged.connect(self._refresh_tester)
        tester_layout.addWidget(self._tester_edit)
        self._tester_result = QLabel(t("fcs_tester_empty"))
        self._tester_result.setWordWrap(True)
        self._tester_result.setStyleSheet("color: #888;")
        tester_layout.addWidget(self._tester_result)
        self._examples_caption = QLabel(t("fcs_examples_label"))
        self._examples_caption.setStyleSheet("color: #999; font-size: 12px; margin-top: 4px;")
        tester_layout.addWidget(self._examples_caption)
        self._examples_lbl = QLabel()
        self._examples_lbl.setWordWrap(True)
        self._examples_lbl.setStyleSheet("color: #8fbf8f; font-size: 12px;")
        tester_layout.addWidget(self._examples_lbl)
        self._form_layout.addWidget(tester_group)

        self._form_layout.addStretch()

        # --- Bottom: save + close ---
        bottom = QHBoxLayout()
        self._save_btn = QPushButton(t("fcs_btn_save"))
        self._save_btn.clicked.connect(self._on_save)
        bottom.addWidget(self._save_btn)
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color: #8fbf8f;")
        bottom.addWidget(self._status_lbl)
        bottom.addStretch()
        close_btn = QPushButton(t("close"))
        close_btn.clicked.connect(self.close)
        bottom.addWidget(close_btn)
        outer.addLayout(bottom)

    # ------------------------------------------------------------------
    # List handling
    # ------------------------------------------------------------------

    def _reload_list(self, select_id: Optional[str] = None) -> None:
        """Refill the scheme list; selection triggers loading the form."""
        self._loading = True
        try:
            self._list.clear()
            active_id = self._store.active_scheme_id()
            for scheme in self._store.list_schemes():
                suffixes = []
                if scheme.scheme_id == active_id:
                    suffixes.append(t("fcs_active_suffix"))
                if scheme.is_builtin:
                    suffixes.append(t("fcs_builtin_suffix"))
                text = scheme.name
                if suffixes:
                    text += f"  ({', '.join(suffixes)})"
                item = QListWidgetItem(text)
                item.setData(_ROLE_ID, scheme.scheme_id)
                if scheme.scheme_id == active_id:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self._list.addItem(item)
        finally:
            self._loading = False
        # Select the requested scheme (or the first one); this loads the form.
        target = select_id or BUILTIN_SCHEME_ID
        for i in range(self._list.count()):
            if self._list.item(i).data(_ROLE_ID) == target:
                self._list.setCurrentRow(i)
                break
        else:
            if self._list.count():
                self._list.setCurrentRow(0)

    @Slot()
    def _on_selection_changed(self, current, previous) -> None:  # noqa: ANN001
        if self._loading:
            return
        if current is None:
            return
        if self._dirty and previous is not None and not self._confirm_discard():
            # Restore the previous selection without re-triggering this slot.
            self._loading = True
            try:
                self._list.setCurrentItem(previous)
            finally:
                self._loading = False
            return
        scheme_id = current.data(_ROLE_ID)
        scheme = self._store.get_scheme(scheme_id)
        if scheme is not None:
            self._load_scheme(scheme)

    def _confirm_discard(self) -> bool:
        name = self._current.name if self._current else ""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle(t("fcs_unsaved_title"))
        box.setText(t("fcs_unsaved_msg", name=name))
        discard = box.addButton(t("fcs_discard_btn"), QMessageBox.DestructiveRole)
        box.addButton(t("fcs_keep_btn"), QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is discard:
            self._dirty = False
            return True
        return False

    # ------------------------------------------------------------------
    # Form ↔ scheme
    # ------------------------------------------------------------------

    def _load_scheme(self, scheme: FamilyCodeScheme) -> None:
        self._loading = True
        try:
            self._current = scheme
            self._dirty = False
            self._status_lbl.setText("")
            editable = not scheme.is_builtin
            self._builtin_note.setVisible(not editable)

            self._name_edit.setText(scheme.name)
            self._desc_edit.setPlainText(scheme.description)

            self._roots_table.setRowCount(0)
            for root in scheme.roots:
                self._append_root_row(root)

            marker_values = {
                "ancestor": scheme.ancestor_letter,
                "sibling": scheme.sibling_letter,
                "spouse": scheme.spouse_letter,
                "friend": scheme.friend_letter,
            }
            defaults = {"ancestor": "F", "sibling": "T", "spouse": "H", "friend": "B"}
            for kind, (check, letter_edit, _example, _hint) in self._marker_rows.items():
                letter = marker_values[kind]
                check.setChecked(bool(letter))
                letter_edit.setText(letter or defaults[kind])
                letter_edit.setEnabled(editable and bool(letter))

            self._opt_unlisted.setChecked(scheme.allow_unlisted_roots)
            self._opt_multi.setChecked(scheme.allow_multi_codes)
            self._opt_ranges.setChecked(scheme.allow_ranges)
            self._opt_braces.setChecked(scheme.allow_braces)
            self._opt_external.setChecked(scheme.allow_external)

            for widget in (
                self._name_edit,
                self._desc_edit,
                self._roots_table,
                self._add_root_btn,
                self._remove_root_btn,
                self._opt_unlisted,
                self._opt_multi,
                self._opt_ranges,
                self._opt_braces,
                self._opt_external,
                self._save_btn,
            ):
                widget.setEnabled(editable)
            for kind, (check, _letter_edit, _example, _hint) in self._marker_rows.items():
                check.setEnabled(editable)

            self._delete_btn.setEnabled(editable)
            active_id = self._store.active_scheme_id()
            self._activate_btn.setEnabled(scheme.scheme_id != active_id)
        finally:
            self._loading = False
        self._refresh_dynamic_labels()

    def _append_root_row(self, root: SchemeRoot) -> None:
        row = self._roots_table.rowCount()
        self._roots_table.insertRow(row)
        letter_item = QTableWidgetItem(root.letter)
        letter_item.setTextAlignment(Qt.AlignCenter)
        self._roots_table.setItem(row, 0, letter_item)
        self._roots_table.setItem(row, 1, QTableWidgetItem(root.name))
        self._roots_table.setItem(row, 2, QTableWidgetItem(root.note))

    def _collect_scheme(self) -> FamilyCodeScheme:
        """Build a scheme object from the current form state."""
        base = self._current
        roots: list[SchemeRoot] = []
        for row in range(self._roots_table.rowCount()):
            letter_item = self._roots_table.item(row, 0)
            name_item = self._roots_table.item(row, 1)
            note_item = self._roots_table.item(row, 2)
            letter = (letter_item.text() if letter_item else "").strip().upper()[:1]
            if not letter:
                continue
            roots.append(
                SchemeRoot(
                    letter=letter,
                    name=(name_item.text() if name_item else "").strip(),
                    note=(note_item.text() if note_item else "").strip(),
                )
            )

        def marker(kind: str) -> str:
            check, letter_edit, _example, _hint = self._marker_rows[kind]
            if not check.isChecked():
                return ""
            return letter_edit.text().strip().upper()[:1]

        return FamilyCodeScheme(
            scheme_id=base.scheme_id if base else new_scheme_id(),
            name=self._name_edit.text().strip(),
            description=self._desc_edit.toPlainText().strip(),
            roots=roots,
            ancestor_letter=marker("ancestor"),
            sibling_letter=marker("sibling"),
            spouse_letter=marker("spouse"),
            friend_letter=marker("friend"),
            allow_unlisted_roots=self._opt_unlisted.isChecked(),
            allow_multi_codes=self._opt_multi.isChecked(),
            allow_ranges=self._opt_ranges.isChecked(),
            allow_braces=self._opt_braces.isChecked(),
            allow_external=self._opt_external.isChecked(),
            is_builtin=base.is_builtin if base else False,
        )

    # ------------------------------------------------------------------
    # Form change handlers
    # ------------------------------------------------------------------

    @Slot()
    def _on_form_changed(self) -> None:
        if self._loading or self._current is None:
            return
        self._current = self._collect_scheme()
        self._dirty = True
        self._status_lbl.setText("")
        self._refresh_dynamic_labels()

    @Slot()
    def _on_marker_toggled(self) -> None:
        if self._loading:
            return
        editable = self._current is not None and not self._current.is_builtin
        for kind, (check, letter_edit, _example, _hint) in self._marker_rows.items():
            letter_edit.setEnabled(editable and check.isChecked())
        self._on_form_changed()

    @Slot(str)
    def _on_marker_letter_edited(self, text: str) -> None:
        edit = self.sender()
        upper = text.strip().upper()
        if upper != text and isinstance(edit, QLineEdit):
            edit.blockSignals(True)
            edit.setText(upper)
            edit.blockSignals(False)
        self._on_form_changed()

    def _on_root_item_changed(self, item: QTableWidgetItem) -> None:
        if self._loading:
            return
        if item.column() == 0:
            upper = item.text().strip().upper()[:1]
            if upper != item.text():
                self._roots_table.blockSignals(True)
                item.setText(upper)
                self._roots_table.blockSignals(False)
        self._on_form_changed()

    @Slot()
    def _on_add_root(self) -> None:
        if self._current is None or self._current.is_builtin:
            return
        used = {
            (self._roots_table.item(r, 0).text() if self._roots_table.item(r, 0) else "")
            for r in range(self._roots_table.rowCount())
        }
        next_letter = next(
            (ch for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if ch not in used), "A"
        )
        self._append_root_row(SchemeRoot(letter=next_letter, name="", note=""))
        self._on_form_changed()

    @Slot()
    def _on_remove_root(self) -> None:
        if self._current is None or self._current.is_builtin:
            return
        row = self._roots_table.currentRow()
        if row >= 0:
            self._roots_table.removeRow(row)
            self._on_form_changed()

    @Slot(bool)
    def _on_guide_toggled(self, checked: bool) -> None:
        self._guide_btn.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self._guide_lbl.setVisible(checked)

    # ------------------------------------------------------------------
    # Dynamic labels (examples, guide, tester)
    # ------------------------------------------------------------------

    def _refresh_dynamic_labels(self) -> None:
        scheme = self._current
        if scheme is None:
            return
        root = next(iter(sorted(scheme.root_letters())), "C")

        # Per-marker live examples
        marker_examples = {
            "ancestor": f"{root}0{scheme.ancestor_letter}1" if scheme.ancestor_letter else "",
            "sibling": f"{root}00{scheme.sibling_letter}1" if scheme.sibling_letter else "",
            "spouse": f"{root}1{scheme.spouse_letter}2" if scheme.spouse_letter else "",
            "friend": f"{root}81{scheme.friend_letter}" if scheme.friend_letter else "",
        }
        for kind, (_check, _edit, example_lbl, _hint) in self._marker_rows.items():
            code = marker_examples[kind]
            example_lbl.setText(f"{code} = {self._safe_describe(code, scheme)}" if code else "—")

        # Example list under the tester
        examples = scheme_example_codes(scheme)
        self._examples_lbl.setText(
            "\n".join(f"{code} = {desc}" for code, desc in examples) or "—"
        )

        self._refresh_guide(scheme, root)
        self._refresh_tester()

    def _safe_describe(self, code: str, scheme: FamilyCodeScheme) -> str:
        try:
            return describe_family_code(code, scheme=scheme)
        except ValueError:
            return "—"

    def _refresh_guide(self, scheme: FamilyCodeScheme, root: str) -> None:
        root_name = scheme.root_names().get(root, root)
        lines = [t("fcs_guide_digits", root=root, rootname=root_name)]

        marker_letters = {
            "ancestor": scheme.ancestor_letter,
            "sibling": scheme.sibling_letter,
            "spouse": scheme.spouse_letter,
            "friend": scheme.friend_letter,
        }
        marker_example_codes = {
            "ancestor": f"{root}0{scheme.ancestor_letter}12" if scheme.ancestor_letter else "",
            "sibling": f"{root}00{scheme.sibling_letter}1" if scheme.sibling_letter else "",
            "spouse": f"{root}1{scheme.spouse_letter}2" if scheme.spouse_letter else "",
            "friend": f"{root}81{scheme.friend_letter}" if scheme.friend_letter else "",
        }
        marker_lines = []
        for kind, _label_key, _hint_key, guide_key in _MARKER_KINDS:
            letter = marker_letters[kind]
            if not letter:
                continue
            code = marker_example_codes[kind]
            example = f"{code} = {self._safe_describe(code, scheme)}"
            marker_lines.append(t(guide_key, letter=letter, example=example))
        if marker_lines:
            lines.append(t("fcs_guide_markers_title"))
            lines.extend(marker_lines)

        extra_lines = []
        if scheme.allow_multi_codes:
            extra_lines.append(f"&nbsp;&nbsp;• {t('fcs_opt_multi')}")
        if scheme.allow_ranges and scheme.friend_letter:
            extra_lines.append(f"&nbsp;&nbsp;• {t('fcs_opt_ranges')}")
        if scheme.allow_braces:
            extra_lines.append(f"&nbsp;&nbsp;• {t('fcs_opt_braces')}")
        if scheme.allow_external:
            extra_lines.append(f"&nbsp;&nbsp;• {t('fcs_opt_external')}")
        if extra_lines:
            lines.append(t("fcs_guide_extras_title"))
            lines.extend(extra_lines)

        self._guide_lbl.setText("<br>".join(lines))

    @Slot()
    def _refresh_tester(self) -> None:
        scheme = self._current
        if scheme is None:
            return
        examples = scheme_example_codes(scheme)
        sample = examples[1][0] if len(examples) > 1 else "C85"
        self._tester_edit.setPlaceholderText(t("placeholder_example", code=sample))

        text = self._tester_edit.text().strip()
        if not text:
            self._tester_result.setText(t("fcs_tester_empty"))
            self._tester_result.setStyleSheet("color: #888;")
            return
        try:
            desc = describe_family_code(text, scheme=scheme)
        except ValueError as exc:
            self._tester_result.setText(t("fcs_tester_bad", error=str(exc)))
            self._tester_result.setStyleSheet("color: #e08080;")
            return
        self._tester_result.setText(t("fcs_tester_ok", desc=desc))
        self._tester_result.setStyleSheet("color: #8fbf8f;")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _validated_for_save(self) -> Optional[FamilyCodeScheme]:
        scheme = self._collect_scheme()
        problems = scheme_problems(scheme)
        if problems:
            QMessageBox.warning(
                self,
                t("fcs_problems_title"),
                t("fcs_problems_msg", problems="\n".join(f"• {p}" for p in problems)),
            )
            return None
        return scheme

    @Slot()
    def _on_save(self) -> bool:
        if self._current is None or self._current.is_builtin:
            return False
        scheme = self._validated_for_save()
        if scheme is None:
            return False
        try:
            self._store.save_scheme(scheme)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, t("fcs_title"), str(exc))
            return False
        self._current = scheme
        self._dirty = False
        self._status_lbl.setText(t("fcs_saved_status"))
        self._reload_list(select_id=scheme.scheme_id)
        return True

    def _save_if_dirty(self) -> bool:
        """Persist pending edits before activate/export; True when safe to go on."""
        if self._current is None or self._current.is_builtin or not self._dirty:
            return True
        return bool(self._on_save())

    @Slot()
    def _on_new(self) -> None:
        if self._dirty and not self._confirm_discard():
            return
        scheme = FamilyCodeScheme(
            scheme_id=new_scheme_id(),
            name=t("fcs_new_scheme_name"),
            roots=[SchemeRoot(letter="A", name="", note="")],
        )
        try:
            self._store.save_scheme(scheme)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, t("fcs_title"), str(exc))
            return
        self._reload_list(select_id=scheme.scheme_id)

    @Slot()
    def _on_duplicate(self) -> None:
        if self._current is None:
            return
        if self._dirty and not self._confirm_discard():
            return
        try:
            copy = self._store.duplicate_scheme(self._current.scheme_id)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, t("fcs_title"), str(exc))
            return
        self._reload_list(select_id=copy.scheme_id)

    @Slot()
    def _on_delete(self) -> None:
        if self._current is None or self._current.is_builtin:
            return
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle(t("fcs_delete_title"))
        box.setText(t("fcs_delete_msg", name=self._current.name))
        delete = box.addButton(t("fcs_btn_delete"), QMessageBox.DestructiveRole)
        box.addButton(t("fcs_keep_btn"), QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is not delete:
            return
        try:
            self._store.delete_scheme(self._current.scheme_id)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, t("fcs_title"), str(exc))
            return
        self._dirty = False
        self._reload_list(select_id=self._store.active_scheme_id())

    @Slot()
    def _on_activate(self) -> None:
        if self._current is None:
            return
        if not self._save_if_dirty():
            return
        try:
            self._store.set_active_scheme_id(self._current.scheme_id)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, t("fcs_title"), str(exc))
            return
        self._reload_list(select_id=self._current.scheme_id)

    @Slot()
    def _on_import(self) -> None:
        if self._dirty and not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, t("fcs_import_title"), "", t("fcs_file_filter")
        )
        if not path:
            return
        try:
            scheme = self._store.import_scheme(path)
        except Exception as exc:  # noqa: BLE001 — bad JSON, IO errors, …
            log.exception("Family code scheme import failed: %s", path)
            QMessageBox.critical(self, t("fcs_import_error_title"), str(exc))
            return
        self._reload_list(select_id=scheme.scheme_id)

    @Slot()
    def _on_export(self) -> None:
        if self._current is None:
            return
        if not self._save_if_dirty():
            return
        safe_name = re.sub(r"[^\w\- ]+", "_", self._current.name).strip() or "scheme"
        path, _ = QFileDialog.getSaveFileName(
            self,
            t("fcs_export_title"),
            f"{safe_name}.json",
            t("fcs_file_filter"),
        )
        if not path:
            return
        try:
            self._store.export_scheme(self._current.scheme_id, path)
        except Exception as exc:  # noqa: BLE001
            log.exception("Family code scheme export failed: %s", path)
            QMessageBox.critical(self, t("fcs_title"), str(exc))
            return
        QMessageBox.information(
            self, t("fcs_export_title"), t("fcs_export_done_msg", path=path)
        )

    # ------------------------------------------------------------------
    # Closing
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: ANN001
        if self._dirty and not self._confirm_discard():
            event.ignore()
            return
        super().closeEvent(event)

    def reject(self) -> None:
        if self._dirty and not self._confirm_discard():
            return
        super().reject()
