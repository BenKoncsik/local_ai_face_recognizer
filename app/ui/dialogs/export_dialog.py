"""Export dialog — CSV, JSON, image, collage, and image metadata export."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Set

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.db.database import session_scope
from app.services.export_service import ExportService
from app.services.image_metadata_export_service import (
    ALL_FIELDS,
    FIELD_DATE,
    FIELD_FILENAME,
    FIELD_GPS,
    FIELD_LOCATION,
    FIELD_PERSONS,
    FIELD_RELPATH,
    PERSON_MODE_COLS,
    PERSON_MODE_LIST,
    ImageMetadataExportService,
)
from app.ui.i18n import t


class ExportDialog(QDialog):
    """Modal export window with CSV, JSON, image and collage export options."""

    def __init__(
        self,
        current_person_id: Optional[int] = None,
        current_person_name: Optional[str] = None,
        current_image_id: Optional[int] = None,
        on_collage_import: Optional[Callable[[], None]] = None,
        on_collage_html_export: Optional[Callable[[], None]] = None,
        on_project_export: Optional[Callable[[], None]] = None,
        on_project_import: Optional[Callable[[], None]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._person_id = current_person_id
        self._person_name = current_person_name
        self._image_id = current_image_id
        self._on_collage_import_cb = on_collage_import
        self._on_collage_html_export_cb = on_collage_html_export
        self._on_project_export_cb = on_project_export
        self._on_project_import_cb = on_project_import
        self.setWindowTitle(t("export_title"))
        self.setMinimumWidth(500)
        self.setMinimumHeight(300)
        self.resize(520, 680)
        self._build_ui()

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # Outer layout: scroll area + sticky Close button
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # ── Scroll area ───────────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(12)
        layout.setContentsMargins(2, 2, 2, 2)

        # --- Scope ---
        scope_box = QGroupBox(t("export_scope"))
        scope_layout = QVBoxLayout(scope_box)
        self._all_radio = QRadioButton(t("export_all_persons"))
        self._cur_radio = QRadioButton(
            t("export_selected_person", name=self._person_name or "—")
        )
        self._cur_radio.setEnabled(self._person_id is not None)
        self._all_radio.setChecked(True)
        scope_layout.addWidget(self._all_radio)
        scope_layout.addWidget(self._cur_radio)
        layout.addWidget(scope_box)

        # --- Full project package (.facepack) ---
        pkg_box = QGroupBox(t("pkg_group"))
        pkg_layout = QVBoxLayout(pkg_box)
        pkg_export_desc = QLabel(t("pkg_export_desc"))
        pkg_export_desc.setWordWrap(True)
        pkg_export_desc.setStyleSheet("color: #aaa; font-size: 11px;")
        pkg_layout.addWidget(pkg_export_desc)
        self._project_export_btn = QPushButton(f"📦  {t('pkg_export_btn')}")
        self._project_export_btn.setEnabled(self._on_project_export_cb is not None)
        self._project_export_btn.clicked.connect(self._on_project_export_clicked)
        pkg_layout.addWidget(self._project_export_btn)

        pkg_import_desc = QLabel(t("pkg_import_desc"))
        pkg_import_desc.setWordWrap(True)
        pkg_import_desc.setStyleSheet("color: #aaa; font-size: 11px;")
        pkg_layout.addWidget(pkg_import_desc)
        self._project_import_btn = QPushButton(f"📂  {t('pkg_import_btn')}")
        self._project_import_btn.setEnabled(self._on_project_import_cb is not None)
        self._project_import_btn.clicked.connect(self._on_project_import_clicked)
        pkg_layout.addWidget(self._project_import_btn)
        layout.addWidget(pkg_box)

        # --- CSV ---
        csv_box = QGroupBox(t("csv_export"))
        csv_layout = QVBoxLayout(csv_box)
        csv_desc = QLabel(t("export_csv_desc"))
        csv_desc.setWordWrap(True)
        csv_desc.setStyleSheet("color: #aaa; font-size: 11px;")
        csv_layout.addWidget(csv_desc)
        self._csv_btn = QPushButton(f"💾  {t('export_save_csv')}")
        self._csv_btn.clicked.connect(self._on_export_csv)
        csv_layout.addWidget(self._csv_btn)
        layout.addWidget(csv_box)

        # --- JSON ---
        json_box = QGroupBox(t("json_export"))
        json_layout = QVBoxLayout(json_box)
        json_desc = QLabel(t("export_json_desc"))
        json_desc.setWordWrap(True)
        json_desc.setStyleSheet("color: #aaa; font-size: 11px;")
        json_layout.addWidget(json_desc)
        self._json_btn = QPushButton(f"💾  {t('export_save_json')}")
        self._json_btn.clicked.connect(self._on_export_json)
        json_layout.addWidget(self._json_btn)
        layout.addWidget(json_box)

        # --- Images ---
        img_box = QGroupBox(t("export_images"))
        img_layout = QVBoxLayout(img_box)
        img_desc = QLabel(t("export_images_desc"))
        img_desc.setWordWrap(True)
        img_desc.setStyleSheet("color: #aaa; font-size: 11px;")
        img_layout.addWidget(img_desc)
        self._images_btn = QPushButton(f"📁  {t('export_choose_folder')}")
        self._images_btn.setEnabled(self._person_id is not None)
        if self._person_id is None:
            self._images_btn.setToolTip(t("export_need_person_tip"))
        self._images_btn.clicked.connect(self._on_export_images)
        img_layout.addWidget(self._images_btn)
        layout.addWidget(img_box)

        # --- HTML gallery ---
        html_box = QGroupBox(t("export_html_group"))
        html_layout = QVBoxLayout(html_box)
        html_desc = QLabel(t("export_html_desc"))
        html_desc.setWordWrap(True)
        html_desc.setStyleSheet("color: #aaa; font-size: 11px;")
        html_layout.addWidget(html_desc)
        self._html_btn = QPushButton(f"🌐  {t('export_generate_html')}")
        self._html_btn.clicked.connect(self._on_export_html)
        html_layout.addWidget(self._html_btn)
        layout.addWidget(html_box)

        # --- Collage Import ---
        col_import_box = QGroupBox(t("export_collage_import_group"))
        col_import_layout = QVBoxLayout(col_import_box)
        col_import_desc = QLabel(t("export_collage_import_desc"))
        col_import_desc.setWordWrap(True)
        col_import_desc.setStyleSheet("color: #aaa; font-size: 11px;")
        col_import_layout.addWidget(col_import_desc)
        self._collage_import_btn = QPushButton(f"📂  {t('export_collage_import_btn')}")
        self._collage_import_btn.setEnabled(self._on_collage_import_cb is not None)
        self._collage_import_btn.clicked.connect(self._on_collage_import_clicked)
        col_import_layout.addWidget(self._collage_import_btn)
        layout.addWidget(col_import_box)

        # --- Collage HTML Gallery ---
        col_html_box = QGroupBox(t("export_collage_html_group"))
        col_html_layout = QVBoxLayout(col_html_box)
        col_html_desc = QLabel(t("export_collage_html_desc"))
        col_html_desc.setWordWrap(True)
        col_html_desc.setStyleSheet("color: #aaa; font-size: 11px;")
        col_html_layout.addWidget(col_html_desc)
        self._collage_html_export_btn = QPushButton(f"🌍  {t('export_collage_html_btn')}")
        self._collage_html_export_btn.setEnabled(self._on_collage_html_export_cb is not None)
        self._collage_html_export_btn.clicked.connect(self._on_collage_html_clicked)
        col_html_layout.addWidget(self._collage_html_export_btn)
        layout.addWidget(col_html_box)

        # --- Image Metadata Export ---
        meta_box = QGroupBox(t("export_metadata_group"))
        meta_layout = QVBoxLayout(meta_box)

        meta_desc = QLabel(t("export_metadata_desc"))
        meta_desc.setWordWrap(True)
        meta_desc.setStyleSheet("color: #aaa; font-size: 11px;")
        meta_layout.addWidget(meta_desc)

        # Format
        fmt_label = QLabel(t("export_metadata_format"))
        meta_layout.addWidget(fmt_label)
        fmt_row = QHBoxLayout()
        self._meta_fmt_csv = QRadioButton(t("export_metadata_csv"))
        self._meta_fmt_xlsx = QRadioButton(t("export_metadata_xlsx"))
        self._meta_fmt_csv.setChecked(True)
        self._meta_fmt_group = QButtonGroup(self)
        self._meta_fmt_group.addButton(self._meta_fmt_csv)
        self._meta_fmt_group.addButton(self._meta_fmt_xlsx)
        fmt_row.addWidget(self._meta_fmt_csv)
        fmt_row.addWidget(self._meta_fmt_xlsx)
        fmt_row.addStretch()
        meta_layout.addLayout(fmt_row)

        # Person mode
        pmode_label = QLabel(t("export_metadata_person_mode"))
        meta_layout.addWidget(pmode_label)
        pmode_row = QHBoxLayout()
        self._meta_persons_list = QRadioButton(t("export_metadata_persons_list"))
        self._meta_persons_cols = QRadioButton(t("export_metadata_persons_cols"))
        self._meta_persons_list.setChecked(True)
        self._meta_pmode_group = QButtonGroup(self)
        self._meta_pmode_group.addButton(self._meta_persons_list)
        self._meta_pmode_group.addButton(self._meta_persons_cols)
        pmode_row.addWidget(self._meta_persons_list)
        pmode_row.addWidget(self._meta_persons_cols)
        pmode_row.addStretch()
        meta_layout.addLayout(pmode_row)

        # Field checkboxes
        fields_label = QLabel(t("export_metadata_fields"))
        meta_layout.addWidget(fields_label)
        fields_grid = QHBoxLayout()
        left_col = QVBoxLayout()
        right_col = QVBoxLayout()

        self._cb_filename = QCheckBox(t("export_metadata_filename"))
        self._cb_relpath = QCheckBox(t("export_metadata_relpath"))
        self._cb_persons = QCheckBox(t("export_metadata_persons"))
        self._cb_date = QCheckBox(t("export_metadata_date"))
        self._cb_location = QCheckBox(t("export_metadata_location"))
        self._cb_gps = QCheckBox(t("export_metadata_gps"))

        for cb in (
            self._cb_filename,
            self._cb_relpath,
            self._cb_persons,
            self._cb_date,
            self._cb_location,
            self._cb_gps,
        ):
            cb.setChecked(True)

        left_col.addWidget(self._cb_filename)
        left_col.addWidget(self._cb_relpath)
        left_col.addWidget(self._cb_persons)
        right_col.addWidget(self._cb_date)
        right_col.addWidget(self._cb_location)
        right_col.addWidget(self._cb_gps)

        fields_grid.addLayout(left_col)
        fields_grid.addLayout(right_col)
        fields_grid.addStretch()
        meta_layout.addLayout(fields_grid)

        self._meta_export_btn = QPushButton(f"💾  {t('export_metadata_btn')}")
        self._meta_export_btn.clicked.connect(self._on_export_metadata)
        meta_layout.addWidget(self._meta_export_btn)

        layout.addWidget(meta_box)

        # --- Embed persons into image files / sidecar JSON ---
        fmeta_box = QGroupBox(t("fmeta_group"))
        fmeta_layout = QVBoxLayout(fmeta_box)

        fmeta_desc = QLabel(t("fmeta_desc"))
        fmeta_desc.setWordWrap(True)
        fmeta_desc.setStyleSheet("color: #aaa; font-size: 11px;")
        fmeta_layout.addWidget(fmeta_desc)

        fmeta_warn = QLabel("⚠️  " + t("fmeta_warning"))
        fmeta_warn.setWordWrap(True)
        fmeta_warn.setStyleSheet("color: #d08b00; font-size: 11px;")
        fmeta_layout.addWidget(fmeta_warn)

        self._fmeta_name = QCheckBox(t("fmeta_opt_name"))
        self._fmeta_name.setChecked(True)
        self._fmeta_notes = QCheckBox(t("fmeta_opt_notes"))
        self._fmeta_notes.setChecked(True)
        self._fmeta_sidecar = QCheckBox(t("fmeta_opt_sidecar"))
        for cb in (self._fmeta_name, self._fmeta_notes, self._fmeta_sidecar):
            fmeta_layout.addWidget(cb)

        self._fmeta_current_btn = QPushButton(t("fmeta_btn_current"))
        self._fmeta_current_btn.setEnabled(self._image_id is not None)
        if self._image_id is None:
            self._fmeta_current_btn.setToolTip(t("fmeta_no_current"))
        self._fmeta_current_btn.clicked.connect(self._on_embed_current)
        fmeta_layout.addWidget(self._fmeta_current_btn)

        self._fmeta_all_btn = QPushButton(t("fmeta_btn_all"))
        self._fmeta_all_btn.clicked.connect(self._on_embed_all)
        fmeta_layout.addWidget(self._fmeta_all_btn)

        layout.addWidget(fmeta_box)

        layout.addStretch()
        scroll.setWidget(inner)
        outer.addWidget(scroll, stretch=1)

        # --- Close — always visible outside the scroll area ---
        close_btn = QPushButton(t("close"))
        close_btn.clicked.connect(self.accept)
        outer.addWidget(close_btn, alignment=Qt.AlignRight)

    # ------------------------------------------------------------------

    def _scope_person_id(self) -> Optional[int]:
        return self._person_id if self._cur_radio.isChecked() else None

    def _on_export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, t("export_save_csv"), "export.csv", "CSV Files (*.csv)"
        )
        if not path:
            return
        with session_scope() as session:
            out = ExportService(session).export_csv(
                target_path=path, person_id=self._scope_person_id()
            )
        QMessageBox.information(self, t("csv_exported"), t("export_csv_saved", path=out))

    def _on_export_json(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, t("export_save_json"), "export.json", "JSON Files (*.json)"
        )
        if not path:
            return
        with session_scope() as session:
            out = ExportService(session).export_json(
                target_path=path, person_id=self._scope_person_id()
            )
        QMessageBox.information(self, t("json_exported"), t("json_saved", path=out))

    def _on_export_html(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, t("html_gallery_folder"), str(Path.home())
        )
        if not folder:
            return
        import subprocess, sys
        with session_scope() as session:
            out = ExportService(session).export_html(
                target_dir=folder, person_id=self._scope_person_id()
            )
        index = out / "index.html"
        reply = QMessageBox.information(
            self,
            t("html_gallery_done"),
            t("html_gallery_open", path=index),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(index)])
            elif sys.platform == "win32":
                subprocess.Popen(["start", str(index)], shell=True)
            else:
                subprocess.Popen(["xdg-open", str(index)])

    def _on_export_images(self) -> None:
        if self._person_id is None:
            return
        folder = QFileDialog.getExistingDirectory(
            self, t("export_choose_folder"), str(Path.home())
        )
        if not folder:
            return
        with session_scope() as session:
            n = ExportService(session).export_person_images(
                person_id=self._person_id, target_dir=folder
            )
        QMessageBox.information(
            self, t("images_exported"), t("files_copied", n=n, folder=folder)
        )

    def _on_export_metadata(self) -> None:
        fields = self._selected_fields()
        if not fields:
            QMessageBox.warning(
                self,
                t("export_metadata_no_fields_title"),
                t("export_metadata_no_fields"),
            )
            return

        use_xlsx = self._meta_fmt_xlsx.isChecked()
        if use_xlsx:
            path, _ = QFileDialog.getSaveFileName(
                self, t("export_metadata_group"), "image_metadata.xlsx",
                "Excel Files (*.xlsx)"
            )
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, t("export_metadata_group"), "image_metadata.csv",
                "CSV Files (*.csv)"
            )
        if not path:
            return

        person_mode = PERSON_MODE_COLS if self._meta_persons_cols.isChecked() else PERSON_MODE_LIST

        try:
            with session_scope() as session:
                svc = ImageMetadataExportService(session)
                if use_xlsx:
                    out = svc.export_xlsx(path, fields, person_mode)
                else:
                    out = svc.export_csv(path, fields, person_mode)
        except ImportError as exc:
            QMessageBox.critical(self, t("export_error"), str(exc))
            return

        QMessageBox.information(
            self, t("export_metadata_done"), t("export_metadata_saved", path=out)
        )

    def _selected_fields(self) -> Set[str]:
        fields: Set[str] = set()
        if self._cb_filename.isChecked():
            fields.add(FIELD_FILENAME)
        if self._cb_relpath.isChecked():
            fields.add(FIELD_RELPATH)
        if self._cb_persons.isChecked():
            fields.add(FIELD_PERSONS)
        if self._cb_date.isChecked():
            fields.add(FIELD_DATE)
        if self._cb_location.isChecked():
            fields.add(FIELD_LOCATION)
        if self._cb_gps.isChecked():
            fields.add(FIELD_GPS)
        return fields

    # ------------------------------------------------------------------
    # Embed persons into image files / sidecar JSON
    # ------------------------------------------------------------------

    def _fmeta_options(self):
        from app.services.face_metadata_export_service import FaceMetadataExportOptions

        return FaceMetadataExportOptions(
            include_person_name=self._fmeta_name.isChecked(),
            include_notes=self._fmeta_notes.isChecked(),
            prefer_sidecar_only=self._fmeta_sidecar.isChecked(),
        )

    def _confirm_embed(self) -> bool:
        reply = QMessageBox.warning(
            self,
            t("fmeta_confirm_title"),
            t("fmeta_warning"),
            QMessageBox.Ok | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        return reply == QMessageBox.Ok

    def _on_embed_current(self) -> None:
        if self._image_id is None or not self._confirm_embed():
            return
        from app.services.face_metadata_export_service import (
            FaceMetadataExportService,
            FaceMetadataExportSummary,
        )

        with session_scope() as session:
            result = FaceMetadataExportService(session).export_image(
                self._image_id, self._fmeta_options()
            )
        summary = FaceMetadataExportSummary(results=[result])
        self._show_embed_summary(summary)

    def _on_embed_all(self) -> None:
        if not self._confirm_embed():
            return
        from app.services.face_metadata_export_service import FaceMetadataExportService

        progress = QProgressDialog(t("fmeta_group"), t("close"), 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        def cb(done: int, total: int) -> None:
            progress.setMaximum(max(total, 1))
            progress.setValue(done)

        with session_scope() as session:
            summary = FaceMetadataExportService(session).export_all(
                self._fmeta_options(), progress_cb=cb
            )
        progress.close()
        self._show_embed_summary(summary)

    def _show_embed_summary(self, summary) -> None:
        msg = t(
            "fmeta_summary",
            total=summary.total,
            embedded=summary.embedded_count,
            sidecar=summary.sidecar_count,
            skipped=summary.skipped_count,
            failed=summary.failed_count,
        )
        errors = summary.errors
        if errors:
            msg += "\n\n" + "\n".join(errors[:10])
            if len(errors) > 10:
                msg += f"\n… (+{len(errors) - 10})"
        QMessageBox.information(self, t("fmeta_done_title"), msg)

    def _on_collage_import_clicked(self) -> None:
        if self._on_collage_import_cb is None:
            return
        self.accept()
        self._on_collage_import_cb()

    def _on_collage_html_clicked(self) -> None:
        if self._on_collage_html_export_cb is None:
            return
        self.accept()
        self._on_collage_html_export_cb()

    def _on_project_export_clicked(self) -> None:
        if self._on_project_export_cb is None:
            return
        self.accept()
        self._on_project_export_cb()

    def _on_project_import_clicked(self) -> None:
        if self._on_project_import_cb is None:
            return
        self.accept()
        self._on_project_import_cb()
