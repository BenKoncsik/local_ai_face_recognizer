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
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.db.database import session_scope
from app.services.export_service import ExportService
from app.services.image_metadata_export_service import (
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
        on_deep_model_export: Optional[Callable[[], None]] = None,
        on_deep_model_import: Optional[Callable[[], None]] = None,
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
        self._on_deep_model_export_cb = on_deep_model_export
        self._on_deep_model_import_cb = on_deep_model_import
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

        # --- Scope (shared across all tabs) ---
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
        outer.addWidget(scope_box)

        # ── Tabs: General + Web ───────────────────────────────────────────
        tabs = QTabWidget()

        def _scrolled_tab():
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            inner = QWidget()
            lay = QVBoxLayout(inner)
            lay.setSpacing(12)
            lay.setContentsMargins(2, 2, 2, 2)
            scroll.setWidget(inner)
            return scroll, lay

        # ``layout`` targets the General tab so the existing box code is unchanged;
        # web-export boxes are routed to ``web_layout`` instead.
        gen_scroll, layout = _scrolled_tab()
        web_scroll, web_layout = _scrolled_tab()
        tabs.addTab(gen_scroll, t("export_tab_general"))
        tabs.addTab(web_scroll, t("export_tab_web"))

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

        # --- AI model export / import ---
        model_box = QGroupBox(t("deep_model_group"))
        model_layout = QVBoxLayout(model_box)

        model_export_desc = QLabel(t("deep_model_export_desc"))
        model_export_desc.setWordWrap(True)
        model_export_desc.setStyleSheet("color: #aaa; font-size: 11px;")
        model_layout.addWidget(model_export_desc)
        self._model_export_btn = QPushButton(f"🧠  {t('deep_model_export_btn')}")
        self._model_export_btn.setEnabled(self._on_deep_model_export_cb is not None)
        self._model_export_btn.clicked.connect(self._on_deep_model_export_clicked)
        model_layout.addWidget(self._model_export_btn)

        model_import_desc = QLabel(t("deep_model_import_desc"))
        model_import_desc.setWordWrap(True)
        model_import_desc.setStyleSheet("color: #aaa; font-size: 11px;")
        model_layout.addWidget(model_import_desc)
        self._model_import_btn = QPushButton(f"📂  {t('deep_model_import_btn')}")
        self._model_import_btn.setEnabled(self._on_deep_model_import_cb is not None)
        self._model_import_btn.clicked.connect(self._on_deep_model_import_clicked)
        model_layout.addWidget(self._model_import_btn)

        layout.addWidget(model_box)

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

        # --- Astro static site (scalable) ---
        astro_box = QGroupBox(t("export_astro_group"))
        astro_layout = QVBoxLayout(astro_box)
        astro_desc = QLabel(t("export_astro_desc"))
        astro_desc.setWordWrap(True)
        astro_desc.setStyleSheet("color: #aaa; font-size: 11px;")
        astro_layout.addWidget(astro_desc)
        self._astro_btn = QPushButton(f"⚡  {t('export_generate_astro')}")
        self._astro_btn.clicked.connect(self._on_export_astro)
        astro_layout.addWidget(self._astro_btn)
        web_layout.addWidget(astro_box)

        # --- Local web server (Python + Node, no Docker) ---
        local_box = QGroupBox(t("export_local_server_group"))
        local_layout = QVBoxLayout(local_box)
        local_desc = QLabel(t("export_local_server_desc"))
        local_desc.setWordWrap(True)
        local_desc.setStyleSheet("color: #aaa; font-size: 11px;")
        local_layout.addWidget(local_desc)
        self._local_server_btn = QPushButton(f"🖥️  {t('export_generate_local_server')}")
        self._local_server_btn.clicked.connect(self._on_export_local_server)
        local_layout.addWidget(self._local_server_btn)
        web_layout.addWidget(local_box)

        # --- Update package (uploadable to a running local server) ---
        update_box = QGroupBox(t("export_update_pkg_group"))
        update_layout = QVBoxLayout(update_box)
        update_desc = QLabel(t("export_update_pkg_desc"))
        update_desc.setWordWrap(True)
        update_desc.setStyleSheet("color: #aaa; font-size: 11px;")
        update_layout.addWidget(update_desc)
        self._update_pkg_csv_chk = QCheckBox(t("export_update_pkg_include_csv"))
        update_layout.addWidget(self._update_pkg_csv_chk)
        self._update_pkg_btn = QPushButton(f"📦  {t('export_generate_update_pkg')}")
        self._update_pkg_btn.clicked.connect(self._on_export_update_package)
        update_layout.addWidget(self._update_pkg_btn)
        web_layout.addWidget(update_box)

        # --- Docker / Kubernetes web server ---
        docker_box = QGroupBox(t("export_docker_group"))
        docker_layout = QVBoxLayout(docker_box)
        docker_desc = QLabel(t("export_docker_desc"))
        docker_desc.setWordWrap(True)
        docker_desc.setStyleSheet("color: #aaa; font-size: 11px;")
        docker_layout.addWidget(docker_desc)
        self._docker_btn = QPushButton(f"🐳  {t('export_generate_docker')}")
        self._docker_btn.clicked.connect(self._on_export_docker)
        docker_layout.addWidget(self._docker_btn)
        web_layout.addWidget(docker_box)

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
        web_layout.addWidget(col_import_box)

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
        web_layout.addWidget(col_html_box)

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
        web_layout.addStretch()
        outer.addWidget(tabs, stretch=1)

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
        person_id = self._scope_person_id()

        def work(ctx):  # noqa: ANN001 — worker thread
            with session_scope() as session:
                return ExportService(session).export_csv(
                    target_path=path, person_id=person_id
                )

        from app.tasks import TaskPriority, get_task_manager
        get_task_manager().submit(
            t("task_csv_export"),
            work,
            priority=TaskPriority.LOW,
            on_done=lambda out: QMessageBox.information(
                self, t("csv_exported"), t("export_csv_saved", path=out)
            ),
            on_error=lambda msg: QMessageBox.critical(self, t("export_error"), msg),
        )

    def _on_export_json(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, t("export_save_json"), "export.json", "JSON Files (*.json)"
        )
        if not path:
            return
        person_id = self._scope_person_id()

        def work(ctx):  # noqa: ANN001 — worker thread
            with session_scope() as session:
                return ExportService(session).export_json(
                    target_path=path, person_id=person_id
                )

        from app.tasks import TaskPriority, get_task_manager
        get_task_manager().submit(
            t("task_json_export"),
            work,
            priority=TaskPriority.LOW,
            on_done=lambda out: QMessageBox.information(
                self, t("json_exported"), t("json_saved", path=out)
            ),
            on_error=lambda msg: QMessageBox.critical(self, t("export_error"), msg),
        )

    def _on_export_astro(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, t("astro_folder"), str(Path.home())
        )
        if not folder:
            return
        person_id = self._scope_person_id()

        # The website export (image resizing + npm build) is slow, so it runs as
        # a low-priority background task: it never blocks the UI and is preempted
        # automatically by the AI pipeline.  Image resizing is parallelised
        # inside the service; the image phase is pause/cancel-able, the opaque
        # npm build is not (it reports an indeterminate -1 percent).
        def work(ctx):  # noqa: ANN001 — runs on the task thread
            from app.services.export_service import ExportService

            def progress(pct, msg):
                if pct is not None and pct >= 0:
                    ctx.report(min(int(pct), 100), str(msg))
                    ctx.checkpoint()  # pause/cancel only in the image phase
                else:
                    # Opaque npm build — keep the bar near full, show the message.
                    ctx.report(99, str(msg))

            with session_scope() as session:
                return str(
                    ExportService(session).export_astro(
                        target_dir=folder,
                        person_id=person_id,
                        progress_callback=progress,
                    )
                )

        from app.tasks import TaskPriority, get_task_manager
        get_task_manager().submit(
            t("task_astro_export"),
            work,
            supports_pause=True,
            priority=TaskPriority.LOW,
            on_done=lambda out: self._astro_open_prompt(out),
            on_error=self._on_astro_failed,
        )

    def _on_export_local_server(self) -> None:
        from app.ui.dialogs.local_server_export_dialog import LocalServerExportDialog
        dlg = LocalServerExportDialog(self)
        if dlg.exec() != LocalServerExportDialog.DialogCode.Accepted:
            return

        folder = QFileDialog.getExistingDirectory(
            self, t("astro_folder"), str(Path.home())
        )
        if not folder:
            return

        admin_email  = dlg.admin_email
        app_password = dlg.gmail_app_password
        allowed_csv  = dlg.allowed_emails_csv
        extra_admins = dlg.extra_admins_csv
        port         = dlg.port
        domain       = dlg.domain
        https_mode   = dlg.https_mode
        smtp_allow_self_signed = dlg.smtp_allow_self_signed

        def work(ctx):
            from app.services.local_server_export_service import LocalServerExportService

            def progress(pct, msg):
                if pct is not None and pct >= 0:
                    ctx.report(min(int(pct), 100), str(msg))
                    ctx.checkpoint()
                else:
                    ctx.report(99, str(msg))

            with session_scope() as session:
                return str(
                    LocalServerExportService(session).export_local(
                        target_dir=folder,
                        admin_email=admin_email,
                        gmail_app_password=app_password,
                        allowed_emails_csv=allowed_csv,
                        port=port,
                        domain=domain,
                        https_mode=https_mode,
                        smtp_allow_self_signed=smtp_allow_self_signed,
                        extra_admins_csv=extra_admins,
                        progress_callback=progress,
                    )
                )

        from app.tasks import TaskPriority, get_task_manager
        get_task_manager().submit(
            t("task_local_server_export"),
            work,
            supports_pause=True,
            priority=TaskPriority.LOW,
            on_done=lambda out: QMessageBox.information(
                self,
                t("local_server_export_done_title"),
                t("local_server_export_done_msg", path=out),
            ),
            on_error=lambda msg: QMessageBox.critical(
                self, t("local_server_export_done_title"), msg
            ),
        )

    def _on_export_update_package(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            t("export_generate_update_pkg"),
            str(Path.home() / "frissites.zip"),
            "ZIP (*.zip)",
        )
        if not path:
            return
        if not path.lower().endswith(".zip"):
            path += ".zip"

        allowed_csv = ""
        if self._update_pkg_csv_chk.isChecked():
            allowed_csv, _ = QFileDialog.getOpenFileName(
                self, t("docker_export_allowed_csv"), str(Path.home()),
                "CSV (*.csv);;All files (*)",
            )
            if not allowed_csv:
                return  # user cancelled the CSV pick

        def work(ctx):
            from app.services.local_server_export_service import LocalServerExportService

            def progress(pct, msg):
                if pct is not None and pct >= 0:
                    ctx.report(min(int(pct), 100), str(msg))
                    ctx.checkpoint()
                else:
                    ctx.report(99, str(msg))

            with session_scope() as session:
                return str(
                    LocalServerExportService(session).export_update_package(
                        target_file=path,
                        allowed_emails_csv=allowed_csv,
                        progress_callback=progress,
                    )
                )

        from app.tasks import TaskPriority, get_task_manager
        get_task_manager().submit(
            t("task_update_pkg_export"),
            work,
            supports_pause=True,
            priority=TaskPriority.LOW,
            on_done=lambda out: QMessageBox.information(
                self,
                t("export_update_pkg_done_title"),
                t("export_update_pkg_done_msg", path=out),
            ),
            on_error=lambda msg: QMessageBox.critical(
                self, t("export_update_pkg_done_title"), msg
            ),
        )

    def _on_export_docker(self) -> None:
        from app.ui.dialogs.docker_export_dialog import DockerExportDialog
        dlg = DockerExportDialog(self)
        if dlg.exec() != DockerExportDialog.DialogCode.Accepted:
            return

        folder = QFileDialog.getExistingDirectory(
            self, t("astro_folder"), str(Path.home())
        )
        if not folder:
            return

        admin_email      = dlg.admin_email
        app_password     = dlg.gmail_app_password
        allowed_csv      = dlg.allowed_emails_csv
        domain           = dlg.domain
        cluster_issuer   = dlg.cluster_issuer
        port             = dlg.port

        def work(ctx):
            from app.services.docker_export_service import DockerExportService

            def progress(pct, msg):
                if pct is not None and pct >= 0:
                    ctx.report(min(int(pct), 100), str(msg))
                    ctx.checkpoint()
                else:
                    ctx.report(99, str(msg))

            with session_scope() as session:
                return str(
                    DockerExportService(session).export(
                        target_dir=folder,
                        admin_email=admin_email,
                        gmail_app_password=app_password,
                        allowed_emails_csv=allowed_csv,
                        domain=domain,
                        cluster_issuer=cluster_issuer,
                        port=port,
                        progress_callback=progress,
                    )
                )

        from app.tasks import TaskPriority, get_task_manager
        get_task_manager().submit(
            t("task_docker_export"),
            work,
            supports_pause=True,
            priority=TaskPriority.LOW,
            on_done=lambda out: QMessageBox.information(
                self,
                t("docker_export_done_title"),
                t("docker_export_done_msg", path=out),
            ),
            on_error=lambda msg: QMessageBox.critical(self, t("docker_export_done_title"), msg),
        )

    def _on_astro_failed(self, message: str) -> None:
        # _run_astro_build raises RuntimeError mentioning npm/node when missing.
        low = message.lower()
        if "npm" in low or "node" in low:
            QMessageBox.warning(self, t("astro_failed"), t("astro_need_node", err=message))
        else:
            QMessageBox.critical(self, t("astro_failed"), message)

    def _astro_open_prompt(self, out: str) -> None:
        import subprocess
        import sys

        index = Path(out) / "index.html"
        reply = QMessageBox.information(
            self,
            t("astro_done"),
            t("astro_open", path=index),
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
        person_id = self._person_id

        def work(ctx):  # noqa: ANN001 — worker thread
            with session_scope() as session:
                return ExportService(session).export_person_images(
                    person_id=person_id, target_dir=folder
                )

        from app.tasks import TaskPriority, get_task_manager
        get_task_manager().submit(
            t("task_image_export"),
            work,
            priority=TaskPriority.LOW,
            on_done=lambda n: QMessageBox.information(
                self, t("images_exported"), t("files_copied", n=n, folder=folder)
            ),
            on_error=lambda msg: QMessageBox.critical(self, t("export_error"), msg),
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

        # Building the table can be slow on a large library, so run it as a
        # low-priority background task instead of blocking the UI thread.
        def work(ctx):  # noqa: ANN001 — runs on the task thread
            with session_scope() as session:
                svc = ImageMetadataExportService(session)
                if use_xlsx:
                    return svc.export_xlsx(path, fields, person_mode)
                return svc.export_csv(path, fields, person_mode)

        from app.tasks import TaskPriority, get_task_manager
        get_task_manager().submit(
            t("task_metadata_table"),
            work,
            priority=TaskPriority.LOW,
            on_done=lambda out: QMessageBox.information(
                self, t("export_metadata_done"), t("export_metadata_saved", path=out)
            ),
            on_error=lambda msg: QMessageBox.critical(self, t("export_error"), msg),
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

        options = self._fmeta_options()

        # Writing persons into every image file is slow and must not block the
        # UI, so it runs as a low-priority background task in the Task Manager
        # (pause/resume/stop there).  Cancellation flows through the task's token
        # so the service returns its partial "X of Y embedded" summary cleanly.
        def work(ctx):  # noqa: ANN001 — runs on the task thread
            import time

            from app.services.face_metadata_export_service import (
                FaceMetadataExportService,
            )

            def cb(done, total, name):
                ctx.report(int(done / total * 100) if total else 0, name or "")
                # Block while paused without raising; let a cancel reach the
                # service so it returns its clean cancelled summary.
                while ctx.token.paused:
                    time.sleep(0.1)

            with session_scope() as session:
                return FaceMetadataExportService(session).export_all(
                    options, progress_cb=cb, cancel_token=ctx.token
                )

        from app.tasks import TaskPriority, get_task_manager
        get_task_manager().submit(
            t("task_metadata_export"),
            work,
            supports_pause=True,
            priority=TaskPriority.LOW,
            on_done=lambda summary: self._show_embed_summary(summary),
            on_error=lambda msg: QMessageBox.critical(self, t("fmeta_done_title"), msg),
        )

    def _show_embed_summary(self, summary) -> None:
        if getattr(summary, "cancelled", False):
            msg = t(
                "fmeta_summary_cancelled",
                total=summary.total,
                remaining=summary.remaining_count,
                embedded=summary.embedded_count,
                sidecar=summary.sidecar_count,
                skipped=summary.skipped_count,
                failed=summary.failed_count,
            )
        else:
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

    def _on_deep_model_export_clicked(self) -> None:
        if self._on_deep_model_export_cb is None:
            return
        self.accept()
        self._on_deep_model_export_cb()

    def _on_deep_model_import_clicked(self) -> None:
        if self._on_deep_model_import_cb is None:
            return
        self.accept()
        self._on_deep_model_import_cb()
