"""Export dialog — CSV, JSON, image and collage export in one place."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.db.database import session_scope
from app.services.export_service import ExportService
from app.ui.i18n import t


class ExportDialog(QDialog):
    """Modal export window with CSV, JSON, image and collage export options."""

    def __init__(
        self,
        current_person_id: Optional[int] = None,
        current_person_name: Optional[str] = None,
        on_collage_import: Optional[Callable[[], None]] = None,
        on_collage_html_export: Optional[Callable[[], None]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._person_id = current_person_id
        self._person_name = current_person_name
        self._on_collage_import_cb = on_collage_import
        self._on_collage_html_export_cb = on_collage_html_export
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
