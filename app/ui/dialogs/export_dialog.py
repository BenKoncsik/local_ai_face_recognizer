"""Export dialog — CSV, JSON and image export in one place."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

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
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.db.database import session_scope
from app.services.export_service import ExportService
from app.ui.i18n import t


class ExportDialog(QDialog):
    """Modal export window with CSV, JSON and image export options."""

    def __init__(
        self,
        current_person_id: Optional[int] = None,
        current_person_name: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._person_id = current_person_id
        self._person_name = current_person_name
        self.setWindowTitle(t("export_title"))
        self.setMinimumWidth(500)
        self._build_ui()

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

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

        # --- Close ---
        close_btn = QPushButton(t("close"))
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)

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
