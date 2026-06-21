"""Dialog for local server export configuration."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from app.services.gmail_check import check_credentials, send_test_email
from app.ui.i18n import t


class LocalServerExportDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("local_server_export_title"))
        self.setMinimumWidth(460)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._admin_email = QLineEdit()
        self._admin_email.setPlaceholderText("admin@gmail.com")
        form.addRow(t("docker_export_admin_email"), self._admin_email)

        self._app_password = QLineEdit()
        self._app_password.setEchoMode(QLineEdit.EchoMode.Password)
        self._app_password.setPlaceholderText("xxxx xxxx xxxx xxxx")
        pw_row = QHBoxLayout()
        pw_row.addWidget(self._app_password)
        self._test_btn = QPushButton(t("smtp_test_btn"))
        self._test_btn.clicked.connect(self._test_email)
        pw_row.addWidget(self._test_btn)
        form.addRow(t("docker_export_app_password"), pw_row)

        csv_row = QHBoxLayout()
        self._csv_path = QLineEdit()
        self._csv_path.setPlaceholderText(t("docker_export_csv_placeholder"))
        self._csv_path.setReadOnly(True)
        csv_browse = QPushButton("…")
        csv_browse.setFixedWidth(32)
        csv_browse.clicked.connect(self._browse_csv)
        csv_row.addWidget(self._csv_path)
        csv_row.addWidget(csv_browse)
        form.addRow(t("docker_export_allowed_csv"), csv_row)

        self._port = QSpinBox()
        self._port.setRange(1024, 65535)
        self._port.setValue(3000)
        form.addRow(t("docker_export_port"), self._port)

        layout.addLayout(form)

        info = QLabel(t("docker_export_info"))
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(info)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._validate)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def showEvent(self, event) -> None:  # noqa: N802 — Qt override
        # Opened nested inside the modal ExportDialog's exec() loop. On macOS a
        # modal dialog stacked over another modal dialog can fail to become the
        # key window, so its inputs render but won't accept clicks/typing. Force
        # activation and put the cursor in the first field.
        super().showEvent(event)
        self.raise_()
        self.activateWindow()
        self._admin_email.setFocus()

    def _browse_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, t("docker_export_csv_dialog_title"), str(Path.home()),
            "CSV files (*.csv);;All files (*)",
        )
        if path:
            self._csv_path.setText(path)

    def _test_email(self) -> None:
        """Send a real test email to the admin address to prove the password works."""
        email = self._admin_email.text().strip().lower()
        pw = self._app_password.text().strip()
        if "@" not in email or len(pw) < 8:
            QMessageBox.warning(self, t("smtp_test_title"), t("smtp_test_need_input"))
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._test_btn.setEnabled(False)
        try:
            ok, detail = send_test_email(email, pw, to=email)
        finally:
            self._test_btn.setEnabled(True)
            QApplication.restoreOverrideCursor()
        if ok:
            QMessageBox.information(self, t("smtp_test_title"), t("smtp_test_ok", to=detail))
        else:
            QMessageBox.critical(self, t("smtp_test_title"), detail)

    def _validate(self) -> None:
        errors = []
        if "@gmail.com" not in self._admin_email.text().lower():
            errors.append(t("docker_export_err_email"))
        if len(self._app_password.text().strip()) < 8:
            errors.append(t("docker_export_err_password"))
        if not self._csv_path.text() or not Path(self._csv_path.text()).is_file():
            errors.append(t("docker_export_err_csv"))
        if errors:
            QMessageBox.warning(self, t("local_server_export_title"), "\n".join(errors))
            return
        if not self._confirm_smtp():
            return
        self.accept()

    def _confirm_smtp(self) -> bool:
        """Verify the Gmail login before generating; let the user override on failure."""
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            ok, detail = check_credentials(
                self._admin_email.text().strip().lower(),
                self._app_password.text().strip(),
            )
        finally:
            QApplication.restoreOverrideCursor()
        if ok:
            return True
        reply = QMessageBox.warning(
            self,
            t("smtp_test_title"),
            t("smtp_check_failed", detail=detail),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    @property
    def admin_email(self) -> str:
        return self._admin_email.text().strip().lower()

    @property
    def gmail_app_password(self) -> str:
        return self._app_password.text().strip()

    @property
    def allowed_emails_csv(self) -> str:
        return self._csv_path.text().strip()

    @property
    def port(self) -> int:
        return self._port.value()
