"""Dialog for configuring Unknown persons reset options."""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
	QCheckBox,
	QDialog,
	QHBoxLayout,
	QLabel,
	QPushButton,
	QVBoxLayout,
	QWidget,
)

from app.services.unknown_person_reset_service import UnknownPersonResetOptions
from app.ui.i18n import t


class ResetUnknownOptionsDialog(QDialog):
	"""Dialog to configure which steps to perform when resetting Unknown identities."""

	def __init__(self, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)
		self.setWindowTitle(t("resetUnknownOptions.title"))
		self.setMinimumWidth(500)
		self._build_ui()
		self._load_defaults()

	def _build_ui(self) -> None:
		"""Build the dialog UI."""
		outer = QVBoxLayout(self)
		outer.setSpacing(12)
		outer.setContentsMargins(12, 12, 12, 12)

		# Title/description
		desc = QLabel(t("resetUnknownOptions.description"))
		desc.setWordWrap(True)
		outer.addWidget(desc)

		# Step 1: Delete Unknown persons
		self._chk_delete_persons = QCheckBox(t("resetUnknownOptions.deletePersons"))
		self._chk_delete_persons.setToolTip(
			t("resetUnknownOptions.deletePersonsTooltip")
		)
		outer.addWidget(self._chk_delete_persons)

		# Step 2: Delete face assignments
		self._chk_delete_assignments = QCheckBox(
			t("resetUnknownOptions.deleteFaceAssignments")
		)
		self._chk_delete_assignments.setToolTip(
			t("resetUnknownOptions.deleteFaceAssignmentsTooltip")
		)
		outer.addWidget(self._chk_delete_assignments)

		# Step 3: Delete face data (advanced)
		self._chk_delete_face_data = QCheckBox(
			t("resetUnknownOptions.deleteFaceData")
		)
		self._chk_delete_face_data.setToolTip(
			t("resetUnknownOptions.deleteFaceDataTooltip")
		)
		outer.addWidget(self._chk_delete_face_data)

		# Step 4: Rebuild Unknown clusters after reset
		self._chk_rebuild_clusters = QCheckBox(t("resetUnknownOptions.rebuildClusters"))
		self._chk_rebuild_clusters.setToolTip(
			t("resetUnknownOptions.rebuildClustersTooltip")
		)
		outer.addWidget(self._chk_rebuild_clusters)

		outer.addStretch()

		# Action buttons
		btn_layout = QHBoxLayout()
		btn_layout.addStretch()

		reset_btn = QPushButton(t("resetUnknownOptions.reset"))
		reset_btn.clicked.connect(self._load_defaults)
		btn_layout.addWidget(reset_btn)

		cancel_btn = QPushButton(t("resetUnknownOptions.cancel"))
		cancel_btn.clicked.connect(self.reject)
		btn_layout.addWidget(cancel_btn)

		ok_btn = QPushButton(t("resetUnknownOptions.ok"))
		ok_btn.clicked.connect(self.accept)
		ok_btn.setDefault(True)
		btn_layout.addWidget(ok_btn)

		outer.addLayout(btn_layout)

	def _load_defaults(self) -> None:
		"""Load default values."""
		defaults = UnknownPersonResetOptions()
		self._chk_delete_persons.setChecked(defaults.delete_unknown_persons)
		self._chk_delete_assignments.setChecked(defaults.delete_face_assignments)
		self._chk_delete_face_data.setChecked(defaults.delete_face_data)
		self._chk_rebuild_clusters.setChecked(defaults.rebuild_clusters)

	def get_options(self) -> UnknownPersonResetOptions:
		"""Get the configured options."""
		return UnknownPersonResetOptions(
			delete_unknown_persons=self._chk_delete_persons.isChecked(),
			delete_face_assignments=self._chk_delete_assignments.isChecked(),
			delete_face_data=self._chk_delete_face_data.isChecked(),
			rebuild_clusters=self._chk_rebuild_clusters.isChecked(),
		)
