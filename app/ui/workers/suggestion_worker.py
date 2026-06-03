"""Background worker for merge suggestion decisions (accept/reject/defer)."""

from __future__ import annotations

import logging
import threading
from typing import Callable

from PySide6.QtCore import QMetaObject, QObject, Qt, Signal

from app.config import MatchingConfig
from app.db.database import session_scope
from app.services.merge_suggestion_service import MergeSuggestionService

log = logging.getLogger(__name__)


class SuggestionWorker(QObject):
	"""Worker that runs merge decisions on a background thread.

	Signals:
		decision_finished(bool): True if successful, False on error.
		error_occurred(str): Error message if decision failed.
	"""

	decision_finished = Signal(bool)
	error_occurred = Signal(str)

	def __init__(self, matching: MatchingConfig) -> None:
		super().__init__()
		self._matching = matching

	def run_decision(
		self,
		action: Callable[[MergeSuggestionService], None],
		label: str,
	) -> None:
		"""Queue a decision to run on a background thread (non-blocking)."""
		thread = threading.Thread(
			target=self._worker_run,
			args=(action, label),
			daemon=True,
		)
		thread.start()

	def _worker_run(
		self,
		action: Callable[[MergeSuggestionService], None],
		label: str,
	) -> None:
		"""Run the decision on the background thread, then emit signals on UI thread."""
		success = False
		error_msg = None
		try:
			with session_scope() as session:
				service = MergeSuggestionService(session, self._matching)
				action(service)
			log.info("Merge suggestion %s completed successfully", label)
			success = True
		except Exception as exc:  # noqa: BLE001
			log.exception("Suggestion %s failed", label)
			error_msg = str(exc)

		# Emit signals on the UI thread using QMetaObject.invokeMethod
		QMetaObject.invokeMethod(
			self,
			"_emit_decision_finished",
			Qt.QueuedConnection,
			success,
		)
		if error_msg:
			QMetaObject.invokeMethod(
				self,
				"_emit_error_occurred",
				Qt.QueuedConnection,
				error_msg,
			)

	def _emit_decision_finished(self, success: bool) -> None:
		"""Emit decision_finished signal (called on UI thread)."""
		self.decision_finished.emit(success)

	def _emit_error_occurred(self, error_msg: str) -> None:
		"""Emit error_occurred signal (called on UI thread)."""
		self.error_occurred.emit(error_msg)



