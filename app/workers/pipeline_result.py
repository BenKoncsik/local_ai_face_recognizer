"""Shared result type for processing-pipeline workers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineResult:
    """Outcome of a pipeline run, returned to the Task Manager's ``on_done``.

    ``success`` is ``False`` for soft failures that should warn the user but are
    not crashes (e.g. a stage degraded gracefully).  Hard failures raise; user
    cancellation raises :class:`OperationCancelled` and never produces a result.
    """

    success: bool
    summary: str
    n_suggestions: int = 0
    n_auto_assignments: int = 0
