"""Background job primitives (cancellation, pause/resume, job descriptors)."""

from app.jobs.cancellation import CancellationToken, OperationCancelled
from app.jobs.job_types import JobPriority, JobSpec, JobState, JobStatus

__all__ = [
    "CancellationToken",
    "OperationCancelled",
    "JobPriority",
    "JobSpec",
    "JobState",
    "JobStatus",
]
