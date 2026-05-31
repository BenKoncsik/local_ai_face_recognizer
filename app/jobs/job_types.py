"""Descriptors and enums for background jobs."""

from __future__ import annotations

import enum
import itertools
import time
from dataclasses import dataclass, field
from typing import List, Optional


class JobPriority(enum.IntEnum):
    """Lower value == higher priority (matches ``queue.PriorityQueue``).

    SCOPED jobs (the currently viewed image / folder) preempt the broad
    FULL archive scan so the user gets relevant suggestions first.
    """

    SCOPED = 0
    FULL = 10


class JobState(enum.Enum):
    """Lifecycle of a single job."""

    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Job kinds understood by the matcher worker.
JOB_KIND_SCOPED = "scoped_match"   # restricted to a set of person ids
JOB_KIND_FULL = "full_match"       # whole-archive scan


_seq = itertools.count(1)


@dataclass(order=True)
class JobSpec:
    """An enqueued unit of background work.

    ``sort_index`` (priority, then FIFO sequence) drives the priority queue;
    all other fields are excluded from comparison.
    """

    sort_index: tuple = field(init=False, repr=False)
    priority: JobPriority = field(compare=False, default=JobPriority.FULL)
    kind: str = field(compare=False, default=JOB_KIND_FULL)
    job_id: str = field(compare=False, default="")
    # For scoped jobs: the candidate (auto-named) person ids to match.
    person_ids: Optional[List[int]] = field(compare=False, default=None)
    label: str = field(compare=False, default="")

    def __post_init__(self) -> None:
        if not self.job_id:
            self.job_id = f"{self.kind}-{next(_seq)}-{int(time.time())}"
        self.sort_index = (int(self.priority), next(_seq))


@dataclass
class JobStatus:
    """Snapshot of a job's progress, emitted to the UI (throttled)."""

    job_id: str
    label: str
    state: JobState
    processed: int = 0
    total: int = 0
    found: int = 0           # suggestions found so far
    error: Optional[str] = None

    def percent(self) -> int:
        if self.total <= 0:
            return 0
        return int(min(100, round(100 * self.processed / self.total)))
