"""Tests for background job descriptors."""

from __future__ import annotations

from app.jobs.job_types import (
    JOB_KIND_FULL,
    JOB_KIND_SCOPED,
    JobPriority,
    JobSpec,
    JobState,
    JobStatus,
)


class TestJobSpecOrdering:
    def test_scoped_jobs_sort_before_full_jobs(self) -> None:
        full = JobSpec(priority=JobPriority.FULL, kind=JOB_KIND_FULL, label="full")
        scoped = JobSpec(priority=JobPriority.SCOPED, kind=JOB_KIND_SCOPED, label="scoped")
        assert scoped < full

    def test_same_priority_uses_fifo_sequence(self) -> None:
        first = JobSpec(priority=JobPriority.FULL, label="first")
        second = JobSpec(priority=JobPriority.FULL, label="second")
        assert first < second

    def test_job_id_is_generated_when_missing(self) -> None:
        spec = JobSpec(kind=JOB_KIND_SCOPED, person_ids=[1, 2])
        assert spec.job_id.startswith("scoped_match-")
        assert spec.person_ids == [1, 2]


class TestJobStatusPercent:
    def test_zero_when_total_not_positive(self) -> None:
        status = JobStatus(
            job_id="j1",
            label="scan",
            state=JobState.RUNNING,
            processed=10,
            total=0,
        )
        assert status.percent() == 0

    def test_rounds_processed_fraction(self) -> None:
        status = JobStatus(
            job_id="j1",
            label="scan",
            state=JobState.RUNNING,
            processed=25,
            total=100,
        )
        assert status.percent() == 25

    def test_caps_at_one_hundred(self) -> None:
        status = JobStatus(
            job_id="j1",
            label="scan",
            state=JobState.DONE,
            processed=150,
            total=100,
        )
        assert status.percent() == 100
