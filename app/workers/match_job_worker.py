"""Background merge-suggestion job worker.

A single :class:`QThread` owns a priority queue of :class:`JobSpec`s and a
:class:`~concurrent.futures.ThreadPoolExecutor` used to score person chunks in
parallel.  The heavy work (centroid matmuls + fuzzy name comparison) is pure
and read-only, so it parallelises safely; the database *writes* are serialised
back on the job thread in batched transactions to avoid SQLite lock churn.

Design highlights
-----------------
* **Queue-based** — jobs are enqueued from the UI thread and drained FIFO
  within each priority band.
* **Priorities** — SCOPED jobs (current image / folder) preempt the broad
  FULL archive scan at chunk granularity, so the user sees relevant
  suggestions first.
* **Single full scan** — a FULL job is refused while one is queued or running.
* **Pause / resume / cancel** — via a per-job :class:`CancellationToken`.
* **Incremental** — suggestions are persisted and signalled per chunk.
* **Throttled progress** — status signals are rate-limited.
* **Graceful shutdown** — :meth:`shutdown` cancels in-flight work and joins the
  pool so no half-written transaction is left behind.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import List, Optional, Sequence

from PySide6.QtCore import QThread, Signal

from app.config import AppConfig
from app.db.database import session_scope
from app.jobs.cancellation import CancellationToken, OperationCancelled
from app.jobs.job_types import (
    JOB_KIND_FULL,
    JOB_KIND_SCOPED,
    JobPriority,
    JobSpec,
    JobState,
    JobStatus,
)
from app.services.merge_suggestion_service import (
    MergeProfile,
    MergeSuggestionService,
)

log = logging.getLogger(__name__)


def _utcnow_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class MatchJobWorker(QThread):
    """Drains a priority queue of merge-matching jobs in the background.

    Signals:
        status_changed:    ``JobStatus`` — throttled per-job progress.
        suggestions_found: ``(total_open: int)`` — open suggestion count after
                           a chunk was persisted (UI can refresh incrementally).
        queue_changed:     ``(pending_jobs: int)``
        idle:              emitted when the queue drains and no job is running.
    """

    status_changed = Signal(object)
    suggestions_found = Signal(int)
    queue_changed = Signal(int)
    idle = Signal()

    def __init__(self, config: AppConfig, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._mcfg = config.matching
        self._queue: "queue.PriorityQueue[JobSpec]" = queue.PriorityQueue()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._full_pending = 0
        self._full_running = False
        # Token of the job currently executing (for pause/resume/cancel).
        self._current_token: Optional[CancellationToken] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._last_emit = 0.0

    # ------------------------------------------------------------------
    # Public API (UI thread)
    # ------------------------------------------------------------------

    def enqueue_full_scan(self, label: str = "Full archive scan") -> bool:
        """Queue a whole-archive scan unless one is already queued/running."""
        with self._lock:
            if self._full_pending > 0 or self._full_running:
                log.debug("Full scan refused — one already queued/running")
                return False
            self._full_pending += 1
        self._queue.put(
            JobSpec(priority=JobPriority.FULL, kind=JOB_KIND_FULL, label=label)
        )
        self._emit_queue_size()
        return True

    def enqueue_scoped(
        self, person_ids: Sequence[int], label: str = "Current view"
    ) -> Optional[str]:
        """Queue a high-priority scoped match for *person_ids*."""
        ids = [int(p) for p in person_ids]
        if not ids:
            return None
        spec = JobSpec(
            priority=JobPriority.SCOPED,
            kind=JOB_KIND_SCOPED,
            person_ids=ids,
            label=label,
        )
        self._queue.put(spec)
        self._emit_queue_size()
        return spec.job_id

    def pause(self) -> None:
        token = self._current_token
        if token is not None:
            token.pause()

    def resume(self) -> None:
        token = self._current_token
        if token is not None:
            token.resume()

    def cancel_current(self) -> None:
        token = self._current_token
        if token is not None:
            token.cancel()

    def shutdown(self, drain: bool = False) -> None:
        """Request graceful stop.  ``drain=False`` cancels in-flight work."""
        if not drain:
            self.cancel_current()
            self._clear_queue()
        self._stop.set()

    # ------------------------------------------------------------------
    # Thread body
    # ------------------------------------------------------------------

    def run(self) -> None:
        workers = self._worker_count()
        log.info("MatchJobWorker started with %d scoring worker(s)", workers)
        self._executor = ThreadPoolExecutor(max_workers=workers)
        try:
            while not self._stop.is_set():
                try:
                    job = self._queue.get(timeout=0.25)
                except queue.Empty:
                    continue
                self._consume_job(job)
                if self._queue.empty():
                    self.idle.emit()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
            log.info("MatchJobWorker stopped")

    def _consume_job(self, job: JobSpec) -> None:
        if job.kind == JOB_KIND_FULL:
            with self._lock:
                self._full_pending = max(0, self._full_pending - 1)
                self._full_running = True
        token = CancellationToken()
        self._current_token = token
        try:
            self._run_job(job, token)
        except OperationCancelled:
            log.info("Job %s cancelled", job.job_id)
            self._emit_status(job, JobState.CANCELLED)
        except Exception as exc:  # noqa: BLE001
            log.exception("Job %s failed", job.job_id)
            self._emit_status(job, JobState.FAILED, error=str(exc))
        finally:
            self._current_token = None
            if job.kind == JOB_KIND_FULL:
                with self._lock:
                    self._full_running = False
            self._emit_queue_size()

    # ------------------------------------------------------------------
    # Job execution
    # ------------------------------------------------------------------

    def _run_job(self, job: JobSpec, token: CancellationToken) -> None:
        started_at = _utcnow_naive()
        self._emit_status(job, JobState.RUNNING, force=True)

        # --- Load profiles + suppressed pairs once (read-only snapshot) ---
        with session_scope() as session:
            svc = MergeSuggestionService(session, self._mcfg)
            targets = svc.load_targets()
            candidates = svc.load_candidates(
                job.person_ids if job.kind == JOB_KIND_SCOPED else None
            )
            suppressed = svc.load_suppressed_pairs()

        total = len(candidates)
        if total == 0 or not targets:
            log.info(
                "Job %s: nothing to do (%d candidates, %d targets)",
                job.job_id, total, len(targets),
            )
            self._emit_status(job, JobState.DONE, processed=0, total=0, force=True)
            return

        status = JobStatus(
            job_id=job.job_id, label=job.label, state=JobState.RUNNING, total=total
        )
        chunk_size = max(1, self._mcfg.chunk_size)
        chunks = [
            candidates[i : i + chunk_size] for i in range(0, total, chunk_size)
        ]

        for chunk in chunks:
            token.wait_if_paused()
            token.raise_if_cancelled()

            # Let queued scoped jobs preempt a long full scan at chunk edges.
            if job.kind == JOB_KIND_FULL:
                self._run_pending_scoped()

            found = self._process_chunk(
                chunk, targets, suppressed, job.job_id, started_at, token
            )
            status.processed += len(chunk)
            status.found += found
            self._emit_status_obj(status)

        self._emit_status_obj(status, state=JobState.DONE, force=True)
        log.info(
            "Job %s done: %d candidate(s), %d suggestion(s) written",
            job.job_id, status.processed, status.found,
        )

    def _process_chunk(
        self,
        chunk: List[MergeProfile],
        targets: List[MergeProfile],
        suppressed,
        job_id: str,
        started_at: datetime,
        token: CancellationToken,
    ) -> int:
        """Score one chunk on the pool, then persist serialised on this thread."""
        assert self._executor is not None
        future = self._executor.submit(
            MergeSuggestionService(None, self._mcfg).score_candidates,  # type: ignore[arg-type]
            chunk,
            targets,
            suppressed,
        )
        # score_candidates is pure (no DB); a None session is never touched.
        results = future.result()
        token.raise_if_cancelled()
        if not results:
            return 0

        with session_scope() as session:
            written = MergeSuggestionService(session, self._mcfg).persist_results(
                results, job_id, started_at
            )
            open_count = MergeSuggestionService(session, self._mcfg).count_open()
        if written:
            self.suggestions_found.emit(open_count)
        return written

    def _run_pending_scoped(self) -> None:
        """Run any queued SCOPED jobs immediately (preempt the full scan)."""
        while True:
            try:
                spec = self._queue.get_nowait()
            except queue.Empty:
                return
            if spec.priority != JobPriority.SCOPED:
                # Not a scoped job — put it back and stop draining.
                self._queue.put(spec)
                return
            token = CancellationToken()
            prev = self._current_token
            self._current_token = token
            try:
                self._run_job(spec, token)
            except OperationCancelled:
                self._emit_status(spec, JobState.CANCELLED)
            except Exception as exc:  # noqa: BLE001
                log.exception("Scoped job %s failed", spec.job_id)
                self._emit_status(spec, JobState.FAILED, error=str(exc))
            finally:
                self._current_token = prev
                self._emit_queue_size()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _worker_count(self) -> int:
        cpu = os.cpu_count() or 2
        derived = max(1, cpu - max(0, self._mcfg.reserved_cpus))
        if self._mcfg.max_workers:
            derived = min(derived, max(1, self._mcfg.max_workers))
        return derived

    def _clear_queue(self) -> None:
        with self._lock:
            self._full_pending = 0
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        self._emit_queue_size()

    def _emit_queue_size(self) -> None:
        self.queue_changed.emit(self._queue.qsize())

    def _emit_status(
        self,
        job: JobSpec,
        state: JobState,
        processed: int = 0,
        total: int = 0,
        error: Optional[str] = None,
        force: bool = False,
    ) -> None:
        self._emit_status_obj(
            JobStatus(
                job_id=job.job_id,
                label=job.label,
                state=state,
                processed=processed,
                total=total,
                error=error,
            ),
            state=state,
            force=force,
        )

    def _emit_status_obj(
        self,
        status: JobStatus,
        state: Optional[JobState] = None,
        force: bool = False,
    ) -> None:
        if state is not None:
            status.state = state
        now = time.monotonic() * 1000.0
        terminal = status.state in (
            JobState.DONE, JobState.FAILED, JobState.CANCELLED
        )
        if not force and not terminal:
            if now - self._last_emit < self._mcfg.progress_throttle_ms:
                return
        self._last_emit = now
        self.status_changed.emit(status)
