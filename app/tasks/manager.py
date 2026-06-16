"""Unified background-task manager.

Long-running work is wrapped in a :class:`BackgroundTask` and executed on a
worker thread, keeping the UI responsive.  The manager enforces a small
concurrency cap, schedules by **priority** (higher first, FIFO within a level),
**preempts** lower-priority work when an important task needs a slot, and tracks
progress / state / CPU time per task.  Qt signals let the Task Manager window
and status bar mirror what is happening.

Task functions receive a :class:`TaskContext` and must:

* call ``ctx.report(percent, message)`` at sensible chunk boundaries,
* call ``ctx.checkpoint()`` regularly — this raises ``OperationCancelled``
  after a cancel request and blocks while paused,
* open their own DB sessions (never reuse UI-thread sessions).

Usage::

    from app.tasks import get_task_manager, TaskPriority

    def work(ctx):
        for i, item in enumerate(items):
            ctx.checkpoint()
            process(item)
            ctx.report(int(i / len(items) * 100), item.name)
        return summary

    task = get_task_manager().submit(
        "HTML export", work, priority=TaskPriority.LOW, on_done=show_result
    )

Scheduling model
----------------
Three lists hold tasks: ``_running`` (active, counts toward the concurrency
cap), ``_paused`` (thread alive but blocked on the pause event — does **not**
count toward the cap) and ``_queue`` (waiting to start).  ``_pump`` repeatedly:

1. **Preempts** — if no slot is free and a queued task outranks a running,
   pausable, lower-priority task, that task is auto-paused to free a slot.
2. **Resumes** — an eligible paused task is resumed into a free slot, but only
   when it ranks at least as high as the front of the queue.
3. **Starts** — the highest-priority queued task is started into a free slot.
"""

from __future__ import annotations

import itertools
import logging
import time
from enum import IntEnum
from typing import Callable, List, Optional

from PySide6.QtCore import QObject, QTimer, Qt, QThread, Signal

from app.jobs.cancellation import CancellationToken, OperationCancelled

log = logging.getLogger(__name__)

_task_ids = itertools.count(1)

#: Time (seconds) to keep finished tasks in history before auto-cleanup
_AUTO_CLEANUP_SECONDS = 5 * 60


def _auto_cleanup_enabled() -> bool:
    """Whether finished tasks should be auto-removed after a few minutes.

    Controlled from Settings → Task Manager; defaults to on.  Read live so the
    toggle takes effect without restarting the app.
    """
    try:
        from app.app_settings import app_qsettings

        return app_qsettings().value(
            "tasks/auto_cleanup_enabled", True, type=bool
        )
    except Exception:  # noqa: BLE001 — settings optional, never break cleanup
        return True


class TaskPriority(IntEnum):
    """Scheduling priority — higher value runs first.

    Auto-assigned by category at submit time: app-critical AI work (scan,
    detection, recognition) is :attr:`HIGH`; maintenance passes are
    :attr:`NORMAL`; exports/reports are :attr:`LOW`.  A higher-priority task
    preempts a running lower-priority one when it needs a slot.
    """

    LOW = 10
    NORMAL = 20
    HIGH = 30

    @property
    def next_up(self) -> "TaskPriority":
        if self is TaskPriority.LOW:
            return TaskPriority.NORMAL
        return TaskPriority.HIGH

    @property
    def next_down(self) -> "TaskPriority":
        if self is TaskPriority.HIGH:
            return TaskPriority.NORMAL
        return TaskPriority.LOW


class TaskState(IntEnum):
    QUEUED = 0
    RUNNING = 1
    PAUSED = 2
    COMPLETED = 3
    FAILED = 4
    CANCELLED = 5

    @property
    def is_final(self) -> bool:
        return self in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED)

    # Back-compat with the previous string-valued enum (logs/tests used .value).
    @property
    def value_str(self) -> str:
        return self.name.lower()


class TaskContext:
    """What a task function sees: progress reporting + cooperative control."""

    def __init__(self, task: "BackgroundTask") -> None:
        self._task = task
        self.token: CancellationToken = task.token

    def report(self, percent: int, message: str = "") -> None:
        self._task._set_progress(percent, message)

    def checkpoint(self) -> None:
        """Raise on cancel; block while paused.  Call at chunk boundaries."""
        self.token.wait_if_paused()


class _TaskThread(QThread):
    """Runs one task function; lifetime == one execution."""

    def __init__(self, task: "BackgroundTask") -> None:
        super().__init__()
        self._task = task

    def run(self) -> None:  # noqa: D102
        task = self._task
        ctx = TaskContext(task)
        cpu_start = time.thread_time()
        try:
            result = task._fn(ctx)
        except OperationCancelled:
            task._finish(TaskState.CANCELLED, cpu_seconds=time.thread_time() - cpu_start)
        except Exception as exc:  # noqa: BLE001 — surface every failure in the UI
            log.exception("Background task failed: %s", task.name)
            task._finish(
                TaskState.FAILED,
                error=str(exc),
                cpu_seconds=time.thread_time() - cpu_start,
            )
        else:
            task._finish(
                TaskState.COMPLETED,
                result=result,
                cpu_seconds=time.thread_time() - cpu_start,
            )


class BackgroundTask(QObject):
    """One unit of background work with observable state."""

    progress_changed = Signal(int, str)      # percent, message
    state_changed = Signal(object)           # TaskState
    finished = Signal(object)                # self

    def __init__(
        self,
        name: str,
        fn: Callable[[TaskContext], object],
        *,
        supports_pause: bool = False,
        priority: TaskPriority = TaskPriority.NORMAL,
        transient: bool = False,
    ) -> None:
        super().__init__()
        self.id: int = next(_task_ids)
        self.name = name
        self._fn = fn
        self.supports_pause = supports_pause
        self.priority = priority
        # Transient tasks (e.g. startup maintenance) are not kept in the
        # finished-task history — they vanish from the list once done.
        self.transient = transient
        self.token = CancellationToken()
        self.state: TaskState = TaskState.QUEUED
        self.progress: int = 0
        self.message: str = ""
        self.result: object = None
        self.error: Optional[str] = None
        self.cpu_seconds: float = 0.0
        self.created_at: float = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self._thread: Optional[_TaskThread] = None
        # Scheduler bookkeeping (managed by TaskManager on the UI thread).
        self._manager: Optional["TaskManager"] = None
        self._auto_paused: bool = False      # paused by preemption (not the user)
        self._resume_eligible: bool = False  # scheduler may resume when a slot frees

    # -- observable properties -----------------------------------------

    @property
    def elapsed_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at if self.finished_at is not None else time.time()
        return end - self.started_at

    @property
    def is_preempted(self) -> bool:
        """Paused automatically to make room for a higher-priority task."""
        return self.state is TaskState.PAUSED and self._auto_paused

    # -- control ---------------------------------------------------------

    def cancel(self) -> None:
        self.token.cancel()
        if self.state is TaskState.QUEUED:
            self._finish(TaskState.CANCELLED, cpu_seconds=0.0)
        # A paused thread is blocked in wait_if_paused(); token.cancel() above
        # wakes it so it raises OperationCancelled and finishes on its own.

    def pause(self) -> None:
        """User-requested pause (delegated to the manager when scheduled)."""
        if self._manager is not None:
            self._manager._request_pause(self)
        elif self.supports_pause and self.state is TaskState.RUNNING:
            self.token.pause()
            self._auto_paused = False
            self._resume_eligible = False
            self._set_state(TaskState.PAUSED)

    def resume(self) -> None:
        """User-requested resume (delegated to the manager when scheduled)."""
        if self._manager is not None:
            self._manager._request_resume(self)
        elif self.state is TaskState.PAUSED:
            self.token.resume()
            self._auto_paused = False
            self._resume_eligible = False
            self._set_state(TaskState.RUNNING)

    def set_priority(self, priority: TaskPriority) -> None:
        """Change priority and let the scheduler re-evaluate ordering."""
        if priority == self.priority:
            return
        self.priority = priority
        self.state_changed.emit(self.state)  # nudge the UI to repaint the row
        if self._manager is not None:
            self._manager._reschedule()

    # -- internal (called from manager / worker thread) -------------------

    def _start(self) -> None:
        self.started_at = time.time()
        self._set_state(TaskState.RUNNING)
        self._thread = _TaskThread(self)
        self._thread.start()

    def _set_progress(self, percent: int, message: str) -> None:
        self.progress = max(0, min(100, percent))
        if message:
            self.message = message
        self.progress_changed.emit(self.progress, self.message)

    def _set_state(self, state: TaskState) -> None:
        self.state = state
        self.state_changed.emit(state)

    def _finish(
        self,
        state: TaskState,
        *,
        result: object = None,
        error: Optional[str] = None,
        cpu_seconds: float = 0.0,
    ) -> None:
        self.result = result
        self.error = error
        self.cpu_seconds = cpu_seconds
        self.finished_at = time.time()
        if state is TaskState.COMPLETED:
            self.progress = 100
        self._set_state(state)
        self.finished.emit(self)


class TaskManager(QObject):
    """Priority scheduler for :class:`BackgroundTask` with a concurrency cap."""

    task_added = Signal(object)             # BackgroundTask
    task_finished = Signal(object)          # BackgroundTask
    counts_changed = Signal(int, int, int)  # running, queued, paused

    #: Completed/failed/cancelled tasks kept for the Task Manager history.
    HISTORY_LIMIT = 50

    def __init__(self, max_concurrent: int = 1) -> None:
        # One heavy task at a time by default: the highest-priority task gets the
        # machine to itself, and a higher-priority arrival preempts a running
        # lower-priority one (which resumes automatically afterwards).  This
        # matches the product rule "AI recognition outranks export" and avoids
        # thrashing the CPU with two heavy ML/image jobs at once — intra-task
        # parallelism (e.g. the export thread pool) provides throughput instead.
        super().__init__()
        self.max_concurrent = max_concurrent
        self._queue: List[BackgroundTask] = []
        self._running: List[BackgroundTask] = []
        self._paused: List[BackgroundTask] = []
        self._history: List[BackgroundTask] = []
        # Auto-cleanup finished tasks after 5 minutes
        self._cleanup_timer = QTimer()
        self._cleanup_timer.timeout.connect(self._cleanup_expired_tasks)
        self._cleanup_timer.start(10_000)  # check every 10 seconds

    # -- public API --------------------------------------------------------

    def submit(
        self,
        name: str,
        fn: Callable[[TaskContext], object],
        *,
        supports_pause: bool = False,
        priority: TaskPriority = TaskPriority.NORMAL,
        transient: bool = False,
        on_done: Optional[Callable[[object], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_cancelled: Optional[Callable[[], None]] = None,
    ) -> BackgroundTask:
        """Queue *fn* for background execution and return its task handle.

        ``on_done(result)`` / ``on_error(message)`` / ``on_cancelled()`` are
        invoked on the UI thread via the task's ``finished`` signal.  Set
        ``transient=True`` for background maintenance that should not linger in
        the finished-task history once complete.
        """
        task = BackgroundTask(
            name, fn, supports_pause=supports_pause, priority=priority,
            transient=transient,
        )
        task._manager = self

        def _dispatch(t: BackgroundTask) -> None:
            if t.state is TaskState.COMPLETED and on_done is not None:
                on_done(t.result)
            elif t.state is TaskState.FAILED and on_error is not None:
                on_error(t.error or "")
            elif t.state is TaskState.CANCELLED and on_cancelled is not None:
                on_cancelled()

        # ``finished`` is emitted from the worker thread (_TaskThread.run).
        # Force QueuedConnection so the user callbacks and the manager
        # bookkeeping always run on the UI thread (where the BackgroundTask
        # QObject lives) — the callbacks touch widgets.  A plain Python
        # callable connected under the default AutoConnection would otherwise
        # risk running on the worker thread.
        task.finished.connect(_dispatch, Qt.QueuedConnection)
        task.finished.connect(self._on_task_finished, Qt.QueuedConnection)
        self._queue.append(task)
        self.task_added.emit(task)
        log.info(
            "Task queued: #%d %s [%s]", task.id, name, task.priority.name
        )
        self._pump()
        return task

    def restart(self, task: BackgroundTask) -> Optional[BackgroundTask]:
        """Re-submit a finished task with the same name/function/priority."""
        if not task.state.is_final:
            return None
        return self.submit(
            task.name,
            task._fn,
            supports_pause=task.supports_pause,
            priority=task.priority,
            transient=task.transient,
        )

    def clear_finished(self) -> int:
        """Drop every finished task from the history (manual "clear" action).

        Returns the number of tasks removed.  Emits ``counts_changed`` so the
        Task Manager window rebuilds its list immediately.
        """
        n = len(self._history)
        if n:
            self._history.clear()
            log.info("Cleared %d finished task(s) from history", n)
            self._emit_counts()
        return n

    def all_tasks(self) -> List[BackgroundTask]:
        """Running + paused + queued + recent history (newest history first)."""
        return (
            list(self._running)
            + list(self._paused)
            + list(self._queue)
            + list(reversed(self._history))
        )

    @property
    def running_count(self) -> int:
        return len(self._running)

    @property
    def queued_count(self) -> int:
        return len(self._queue)

    @property
    def paused_count(self) -> int:
        return len(self._paused)

    @property
    def active_count(self) -> int:
        """Running + queued + paused — everything not yet finished."""
        return len(self._running) + len(self._queue) + len(self._paused)

    def cancel_all(self) -> None:
        for task in list(self._queue) + list(self._running) + list(self._paused):
            task.cancel()

    def has_active_tasks(self) -> bool:
        return bool(self._running or self._queue or self._paused)

    def wait_for_idle(self, timeout_s: float = 5.0) -> bool:
        """Block until all tasks reach a final state, processing UI events.

        Intended for shutdown after :meth:`cancel_all`: it gives worker
        threads a bounded window to observe the cancel and exit, so QThreads
        are not torn down mid-run.  Stays responsive by pumping the event loop
        (the ``finished`` signals are queued to the UI thread).  Returns True
        if everything settled within *timeout_s*, False on timeout.
        """
        import time as _time

        from PySide6.QtWidgets import QApplication

        deadline = _time.monotonic() + timeout_s
        app = QApplication.instance()
        while self.has_active_tasks() and _time.monotonic() < deadline:
            for task in list(self._running) + list(self._paused):
                thread = task._thread
                if thread is not None and thread.isRunning():
                    thread.wait(50)  # ms — lets a finishing worker exit
            if app is not None:
                app.processEvents()  # deliver queued finished() → bookkeeping
        return not self.has_active_tasks()

    # -- scheduler internals -------------------------------------------------

    def _pump(self) -> None:
        """Run the preempt → resume → start cycle until no further progress."""
        progressed = True
        while progressed:
            progressed = False

            # (1) Preemption: free a slot for a higher-priority *waiter* — either
            # a queued task or a resume-eligible paused one (e.g. the user just
            # resumed a HIGH task that should reclaim the slot from a running
            # lower-priority task).  Each preemption replaces a running task with
            # a strictly higher-priority one, so this always terminates.
            if len(self._running) >= self.max_concurrent:
                waiter = self._highest_waiting_priority()
                if waiter is not None:
                    victim = self._preemption_victim(waiter)
                    if victim is not None:
                        self._auto_pause(victim)
                        progressed = True
                        continue

            # (2) Resume an eligible paused task into a free slot.
            if len(self._running) < self.max_concurrent:
                cand = self._resume_candidate()
                if cand is not None:
                    self._do_resume(cand)
                    progressed = True
                    continue

            # (3) Start the highest-priority queued task into a free slot.
            if self._queue and len(self._running) < self.max_concurrent:
                task = self._pop_top_queued()
                if task.state.is_final:  # cancelled while queued
                    progressed = True
                    continue
                self._running.append(task)
                log.info(
                    "Task started: #%d %s [%s]",
                    task.id, task.name, task.priority.name,
                )
                task._start()
                progressed = True
                continue

        self._emit_counts()

    def _reschedule(self) -> None:
        """Re-evaluate scheduling after a priority change."""
        self._pump()

    def _top_queued(self) -> BackgroundTask:
        """Highest-priority queued task (FIFO within a priority level)."""
        return max(self._queue, key=lambda t: (t.priority.value, -t.id))

    def _highest_waiting_priority(self) -> Optional[TaskPriority]:
        """Top priority among tasks wanting a slot (queued + resume-eligible)."""
        waiting = list(self._queue) + [
            t for t in self._paused if t._resume_eligible
        ]
        if not waiting:
            return None
        return max(t.priority for t in waiting)

    def _pop_top_queued(self) -> BackgroundTask:
        task = self._top_queued()
        self._queue.remove(task)
        return task

    def _preemption_victim(
        self, incoming_priority: TaskPriority
    ) -> Optional[BackgroundTask]:
        """Lowest-priority running, pausable task that *incoming* outranks."""
        candidates = [
            t
            for t in self._running
            if t.supports_pause
            and t.state is TaskState.RUNNING
            and t.priority.value < incoming_priority.value
        ]
        if not candidates:
            return None
        # Lowest priority first; among ties preempt the most recently started.
        return min(candidates, key=lambda t: (t.priority.value, -t.id))

    def _resume_candidate(self) -> Optional[BackgroundTask]:
        """Highest-priority eligible paused task, if it outranks the queue."""
        eligible = [t for t in self._paused if t._resume_eligible]
        if not eligible:
            return None
        cand = max(eligible, key=lambda t: (t.priority.value, -t.id))
        if self._queue and self._top_queued().priority.value > cand.priority.value:
            return None  # a higher-priority queued task should take the slot
        return cand

    def _auto_pause(self, task: BackgroundTask) -> None:
        """Preempt a running task: pause it and free its slot."""
        task.token.pause()
        task._auto_paused = True
        task._resume_eligible = True
        if task in self._running:
            self._running.remove(task)
        self._paused.append(task)
        task._set_state(TaskState.PAUSED)
        log.info("Task preempted: #%d %s", task.id, task.name)

    def _do_resume(self, task: BackgroundTask) -> None:
        """Resume a paused task into a running slot."""
        task.token.resume()
        task._auto_paused = False
        task._resume_eligible = False
        if task in self._paused:
            self._paused.remove(task)
        self._running.append(task)
        task._set_state(TaskState.RUNNING)
        log.info("Task resumed: #%d %s", task.id, task.name)

    def _request_pause(self, task: BackgroundTask) -> None:
        """User pause: free the slot but keep the task off the resume list."""
        if not task.supports_pause or task.state is not TaskState.RUNNING:
            return
        task.token.pause()
        task._auto_paused = False
        task._resume_eligible = False
        if task in self._running:
            self._running.remove(task)
        self._paused.append(task)
        task._set_state(TaskState.PAUSED)
        log.info("Task paused: #%d %s", task.id, task.name)
        self._pump()

    def _request_resume(self, task: BackgroundTask) -> None:
        """User resume: mark eligible and let the scheduler place it."""
        if task.state is not TaskState.PAUSED:
            return
        task._resume_eligible = True
        self._pump()

    def _emit_counts(self) -> None:
        self.counts_changed.emit(
            len(self._running), len(self._queue), len(self._paused)
        )

    def _on_task_finished(self, task: BackgroundTask) -> None:
        for bucket in (self._running, self._queue, self._paused):
            if task in bucket:
                bucket.remove(task)
        # Transient maintenance tasks are not kept around once finished.
        if not task.transient:
            self._history.append(task)
            del self._history[: -self.HISTORY_LIMIT]
        log.info(
            "Task finished: #%d %s → %s (%.1fs wall, %.1fs CPU)",
            task.id,
            task.name,
            task.state.name.lower(),
            task.elapsed_seconds,
            task.cpu_seconds,
        )
        self.task_finished.emit(task)
        self._pump()

    def _cleanup_expired_tasks(self) -> None:
        """Remove finished tasks from history if they've been done for 5 minutes."""
        if not _auto_cleanup_enabled():
            return
        now = time.time()
        expired = []
        for task in self._history:
            if (
                task.finished_at is not None
                and now - task.finished_at >= _AUTO_CLEANUP_SECONDS
            ):
                expired.append(task)
        for task in expired:
            self._history.remove(task)
            log.info("Auto-cleanup: #%d %s removed from history", task.id, task.name)
        if expired:
            self._emit_counts()  # nudge the Task Manager window to rebuild


_manager: Optional[TaskManager] = None


def get_task_manager() -> TaskManager:
    """Application-wide singleton (created on first use, UI thread)."""
    global _manager
    if _manager is None:
        _manager = TaskManager()
    return _manager
