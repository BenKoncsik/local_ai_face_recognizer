"""Unit tests for the priority + preemption scheduler in app.tasks.manager.

The scheduling *decisions* (priority ordering, preempting a lower-priority
running task, auto-resume when a slot frees, manual pause/resume) are tested
deterministically without real worker threads: ``BackgroundTask._start`` is
faked to flip the task to RUNNING in place, and task completion is simulated by
calling ``TaskManager._on_task_finished`` directly.  This isolates the
algorithm from QThread/event-loop timing.
"""

from __future__ import annotations

import time

import pytest

from app.tasks.manager import (
    BackgroundTask,
    TaskManager,
    TaskPriority,
    TaskState,
)


@pytest.fixture(autouse=True)
def _fake_start(monkeypatch):
    """Make ``_start`` flip to RUNNING in place — no real QThread."""

    def fake_start(self: BackgroundTask) -> None:
        self.started_at = time.time()
        self._set_state(TaskState.RUNNING)

    monkeypatch.setattr(BackgroundTask, "_start", fake_start)


def _noop(ctx):  # noqa: ANN001 — never actually invoked (start is faked)
    return None


def _finish(mgr: TaskManager, task: BackgroundTask,
            state: TaskState = TaskState.COMPLETED) -> None:
    """Simulate a task reaching a final state and notify the manager."""
    task.finished_at = time.time()
    task._set_state(state)
    mgr._on_task_finished(task)


def test_priority_ordering(qapp):
    """Queued tasks start highest-priority first, FIFO within a level."""
    mgr = TaskManager(max_concurrent=1)
    blocker = mgr.submit("block", _noop, priority=TaskPriority.NORMAL)
    assert blocker.state is TaskState.RUNNING

    low = mgr.submit("low", _noop, priority=TaskPriority.LOW)
    high = mgr.submit("high", _noop, priority=TaskPriority.HIGH)
    norm = mgr.submit("norm", _noop, priority=TaskPriority.NORMAL)
    assert {low.state, high.state, norm.state} == {TaskState.QUEUED}

    _finish(mgr, blocker)
    assert high.state is TaskState.RUNNING  # HIGH jumps the queue

    _finish(mgr, high)
    assert norm.state is TaskState.RUNNING  # NORMAL before LOW

    _finish(mgr, norm)
    assert low.state is TaskState.RUNNING


def test_high_priority_preempts_running_low(qapp):
    """A HIGH task auto-pauses a running, pausable LOW task to get a slot."""
    mgr = TaskManager(max_concurrent=1)
    export = mgr.submit("export", _noop, supports_pause=True, priority=TaskPriority.LOW)
    assert export.state is TaskState.RUNNING

    scan = mgr.submit("scan", _noop, supports_pause=True, priority=TaskPriority.HIGH)
    assert scan.state is TaskState.RUNNING
    assert export.state is TaskState.PAUSED
    assert export.is_preempted
    assert export in mgr._paused and scan in mgr._running


def test_preempted_task_auto_resumes(qapp):
    """When the preempting task finishes, the paused task resumes by itself."""
    mgr = TaskManager(max_concurrent=1)
    export = mgr.submit("export", _noop, supports_pause=True, priority=TaskPriority.LOW)
    scan = mgr.submit("scan", _noop, supports_pause=True, priority=TaskPriority.HIGH)
    assert export.is_preempted

    _finish(mgr, scan)
    assert export.state is TaskState.RUNNING
    assert not export.is_preempted
    assert export in mgr._running


def test_non_pausable_low_is_not_preempted(qapp):
    """A running task that cannot pause blocks a higher-priority arrival."""
    mgr = TaskManager(max_concurrent=1)
    export = mgr.submit("export", _noop, supports_pause=False, priority=TaskPriority.LOW)
    scan = mgr.submit("scan", _noop, supports_pause=True, priority=TaskPriority.HIGH)
    assert export.state is TaskState.RUNNING
    assert scan.state is TaskState.QUEUED  # waits — export can't be preempted

    _finish(mgr, export)
    assert scan.state is TaskState.RUNNING


def test_manual_pause_frees_slot_and_resume_waits(qapp):
    """User pause frees the slot; resume only takes effect when one is free."""
    mgr = TaskManager(max_concurrent=1)
    first = mgr.submit("first", _noop, supports_pause=True, priority=TaskPriority.NORMAL)
    second = mgr.submit("second", _noop, priority=TaskPriority.NORMAL)
    assert first.state is TaskState.RUNNING and second.state is TaskState.QUEUED

    first.pause()
    assert first.state is TaskState.PAUSED
    assert not first.is_preempted          # manual pause, not preemption
    assert second.state is TaskState.RUNNING  # took the freed slot

    first.resume()
    assert first.state is TaskState.PAUSED  # no free slot yet — still waiting

    _finish(mgr, second)
    assert first.state is TaskState.RUNNING


def test_resume_reclaims_slot_from_lower_priority(qapp):
    """Resuming a paused HIGH task preempts a running lower-priority task."""
    mgr = TaskManager(max_concurrent=1)
    low = mgr.submit("export", _noop, supports_pause=True, priority=TaskPriority.LOW)
    high = mgr.submit("scan", _noop, supports_pause=True, priority=TaskPriority.HIGH)
    assert high.state is TaskState.RUNNING and low.is_preempted

    # User pauses the scan → the preempted export resumes to fill the slot.
    high.pause()
    assert high.state is TaskState.PAUSED
    assert low.state is TaskState.RUNNING

    # User resumes the scan → it must reclaim the slot from the lower export.
    high.resume()
    assert high.state is TaskState.RUNNING
    assert low.state is TaskState.PAUSED


def test_set_priority_reorders_queue(qapp):
    """Raising a queued task's priority moves it ahead in the queue."""
    mgr = TaskManager(max_concurrent=1)
    blocker = mgr.submit("block", _noop, priority=TaskPriority.NORMAL)
    a = mgr.submit("a", _noop, priority=TaskPriority.LOW)
    b = mgr.submit("b", _noop, priority=TaskPriority.LOW)

    b.set_priority(TaskPriority.HIGH)
    _finish(mgr, blocker)
    assert b.state is TaskState.RUNNING   # promoted past the earlier LOW task
    assert a.state is TaskState.QUEUED


def test_counts_and_active_count(qapp):
    """active_count reflects running + queued + paused."""
    mgr = TaskManager(max_concurrent=1)
    mgr.submit("r", _noop, supports_pause=True, priority=TaskPriority.LOW)
    mgr.submit("q", _noop, priority=TaskPriority.LOW)
    assert mgr.running_count == 1
    assert mgr.queued_count == 1
    assert mgr.active_count == 2

    high = mgr.submit("h", _noop, supports_pause=True, priority=TaskPriority.HIGH)
    # 'running' preempted → paused; high runs; queued still waits.
    assert mgr.running_count == 1
    assert mgr.paused_count == 1
    assert mgr.queued_count == 1
    assert mgr.active_count == 3
    assert high.state is TaskState.RUNNING


def test_transient_task_not_kept_in_history(qapp):
    """A transient task disappears once finished; a normal one stays in history."""
    mgr = TaskManager(max_concurrent=1)
    normal = mgr.submit("repair-visible", _noop, priority=TaskPriority.NORMAL)
    _finish(mgr, normal)
    assert normal in mgr.all_tasks()  # normal tasks linger in history

    transient = mgr.submit("startup-maintenance", _noop, transient=True)
    _finish(mgr, transient)
    assert transient not in mgr.all_tasks()  # vanishes once done


def test_cancel_queued_task(qapp):
    """Cancelling a queued task removes it without starting it."""
    mgr = TaskManager(max_concurrent=1)
    blocker = mgr.submit("block", _noop, priority=TaskPriority.NORMAL)
    queued = mgr.submit("q", _noop, priority=TaskPriority.NORMAL)
    assert queued.state is TaskState.QUEUED

    queued.cancel()
    assert queued.state is TaskState.CANCELLED
    # Finishing the blocker pumps the scheduler, which discards the cancelled
    # task instead of starting it.
    _finish(mgr, blocker)
    assert queued.state is TaskState.CANCELLED
    assert queued not in mgr._queue
    assert mgr.running_count == 0


def test_auto_cleanup_finished_tasks(qapp, monkeypatch):
    """Finished tasks are removed from history 5 minutes after completion."""
    mgr = TaskManager(max_concurrent=1)
    task = mgr.submit("work", _noop)
    _finish(mgr, task)
    assert task in mgr._history

    # Simulate 5 minutes passing — set finished_at to 5+ minutes ago
    monkeypatch.setattr(
        task, "finished_at", time.time() - (5 * 60 + 1),
        raising=False
    )
    mgr._cleanup_expired_tasks()
    assert task not in mgr._history  # removed after 5 minutes

    # Task finished < 5 minutes ago should not be removed
    task2 = mgr.submit("work2", _noop)
    _finish(mgr, task2)
    monkeypatch.setattr(
        task2, "finished_at", time.time() - (4 * 60),  # 4 minutes ago
        raising=False
    )
    mgr._cleanup_expired_tasks()
    assert task2 in mgr._history  # not yet removed
