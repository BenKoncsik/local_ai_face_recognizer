"""Unified background-task system.

Every long-running operation (export, backup, batch re-analysis, …) should be
submitted through :func:`app.tasks.manager.get_task_manager` so the UI thread
never blocks and the Task Manager window can show/pause/cancel it.
"""

from app.tasks.manager import (  # noqa: F401
    BackgroundTask,
    TaskContext,
    TaskManager,
    TaskPriority,
    TaskState,
    get_task_manager,
)
