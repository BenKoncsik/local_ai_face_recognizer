"""Task Manager window — live view of all background tasks.

Non-modal dialog showing running / paused / queued / finished tasks with
progress, elapsed wall-clock and CPU time, a per-task **priority** control
(raise / lower) plus pause / resume / stop / restart.  Rows refresh from the
:class:`~app.tasks.manager.TaskManager` once a second (and immediately on task
add/finish/count changes); the header shows process-level CPU and RAM usage
plus a running/queued/paused summary.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.tasks import BackgroundTask, TaskPriority, TaskState, get_task_manager
from app.ui.i18n import t

_STATE_KEYS = {
    TaskState.QUEUED: "tasks_state_queued",
    TaskState.RUNNING: "tasks_state_running",
    TaskState.PAUSED: "tasks_state_paused",
    TaskState.COMPLETED: "tasks_state_completed",
    TaskState.FAILED: "tasks_state_failed",
    TaskState.CANCELLED: "tasks_state_cancelled",
}

_STATE_COLORS = {
    TaskState.QUEUED: "#888",
    TaskState.RUNNING: "#4caf50",
    TaskState.PAUSED: "#ffcc00",
    TaskState.COMPLETED: "#4caf50",
    TaskState.FAILED: "#f44336",
    TaskState.CANCELLED: "#f57c00",
}

_PRIORITY_KEYS = {
    TaskPriority.HIGH: "tasks_priority_high",
    TaskPriority.NORMAL: "tasks_priority_normal",
    TaskPriority.LOW: "tasks_priority_low",
}

_PRIORITY_COLORS = {
    TaskPriority.HIGH: "#ff7043",
    TaskPriority.NORMAL: "#90caf9",
    TaskPriority.LOW: "#9e9e9e",
}

(
    COL_NAME,
    COL_PRIORITY,
    COL_STATE,
    COL_PROGRESS,
    COL_MESSAGE,
    COL_STARTED,
    COL_ELAPSED,
    COL_CPU,
    COL_ACTIONS,
) = range(9)


class TaskManagerDialog(QDialog):
    """Live list of background tasks with priority + lifecycle controls."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("tasks_title"))
        self.setMinimumSize(900, 360)
        self.resize(1040, 460)
        self.setModal(False)

        layout = QVBoxLayout(self)

        self._stats_label = QLabel()
        self._stats_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._stats_label)

        self._table = QTableWidget(0, 9)
        self._table.setHorizontalHeaderLabels(
            [
                t("tasks_col_name"),
                t("tasks_col_priority"),
                t("tasks_col_state"),
                t("tasks_col_progress"),
                t("tasks_col_message"),
                t("tasks_col_started"),
                t("tasks_col_elapsed"),
                t("tasks_col_cpu"),
                "",
            ]
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.NoSelection)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(COL_NAME, QHeaderView.Stretch)
        header.setSectionResizeMode(COL_MESSAGE, QHeaderView.Stretch)
        for col in (
            COL_PRIORITY,
            COL_STATE,
            COL_PROGRESS,
            COL_STARTED,
            COL_ELAPSED,
            COL_CPU,
            COL_ACTIONS,
        ):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        layout.addWidget(self._table)

        self._empty_label = QLabel(t("tasks_empty"))
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet("color: #888;")
        layout.addWidget(self._empty_label)

        manager = get_task_manager()
        manager.task_added.connect(self._rebuild)
        manager.task_finished.connect(self._rebuild)
        manager.counts_changed.connect(lambda *_: self._rebuild())

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._refresh_dynamic)
        self._timer.start()

        self._psutil_proc = None
        try:
            import psutil
            self._psutil_proc = psutil.Process()
            self._psutil_proc.cpu_percent(None)  # prime the sampler
        except Exception:  # noqa: BLE001 — psutil optional
            pass

        self._rebuild()

    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: ANN001
        self._timer.stop()
        super().closeEvent(event)

    def _rebuild(self) -> None:
        """Recreate all rows from the manager's current task list."""
        tasks = get_task_manager().all_tasks()
        self._table.setRowCount(0)
        for task in tasks:
            self._add_row(task)
        has_tasks = bool(tasks)
        self._table.setVisible(has_tasks)
        self._empty_label.setVisible(not has_tasks)
        self._refresh_dynamic()

    def _add_row(self, task: BackgroundTask) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)

        name_item = QTableWidgetItem(task.name)
        name_item.setData(Qt.UserRole, task)
        self._table.setItem(row, COL_NAME, name_item)

        # Priority cell: ▲ <label> ▼
        prio = QWidget()
        ph = QHBoxLayout(prio)
        ph.setContentsMargins(2, 0, 2, 0)
        ph.setSpacing(2)
        up_btn = QPushButton("▲")
        up_btn.setFixedWidth(22)
        up_btn.setToolTip(t("tasks_raise_btn"))
        up_btn.clicked.connect(
            lambda _=False, tk=task: self._on_change_priority(tk, raise_it=True)
        )
        prio_label = QLabel()
        prio_label.setMinimumWidth(64)
        prio_label.setAlignment(Qt.AlignCenter)
        down_btn = QPushButton("▼")
        down_btn.setFixedWidth(22)
        down_btn.setToolTip(t("tasks_lower_btn"))
        down_btn.clicked.connect(
            lambda _=False, tk=task: self._on_change_priority(tk, raise_it=False)
        )
        ph.addWidget(up_btn)
        ph.addWidget(prio_label)
        ph.addWidget(down_btn)
        self._table.setCellWidget(row, COL_PRIORITY, prio)

        self._table.setItem(row, COL_STATE, QTableWidgetItem())
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setTextVisible(True)
        bar.setMinimumWidth(140)
        self._table.setCellWidget(row, COL_PROGRESS, bar)
        self._table.setItem(row, COL_MESSAGE, QTableWidgetItem())

        started = (
            datetime.fromtimestamp(task.started_at).strftime("%H:%M:%S")
            if task.started_at
            else "—"
        )
        self._table.setItem(row, COL_STARTED, QTableWidgetItem(started))
        self._table.setItem(row, COL_ELAPSED, QTableWidgetItem())
        self._table.setItem(row, COL_CPU, QTableWidgetItem())

        actions = QWidget()
        h = QHBoxLayout(actions)
        h.setContentsMargins(2, 0, 2, 0)
        h.setSpacing(4)

        pause_btn = QPushButton(t("tasks_pause_btn"))
        pause_btn.setMinimumWidth(86)
        pause_btn.clicked.connect(lambda _=False, tk=task: self._on_pause_resume(tk))
        h.addWidget(pause_btn)

        cancel_btn = QPushButton(t("tasks_cancel_btn"))
        cancel_btn.setMinimumWidth(86)
        cancel_btn.clicked.connect(lambda _=False, tk=task: tk.cancel())
        h.addWidget(cancel_btn)

        restart_btn = QPushButton(t("tasks_restart_btn"))
        restart_btn.setMinimumWidth(96)
        restart_btn.clicked.connect(lambda _=False, tk=task: self._on_restart(tk))
        h.addWidget(restart_btn)

        self._table.setCellWidget(row, COL_ACTIONS, actions)
        self._update_row(row, task)

    def _update_row(self, row: int, task: BackgroundTask) -> None:
        state_item = self._table.item(row, COL_STATE)
        if task.is_preempted:
            state_item.setText(t("tasks_state_preempted"))
        else:
            state_item.setText(t(_STATE_KEYS[task.state]))
        state_item.setForeground(QColor(_STATE_COLORS[task.state]))

        # Priority cell
        prio_widget = self._table.cellWidget(row, COL_PRIORITY)
        if prio_widget is not None:
            up_btn, prio_label, down_btn = (
                prio_widget.layout().itemAt(i).widget() for i in range(3)
            )
            prio_label.setText(t(_PRIORITY_KEYS[task.priority]))
            prio_label.setStyleSheet(
                f"color: {_PRIORITY_COLORS[task.priority]}; font-weight: bold;"
            )
            adjustable = not task.state.is_final
            up_btn.setVisible(adjustable)
            down_btn.setVisible(adjustable)
            up_btn.setEnabled(adjustable and task.priority is not TaskPriority.HIGH)
            down_btn.setEnabled(adjustable and task.priority is not TaskPriority.LOW)

        bar = self._table.cellWidget(row, COL_PROGRESS)
        if isinstance(bar, QProgressBar):
            if task.state is TaskState.RUNNING and task.progress == 0:
                bar.setRange(0, 0)  # indeterminate until first report
            elif task.state is TaskState.COMPLETED:
                bar.setRange(0, 100)
                bar.setValue(100)
            else:
                bar.setRange(0, 100)
                bar.setValue(task.progress)

        message = task.error if task.state is TaskState.FAILED else task.message
        self._table.item(row, COL_MESSAGE).setText(message or "")

        if task.started_at:
            self._table.item(row, COL_STARTED).setText(
                datetime.fromtimestamp(task.started_at).strftime("%H:%M:%S")
            )
        self._table.item(row, COL_ELAPSED).setText(_fmt_duration(task.elapsed_seconds))
        cpu = task.cpu_seconds
        self._table.item(row, COL_CPU).setText(_fmt_duration(cpu) if cpu else "—")

        actions = self._table.cellWidget(row, COL_ACTIONS)
        if actions is not None:
            pause_btn, cancel_btn, restart_btn = (
                actions.layout().itemAt(i).widget() for i in range(3)
            )
            pause_btn.setVisible(task.supports_pause and not task.state.is_final)
            pause_btn.setText(
                t("tasks_resume_btn")
                if task.state is TaskState.PAUSED
                else t("tasks_pause_btn")
            )
            cancel_btn.setVisible(not task.state.is_final)
            restart_btn.setVisible(task.state.is_final)

    def _refresh_dynamic(self) -> None:
        """Per-second tick: progress, elapsed, process + summary stats."""
        for row in range(self._table.rowCount()):
            item = self._table.item(row, COL_NAME)
            task = item.data(Qt.UserRole) if item else None
            if isinstance(task, BackgroundTask):
                self._update_row(row, task)

        manager = get_task_manager()
        summary = t(
            "tasks_header_summary",
            running=manager.running_count,
            queued=manager.queued_count,
            paused=manager.paused_count,
        )
        proc = ""
        if self._psutil_proc is not None:
            try:
                cpu = self._psutil_proc.cpu_percent(None)
                ram = self._psutil_proc.memory_info().rss / (1024 * 1024)
                proc = t("tasks_process_stats", cpu=cpu, ram=ram)
            except Exception:  # noqa: BLE001
                proc = ""
        else:
            from app.perf import memory_mb
            ram = memory_mb()
            if ram:
                proc = t("tasks_process_stats", cpu=0.0, ram=ram)
        self._stats_label.setText(f"{summary}    {proc}".strip())

    def _on_change_priority(self, task: BackgroundTask, *, raise_it: bool) -> None:
        new_priority = task.priority.next_up if raise_it else task.priority.next_down
        task.set_priority(new_priority)
        self._rebuild()

    def _on_pause_resume(self, task: BackgroundTask) -> None:
        if task.state is TaskState.PAUSED:
            task.resume()
        else:
            task.pause()

    def _on_restart(self, task: BackgroundTask) -> None:
        get_task_manager().restart(task)


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"
