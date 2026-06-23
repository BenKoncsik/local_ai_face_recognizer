"""Task Manager window — background tasks + a live performance monitor.

Two tabs:

* **Tasks** — running / paused / queued / finished tasks with progress, elapsed
  wall-clock and CPU time, a per-task priority control (raise / lower) and
  pause / resume / stop / restart.  Rows refresh once a second.
* **Performance** — a Windows-Task-Manager-style monitor: per-core CPU graphs
  (or a single app-CPU graph), a memory-over-time graph, and CPU / GPU / RAM
  readouts.  A toggle switches between *this app only* and *the whole machine*.
"""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime
from typing import Deque, List, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
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

_GRAPH_HISTORY = 60  # samples kept per graph (≈ last minute at 1 s tick)


def _fit_button(btn: QPushButton) -> None:
    """Widen a button so its label is never clipped (incl. macOS end-caps).

    Qt's native ``sizeHint`` underestimates the width some platform styles need
    for their rounded end-caps, which clipped the last character of longer
    labels (e.g. *Újraindítás*).  Sizing from the font metrics plus generous
    horizontal padding keeps every label fully visible across platforms and
    languages.
    """
    btn.setMinimumHeight(28)
    width = btn.fontMetrics().horizontalAdvance(btn.text()) + 32
    btn.setMinimumWidth(width)


class _Sparkline(QWidget):
    """Lightweight filled time-series graph (Task-Manager style, 0..maximum)."""

    def __init__(
        self,
        title: str = "",
        color: str = "#4caf50",
        maximum: float = 100.0,
        history: int = _GRAPH_HISTORY,
        autoscale: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._title = title
        self._color = QColor(color)
        self._maximum = maximum
        # Autoscaling graphs (e.g. disk I/O, whose rate has no fixed ceiling)
        # grow their maximum to fit the largest sample seen so far.
        self._autoscale = autoscale
        self._base_maximum = maximum
        self._value_text = ""
        self._values: Deque[float] = deque([0.0] * history, maxlen=history)
        self.setMinimumSize(120, 64)

    def set_title(self, title: str) -> None:
        self._title = title
        self.update()

    def set_value_text(self, text: str) -> None:
        self._value_text = text
        self.update()

    def add_sample(self, value: float) -> None:
        v = max(0.0, float(value))
        self._values.append(v)
        if self._autoscale:
            peak = max(self._values)
            # Round the ceiling up to a tidy headroom so the line never pins.
            self._maximum = max(self._base_maximum, peak * 1.15)
        self.update()

    def clear_samples(self) -> None:
        self._values = deque([0.0] * self._values.maxlen, maxlen=self._values.maxlen)
        if self._autoscale:
            self._maximum = self._base_maximum
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ANN001
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        p.fillRect(self.rect(), QColor("#181825"))
        p.setPen(QPen(QColor("#313244")))
        p.drawRect(0, 0, w - 1, h - 1)

        # Horizontal grid (25 / 50 / 75 %).
        p.setPen(QPen(QColor("#26263a")))
        for frac in (0.25, 0.5, 0.75):
            y = int(h - frac * h)
            p.drawLine(1, y, w - 2, y)

        n = len(self._values)
        if n >= 2 and self._maximum > 0:
            step = (w - 2) / (n - 1)

            def _y(v: float) -> float:
                return h - 2 - (min(v, self._maximum) / self._maximum) * (h - 4)

            path = QPainterPath()
            path.moveTo(1, h - 1)
            for i, v in enumerate(self._values):
                path.lineTo(1 + i * step, _y(v))
            path.lineTo(1 + (n - 1) * step, h - 1)
            path.closeSubpath()
            fill = QColor(self._color)
            fill.setAlpha(70)
            p.fillPath(path, fill)

            p.setPen(QPen(self._color, 1.5))
            prev = None
            for i, v in enumerate(self._values):
                cur = (1 + i * step, _y(v))
                if prev is not None:
                    p.drawLine(int(prev[0]), int(prev[1]), int(cur[0]), int(cur[1]))
                prev = cur

        p.setPen(QPen(QColor("#cdd6f4")))
        if self._title:
            p.drawText(5, 14, self._title)
        if self._value_text:
            p.drawText(
                self.rect().adjusted(0, 0, -5, 0),
                Qt.AlignRight | Qt.AlignTop,
                self._value_text,
            )


def _read_gpu_percent() -> Optional[float]:
    """Best-effort GPU utilisation %, or None when not available.

    NVIDIA GPUs are read via ``pynvml`` when present.  macOS / other GPUs have
    no dependency-free way to read utilisation, so this returns None there.
    """
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
    except Exception:  # noqa: BLE001 — optional, platform-dependent
        return None


class PerformancePanel(QWidget):
    """Live machine/app resource monitor.

    Two independent toggles drive what is shown:

    * **Scope** — *App only* vs. *Whole machine* (drives the memory, I/O and the
      CPU readout / combined CPU graph).
    * **View** — *Per core* (one graph per logical core — e.g. two on a
      1-core / 2-thread CPU) vs. *Combined* (a single total graph).

    Below the CPU section there are always a memory-over-time graph and a disk
    read/write throughput graph.  Every metric is read through ``psutil`` and is
    guarded so unsupported combinations (e.g. per-process I/O on macOS) simply
    show *n/a* instead of breaking the panel.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._app_only = False
        self._combined = False
        self._proc = None
        self._ncores = 1
        self._core_graphs: List[_Sparkline] = []
        self._app_cpu_graph: Optional[_Sparkline] = None
        self._cpu_total_graph: Optional[_Sparkline] = None
        self._io_last: Optional[tuple] = None       # (read_bytes, write_bytes)
        self._io_last_wall: Optional[float] = None

        try:
            import psutil

            self._psutil = psutil
            self._proc = psutil.Process()
            self._proc.cpu_percent(None)            # prime per-process sampler
            psutil.cpu_percent(percpu=True)         # prime per-core sampler
            self._ncores = psutil.cpu_count(logical=True) or 1
        except Exception:  # noqa: BLE001 — psutil optional
            self._psutil = None

        layout = QVBoxLayout(self)

        # --- Header: toggles + readouts ---
        header = QHBoxLayout()
        self._toggle = QPushButton()
        self._toggle.setCheckable(True)
        self._toggle.setToolTip(t("perf_app_only_tip"))
        self._toggle.toggled.connect(self._on_scope_toggle)
        header.addWidget(self._toggle)
        self._view_toggle = QPushButton()
        self._view_toggle.setCheckable(True)
        self._view_toggle.setToolTip(t("perf_view_combined_tip"))
        self._view_toggle.toggled.connect(self._on_view_toggle)
        header.addWidget(self._view_toggle)
        self._update_toggle_labels()
        header.addStretch(1)
        self._cpu_label = QLabel()
        self._gpu_label = QLabel()
        self._ram_label = QLabel()
        self._io_label = QLabel()
        for lbl in (self._cpu_label, self._gpu_label, self._ram_label,
                    self._io_label):
            lbl.setStyleSheet("font-weight: bold; padding: 0 8px;")
            header.addWidget(lbl)
        layout.addLayout(header)

        if self._psutil is None:
            layout.addWidget(QLabel(t("perf_psutil_missing")))
            layout.addStretch(1)
            return

        # --- CPU section ---
        self._cpu_section_label = QLabel()
        self._cpu_section_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._cpu_section_label)
        self._cpu_container = QWidget()
        self._cpu_grid = QGridLayout(self._cpu_container)
        self._cpu_grid.setContentsMargins(0, 0, 0, 0)
        self._cpu_grid.setSpacing(4)
        layout.addWidget(self._cpu_container, stretch=3)

        # --- Memory section ---
        self._mem_section_label = QLabel()
        self._mem_section_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._mem_section_label)
        self._mem_graph = _Sparkline(t("perf_ram"), color="#89b4fa")
        layout.addWidget(self._mem_graph, stretch=2)

        # --- Disk I/O section ---
        self._io_section_label = QLabel()
        self._io_section_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._io_section_label)
        io_row = QHBoxLayout()
        io_row.setContentsMargins(0, 0, 0, 0)
        io_row.setSpacing(4)
        self._io_read_graph = _Sparkline(
            t("perf_io_read"), color="#94e2d5", maximum=1.0, autoscale=True
        )
        self._io_write_graph = _Sparkline(
            t("perf_io_write"), color="#f9e2af", maximum=1.0, autoscale=True
        )
        io_row.addWidget(self._io_read_graph)
        io_row.addWidget(self._io_write_graph)
        layout.addLayout(io_row, stretch=2)

        self._build_cpu_graphs()
        self._update_scope_labels()

    # ------------------------------------------------------------------

    def _clear_cpu_graphs(self) -> None:
        while self._cpu_grid.count():
            item = self._cpu_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._core_graphs = []
        self._app_cpu_graph = None
        self._cpu_total_graph = None

    def _build_cpu_graphs(self) -> None:
        self._clear_cpu_graphs()
        if self._combined:
            # One combined graph; its meaning follows the scope toggle.
            if self._app_only:
                self._cpu_section_label.setText(t("perf_cpu_section_app"))
                self._app_cpu_graph = _Sparkline(t("perf_app_cpu"), color="#a6e3a1")
                self._cpu_grid.addWidget(self._app_cpu_graph, 0, 0)
            else:
                self._cpu_section_label.setText(t("perf_cpu_total"))
                self._cpu_total_graph = _Sparkline(
                    t("perf_cpu_total"), color="#4caf50"
                )
                self._cpu_grid.addWidget(self._cpu_total_graph, 0, 0)
        else:
            # Per logical core — one graph per hardware thread (e.g. two on a
            # 1-core / 2-thread CPU).  Always machine-wide: that is the only
            # per-core breakdown psutil exposes.
            self._cpu_section_label.setText(t("perf_cpu_section"))
            cols = 4 if self._ncores > 4 else max(1, self._ncores)
            for i in range(self._ncores):
                g = _Sparkline(t("perf_core", n=i + 1), color="#4caf50")
                self._core_graphs.append(g)
                self._cpu_grid.addWidget(g, i // cols, i % cols)

    def _update_toggle_labels(self) -> None:
        """Each toggle's label reflects what it is currently showing."""
        self._toggle.setText(
            t("perf_app_only") if self._app_only else t("perf_scope_machine")
        )
        self._view_toggle.setText(
            t("perf_view_combined") if self._combined else t("perf_view_percore")
        )
        _fit_button(self._toggle)
        _fit_button(self._view_toggle)

    def _on_scope_toggle(self, checked: bool) -> None:
        self._app_only = checked
        self._update_toggle_labels()
        self._rebuild_after_toggle()

    def _on_view_toggle(self, checked: bool) -> None:
        self._combined = checked
        self._update_toggle_labels()
        self._rebuild_after_toggle()

    def _update_scope_labels(self) -> None:
        """Refresh section headers to explicitly show whether scope is app or machine."""
        if not hasattr(self, "_mem_section_label"):
            return
        if self._app_only:
            self._mem_section_label.setText(t("perf_ram_section_app"))
            self._io_section_label.setText(t("perf_io_section_app"))
        else:
            self._mem_section_label.setText(t("perf_ram_section_machine"))
            self._io_section_label.setText(t("perf_io_section_machine"))

    def _rebuild_after_toggle(self) -> None:
        if self._psutil is None:
            return
        self._build_cpu_graphs()
        self._update_scope_labels()
        self._mem_graph.clear_samples()
        self._io_read_graph.clear_samples()
        self._io_write_graph.clear_samples()
        # Re-prime so the first post-toggle sample is sane.
        try:
            self._proc.cpu_percent(None)
            self._psutil.cpu_percent(percpu=True)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------

    def _sample_io(self) -> None:
        """Push disk read/write throughput (MB/s) into the I/O graphs."""
        ps = self._psutil
        now = time.time()
        counters = None
        try:
            if self._app_only:
                io = self._proc.io_counters()  # unavailable on macOS
                counters = (io.read_bytes, io.write_bytes)
            else:
                io = ps.disk_io_counters()
                if io is not None:
                    counters = (io.read_bytes, io.write_bytes)
        except Exception:  # noqa: BLE001 — io_counters absent on macOS, etc.
            counters = None

        io_scope = t("perf_io_app") if self._app_only else f"{t('perf_io')} {t('perf_scope_sys_short')}"
        if counters is None:
            na = t("perf_na")
            self._io_read_graph.set_value_text(na)
            self._io_write_graph.set_value_text(na)
            self._io_label.setText(f"{io_scope} {na}")
            self._io_last = None
            self._io_last_wall = None
            return

        dt = (now - self._io_last_wall) if self._io_last_wall else 0.0
        if self._io_last is not None and dt > 0:
            read_mb = max(0.0, (counters[0] - self._io_last[0]) / dt / (1024 * 1024))
            write_mb = max(0.0, (counters[1] - self._io_last[1]) / dt / (1024 * 1024))
            self._io_read_graph.add_sample(read_mb)
            self._io_read_graph.set_value_text(t("perf_io_rate", mb=read_mb))
            self._io_write_graph.add_sample(write_mb)
            self._io_write_graph.set_value_text(t("perf_io_rate", mb=write_mb))
            self._io_label.setText(f"{io_scope} ↓{read_mb:.1f} ↑{write_mb:.1f} MB/s")
        self._io_last = counters
        self._io_last_wall = now

    def sample(self) -> None:
        """Pull one sample from psutil and push it into the graphs/labels."""
        if self._psutil is None:
            return
        ps = self._psutil
        gpu = _read_gpu_percent()
        gpu_text = f"{gpu:.0f}%" if gpu is not None else t("perf_na")
        self._gpu_label.setText(f"{t('perf_gpu')} {gpu_text}")

        # Machine per-logical-core sample feeds the per-core graphs (in either
        # scope) and the machine total readout/combined graph.
        try:
            per = ps.cpu_percent(percpu=True)
        except Exception:  # noqa: BLE001
            per = []
        cpu_total = sum(per) / len(per) if per else 0.0
        if not self._combined:
            for g, v in zip(self._core_graphs, per):
                g.add_sample(v)
                g.set_value_text(f"{v:.0f}%")

        if self._app_only:
            try:
                app_cpu = self._proc.cpu_percent(None) / max(self._ncores, 1)
                mi = self._proc.memory_info().rss
            except Exception:  # noqa: BLE001
                return
            total = ps.virtual_memory().total
            mem_pct = (mi / total * 100.0) if total else 0.0
            if self._combined and self._app_cpu_graph is not None:
                self._app_cpu_graph.add_sample(app_cpu)
                self._app_cpu_graph.set_value_text(f"{app_cpu:.0f}%")
            self._mem_graph.add_sample(mem_pct)
            self._mem_graph.set_value_text(f"{mi // (1024 * 1024)} MB")
            self._cpu_label.setText(f"{t('perf_app_cpu')} {app_cpu:.0f}%")
            self._ram_label.setText(
                f"{t('perf_ram_app')} "
                + t("perf_ram_value_app", used=mi // (1024 * 1024), pct=mem_pct)
            )
        else:
            try:
                vm = ps.virtual_memory()
            except Exception:  # noqa: BLE001
                return
            if self._combined and self._cpu_total_graph is not None:
                self._cpu_total_graph.add_sample(cpu_total)
                self._cpu_total_graph.set_value_text(f"{cpu_total:.0f}%")
            self._mem_graph.add_sample(vm.percent)
            self._mem_graph.set_value_text(f"{vm.percent:.0f}%")
            self._cpu_label.setText(
                f"{t('perf_cpu')} {t('perf_scope_sys_short')} {cpu_total:.0f}%"
            )
            self._ram_label.setText(
                f"{t('perf_ram')} {t('perf_scope_sys_short')} "
                + t(
                    "perf_ram_value",
                    used=vm.used // (1024 * 1024),
                    total=vm.total // (1024 * 1024),
                    pct=vm.percent,
                )
            )

        self._sample_io()


class TaskManagerDialog(QDialog):
    """Live list of background tasks + a performance monitor tab."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("tasks_title"))
        self.setMinimumSize(900, 380)
        self.resize(1080, 480)
        self.setModal(False)

        layout = QVBoxLayout(self)
        self._tabs = QTabWidget()
        layout.addWidget(self._tabs)

        # --- Tasks tab ---
        tasks_tab = QWidget()
        tasks_layout = QVBoxLayout(tasks_tab)
        tasks_layout.setContentsMargins(0, 4, 0, 0)

        header_row = QHBoxLayout()
        self._summary_label = QLabel()
        self._summary_label.setStyleSheet("color: #888; font-size: 11px;")
        header_row.addWidget(self._summary_label)
        header_row.addStretch(1)
        self._clear_btn = QPushButton(t("tasks_clear_finished"))
        self._clear_btn.setToolTip(t("tasks_clear_finished_tip"))
        self._clear_btn.clicked.connect(self._on_clear_finished)
        _fit_button(self._clear_btn)
        header_row.addWidget(self._clear_btn)
        tasks_layout.addLayout(header_row)

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
        # Taller rows so the action buttons (and their text) are never clipped.
        self._table.verticalHeader().setDefaultSectionSize(38)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.NoSelection)
        header = self._table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(COL_NAME, QHeaderView.Interactive)
        self._table.setColumnWidth(COL_NAME, 200)
        header.setSectionResizeMode(COL_MESSAGE, QHeaderView.Stretch)
        header.setSectionResizeMode(COL_PROGRESS, QHeaderView.Fixed)
        self._table.setColumnWidth(COL_PROGRESS, 130)
        for col in (COL_PRIORITY, COL_STATE, COL_STARTED, COL_ELAPSED, COL_CPU):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        # ResizeToContents mismeasures cells that hold a widget (Qt sizes the
        # delegate, not the button row), which clipped the action buttons.  A
        # fixed, generous width keeps every label fully visible on EN and HU.
        header.setSectionResizeMode(COL_ACTIONS, QHeaderView.Fixed)
        self._table.setColumnWidth(COL_ACTIONS, 250)
        tasks_layout.addWidget(self._table)

        self._empty_label = QLabel(t("tasks_empty"))
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet("color: #888;")
        tasks_layout.addWidget(self._empty_label)

        self._tabs.addTab(tasks_tab, t("tasks_tab_tasks"))

        # --- Performance tab ---
        self._perf = PerformancePanel()
        self._tabs.addTab(self._perf, t("tasks_tab_perf"))

        manager = get_task_manager()
        manager.task_added.connect(self._rebuild)
        manager.task_finished.connect(self._rebuild)
        manager.counts_changed.connect(lambda *_: self._rebuild())

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        self._rebuild()

    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: ANN001
        self._timer.stop()
        super().closeEvent(event)

    def _tick(self) -> None:
        self._refresh_dynamic()
        # Only sample the perf monitor while its tab is visible (cheap, but no
        # point churning when hidden).
        if self._tabs.currentWidget() is self._perf:
            self._perf.sample()

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
        name_item.setToolTip(task.name)
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
        prio_label.setMinimumWidth(60)
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
        h.setContentsMargins(4, 2, 4, 2)
        h.setSpacing(6)

        pause_btn = QPushButton(t("tasks_pause_btn"))
        pause_btn.clicked.connect(lambda _=False, tk=task: self._on_pause_resume(tk))
        h.addWidget(pause_btn)

        cancel_btn = QPushButton(t("tasks_cancel_btn"))
        cancel_btn.clicked.connect(lambda _=False, tk=task: tk.cancel())
        h.addWidget(cancel_btn)

        restart_btn = QPushButton(t("tasks_restart_btn"))
        restart_btn.clicked.connect(lambda _=False, tk=task: self._on_restart(tk))
        h.addWidget(restart_btn)

        # Taller, text-fitted buttons so labels are never clipped (the action
        # column holds widgets, which Qt's ResizeToContents mismeasures).
        h.addStretch(1)
        for btn in (pause_btn, cancel_btn, restart_btn):
            _fit_button(btn)

        self._table.setCellWidget(row, COL_ACTIONS, actions)
        self._update_row(row, task)

    def _update_row(self, row: int, task: BackgroundTask) -> None:
        state_item = self._table.item(row, COL_STATE)
        if task.is_preempted:
            state_item.setText(t("tasks_state_preempted"))
        else:
            state_item.setText(t(_STATE_KEYS[task.state]))
        state_item.setForeground(QColor(_STATE_COLORS[task.state]))

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
        msg_item = self._table.item(row, COL_MESSAGE)
        msg_item.setText(message or "")
        msg_item.setToolTip(message or "")

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
            _fit_button(pause_btn)  # Resume/Pause differ in width — refit on change
            cancel_btn.setVisible(not task.state.is_final)
            restart_btn.setVisible(task.state.is_final)

    def _refresh_dynamic(self) -> None:
        """Per-second tick: progress, elapsed, task-count summary."""
        for row in range(self._table.rowCount()):
            item = self._table.item(row, COL_NAME)
            task = item.data(Qt.UserRole) if item else None
            if isinstance(task, BackgroundTask):
                self._update_row(row, task)

        manager = get_task_manager()
        self._summary_label.setText(
            t(
                "tasks_header_summary",
                running=manager.running_count,
                queued=manager.queued_count,
                paused=manager.paused_count,
            )
        )
        # Only enable "Clear finished" when there is finished history to clear.
        finished = len(manager.all_tasks()) - manager.active_count
        self._clear_btn.setEnabled(finished > 0)

    def _on_clear_finished(self) -> None:
        get_task_manager().clear_finished()
        self._rebuild()

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
