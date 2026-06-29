"""Crash & resource diagnostics — capture evidence for hard freezes.

The deep scan pipeline can drive a machine (notably Windows) into a
memory/CPU-exhaustion *hard freeze* that leaves **no Python traceback**: the
OS either reaps the process or the whole desktop hangs and the user can only
recover with a power-cycle.  When that happens the normal rotating file log
shows nothing useful — the last thing written is whatever stage was running.

This module installs three observability-only aids (it never changes pipeline
behaviour):

* **faulthandler** → a persistent crash log so a *native* fault (segfault,
  access violation, C-level ``abort``) dumps every thread's stack to disk,
  even when Python-level logging can no longer run.
* **sys / threading excepthooks** → uncaught Python exceptions on the main
  thread *and* worker threads are written to the app log and the crash log.
* a low-overhead **resource watchdog** daemon thread that periodically records
  process RSS, system memory and CPU into a dedicated ``resource_monitor.log``,
  and — when free memory crosses a danger threshold — dumps every thread's
  stack to the crash log *just before* the likely freeze, so we can see which
  stage (DBSCAN clustering, multi-variant detection, …) was executing.

Disable with ``FL_RESMON=0``; tune the cadence with ``FL_RESMON_INTERVAL``
(seconds) and the danger threshold with ``FL_RESMON_DANGER_MB`` (MiB free).
"""

from __future__ import annotations

import faulthandler
import logging
import os
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, TextIO

log = logging.getLogger(__name__)

# Kept open for the whole process lifetime: faulthandler writes to it from a
# native fault / signal context, so the file (with a real fileno) must outlive
# any single function call.
_crash_file: Optional[TextIO] = None
_watchdog_started = False


def install_crash_handlers(log_dir: Path) -> Optional[Path]:
    """Enable faulthandler and exception hooks.  Returns the crash-log path.

    Best-effort: any failure is logged and swallowed so diagnostics can never
    prevent the application from starting.
    """
    global _crash_file
    if _crash_file is not None:
        return Path(_crash_file.name)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        crash_path = log_dir / "faulthandler.log"
        # Line-buffered append; explicit file (not sys.stderr, which is None in
        # a windowed frozen build) so faulthandler always has a valid fileno.
        _crash_file = open(crash_path, "a", buffering=1, encoding="utf-8")
        _crash_file.write(
            f"\n===== session start {time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"pid={os.getpid()} platform={sys.platform} =====\n"
        )
        _crash_file.flush()
        faulthandler.enable(file=_crash_file, all_threads=True)
        log.info("faulthandler enabled -> %s", crash_path)
    except Exception as exc:  # noqa: BLE001 — diagnostics must never break startup
        log.warning("Could not enable faulthandler: %s", exc)
        return None

    _install_excepthooks()
    return Path(_crash_file.name)


def _dump_exc(prefix: str, exc_type, exc_value, exc_tb) -> None:
    if _crash_file is None:
        return
    try:
        import traceback

        _crash_file.write(f"\n--- {prefix} {time.strftime('%H:%M:%S')} ---\n")
        traceback.print_exception(exc_type, exc_value, exc_tb, file=_crash_file)
        _crash_file.flush()
    except Exception:  # noqa: BLE001
        pass


def _install_excepthooks() -> None:
    prev_main = sys.excepthook

    def _main_hook(exc_type, exc_value, exc_tb):
        log.critical(
            "UNCAUGHT exception on main thread",
            exc_info=(exc_type, exc_value, exc_tb),
        )
        _dump_exc("uncaught main-thread exception", exc_type, exc_value, exc_tb)
        prev_main(exc_type, exc_value, exc_tb)

    sys.excepthook = _main_hook

    # Worker-thread exceptions (Python 3.8+).  QThread.run() bodies and the
    # TaskManager threads surface here when they raise without catching.
    try:
        prev_thread = threading.excepthook

        def _thread_hook(args):  # noqa: ANN001 — threading.ExceptHookArgs
            if args.exc_type is SystemExit:
                return
            log.critical(
                "UNCAUGHT exception on thread %r",
                getattr(getattr(args, "thread", None), "name", "?"),
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
            _dump_exc(
                "uncaught worker-thread exception",
                args.exc_type,
                args.exc_value,
                args.exc_traceback,
            )
            if prev_thread is not None:
                prev_thread(args)

        threading.excepthook = _thread_hook
    except Exception:  # noqa: BLE001
        pass


def _make_resmon_logger(log_dir: Path) -> logging.Logger:
    """A dedicated, isolated logger so the heartbeat never floods the app log."""
    resmon = logging.getLogger("face_local.resmon")
    resmon.setLevel(logging.INFO)
    resmon.propagate = False  # keep heartbeat out of the main rotating log
    if not resmon.handlers:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(
                str(log_dir / "resource_monitor.log"),
                maxBytes=4 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
            )
            resmon.addHandler(handler)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not create resource_monitor.log: %s", exc)
    return resmon


def start_resource_watchdog(
    log_dir: Path,
    interval_s: Optional[float] = None,
    danger_free_mb: Optional[float] = None,
) -> None:
    """Start a daemon thread logging RSS/mem/CPU; dump stacks on memory danger.

    No-op when disabled via ``FL_RESMON=0`` or when ``psutil`` is unavailable.
    """
    global _watchdog_started
    if _watchdog_started:
        return
    if os.environ.get("FL_RESMON", "1") == "0":
        log.info("Resource watchdog disabled via FL_RESMON=0")
        return
    try:
        import psutil  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        log.info("Resource watchdog disabled (psutil unavailable: %s)", exc)
        return

    if interval_s is None:
        try:
            interval_s = float(os.environ.get("FL_RESMON_INTERVAL", "5"))
        except ValueError:
            interval_s = 5.0
    if danger_free_mb is None:
        try:
            danger_free_mb = float(os.environ.get("FL_RESMON_DANGER_MB", "700"))
        except ValueError:
            danger_free_mb = 700.0

    resmon = _make_resmon_logger(log_dir)
    _watchdog_started = True
    thread = threading.Thread(
        target=_watchdog_loop,
        args=(resmon, float(interval_s), float(danger_free_mb)),
        name="resource-watchdog",
        daemon=True,
    )
    thread.start()
    log.info(
        "Resource watchdog started (every %.0fs, danger when free<%.0fMB) -> resource_monitor.log",
        interval_s, danger_free_mb,
    )


def _watchdog_loop(
    resmon: logging.Logger, interval_s: float, danger_free_mb: float
) -> None:
    import psutil

    proc = psutil.Process(os.getpid())
    try:
        proc.cpu_percent(None)  # prime the per-process CPU sampler
    except Exception:  # noqa: BLE001
        pass
    last_dump = 0.0

    while True:
        try:
            vm = psutil.virtual_memory()
            rss_mb = proc.memory_info().rss / (1024 * 1024)
            free_mb = vm.available / (1024 * 1024)
            sys_cpu = psutil.cpu_percent(None)
            nthreads = proc.num_threads()
            resmon.info(
                "rss=%.0fMB sys_mem_used=%.0f%% free=%.0fMB cpu=%.0f%% threads=%d",
                rss_mb, vm.percent, free_mb, sys_cpu, nthreads,
            )

            if free_mb < danger_free_mb or vm.percent >= 92.0:
                now = time.time()
                if now - last_dump > 20.0:  # don't spam during sustained pressure
                    last_dump = now
                    msg = (
                        "MEMORY DANGER: free=%.0fMB used=%.0f%% rss=%.0fMB "
                        "threads=%d — dumping all thread stacks to faulthandler.log"
                        % (free_mb, vm.percent, rss_mb, nthreads)
                    )
                    resmon.warning(msg)
                    log.warning(msg)  # also surface in the main app log
                    if _crash_file is not None:
                        try:
                            _crash_file.write(
                                f"\n--- MEMORY DANGER thread dump {time.strftime('%H:%M:%S')} "
                                f"free={free_mb:.0f}MB used={vm.percent:.0f}% "
                                f"rss={rss_mb:.0f}MB threads={nthreads} ---\n"
                            )
                            faulthandler.dump_traceback(
                                file=_crash_file, all_threads=True
                            )
                            _crash_file.flush()
                        except Exception:  # noqa: BLE001
                            pass
        except Exception as exc:  # noqa: BLE001 — a watchdog must never die noisily
            log.debug("resource watchdog tick failed: %s", exc)

        time.sleep(interval_s)
