"""Main-thread garbage-collection guard.

PySide6 wraps Qt's C++ objects in Python objects.  When CPython's cyclic
garbage collector reclaims one of those wrappers it runs the C++ destructor
*on whatever thread happened to trigger the collection*.  GC can fire on any
thread that allocates — including our background :class:`QThread` workers
(face detection, the deep pipeline, …).  If the collected object is a
``QWidget`` / ``QDialog``, its destructor closes the underlying ``NSWindow``,
and AppKit aborts with *"Must only be used from the main thread"*
(``EXC_BREAKPOINT``).  The crash is intermittent because it depends on *when*
GC happens to run and *which* thread it runs on.

The fix is to make collection deterministic and main-thread-only:

* disable the automatic, allocation-triggered collector, and
* drive :func:`gc.collect` from a :class:`QTimer` that lives on the GUI
  (main) thread.

With automatic collection off, no worker thread can ever reclaim a Qt wrapper;
cyclic garbage simply waits a couple of seconds for the next main-thread tick.
"""

from __future__ import annotations

import gc
import logging

from PySide6.QtCore import QObject, QTimer

log = logging.getLogger(__name__)

# Module-level reference so the timer/owner are not themselves collected.
_guard: "_GcGuard | None" = None


class _GcGuard(QObject):
    """Owns the periodic main-thread collection timer."""

    def __init__(self, interval_ms: int, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._collect)
        self._timer.start()

    def _collect(self) -> None:
        # Runs on the main thread (the timer lives here), so any Qt wrapper
        # reclaimed now is destroyed safely.
        gc.collect()


def install_main_thread_gc(interval_ms: int = 2000) -> None:
    """Route all cyclic GC onto the main thread.

    Must be called from the main (GUI) thread, after the ``QApplication`` has
    been created.  Idempotent — a second call is a no-op.

    Args:
        interval_ms: How often the main thread collects cyclic garbage.
    """
    global _guard
    if _guard is not None:
        return
    gc.disable()
    _guard = _GcGuard(interval_ms)
    log.info(
        "Automatic GC disabled; collecting on the main thread every %d ms",
        interval_ms,
    )
