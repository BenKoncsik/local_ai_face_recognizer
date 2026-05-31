"""Cooperative cancellation + pause/resume primitive for background jobs.

A :class:`CancellationToken` is shared between the controller (UI thread) and
the worker(s).  Worker code must poll :meth:`raise_if_cancelled` (or
:meth:`wait_if_paused`) at chunk boundaries so a long-running job can be
stopped or paused promptly without leaving the database half-written.
"""

from __future__ import annotations

import threading


class OperationCancelled(Exception):
    """Raised inside worker code when a cancellation has been requested."""


class CancellationToken:
    """Thread-safe cancel + pause/resume flag.

    The token starts in the *running, not cancelled* state.  ``pause`` blocks
    workers that call :meth:`wait_if_paused`; ``resume`` releases them.  A
    cancel always wins over a pause so a paused job can still be stopped.
    """

    def __init__(self) -> None:
        self._cancelled = threading.Event()
        # _resumed is set while running; cleared while paused.
        self._resumed = threading.Event()
        self._resumed.set()

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        """Request cancellation and wake any paused workers."""
        self._cancelled.set()
        self._resumed.set()  # never leave a worker blocked after cancel

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def raise_if_cancelled(self) -> None:
        """Raise :class:`OperationCancelled` if cancellation was requested."""
        if self._cancelled.is_set():
            raise OperationCancelled()

    # ------------------------------------------------------------------
    # Pause / resume
    # ------------------------------------------------------------------

    def pause(self) -> None:
        if not self._cancelled.is_set():
            self._resumed.clear()

    def resume(self) -> None:
        self._resumed.set()

    @property
    def paused(self) -> bool:
        return not self._resumed.is_set() and not self._cancelled.is_set()

    def wait_if_paused(self, poll_interval: float = 0.2) -> None:
        """Block while paused; return promptly on resume or cancel.

        Raises:
            OperationCancelled: if a cancel arrives while (or before) waiting.
        """
        self.raise_if_cancelled()
        # Wake periodically so cancellation during a pause is observed quickly.
        while not self._resumed.wait(timeout=poll_interval):
            self.raise_if_cancelled()
        self.raise_if_cancelled()
