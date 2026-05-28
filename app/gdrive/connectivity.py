"""Network connectivity check for Google Drive mode.

Google Drive mode is strictly online-only.  Any operation that requires
Drive access must call :func:`require_online` first.  If the host is not
reachable the function raises :class:`GDriveOfflineError` so callers can
surface a clear, localised error message instead of a confusing socket
exception.
"""

from __future__ import annotations

import logging
import socket

log = logging.getLogger(__name__)

_DRIVE_HOST = "drive.google.com"
_DRIVE_PORT = 443
_TIMEOUT_SECONDS = 5


class GDriveOfflineError(RuntimeError):
    """Raised when Google Drive is not reachable."""


def is_online() -> bool:
    """Return True if Google Drive appears to be reachable over HTTPS."""
    try:
        with socket.create_connection((_DRIVE_HOST, _DRIVE_PORT), timeout=_TIMEOUT_SECONDS):
            return True
    except OSError as exc:
        log.debug("GDrive connectivity check failed: %s", exc)
        return False


def require_online() -> None:
    """Raise :class:`GDriveOfflineError` if Google Drive is not reachable.

    Call this at the start of every Drive operation.  The error message is
    intentionally user-facing (shown directly in the UI).
    """
    if not is_online():
        raise GDriveOfflineError(
            "Google Drive módhoz internetkapcsolat szükséges."
        )
