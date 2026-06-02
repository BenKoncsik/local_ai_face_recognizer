"""Qt-backed monitor enumeration for the screen recorder.

Bridges :class:`PySide6.QtGui.QScreen` to the Qt-free
:class:`~app.services.screen_recorder_service.RecordingDisplayInfo` used by the
recorder's pure capture-resolution logic.  Kept tiny and defensive: a flaky
windowing layer must never crash a recording start or the settings dialog.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from app.services.screen_recorder_service import RecordingDisplayInfo

log = logging.getLogger(__name__)


def enumerate_displays() -> List[RecordingDisplayInfo]:
    """Return the connected monitors, or ``[]`` if enumeration fails.

    ``av_index`` is left ``None`` here — on macOS the avfoundation
    "Capture screen N" device index is **not** the monitor ordinal (capture
    devices sit after the cameras in the device list), so the caller populates
    it from :func:`~app.services.screen_recorder_service.probe_screen_indices`.
    """
    try:
        from PySide6.QtGui import QGuiApplication

        screens = QGuiApplication.screens()
        primary = QGuiApplication.primaryScreen()
    except Exception:  # noqa: BLE001 — no display / headless / Qt issue
        log.debug("display enumeration failed", exc_info=True)
        return []

    displays: List[RecordingDisplayInfo] = []
    for idx, screen in enumerate(screens):
        try:
            geo = screen.geometry()
            name = screen.name() or ""
            display_id = name or f"display-{idx}"
            displays.append(
                RecordingDisplayInfo(
                    id=display_id,
                    name=name,
                    width=int(geo.width()),
                    height=int(geo.height()),
                    is_primary=(screen is primary),
                    x=int(geo.x()),
                    y=int(geo.y()),
                    av_index=None,
                )
            )
        except Exception:  # noqa: BLE001 — skip a misbehaving screen
            log.debug("skipping unreadable screen %d", idx, exc_info=True)
    return displays


def active_window_bounds(widget) -> Optional[tuple]:
    """Return ``(x, y, w, h)`` of *widget*'s top-level window in screen coords.

    Used by the Windows ``gdigrab`` ACTIVE_WINDOW crop path; returns ``None``
    when the geometry cannot be determined.
    """
    try:
        win = widget.window() if widget is not None else None
        if win is None:
            return None
        top_left = win.mapToGlobal(win.rect().topLeft())
        return (
            int(top_left.x()),
            int(top_left.y()),
            int(win.width()),
            int(win.height()),
        )
    except Exception:  # noqa: BLE001
        log.debug("active window bounds lookup failed", exc_info=True)
        return None
