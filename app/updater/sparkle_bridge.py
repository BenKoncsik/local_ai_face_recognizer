"""Python bridge to SparkleHelper.

SparkleHelper is a thin Swift binary that embeds Sparkle.framework and handles
update checking + installation natively.  This module starts it as a subprocess
so that Sparkle's NSApplication run loop never conflicts with PySide6/Qt's own
NSApplication instance.

All public functions are safe to call on every platform; they return False when
not running on macOS, not inside a frozen bundle, or when SparkleHelper is not
bundled.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def _bundle_path() -> Path | None:
    """Return the .app bundle root when running from a PyInstaller freeze."""
    if not getattr(sys, "frozen", False):
        return None
    # sys.executable is .../Face-Local.app/Contents/MacOS/Face-Local
    exe = Path(sys.executable).resolve()
    app = exe.parent.parent.parent
    return app if app.suffix == ".app" else None


def _helper_path(bundle: Path) -> Path:
    return bundle / "Contents" / "MacOS" / "SparkleHelper"


def is_sparkle_available() -> bool:
    """Return True when SparkleHelper is present inside the current bundle."""
    if sys.platform != "darwin":
        return False
    bundle = _bundle_path()
    return bundle is not None and _helper_path(bundle).exists()


def start_background_update_check() -> bool:
    """Launch SparkleHelper for a silent background update check.

    Non-blocking: the helper terminates on its own after checking.
    Returns True if the helper process was started.
    """
    if sys.platform != "darwin":
        return False

    bundle = _bundle_path()
    if bundle is None:
        log.debug("Not in a frozen bundle — skipping Sparkle background check")
        return False

    helper = _helper_path(bundle)
    if not helper.exists():
        log.debug("SparkleHelper not found at %s — skipping update check", helper)
        return False

    env = {**os.environ, "FACE_LOCAL_BUNDLE_PATH": str(bundle)}
    try:
        subprocess.Popen(
            [str(helper), "--background"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        log.debug("SparkleHelper launched for background check")
        return True
    except OSError as exc:
        log.warning("Failed to launch SparkleHelper for background check: %s", exc)
        return False


def check_for_updates() -> bool:
    """Launch SparkleHelper with the manual update-check UI.

    Non-blocking: the helper shows Sparkle's native update sheet independently.
    Returns True if the helper process was started.
    """
    if sys.platform != "darwin":
        return False

    bundle = _bundle_path()
    if bundle is None:
        return False

    helper = _helper_path(bundle)
    if not helper.exists():
        return False

    env = {**os.environ, "FACE_LOCAL_BUNDLE_PATH": str(bundle)}
    try:
        subprocess.Popen(
            [str(helper), "--check"],
            env=env,
            close_fds=True,
        )
        return True
    except OSError as exc:
        log.warning("Failed to launch SparkleHelper for manual check: %s", exc)
        return False
