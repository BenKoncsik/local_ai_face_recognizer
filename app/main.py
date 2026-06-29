"""Application entry point.

Usage::

    python -m app.main                      # default config
    python -m app.main --config config.yaml # explicit config
    python -m app.main --debug              # verbose logging
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.paths import app_icon_path, default_log_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Face-Local — offline face grouping and person labeling"
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to config.yaml (default: auto-discover)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    parser.add_argument(
        "--db",
        metavar="PATH",
        default=None,
        help="Override database path (e.g. --db /tmp/test.db)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Logging (before any other imports that log) ---
    from app.logging_setup import setup_logging

    setup_logging(
        level=logging.DEBUG if args.debug else logging.INFO,
        log_file=str(default_log_file()),
    )

    log = logging.getLogger(__name__)
    log.info("Starting Face-Local")

    # --- Crash & resource diagnostics (observability only) ---
    # A scan can drive the machine into a memory/CPU-exhaustion hard freeze that
    # leaves no Python traceback.  Enable faulthandler + exception hooks as early
    # as possible, and start a lightweight resource watchdog that records the
    # RAM/CPU run-up and snapshots all thread stacks just before a likely freeze.
    try:
        from app.crash_diagnostics import (
            install_crash_handlers,
            start_resource_watchdog,
        )

        _log_dir = default_log_file().parent
        install_crash_handlers(_log_dir)
        start_resource_watchdog(_log_dir)
    except Exception:  # noqa: BLE001 — diagnostics must never block startup
        log.exception("Could not install crash/resource diagnostics")

    # --- Load .env (Google OAuth secrets, etc.) ---
    # Must run BEFORE app.gdrive.oauth_config is imported, so the module
    # picks up the values from the environment.  .env is in .gitignore.
    from pathlib import Path as _P

    from app.gdrive.dotenv_loader import load_dotenv
    from app.paths import project_root
    # Look in both the current working directory and the project root —
    # whichever exists wins, with cwd taking priority.
    for _candidate in (_P.cwd() / ".env", project_root() / ".env"):
        if load_dotenv(_candidate) > 0:
            break

    # --- Config ---
    from app.config import load_config

    config = load_config(args.config)

    if args.db:
        config.storage.db_path = args.db

    log.info("Config loaded — DB: %s", config.db_path_resolved)
    log.info("Crops dir:          %s", config.crops_dir_resolved)

    # Ensure data directories exist
    config.db_path_resolved.parent.mkdir(parents=True, exist_ok=True)
    config.crops_dir_resolved.mkdir(parents=True, exist_ok=True)

    # Google Drive cache — remove any leftovers from a previous (possibly crashed) run
    from app.gdrive.cache import get_cache_manager
    _gdrive_cache = get_cache_manager()
    _gdrive_cache.startup_cleanup()

    # --- Language preferences ---
    from app.ui.i18n import load_prefs
    load_prefs()

    # --- Qt application ---
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    # QtWebEngine (used by the Places map widget) requires shared OpenGL
    # contexts to be enabled *before* the QApplication is constructed.
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)

    app = QApplication(sys.argv)
    app.setApplicationName("Face-Local")
    app.setOrganizationName("face-local")

    # Route all cyclic garbage collection onto the main thread.  PySide6 runs a
    # collected Qt wrapper's C++ destructor on whichever thread triggered GC; if
    # that is a background worker and the object is a QWidget/QDialog, AppKit
    # aborts with "Must only be used from the main thread".  See app.ui.gc_guard.
    from app.ui.gc_guard import install_main_thread_gc
    install_main_thread_gc()
    icon_path = app_icon_path()
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    # Theme (palette + QSS)
    from app.ui.theme import apply_theme
    apply_theme(app)

    # Migrate settings from legacy native QSettings store (one-time, no-op afterwards)
    from app.app_settings import migrate_legacy_settings
    migrate_legacy_settings()

    # Family code scheme — make the persisted active scheme the runtime default
    from app.services.family_code_schemes import FamilyCodeSchemeStore
    try:
        active = FamilyCodeSchemeStore().load_active_into_runtime()
        log.info("Active family code scheme: %s", active.name)
    except Exception:
        log.exception("Could not load the active family code scheme")

    from app.ui.main_window import MainWindow

    window = MainWindow(config=config)
    window.show()

    # Warm up onnxruntime on the main thread once the window is up, so its heavy
    # C-extension import (and any GC it provokes) never first runs inside a
    # background worker.  Deferred via the event loop so it doesn't block show().
    from PySide6.QtCore import QTimer

    from app.detectors.factory import warm_up_onnxruntime
    QTimer.singleShot(0, lambda: warm_up_onnxruntime(config.detection))

    # Background install of optional AI packages (ai-edge-litert, insightface).
    # Runs after the event loop starts so it never blocks startup.
    from app.workers.dependency_installer_worker import DependencyInstallerWorker
    _dep_worker = DependencyInstallerWorker()
    _dep_worker.package_installing.connect(window.on_dep_installing)
    _dep_worker.package_installed.connect(window.on_dep_installed)
    _dep_worker.install_failed.connect(window.on_dep_install_failed)
    _dep_worker.all_done.connect(window.on_dep_install_done)
    QTimer.singleShot(2000, _dep_worker.start)  # wait 2 s so startup scan doesn't race

    # Google Drive cache — clean up on normal exit
    app.aboutToQuit.connect(_gdrive_cache.shutdown_cleanup)

    log.info("GUI ready")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
