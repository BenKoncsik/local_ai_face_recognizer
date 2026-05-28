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
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName("Face-Local")
    app.setOrganizationName("face-local")
    icon_path = app_icon_path()
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    # Theme (palette + QSS)
    from app.ui.theme import apply_theme
    apply_theme(app)

    from app.ui.main_window import MainWindow

    window = MainWindow(config=config)
    window.show()

    # Google Drive cache — clean up on normal exit
    app.aboutToQuit.connect(_gdrive_cache.shutdown_cleanup)

    log.info("GUI ready")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
