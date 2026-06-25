"""Background worker that installs missing AI packages via pip.

Runs entirely in a QThread so the UI stays responsive.  Emits progress
signals that the main window uses to update the status bar.

Lifecycle
---------
1. Worker checks which packages from CANDIDATE_PACKAGES are missing.
2. If nothing is missing it emits ``all_done(False)`` immediately.
3. For each missing package it emits ``package_installing``, runs pip,
   then emits ``package_installed`` or ``install_failed``.
4. On any successful install it calls ``refresh_sys_path()`` so the
   new package is importable in the current process without a restart.
5. Emits ``all_done(True)`` when all installs have been attempted.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QThread, Signal

log = logging.getLogger(__name__)


class DependencyInstallerWorker(QThread):
    """QThread that installs missing AI runtime packages in the background.

    Signals:
        package_installing(display_name):      pip install started
        package_installed(display_name):       pip install succeeded
        install_failed(display_name, reason):  pip install failed
        all_done(any_installed):               all packages attempted
    """

    package_installing: Signal = Signal(str)
    package_installed: Signal = Signal(str)
    install_failed: Signal = Signal(str, str)
    all_done: Signal = Signal(bool)

    def run(self) -> None:
        from app.services.dependency_installer import (
            find_python_executable,
            install_package,
            missing_packages,
            refresh_sys_path,
        )

        missing = missing_packages()
        if not missing:
            log.debug("DependencyInstallerWorker: all packages already present.")
            self.all_done.emit(False)
            return

        python_exe = find_python_executable()
        if not python_exe:
            for pkg in missing:
                self.install_failed.emit(pkg.display_name, "Python interpreter not found on PATH")
            self.all_done.emit(False)
            return

        any_installed = False
        for pkg in missing:
            log.info("Background install: %s", pkg.display_name)
            self.package_installing.emit(pkg.display_name)
            ok = install_package(pkg.pip_spec, python_exe)
            if ok:
                refresh_sys_path()
                any_installed = True
                self.package_installed.emit(pkg.display_name)
            else:
                self.install_failed.emit(pkg.display_name, "pip returned non-zero exit")

        self.all_done.emit(any_installed)
