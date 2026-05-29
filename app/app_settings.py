"""Centralized QSettings factory.

All application settings are stored in a single INI file inside the user's
Documents folder:

    Documents/localAIFaceRecognizer/settings/settings.ini

Use :func:`app_qsettings` as a drop-in replacement for every
``QSettings("FaceLocal", "FaceLocal")`` call in the codebase so that the
storage location is managed in one place.

One-time migration from the legacy platform-native store (macOS plist /
Windows registry / Linux XDG INI) is performed by
:func:`migrate_legacy_settings`.  Call it once at application startup,
after the Qt application object has been created.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_SETTINGS_FILENAME = "settings.ini"
_LEGACY_ORG = "FaceLocal"
_LEGACY_APP = "FaceLocal"


def app_qsettings():  # noqa: ANN201
    """Return a QSettings instance backed by the Documents-based INI file.

    The settings directory is created on first call if it does not yet exist.
    """
    from PySide6.QtCore import QSettings

    from app.paths import ensure_settings_dir

    settings_file = str(ensure_settings_dir() / _SETTINGS_FILENAME)
    return QSettings(settings_file, QSettings.Format.IniFormat)


def migrate_legacy_settings() -> None:
    """Copy settings from the legacy native store to the new INI file.

    Runs only when the new INI file does not yet exist AND the legacy store
    contains at least one key.  Safe to call every startup — it is a no-op
    after the first successful migration.
    """
    from PySide6.QtCore import QSettings

    from app.paths import ensure_settings_dir

    new_file = ensure_settings_dir() / _SETTINGS_FILENAME
    if new_file.exists():
        return

    legacy = QSettings(_LEGACY_ORG, _LEGACY_APP)
    keys = legacy.allKeys()
    if not keys:
        return

    new_qs = QSettings(str(new_file), QSettings.Format.IniFormat)
    for key in keys:
        new_qs.setValue(key, legacy.value(key))
    new_qs.sync()
    log.info(
        "Migrated %d settings key(s) from legacy QSettings to %s",
        len(keys),
        new_file,
    )
