"""Build a GeocodingService from user settings (opt-in online geocoding).

Online geocoding is OFF by default: nothing is sent to an external service
unless the user enables ``geocoding/enabled`` in Settings. Reads QSettings so
the UI layer doesn't have to thread provider config through every call.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy.orm import Session

from app.services.geocoding_service import GeocodingService

log = logging.getLogger(__name__)

# QSettings keys.
SETTING_ENABLED = "geocoding/enabled"
SETTING_PROVIDER = "geocoding/provider"
SETTING_BASE_URL = "geocoding/base_url"
SETTING_COUNTRY_CODES = "geocoding/country_codes"


def _read_bool(qs, key: str, default: bool) -> bool:
    value = qs.value(key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def create_geocoding_service(
    session: Session, *, force_online: Optional[bool] = None
) -> GeocodingService:
    """Construct a GeocodingService honouring the user's opt-in setting.

    Args:
        force_online: Override the stored setting (mainly for tests). When None,
            the QSettings ``geocoding/enabled`` value decides (default False).
    """
    online = force_online
    base_url = None
    country_codes = None
    if online is None:
        try:
            from app.app_settings import app_qsettings

            qs = app_qsettings()
            online = _read_bool(qs, SETTING_ENABLED, False)
            base_url = qs.value(SETTING_BASE_URL) or None
            country_codes = qs.value(SETTING_COUNTRY_CODES) or None
        except Exception as exc:  # noqa: BLE001
            log.debug("Geocoding settings unavailable (%s); staying offline", exc)
            online = False

    provider = None
    if online:
        try:
            from app.services.geocoding.nominatim_provider import NominatimProvider

            kwargs = {}
            if base_url:
                kwargs["base_url"] = str(base_url)
            if country_codes:
                kwargs["country_codes"] = str(country_codes)
            provider = NominatimProvider(**kwargs)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to build geocoding provider (%s); staying offline", exc)
            online = False

    return GeocodingService(session, provider, online=bool(online))
