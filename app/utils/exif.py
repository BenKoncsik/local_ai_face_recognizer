"""Small, failure-tolerant EXIF helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

log = logging.getLogger(__name__)


def _rational_to_float(value) -> float:  # noqa: ANN001
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        return float(value.numerator) / float(value.denominator)
    if isinstance(value, tuple) and len(value) == 2:
        return float(value[0]) / float(value[1])
    return float(value)


def _dms_to_decimal(values, ref: str) -> Optional[float]:  # noqa: ANN001
    try:
        deg = _rational_to_float(values[0])
        minutes = _rational_to_float(values[1])
        seconds = _rational_to_float(values[2])
        decimal = deg + minutes / 60.0 + seconds / 3600.0
        if ref in {"S", "W"}:
            decimal *= -1
        return decimal
    except Exception as exc:  # noqa: BLE001
        log.debug("Invalid EXIF GPS DMS data: %s", exc)
        return None


def read_exif_gps(path: str | Path) -> Optional[Tuple[float, float]]:
    """Return ``(latitude, longitude)`` from EXIF GPS data, or ``None``.

    The caller can use this during import safely: unreadable files, missing GPS
    tags, or malformed coordinates are logged and reported as no GPS data.
    """
    try:
        from PIL import Image as PilImage
        from PIL.ExifTags import GPSTAGS

        with PilImage.open(path) as img:
            exif = img.getexif()
            gps_ifd = exif.get_ifd(0x8825) if exif else {}
        if not gps_ifd:
            return None

        gps = {GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}
        lat = _dms_to_decimal(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef", "N"))
        lon = _dms_to_decimal(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef", "E"))
        if lat is None or lon is None:
            return None
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            log.info("Ignoring out-of-range EXIF GPS for %s: %s, %s", path, lat, lon)
            return None
        return lat, lon
    except Exception as exc:  # noqa: BLE001
        log.info("EXIF GPS read failed for %s: %s", path, exc)
        return None
