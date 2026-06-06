"""Background geocoding lookups (QRunnable + QThreadPool).

Keeps autocomplete and address geocoding off the GUI thread so typing never
blocks. Each worker opens its own short-lived DB session (sessions are not
thread-safe across threads) and builds a GeocodingService via the settings
factory. Results come back as plain dicts/lists over signals — the receiving
slot must run on the GUI thread.

Usage
-----
    w = SettlementSuggestWorker(prefix, request_id)
    w.signals.settlements_ready.connect(slot)   # (request_id, list[dict])
    QThreadPool.globalInstance().start(w)
"""

from __future__ import annotations

import dataclasses
import logging
from typing import List, Optional

from PySide6.QtCore import QObject, QRunnable, Signal

from app.db.database import session_scope
from app.services.geocoding.factory import create_geocoding_service

log = logging.getLogger(__name__)


class _GeocodingSignals(QObject):
    # request_id lets the widget ignore stale results from older keystrokes.
    settlements_ready = Signal(int, list)   # request_id, list[dict]
    streets_ready = Signal(int, list)       # request_id, list[dict]
    geocode_ready = Signal(int, object)     # request_id, dict | None
    failed = Signal(int, str)               # request_id, message


def _to_dicts(items) -> List[dict]:
    return [dataclasses.asdict(i) for i in items]


class SettlementSuggestWorker(QRunnable):
    def __init__(self, prefix: str, request_id: int, *, limit: int = 8) -> None:
        super().__init__()
        self.signals = _GeocodingSignals()
        self._prefix = prefix
        self._request_id = request_id
        self._limit = limit
        self.setAutoDelete(True)

    def run(self) -> None:
        try:
            with session_scope() as session:
                svc = create_geocoding_service(session)
                results = svc.suggest_settlements(self._prefix, limit=self._limit)
                payload = _to_dicts(results)
            self.signals.settlements_ready.emit(self._request_id, payload)
        except Exception as exc:  # noqa: BLE001
            log.warning("Settlement suggest worker failed: %s", exc)
            self.signals.failed.emit(self._request_id, str(exc))


class StreetSuggestWorker(QRunnable):
    def __init__(
        self, settlement: str, prefix: str, request_id: int, *, limit: int = 8
    ) -> None:
        super().__init__()
        self.signals = _GeocodingSignals()
        self._settlement = settlement
        self._prefix = prefix
        self._request_id = request_id
        self._limit = limit
        self.setAutoDelete(True)

    def run(self) -> None:
        try:
            with session_scope() as session:
                svc = create_geocoding_service(session)
                results = svc.suggest_streets(
                    self._settlement, self._prefix, limit=self._limit
                )
                payload = _to_dicts(results)
            self.signals.streets_ready.emit(self._request_id, payload)
        except Exception as exc:  # noqa: BLE001
            log.warning("Street suggest worker failed: %s", exc)
            self.signals.failed.emit(self._request_id, str(exc))


class GeocodeWorker(QRunnable):
    def __init__(
        self,
        settlement: str,
        street: Optional[str],
        house_number: Optional[str],
        request_id: int,
    ) -> None:
        super().__init__()
        self.signals = _GeocodingSignals()
        self._settlement = settlement
        self._street = street
        self._house_number = house_number
        self._request_id = request_id
        self.setAutoDelete(True)

    def run(self) -> None:
        try:
            with session_scope() as session:
                svc = create_geocoding_service(session)
                result = svc.geocode(self._settlement, self._street, self._house_number)
                payload = dataclasses.asdict(result) if result is not None else None
            self.signals.geocode_ready.emit(self._request_id, payload)
        except Exception as exc:  # noqa: BLE001
            log.warning("Geocode worker failed: %s", exc)
            self.signals.failed.emit(self._request_id, str(exc))
