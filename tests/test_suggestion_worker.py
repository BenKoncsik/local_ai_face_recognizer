"""Smoke and behaviour tests for SuggestionWorker."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QTimer

from app.config import MatchingConfig
from app.db.database import init_db, session_scope
from app.db.models import MergeSuggestion, Person
from app.services.merge_suggestion_service import MergeSuggestionService
from app.ui.workers.suggestion_worker import SuggestionWorker


@pytest.fixture()
def db(tmp_path):
    init_db(tmp_path / "suggestion_worker.db")


@pytest.fixture()
def run_timers_immediately(monkeypatch):
    """Execute QTimer.singleShot callbacks synchronously in tests."""

    def _immediate(_ms: int, callback) -> None:
        callback()

    monkeypatch.setattr(QTimer, "singleShot", _immediate)


def test_worker_instantiation():
    worker = SuggestionWorker(MatchingConfig())
    assert worker is not None


def test_run_decision_noop_emits_success(qtbot, db, run_timers_immediately):
    worker = SuggestionWorker(MatchingConfig())
    received: list[tuple] = []
    worker.decision_finished.connect(lambda *args: received.append(args))

    worker.run_decision(lambda _svc: None, "noop", suggestion_id=7)
    qtbot.waitUntil(lambda: len(received) > 0, timeout=1000)

    assert received == [(7, "noop", True)]


def test_run_decision_error_emits_failure(qtbot, db, run_timers_immediately):
    worker = SuggestionWorker(MatchingConfig())
    finished: list[tuple] = []
    errors: list[tuple] = []
    worker.decision_finished.connect(lambda *args: finished.append(args))
    worker.error_occurred.connect(lambda *args: errors.append(args))

    def _boom(_svc: MergeSuggestionService) -> None:
        raise RuntimeError("boom")

    worker.run_decision(_boom, "fail", suggestion_id=3)
    qtbot.waitUntil(lambda: len(finished) > 0 and len(errors) > 0, timeout=1000)

    assert finished == [(3, "fail", False)]
    assert errors == [(3, "boom")]


def test_run_decision_calls_service_action(qtbot, db, run_timers_immediately):
    with session_scope() as session:
        source = Person(name="Unknown 1", is_auto_named=True)
        target = Person(name="Anna", is_auto_named=False)
        session.add_all([source, target])
        session.flush()
        suggestion = MergeSuggestion(
            source_person_id=source.id,
            target_person_id=target.id,
            confidence=0.9,
            status="pending",
        )
        session.add(suggestion)
        session.flush()
        suggestion_id = suggestion.id

    worker = SuggestionWorker(MatchingConfig())
    received: list[tuple] = []
    worker.decision_finished.connect(lambda *args: received.append(args))

    worker.run_decision(
        lambda svc: svc.dismiss(suggestion_id),
        "dismiss",
        suggestion_id=suggestion_id,
    )
    qtbot.waitUntil(lambda: len(received) > 0, timeout=1000)

    assert received == [(suggestion_id, "dismiss", True)]

    with session_scope() as session:
        row = session.get(MergeSuggestion, suggestion_id)
        assert row is not None
        assert row.status == "dismissed"
