"""Tests for the background merge-matching worker and cancellation token."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from app.config import AppConfig
from app.db.database import init_db, session_scope
from app.db.models import Face, Image, MergeSuggestion, Person
from app.jobs.cancellation import CancellationToken, OperationCancelled
from app.workers.match_job_worker import MatchJobWorker

DIM = 64


@pytest.fixture()
def tmp_db(tmp_path):
    init_db(tmp_path / "test.db")
    return tmp_path


def _config() -> AppConfig:
    cfg = AppConfig()
    cfg.matching.chunk_size = 1          # exercise the chunk loop
    cfg.matching.progress_throttle_ms = 0
    return cfg


def _axis_vec(index: int, seed: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[index] = 1.0
    v += np.random.default_rng(seed).normal(0, 0.02, DIM).astype(np.float32)
    return v.astype(np.float32)


def _seed_pairs(n: int) -> None:
    """Create *n* look-alike (named, auto) person pairs on distinct axes."""
    with session_scope() as session:
        for axis in range(n):
            named = Person(name=f"Named{axis}", is_auto_named=False)
            auto = Person(name=f"Unknown{axis}", is_auto_named=True)
            session.add_all([named, auto])
            session.flush()
            for s in range(3):
                for pid, off in ((named.id, 0), (auto.id, 100)):
                    img = Image(
                        file_path=f"/img/{axis}-{pid}-{s}.jpg",
                        file_hash="h", file_mtime=0.0,
                    )
                    session.add(img)
                    session.flush()
                    f = Face(
                        image_id=img.id, person_id=pid,
                        bbox_x=0, bbox_y=0, bbox_w=100, bbox_h=100,
                        confidence=0.9, crop_path=f"/c/{axis}-{pid}-{s}.jpg",
                    )
                    f.set_embedding(_axis_vec(axis, seed=s + off))
                    session.add(f)


def _assert_consistent() -> None:
    """No duplicate pairs and every suggestion references existing persons."""
    with session_scope() as session:
        rows = session.query(MergeSuggestion).all()
        pairs = {(r.source_person_id, r.target_person_id) for r in rows}
        assert len(pairs) == len(rows), "duplicate suggestion pairs found"
        valid = {pid for (pid,) in session.query(Person.id).all()}
        for r in rows:
            assert r.source_person_id in valid
            assert r.target_person_id in valid
            assert r.source_person_id < r.target_person_id


# ---------------------------------------------------------------------------
# CancellationToken
# ---------------------------------------------------------------------------

class TestCancellationToken:
    def test_cancel_raises(self):
        tok = CancellationToken()
        tok.raise_if_cancelled()  # no-op
        tok.cancel()
        assert tok.cancelled
        with pytest.raises(OperationCancelled):
            tok.raise_if_cancelled()

    def test_pause_blocks_until_resume(self):
        tok = CancellationToken()
        tok.pause()
        assert tok.paused
        released = threading.Event()

        def waiter():
            tok.wait_if_paused()
            released.set()

        th = threading.Thread(target=waiter)
        th.start()
        assert not released.wait(0.2)  # still blocked
        tok.resume()
        assert released.wait(1.0)
        th.join()

    def test_cancel_releases_paused_waiter(self):
        tok = CancellationToken()
        tok.pause()
        raised = threading.Event()

        def waiter():
            try:
                tok.wait_if_paused()
            except OperationCancelled:
                raised.set()

        th = threading.Thread(target=waiter)
        th.start()
        tok.cancel()
        assert raised.wait(1.0)
        th.join()


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class TestMatchJobWorker:
    def test_full_scan_persists_suggestions(self, qtbot, tmp_db):
        _seed_pairs(4)
        worker = MatchJobWorker(_config())
        worker.enqueue_full_scan()
        with qtbot.waitSignal(worker.idle, timeout=10000):
            worker.start()
        worker.shutdown()
        worker.wait(3000)

        with session_scope() as session:
            assert session.query(MergeSuggestion).count() == 4
        _assert_consistent()

    def test_second_full_scan_refused_while_pending(self, tmp_db):
        worker = MatchJobWorker(_config())
        assert worker.enqueue_full_scan() is True
        assert worker.enqueue_full_scan() is False  # already queued

    def test_cancel_leaves_consistent_state(self, qtbot, tmp_db):
        _seed_pairs(12)
        worker = MatchJobWorker(_config())
        worker.enqueue_full_scan()
        worker.start()
        # Cancel as soon as the job starts running.
        qtbot.waitUntil(lambda: worker._current_token is not None, timeout=5000)
        worker.cancel_current()
        worker.shutdown()
        worker.wait(5000)
        # Whatever got written before the cancel must be internally consistent.
        _assert_consistent()

    def test_scoped_job_runs(self, qtbot, tmp_db):
        _seed_pairs(3)
        with session_scope() as session:
            auto_ids = [
                p.id for p in session.query(Person)
                .filter(Person.is_auto_named == True)  # noqa: E712
                .all()
            ]
        worker = MatchJobWorker(_config())
        worker.enqueue_scoped(auto_ids[:1], label="scoped")
        with qtbot.waitSignal(worker.idle, timeout=10000):
            worker.start()
        worker.shutdown()
        worker.wait(3000)
        with session_scope() as session:
            assert session.query(MergeSuggestion).count() == 1
