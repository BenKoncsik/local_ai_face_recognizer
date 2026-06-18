"""Tests for lightweight performance instrumentation."""

from __future__ import annotations

import logging
import time

import pytest
from sqlalchemy import create_engine, text

from app import perf


@pytest.fixture(autouse=True)
def _reset_perf() -> None:
    perf.reset()
    yield
    perf.reset()


class TestPerfRecord:
    def test_avg_ms(self) -> None:
        rec = perf.PerfRecord(name="op", count=4, total_ms=40.0)
        assert rec.avg_ms == 10.0

    def test_avg_ms_zero_when_no_samples(self) -> None:
        rec = perf.PerfRecord(name="op")
        assert rec.avg_ms == 0.0


class TestRecordAndTimedBlock:
    def test_record_accumulates_stats(self) -> None:
        perf.record("work", 10.0)
        perf.record("work", 30.0)
        snap = perf.snapshot()
        assert snap["work"].count == 2
        assert snap["work"].total_ms == pytest.approx(40.0)
        assert snap["work"].max_ms == pytest.approx(30.0)
        assert snap["work"].last_ms == pytest.approx(30.0)

    def test_timed_block_records_elapsed(self) -> None:
        with perf.timed_block("block"):
            time.sleep(0.01)
        snap = perf.snapshot()
        assert snap["block"].count == 1
        assert snap["block"].total_ms > 0

    def test_timed_decorator_records_function(self) -> None:
        @perf.timed("decorated-op")
        def _work() -> int:
            return 42

        assert _work() == 42
        assert perf.snapshot()["decorated-op"].count == 1

    def test_slow_operation_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="app.perf"):
            perf.record("slow-op", 250.0)
        assert any("SLOW slow-op" in r.message for r in caplog.records)


class TestPerfCounter:
    def test_hit_miss_and_hit_rate(self) -> None:
        counter = perf.counter("thumb-cache")
        counter.hit()
        counter.hit()
        counter.miss()
        assert counter.hits == 2
        assert counter.misses == 1
        assert counter.hit_rate == pytest.approx(2 / 3)

    def test_hit_rate_zero_without_samples(self) -> None:
        counter = perf.counter("empty-cache")
        assert counter.hit_rate == 0.0


class TestResetAndReport:
    def test_reset_clears_records_and_counters(self) -> None:
        perf.record("x", 1.0)
        perf.counter("c").hit()
        perf.reset()
        assert perf.snapshot() == {}
        assert perf.report().startswith("operation")

    def test_report_includes_operations_and_rss(self) -> None:
        perf.record("alpha", 5.0)
        perf.counter("cache").miss()
        report = perf.report()
        assert "alpha" in report
        assert "cache" in report
        assert "process RSS" in report


class TestMemoryMb:
    def test_returns_non_negative_float(self) -> None:
        assert perf.memory_mb() >= 0.0


class TestAttachSqlTiming:
    def test_records_sql_total(self) -> None:
        engine = create_engine("sqlite:///:memory:")
        perf.attach_sql_timing(engine, slow_ms=1000.0)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        assert perf.snapshot()["sql.total"].count == 1
