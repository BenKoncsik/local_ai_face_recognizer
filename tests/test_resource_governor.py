"""Tests for the adaptive resource governor and its checkpoint throttle.

The governor's live sampling (psutil) is bypassed with ``set_test_override`` so
the advice math and the TaskContext throttle can be exercised deterministically.
"""

from __future__ import annotations

import time

import pytest

from app.jobs.cancellation import OperationCancelled
from app.tasks.manager import BackgroundTask, TaskContext, TaskPriority
from app.tasks.resource_governor import (
    ResourceGovernor,
    _ramp,
    get_resource_governor,
)


@pytest.fixture
def gov():
    """A fresh governor plus automatic override cleanup on the singleton."""
    g = ResourceGovernor()
    yield g
    # Ensure the process-wide singleton never keeps a test override set.
    get_resource_governor().set_test_override(None)


# -- pure helpers ---------------------------------------------------------


def test_ramp_clamps_and_interpolates():
    assert _ramp(0.0, 0.2, 0.8) == 0.0
    assert _ramp(0.2, 0.2, 0.8) == 0.0
    assert _ramp(0.8, 0.2, 0.8) == 1.0
    assert _ramp(1.0, 0.2, 0.8) == 1.0
    assert _ramp(0.5, 0.2, 0.8) == pytest.approx(0.5)


# -- throttle advice ------------------------------------------------------


def test_no_throttle_when_idle(gov):
    gov.set_test_override(0.0, True)
    assert gov.throttle_delay(1.0) == 0.0
    assert gov.recommended_workers(8) == 8


def test_throttle_proportional_to_work_slice(gov):
    gov.set_test_override(1.0, True)
    # slice * MAX_DUTY(1.0) * load(1.0) * weight(1.0), capped at 0.25.
    assert gov.throttle_delay(0.1) == pytest.approx(0.1)
    assert gov.throttle_delay(1.0) == 0.25  # capped for cancel responsiveness


def test_throttle_scales_with_weight(gov):
    gov.set_test_override(1.0, True)
    assert gov.throttle_delay(0.2, weight=0.25) == pytest.approx(0.05)
    assert gov.throttle_delay(0.2, weight=0.0) == 0.0


def test_workers_scale_down_with_load(gov):
    gov.set_test_override(1.0, True)
    assert gov.recommended_workers(8) == 1
    gov.set_test_override(0.25, True)
    # round(9 - 0.25 * 8) == round(7.0) == 7
    assert gov.recommended_workers(9) == 7
    # Never below 1, never above the hard cap.
    assert gov.recommended_workers(1) == 1


def test_disabled_override_means_full_speed(gov):
    gov.set_test_override(1.0, enabled=False)
    assert gov.throttle_delay(1.0) == 0.0
    assert gov.recommended_workers(8) == 8


# -- checkpoint integration ----------------------------------------------


def _ctx(priority=TaskPriority.LOW) -> TaskContext:
    task = BackgroundTask("t", lambda ctx: None, priority=priority)
    return TaskContext(task)


def test_checkpoint_sleeps_under_load(qapp):
    get_resource_governor().set_test_override(1.0, True)
    try:
        ctx = _ctx()
        ctx.checkpoint()          # first checkpoint only primes the timer
        time.sleep(0.05)          # simulate a work slice
        start = time.perf_counter()
        ctx.checkpoint()          # should throttle proportional to the slice
        assert time.perf_counter() - start >= 0.03
    finally:
        get_resource_governor().set_test_override(None)


def test_checkpoint_no_sleep_when_idle(qapp):
    get_resource_governor().set_test_override(0.0, True)
    try:
        ctx = _ctx()
        ctx.checkpoint()
        time.sleep(0.02)
        start = time.perf_counter()
        ctx.checkpoint()
        assert time.perf_counter() - start < 0.02
    finally:
        get_resource_governor().set_test_override(None)


def test_throttle_aborts_on_cancel(qapp):
    get_resource_governor().set_test_override(1.0, True)
    try:
        ctx = _ctx()
        # Pretend a work slice already elapsed so a throttle sleep is due.
        ctx._last_checkpoint = time.perf_counter() - 0.2
        ctx._task.token.cancel()
        with pytest.raises(OperationCancelled):
            ctx._apply_adaptive_throttle()
    finally:
        get_resource_governor().set_test_override(None)
