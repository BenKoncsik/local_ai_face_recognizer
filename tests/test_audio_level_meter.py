"""Tests for the pure dBFS→fraction mapping behind the VU meter."""

from __future__ import annotations

from app.ui.widgets.audio_level_meter import level_to_fraction


def test_level_to_fraction_clamps_and_scales() -> None:
    assert level_to_fraction(None) == 0.0
    assert level_to_fraction(-90.0) == 0.0   # below floor
    assert level_to_fraction(-60.0) == 0.0   # at floor
    assert level_to_fraction(0.0) == 1.0     # full scale
    assert level_to_fraction(12.0) == 1.0    # over full scale → clamped
    # Midpoint of the -60..0 range.
    assert abs(level_to_fraction(-30.0) - 0.5) < 1e-6
