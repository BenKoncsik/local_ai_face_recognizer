"""Tests for app.utils.fuzzy_date.FuzzyDate parsing and ordering."""

from __future__ import annotations

from datetime import date

import pytest

from app.utils.fuzzy_date import FuzzyDate, Precision


@pytest.mark.parametrize(
    "raw, precision, earliest, latest, estimated",
    [
        # Exact / month / year
        ("1954.03.12", Precision.EXACT, date(1954, 3, 12), date(1954, 3, 12), False),
        ("1954-03-12", Precision.EXACT, date(1954, 3, 12), date(1954, 3, 12), False),
        ("1954.03", Precision.MONTH, date(1954, 3, 1), date(1954, 3, 31), False),
        ("1954", Precision.YEAR, date(1954, 1, 1), date(1954, 12, 31), False),
        # Estimated year
        ("1920 körül", Precision.YEAR, date(1920, 1, 1), date(1920, 12, 31), True),
        ("kb. 1920", Precision.YEAR, date(1920, 1, 1), date(1920, 12, 31), True),
        ("~1920", Precision.YEAR, date(1920, 1, 1), date(1920, 12, 31), True),
        ("1920?", Precision.YEAR, date(1920, 1, 1), date(1920, 12, 31), True),
        ("cca 1920", Precision.YEAR, date(1920, 1, 1), date(1920, 12, 31), True),
        # Interval
        ("1969-1976", Precision.RANGE, date(1969, 1, 1), date(1976, 12, 31), False),
        ("1969–1976", Precision.RANGE, date(1969, 1, 1), date(1976, 12, 31), False),
        # Decade
        ("1970-es évek", Precision.DECADE, date(1970, 1, 1), date(1979, 12, 31), False),
        ("1970s", Precision.DECADE, date(1970, 1, 1), date(1979, 12, 31), False),
        # Open-ended
        ("1945 előtt", Precision.BEFORE, None, date(1945, 12, 31), False),
        ("1945 után", Precision.AFTER, date(1945, 1, 1), None, False),
        # Date + clock time (EXIF / filename) → exact date, time ignored, not estimated
        ("2008.05.03 14:22:01", Precision.EXACT, date(2008, 5, 3), date(2008, 5, 3), False),
        ("2008:05:03 14:22:01", Precision.EXACT, date(2008, 5, 3), date(2008, 5, 3), False),
        ("2008.05.03 14.22.01", Precision.EXACT, date(2008, 5, 3), date(2008, 5, 3), False),
        ("2008:05:03", Precision.EXACT, date(2008, 5, 3), date(2008, 5, 3), False),
        ("1954-03-12T09:30", Precision.EXACT, date(1954, 3, 12), date(1954, 3, 12), False),
    ],
)
def test_parse_matrix(raw, precision, earliest, latest, estimated):
    fd = FuzzyDate.parse(raw)
    assert fd.precision is precision
    assert fd.earliest == earliest
    assert fd.latest == latest
    assert fd.is_estimated is estimated
    assert fd.raw == raw


@pytest.mark.parametrize("raw", ["", "   ", None, "no date here", "??"])
def test_unknown(raw):
    fd = FuzzyDate.parse(raw)
    assert fd.precision is Precision.UNKNOWN
    assert not fd.is_known
    assert fd.sort_key == float("inf")


def test_messy_year_extraction():
    fd = FuzzyDate.parse("196?")  # not a clean year
    # "?" stripped → "196" is not 4 digits, so no bare-year match → unknown.
    assert fd.precision is Precision.UNKNOWN

    fd2 = FuzzyDate.parse("valamikor 1970 táján")
    assert fd2.precision is Precision.YEAR
    assert fd2.earliest == date(1970, 1, 1)
    assert fd2.is_estimated is True


def test_estimated_with_question_mark_combo():
    fd = FuzzyDate.parse("1970 körül?")
    assert fd.precision is Precision.YEAR
    assert fd.earliest == date(1970, 1, 1)
    assert fd.is_estimated is True


def test_sort_key_orders_chronologically():
    dates = [
        FuzzyDate.parse("1980"),
        FuzzyDate.parse("1954.03.12"),
        FuzzyDate.parse("1970-es évek"),
        FuzzyDate.parse(""),            # unknown → last
        FuzzyDate.parse("1945 előtt"),
        FuzzyDate.parse("1969-1976"),
    ]
    ordered = sorted(dates, key=lambda d: d.sort_key)
    labels = [d.raw for d in ordered]
    # "1945 előtt" (~1935-1945 midpoint) is the earliest; unknown is last.
    assert labels[0] == "1945 előtt"
    assert labels[1] == "1954.03.12"
    assert labels[-1] == ""


def test_range_unordered_input_normalised():
    fd = FuzzyDate.parse("1976-1969")
    assert fd.earliest == date(1969, 1, 1)
    assert fd.latest == date(1976, 12, 31)


def test_display_labels():
    assert FuzzyDate.parse("kb. 1920").display == "kb. 1920"
    assert FuzzyDate.parse("1969-1976").display == "1969–1976"
    assert FuzzyDate.parse("1970-es évek").display == "1970-es évek"
    assert FuzzyDate.parse("1945 előtt").display == "1945 előtt"
    assert FuzzyDate.parse("1945 után").display == "1945 után"
    assert FuzzyDate.parse("1954.03.12").display == "1954.03.12"
