"""Fuzzy / imprecise date parsing for photo and biographical dates.

The app stores dates as free-form strings (``Image.photo_date``,
``Person.birth_date`` …) so a single value may be exact (``1954.03.12``),
year-only (``1954``), a decade (``1970-es évek``), an interval
(``1969-1976``), open-ended (``1945 előtt``) or merely estimated
(``kb. 1920``, ``1920 körül``, ``1920?``).

:class:`FuzzyDate` normalises any such string into a sortable, displayable
value object with an ``earliest``/``latest`` span and a :class:`Precision`.
It builds on :func:`app.utils.exif.parse_flexible_date` for the exact cases
and adds the imprecise ones on top.  Parsing never raises — an
unrecognisable string yields :attr:`Precision.UNKNOWN`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional

from app.utils.exif import parse_flexible_date

# How many years an "estimated" year (``kb. 1920``) spreads either side, and how
# far an open-ended bound (``1945 előtt/után``) is assumed to reach.  These only
# affect the sort key and the rendered span width, never the stored string.
_ESTIMATE_PAD_YEARS = 2
_OPEN_PAD_YEARS = 10


class Precision(Enum):
    """How precisely a :class:`FuzzyDate` pins down a moment in time."""

    EXACT = "exact"      # full Y-M-D
    MONTH = "month"      # Y-M
    YEAR = "year"        # Y
    DECADE = "decade"    # "1970s" → 1970-1979
    RANGE = "range"      # explicit "1969-1976"
    BEFORE = "before"    # "… előtt" — open lower bound
    AFTER = "after"      # "… után" — open upper bound
    UNKNOWN = "unknown"  # could not be parsed


# Estimate markers (Hungarian + English + symbols).  Presence flips
# ``is_estimated`` and is then stripped before structural parsing.
_ESTIMATE_PATTERNS = [
    r"\bk\s*\.?\s*b\.?\b",   # "kb", "kb.", "k.b."
    r"körül",
    r"\bcca\.?\b",
    r"\bcirca\b",
    r"\bca\.?\b",
    r"\bapprox\.?\b",
    r"~",
]

_DECADE_RE = re.compile(
    r"^(?P<year>\d{3,4})\s*-?\s*(?:as|es|ös|os)?\s*évek$|"   # "1970-es évek"
    r"^(?P<year2>\d{3,4})\s*'?s$",                            # "1970s"
    re.IGNORECASE,
)

# Two 3-4 digit years joined by a dash/en-dash → interval.
_RANGE_RE = re.compile(r"^(?P<a>\d{3,4})\s*[-–—]\s*(?P<b>\d{3,4})$")

_BEFORE_RE = re.compile(r"^(?P<rest>.+?)\s*(?:előtt|before)$", re.IGNORECASE)
_AFTER_RE = re.compile(r"^(?P<rest>.+?)\s*(?:után|after)$", re.IGNORECASE)

# A bare year anywhere — last-resort extraction for messy input ("196?", "1970 körül?").
_YEAR_ANYWHERE_RE = re.compile(r"\b(\d{4})\b")


def _frac_year(d: date) -> float:
    """Return *d* as a fractional year (1954-07-02 → ≈1954.5)."""
    start = date(d.year, 1, 1).toordinal()
    end = date(d.year + 1, 1, 1).toordinal()
    return d.year + (d.toordinal() - start) / (end - start)


def _year_span(year: int) -> tuple[date, date]:
    return date(year, 1, 1), date(year, 12, 31)


@dataclass(frozen=True)
class FuzzyDate:
    """A normalised, sortable representation of an imprecise date string."""

    earliest: Optional[date]
    latest: Optional[date]
    precision: Precision
    is_estimated: bool
    raw: str

    # ------------------------------------------------------------------
    @property
    def is_known(self) -> bool:
        """True when at least one bound could be established."""
        return self.earliest is not None or self.latest is not None

    @property
    def span_frac(self) -> Optional[tuple[float, float]]:
        """Return ``(lo, hi)`` fractional-year bounds, or ``None`` if unknown.

        Open-ended bounds (BEFORE/AFTER) are padded by :data:`_OPEN_PAD_YEARS`
        so they have a finite extent for the timeline and ordering.
        """
        lo, hi = self.earliest, self.latest
        if lo is None and hi is None:
            return None
        if lo is None:                              # BEFORE
            lo = date(hi.year - _OPEN_PAD_YEARS, hi.month, hi.day)
        if hi is None:                              # AFTER
            hi = date(lo.year + _OPEN_PAD_YEARS, lo.month, lo.day)
        return _frac_year(lo), _frac_year(hi)

    @property
    def sort_key(self) -> float:
        """A single float for ordering; unknown dates sort last (``+inf``).

        Open-ended bounds are padded by :data:`_OPEN_PAD_YEARS` so they still
        order sensibly relative to closed dates.
        """
        span = self.span_frac
        if span is None:
            return float("inf")
        return (span[0] + span[1]) / 2.0

    @property
    def display(self) -> str:
        """A short, human-friendly label (Hungarian-leaning) for the UI."""
        p = self.precision
        if p is Precision.UNKNOWN:
            return self.raw.strip() or "?"
        prefix = "kb. " if self.is_estimated else ""
        if p is Precision.RANGE and self.earliest and self.latest:
            return f"{self.earliest.year}–{self.latest.year}"
        if p is Precision.DECADE and self.earliest:
            return f"{self.earliest.year}-es évek"
        if p is Precision.BEFORE and self.latest:
            return f"{self.latest.year} előtt"
        if p is Precision.AFTER and self.earliest:
            return f"{self.earliest.year} után"
        if p is Precision.YEAR and self.earliest:
            return f"{prefix}{self.earliest.year}"
        if p is Precision.MONTH and self.earliest:
            return f"{prefix}{self.earliest.year}.{self.earliest.month:02d}"
        if p is Precision.EXACT and self.earliest:
            d = self.earliest
            return f"{prefix}{d.year}.{d.month:02d}.{d.day:02d}"
        return self.raw.strip()

    # ------------------------------------------------------------------
    @classmethod
    def unknown(cls, raw: str = "") -> "FuzzyDate":
        return cls(None, None, Precision.UNKNOWN, False, raw)

    @classmethod
    def parse(cls, date_str: Optional[str]) -> "FuzzyDate":
        """Parse *date_str* into a :class:`FuzzyDate`; never raises."""
        if not date_str:
            return cls.unknown(date_str or "")
        raw = date_str
        s = date_str.strip()
        if not s:
            return cls.unknown(raw)

        # 1) Estimate markers — note then strip them.
        is_estimated = False
        for pat in _ESTIMATE_PATTERNS:
            if re.search(pat, s, re.IGNORECASE):
                is_estimated = True
                s = re.sub(pat, " ", s, flags=re.IGNORECASE)
        # A trailing "?" also signals estimation ("1920?").
        if "?" in s:
            is_estimated = True
            s = s.replace("?", " ")
        s = re.sub(r"\s+", " ", s).strip(" .,-–—")

        if not s:
            return cls.unknown(raw)

        # 2) Open-ended: "… előtt" / "… után".
        m = _BEFORE_RE.match(s)
        if m:
            inner = cls.parse(m.group("rest"))
            bound = inner.latest or inner.earliest
            if bound is not None:
                return cls(None, bound, Precision.BEFORE, is_estimated, raw)
        m = _AFTER_RE.match(s)
        if m:
            inner = cls.parse(m.group("rest"))
            bound = inner.earliest or inner.latest
            if bound is not None:
                return cls(bound, None, Precision.AFTER, is_estimated, raw)

        # 3) Decade: "1970-es évek", "1970s".
        m = _DECADE_RE.match(s)
        if m:
            year = int(m.group("year") or m.group("year2"))
            return cls(
                date(year, 1, 1), date(year + 9, 12, 31),
                Precision.DECADE, is_estimated, raw,
            )

        # 4) Explicit interval: "1969-1976".
        m = _RANGE_RE.match(s)
        if m:
            a, b = int(m.group("a")), int(m.group("b"))
            lo, hi = min(a, b), max(a, b)
            return cls(
                date(lo, 1, 1), date(hi, 12, 31),
                Precision.RANGE, is_estimated, raw,
            )

        # 5) Exact / month / year via the existing flexible parser.
        dt = parse_flexible_date(s)
        if dt is not None:
            digits = re.sub(r"[^\d]", "", s)
            if len(digits) >= 8:           # YYYYMMDD
                d = dt.date()
                return cls(d, d, Precision.EXACT, is_estimated, raw)
            if len(digits) >= 6:           # YYYYMM
                last = _month_last_day(dt.year, dt.month)
                return cls(
                    date(dt.year, dt.month, 1), date(dt.year, dt.month, last),
                    Precision.MONTH, is_estimated, raw,
                )
            lo, hi = _year_span(dt.year)   # YYYY
            return cls(lo, hi, Precision.YEAR, is_estimated, raw)

        # 6) Last resort: any bare 4-digit year inside messy text.
        m = _YEAR_ANYWHERE_RE.search(s)
        if m:
            year = int(m.group(1))
            lo, hi = _year_span(year)
            return cls(lo, hi, Precision.YEAR, True, raw)

        return cls.unknown(raw)


def _month_last_day(year: int, month: int) -> int:
    if month == 12:
        return 31
    return (date(year, month + 1, 1).toordinal() - date(year, month, 1).toordinal())
