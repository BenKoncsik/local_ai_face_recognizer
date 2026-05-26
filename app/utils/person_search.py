"""Accent-insensitive, case-insensitive person name search helpers.

All filtering and ranking logic lives here so it can be tested independently
of any UI code.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional


def normalize(text: str) -> str:
    """Return *text* lowercased with accent marks stripped.

    Examples
    --------
    >>> normalize("Pánnì")
    'panni'
    >>> normalize("ÁBÉCÉ")
    'abece'
    """
    nfd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn").lower()


@dataclass
class PersonEntry:
    """Lightweight DTO used by :class:`PersonSearchSelect`."""

    person_id: int
    name: str
    display_text: str = ""
    _normalized: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.display_text:
            self.display_text = self.name
        self._normalized = normalize(self.name)


def search_persons(
    query: str,
    entries: List[PersonEntry],
    max_results: int = 50,
) -> List[PersonEntry]:
    """Return *entries* that match *query*, ordered by relevance.

    Matching is accent-insensitive and case-insensitive substring search.
    Prefix matches are ranked above infix matches.  An empty *query* returns
    the first *max_results* entries unchanged (preserving caller-provided
    priority ordering).
    """
    if not query.strip():
        return entries[:max_results]
    q = normalize(query.strip())
    matched = [e for e in entries if q in e._normalized]
    matched.sort(key=lambda e: (not e._normalized.startswith(q), e._normalized))
    return matched[:max_results]
