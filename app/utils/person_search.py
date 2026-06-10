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
    #: ``True`` for placeholder identities ("Unknown N" / the protected
    #: "Ismeretlen" bucket).  Such persons are hidden from the *base* list (no
    #: active query) but remain reachable by typing their name — see
    #: :func:`search_persons`.
    is_unknown: bool = False
    _normalized: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.display_text:
            self.display_text = self.name
        self._normalized = normalize(self.name)


def person_is_unknown(person) -> bool:
    """Return ``True`` when *person* is a placeholder/unknown identity.

    Duck-typed so it works for both :class:`app.db.models.Person` and the
    lightweight stand-ins used in tests: a person counts as "unknown" when it is
    still auto-named (an unnamed "Unknown N" cluster) or system-protected (the
    "Ismeretlen" bucket).  Missing attributes default to ``False``.
    """
    return bool(getattr(person, "is_auto_named", False)) or bool(
        getattr(person, "is_protected", False)
    )


def search_persons(
    query: str,
    entries: List[PersonEntry],
    max_results: int = 50,
    *,
    hide_unknown: bool = False,
) -> List[PersonEntry]:
    """Return *entries* that match *query*, ordered by relevance.

    Matching is accent-insensitive and case-insensitive substring search.
    Prefix matches are ranked above infix matches.  An empty *query* returns
    the first *max_results* entries unchanged (preserving caller-provided
    priority ordering).

    *hide_unknown* — when ``True``, entries flagged :attr:`PersonEntry.is_unknown`
    are dropped from the *base* (empty-query) list, but a non-empty query still
    matches them by name.  This implements the shared rule "unknown persons are
    hidden by default but reachable by search".
    """
    if not query.strip():
        base = (
            [e for e in entries if not e.is_unknown] if hide_unknown else entries
        )
        return base[:max_results]
    q = normalize(query.strip())
    matched = [e for e in entries if q in e._normalized]
    matched.sort(key=lambda e: (not e._normalized.startswith(q), e._normalized))
    return matched[:max_results]


def filter_entries(
    query: str,
    entries: List[PersonEntry],
    max_results: int = 50,
) -> List[PersonEntry]:
    """Substring-filter *entries* while preserving their incoming order.

    Unlike :func:`search_persons`, this does **not** re-rank by name relevance.
    It is used when the caller has already ordered *entries* (e.g. by face-match
    similarity) and wants the search to keep that ordering intact.
    """
    if not query.strip():
        return entries[:max_results]
    q = normalize(query.strip())
    return [e for e in entries if q in e._normalized][:max_results]
