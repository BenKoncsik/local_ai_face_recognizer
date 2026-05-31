"""Fuzzy person-name matching.

Pure functions with no DB or Qt dependency so they can be unit-tested in
isolation and reused from the background merge-suggestion engine.

The matching deliberately goes beyond exact string equality.  It handles:

* case differences            ("Anna" == "anna")
* accented / unaccented forms ("Pánni" == "panni")
* extra / collapsed spaces    ("  John   Doe " == "John Doe")
* nicknames                   ("Pista" ~ "István" when a nickname is recorded)
* very similar names          ("Katalin" ~ "Katalín", "Józsefné" ~ "Jozsef")

A name match is intentionally *not* sufficient on its own for an automatic
merge — see :func:`app.services.merge_suggestion_service.compute_confidence`,
which always weights face similarity above name similarity.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable, Optional

from app.utils.person_search import normalize as _strip_accents

__all__ = [
    "normalize_name",
    "name_key",
    "phonetic_key",
    "name_similarity",
    "names_match",
]

# Tokens that carry no identity information and only add noise to comparisons.
# Hungarian married-name suffix "né" is kept (handled separately) — these are
# pure honorifics / generational markers.
_STOPWORDS = frozenset(
    {
        "dr", "ifj", "id", "özv", "ozv", "br", "gr",
        "mr", "mrs", "ms", "jr", "sr",
    }
)

_NON_ALNUM = re.compile(r"[^a-z0-9\s]+")
_WS = re.compile(r"\s+")


def normalize_name(text: Optional[str]) -> str:
    """Return *text* lowercased, accent-stripped, punctuation-free, single-spaced.

    Examples
    --------
    >>> normalize_name("  Dr.  Kovács   Anna ")
    'kovacs anna'
    >>> normalize_name("PÁNNI")
    'panni'
    >>> normalize_name(None)
    ''
    """
    if not text:
        return ""
    folded = _strip_accents(text)            # lowercase + accents removed
    folded = _NON_ALNUM.sub(" ", folded)     # drop punctuation
    tokens = [
        tok for tok in _WS.sub(" ", folded).strip().split(" ")
        if tok and tok not in _STOPWORDS
    ]
    return " ".join(tokens)


def name_key(text: Optional[str]) -> str:
    """Return an order-independent blocking key for *text*.

    Tokens are sorted so "Kovács Anna" and "Anna Kovács" share a key, which
    lets the merge engine cheaply group candidate persons before doing the
    more expensive pairwise comparison.
    """
    norm = normalize_name(text)
    if not norm:
        return ""
    return " ".join(sorted(norm.split(" ")))


# Collapsed digraphs / doubled letters for a coarse phonetic fold.  The goal is
# not linguistic accuracy but to put near-homophones into the same bucket.
_PHONETIC_RULES = (
    ("cs", "c"), ("zs", "z"), ("sz", "s"), ("gy", "g"), ("ly", "j"),
    ("ny", "n"), ("ty", "t"), ("dz", "z"), ("ph", "f"), ("th", "t"),
    ("ck", "k"), ("qu", "k"), ("x", "ks"), ("w", "v"), ("y", "i"),
)


def phonetic_key(text: Optional[str]) -> str:
    """Return a coarse phonetic key (per token, order-independent).

    Doubled consonants and common Hungarian/Latin digraphs are folded so that
    "Kati" / "Katy", "Józsa" / "Jozsa", "Filip" / "Philip" collapse together.
    """
    norm = normalize_name(text)
    if not norm:
        return ""
    folded_tokens = []
    for token in norm.split(" "):
        s = token
        for src, dst in _PHONETIC_RULES:
            s = s.replace(src, dst)
        s = re.sub(r"(.)\1+", r"\1", s)      # collapse doubled letters
        s = re.sub(r"[aeiou]", "", s) or s   # drop vowels (keep if all-vowel)
        folded_tokens.append(s)
    return " ".join(sorted(t for t in folded_tokens if t))


def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def name_similarity(
    name_a: Optional[str],
    name_b: Optional[str],
    *,
    nicknames_a: Iterable[Optional[str]] = (),
    nicknames_b: Iterable[Optional[str]] = (),
) -> float:
    """Return a fuzzy name-similarity score in ``[0.0, 1.0]``.

    The score is the maximum of several comparisons so that any one strong
    signal (exact normalized match, token-set match, phonetic match, nickname
    match, or edit-distance similarity) is enough to score highly.

    Args:
        name_a, name_b: The primary names to compare.
        nicknames_a, nicknames_b: Optional extra known aliases for each person
            (e.g. the ``nickname`` column).  A name matching the other side's
            nickname yields a high score.
    """
    na, nb = normalize_name(name_a), normalize_name(name_b)
    if not na or not nb:
        return 0.0

    scores = [_ratio(na, nb)]

    # Order-independent token comparison ("Kovács Anna" vs "Anna Kovács").
    ka, kb = name_key(name_a), name_key(name_b)
    if ka and kb:
        scores.append(_ratio(ka, kb))

    # Phonetic fold ("Filip" vs "Philip").
    pa, pb = phonetic_key(name_a), phonetic_key(name_b)
    if pa and pb and pa == pb:
        scores.append(0.92)

    # Nickname / alias cross-comparison.
    aliases_a = [normalize_name(n) for n in nicknames_a if normalize_name(n)]
    aliases_b = [normalize_name(n) for n in nicknames_b if normalize_name(n)]
    for alias in aliases_a:
        if alias and (alias == nb or alias in nb.split(" ")):
            scores.append(0.95)
    for alias in aliases_b:
        if alias and (alias == na or alias in na.split(" ")):
            scores.append(0.95)

    return max(0.0, min(1.0, max(scores)))


def names_match(
    name_a: Optional[str],
    name_b: Optional[str],
    *,
    threshold: float = 0.85,
    nicknames_a: Iterable[Optional[str]] = (),
    nicknames_b: Iterable[Optional[str]] = (),
) -> bool:
    """Return True when :func:`name_similarity` is at or above *threshold*."""
    return name_similarity(
        name_a, name_b, nicknames_a=nicknames_a, nicknames_b=nicknames_b
    ) >= threshold
