"""Extended family code parser and Hungarian description generator.

Handles the full family code grammar used in this project:
  - Base person codes: C0, C00, C8, C80, C81, C810, C2201, ...
  - Ancestor suffix (F): C0F1, C0F11, C0F21, C00F2, ...
  - Sibling suffix (T): C00T1, C00T10, C00T11, ...
  - Friend/acquaintance suffix (B): C81B
  - Multi-spouse suffix (H): G1H1, G1H2, G1H21 (spouse's brought child)
  - Parent disambiguation braces: G1{2}3 = G1's 3rd child, from the 2nd spouse
  - Multi-code (shared friend): C81B,C82B  or  C81B C82B
  - Range notation: C[1-9]B, C[81-82]B
  - External family identifiers: #Zoli#, #Zoli#0, #Zoli#31, #47#1

The simpler DB-level codes (stored in Person.family_code) use the
validate_family_code() function in family_service.py. This module adds the
full extended grammar on top for human-readable descriptions and display.

The external identifier (#root#path) describes the *external* person's own
family. It lives in a separate field (Person.external_family_code) and never
appears in the main family code; validate_extended_code() rejects it, while
validate_external_family_code() accepts only that form.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# ── Root person registry ──────────────────────────────────────────────────────

# The four root persons of the family (siblings). Each gets its own initial
# letter rather than a T-sibling code, per the specification. Names use the
# same short/given form as the examples (Cikky, not Pósa Anna Mária).
DEFAULT_ROOT_NAMES: dict[str, str] = {
    "C": "Cikky",
    "G": "Gábor",
    "J": "Jerne",
    "I": "Ildi",
}

# ── Regex ─────────────────────────────────────────────────────────────────────

# External prefix: #root#  followed by the relationship path.
# The root is any non-# run (db id, name, nickname, abbreviation) and is kept
# verbatim (case preserved); the path that follows is upper-cased like internal
# codes.
_EXTERNAL_RE = re.compile(r"^#([^#]+)#(.*)$")

# Internal code: a single root letter, then the path + optional suffix.
_INTERNAL_RE = re.compile(r"^([A-Z])(.*)$")

# Trailing suffix marker (anchored at the end of the path string).
#   F[12]+        ancestors (1=father, 2=mother per step)
#   T[1-9]...     siblings / collateral relatives
#   H[1-9]...     numbered spouse (+ optional brought-child path)
#   B             friend / acquaintance
_SUFFIX_RE = re.compile(r"(F[12]+|T[1-9][0-9]*|H[1-9][0-9]*|B)$")

# A base-path token: a single digit, or a {n} brace annotating the next child.
_BASE_TOKEN_RE = re.compile(r"\{([1-9][0-9]*)\}|([0-9])")

# Range notation: ROOT[start-end]B  e.g. C[1-9]B or C[81-82]B
_RANGE_RE = re.compile(r"^([A-Z])\[([0-9]+)-([0-9]+)\]B$")


# ── Hungarian ancestor terms (nominative, possessive) ────────────────────────

_ANCESTOR_TERMS: dict[tuple[int, ...], tuple[str, str]] = {
    (1,):       ("apja",                          "apjának"),
    (2,):       ("anyja",                         "anyjának"),
    (1, 1):     ("apai nagyapja",                 "apai nagyapjának"),
    (1, 2):     ("apai nagyanyja",                "apai nagyanyjának"),
    (2, 1):     ("anyai nagyapja",                "anyai nagyapjának"),
    (2, 2):     ("anyai nagyanyja",               "anyai nagyanyjának"),
    (1, 1, 1):  ("apai dédapja",                  "apai dédapjának"),
    (1, 1, 2):  ("apai dédanyja",                 "apai dédanyjának"),
    (1, 2, 1):  ("apai anyai ágú dédapja",        "apai anyai ágú dédapjának"),
    (1, 2, 2):  ("apai anyai ágú dédanyja",       "apai anyai ágú dédanyjának"),
    (2, 1, 1):  ("anyai apai ágú dédapja",        "anyai apai ágú dédapjának"),
    (2, 1, 2):  ("anyai apai ágú dédanyja",       "anyai apai ágú dédanyjának"),
    (2, 2, 1):  ("anyai dédapja",                 "anyai dédapjának"),
    (2, 2, 2):  ("anyai dédanyja",                "anyai dédanyjának"),
}


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExtendedCodeInfo:
    raw: str
    is_external: bool                # True for #root#path codes
    external_root: Optional[str]     # the verbatim #...# root, None for internal
    root: str                        # internal letter, or external root string
    path_digits: tuple[int, ...]
    # Parallel to path_digits: the {n} spouse number annotating each child digit
    # (None where no brace was given, or for spouse-0 digits).
    path_spouse_nums: tuple[Optional[int], ...]
    suffix_type: Optional[str]       # "ancestor"|"sibling"|"friend"|"spouse"|None
    ancestor_path: tuple[int, ...]   # digits after F (1=father, 2=mother each step)
    sibling_path: tuple[int, ...]    # digits after T (first = sibling index)
    spouse_path: tuple[int, ...]     # digits after H ([0]=spouse number, rest=children)
    is_stepchild: bool               # last digit follows a non-initial spouse-zero


# ── Low-level parsing ─────────────────────────────────────────────────────────

def _parse_base_path(base: str) -> tuple[tuple[int, ...], tuple[Optional[int], ...]]:
    """Split a base path string into digits and parallel {n} spouse annotations.

    ``base`` is the part between the root and any F/T/H/B suffix — only digits
    and ``{n}`` braces.  A brace annotates the child digit that follows it
    (``G1{2}3`` → the ``3`` came from the 2nd spouse).  Raises ValueError for
    malformed input (stray characters, a brace not followed by a child digit).
    """
    digits: list[int] = []
    spouse_nums: list[Optional[int]] = []
    pending: Optional[int] = None
    pos = 0
    while pos < len(base):
        m = _BASE_TOKEN_RE.match(base, pos)
        if not m:
            raise ValueError(
                f"Érvénytelen karakter a számkódban: '{base[pos]}'"
            )
        if m.group(1) is not None:          # {n} brace
            if pending is not None:
                raise ValueError(
                    "Két kapcsos zárójel között gyermek-sorszámnak kell állnia."
                )
            pending = int(m.group(1))
        else:                               # plain digit
            digit = int(m.group(2))
            if pending is not None and digit == 0:
                raise ValueError(
                    "A {n} kapcsos jelölés csak gyermek-sorszám előtt állhat, "
                    "házastárs-0 előtt nem."
                )
            digits.append(digit)
            spouse_nums.append(pending)
            pending = None
        pos = m.end()
    if pending is not None:
        raise ValueError(
            "A kapcsos zárójeles {n} jelölést gyermek-sorszámnak kell követnie."
        )
    return tuple(digits), tuple(spouse_nums)


def _path_is_valid(path_digits: tuple[int, ...], external: bool = False) -> bool:
    """Validate a descendant/spouse digit path against the spec's 0-semantics.

    For internal codes the first digit is the root-person marker (0) or an
    nth-child index (1-9).  For external codes (#root#) the root person is
    implicit, so the path starts as if a person were already established: a
    leading 0 is that person's spouse and 1-9 is their nth child.

    Afterwards a 0 marks the spouse of the preceding person, and a 1-9 after a
    spouse-0 is that spouse's child from a previous relationship (stepchild).
    Two 0s in a row (spouse of a spouse) and a child index hung directly off the
    bare root marker (e.g. C08) are invalid.
    """
    prev = "person" if external else ""  # "" | "rootmarker" | "person" | "spouse"
    for i, digit in enumerate(path_digits):
        if not external and i == 0:
            prev = "rootmarker" if digit == 0 else "person"
            continue
        if prev == "rootmarker":
            # Only a spouse (0) may follow the bare root person (C00); a child
            # index attaches directly to the root (C8), never as C08.
            if digit != 0:
                return False
            prev = "spouse"
        elif prev == "spouse":
            if digit == 0:
                return False  # spouse of a spouse
            prev = "person"   # stepchild — now a person in their own right
        else:  # prev == "person"
            prev = "spouse" if digit == 0 else "person"
    return True


def _detect_stepchild(path_digits: tuple[int, ...], external: bool = False) -> bool:
    """True when the last path digit comes right after a non-initial spouse-zero."""
    n = len(path_digits)
    if n < 2:
        return False
    if path_digits[-1] == 0:
        return False
    if path_digits[-2] != 0:
        return False
    if external:
        # No root-person marker to exclude; any preceding spouse-0 counts.
        return True
    # The zero must not be the very first digit (root-person marker)
    return (n - 2) > 0


def parse_extended_code(code: str) -> ExtendedCodeInfo:
    """Parse an extended family code into its components.

    Accepts internal codes (``C81B``) and external codes (``#Zoli#31``).
    Internal codes are upper-cased; for external codes the ``#...#`` root keeps
    its original case while the relationship path is upper-cased.  Raises
    ValueError for malformed codes.
    """
    code = code.strip()
    if not code:
        raise ValueError("Üres kód nem fogadható el.")

    ext_m = _EXTERNAL_RE.fullmatch(code)
    if ext_m:
        external_root = ext_m.group(1).strip()
        if not external_root:
            raise ValueError("A külső gyökérelem (#...#) nem lehet üres.")
        rest = ext_m.group(2).strip().upper()
        is_external = True
        root_token = external_root
        raw = f"#{external_root}#{rest}"
    else:
        code_u = code.upper()
        int_m = _INTERNAL_RE.fullmatch(code_u)
        if not int_m or not code_u[0].isalpha():
            raise ValueError(f"Érvénytelen kód: '{code}'")
        root_token = int_m.group(1)
        rest = int_m.group(2)
        is_external = False
        external_root = None
        raw = code_u

    suffix_m = _SUFFIX_RE.search(rest)
    if suffix_m:
        suffix_str = suffix_m.group(1)
        base = rest[: suffix_m.start()]
    else:
        suffix_str = ""
        base = rest

    path_digits, path_spouse_nums = _parse_base_path(base)

    if not _path_is_valid(path_digits, external=is_external):
        raise ValueError(
            f"Érvénytelen kód '{code}': a számkód nem követi a 0-szabályokat "
            "(0 = kiinduló személy vagy házastárs; házastárs után 1-9 = hozott "
            "gyermek; két 0 egymás után nem megengedett)."
        )

    # Internal codes need a person path before a suffix (C0F1, not CF1).
    # External codes attach the suffix directly to the implicit root person
    # (#Zoli#F1 = Zoli's father), so no leading path is required there.
    if suffix_str and not path_digits and not is_external:
        raise ValueError(
            f"Érvénytelen kód '{code}': a jelző (F/T/H/B) előtt meg kell adni "
            "a személy számkódját (pl. C0F1, nem CF1)."
        )

    suffix_type: Optional[str] = None
    ancestor_path: tuple[int, ...] = ()
    sibling_path: tuple[int, ...] = ()
    spouse_path: tuple[int, ...] = ()

    if suffix_str.startswith("F"):
        suffix_type = "ancestor"
        ancestor_path = tuple(int(c) for c in suffix_str[1:])
    elif suffix_str.startswith("T"):
        suffix_type = "sibling"
        sibling_path = tuple(int(c) for c in suffix_str[1:])
    elif suffix_str.startswith("H"):
        suffix_type = "spouse"
        spouse_path = tuple(int(c) for c in suffix_str[1:])
    elif suffix_str == "B":
        suffix_type = "friend"

    return ExtendedCodeInfo(
        raw=raw,
        is_external=is_external,
        external_root=external_root,
        root=root_token,
        path_digits=path_digits,
        path_spouse_nums=path_spouse_nums,
        suffix_type=suffix_type,
        ancestor_path=ancestor_path,
        sibling_path=sibling_path,
        spouse_path=spouse_path,
        is_stepchild=_detect_stepchild(path_digits, external=is_external),
    )


def validate_extended_code(code: str) -> str:
    """Validate and return the canonical form of a *main* family code.

    Accepts single codes, multi-codes and range notation.  External ``#root#``
    codes are rejected here — they belong in the external family code field
    (see :func:`validate_external_family_code`).  Raises ValueError if invalid.
    """
    code = code.strip()
    if not code:
        raise ValueError("Üres kód nem fogadható el.")

    if code.startswith("#"):
        raise ValueError(
            "A külső családi azonosító (#...#) nem a fő családi azonosító mezőbe "
            "tartozik."
        )

    # A comma must always separate two non-empty codes — 'C8B,,C80B' or a
    # trailing/leading comma is a mistake, not an empty token to drop silently.
    for chunk in code.split(","):
        if not chunk.strip():
            raise ValueError(
                "Üres kód a vesszők között — minden vesszővel elválasztott "
                "azonosítónak ki kell töltve lennie."
            )

    # Split into comma/space separated tokens first, then validate each token
    # on its own with the same logic that already accepts a standalone range
    # (e.g. 'C[81-86]B').  This lets a list mix plain and range codes:
    #   'C8B,C80B,C[81-86]B'
    parts = _split_multi(code)
    if not parts:
        raise ValueError("Üres kód nem fogadható el.")

    canonical: list[str] = []
    for part in parts:
        canonical.append(_validate_single_code(part))

    return " ".join(canonical) if len(canonical) > 1 else canonical[0]


def _validate_single_code(part: str) -> str:
    """Validate a single family code token and return its canonical form.

    Accepts either range notation ('C[81-86]B') or a plain extended code
    ('C8B').  Raises ValueError if invalid.  External (#root#) codes are
    rejected here — they belong in the external family code field.
    """
    part = part.strip()
    if not part:
        raise ValueError("Üres kód nem fogadható el.")

    if part.startswith("#"):
        raise ValueError(
            "A külső családi azonosító (#...#) nem a fő családi azonosító "
            "mezőbe tartozik."
        )

    # Range notation?
    if "[" in part or "]" in part:
        if not _RANGE_RE.fullmatch(part.upper()):
            raise ValueError(f"Érvénytelen tartományjelölés: '{part}'")
        expanded = expand_range_code(part)
        if not expanded:
            raise ValueError(f"Érvénytelen tartományjelölés: '{part}'")
        return part.upper()

    info = parse_extended_code(part)   # raises on invalid
    if info.is_external:
        raise ValueError(
            "A külső családi azonosító (#...#) nem a fő családi azonosító "
            "mezőbe tartozik."
        )
    return part.upper()


def validate_external_family_code(code: Optional[str]) -> Optional[str]:
    """Validate and canonicalise an *external* family code (``#root#path``).

    Returns None for an empty value.  Requires the ``#root#`` form and rejects
    plain internal codes.  Raises ValueError if invalid.
    """
    if code is None:
        return None
    code = code.strip()
    if not code:
        return None
    if not code.startswith("#"):
        raise ValueError(
            "A külső családi azonosítónak #gyökér# alakkal kell kezdődnie "
            "(pl. #Zoli# vagy #Zoli#31)."
        )
    info = parse_extended_code(code)
    if not info.is_external:
        raise ValueError(f"Érvénytelen külső családi azonosító: '{code}'")
    return info.raw


# ── Multi-code helpers ────────────────────────────────────────────────────────

def _split_multi(value: str) -> list[str]:
    """Split 'C81B,C82B' or 'C81B C82B' into ['C81B', 'C82B']."""
    normalized = value.replace(",", " ")
    return [p.strip() for p in normalized.split() if p.strip()]


def expand_range_code(value: str) -> list[str]:
    """Expand 'C[1-9]B' → ['C1B', 'C2B', ..., 'C9B'].

    Returns an empty list when the range is invalid or backwards.
    """
    m = _RANGE_RE.fullmatch(value.strip().upper())
    if not m:
        return []
    root = m.group(1)
    try:
        start, end = int(m.group(2)), int(m.group(3))
    except ValueError:
        return []
    if start > end:
        return []
    return [f"{root}{i}B" for i in range(start, end + 1)]


# ── Hungarian article helper ──────────────────────────────────────────────────

# Ordinals whose spoken form starts with a vowel take "az" instead of "a".
# Among realistic spouse/partner counts only "első" (1) and "ötödik" (5) qualify.
_VOWEL_INITIAL_ORDINALS = frozenset({1, 5})


def _hu_article(n: int) -> str:
    """Return the definite article ('a' / 'az') for the n-th ordinal."""
    return "az" if n in _VOWEL_INITIAL_ORDINALS else "a"


# ── Path → segment list ───────────────────────────────────────────────────────

def _path_segments(
    path_digits: tuple[int, ...],
    spouse_nums: Optional[tuple[Optional[int], ...]] = None,
    first_zero_is_root: bool = True,
) -> list[tuple[str, str]]:
    """Convert a digit path to (nominative, possessive) term pairs.

    When first_zero_is_root is True (default), a leading 0 is the
    'root person' marker (C0 = Cikky herself) and produces no segment.
    When False (used inside sibling/external/spouse child paths) a leading 0
    means spouse.

    ``spouse_nums`` (parallel to ``path_digits``) carries the {n} brace
    annotations; when present a child segment gains an "a n. házastárstól"
    clause naming the other parent.
    """
    segments: list[tuple[str, str]] = []
    for i, digit in enumerate(path_digits):
        if digit == 0:
            if i == 0 and first_zero_is_root:
                continue  # C0 = Cikky; no extra segment
            segments.append(("házastársa", "házastársának"))
        else:
            # A child that immediately follows a non-initial spouse-zero is
            # the spouse's child from a previous relationship (hozott gyermek).
            prev_is_spouse_zero = (
                i > 0
                and path_digits[i - 1] == 0
                and not (i == 1 and first_zero_is_root)
            )
            spouse_num = spouse_nums[i] if spouse_nums else None
            if spouse_num:
                article = _hu_article(spouse_num)
                from_clause = f" {article} {spouse_num}. házastárstól"
                poss_clause = f" ({article} {spouse_num}. házastárstól)"
            else:
                from_clause = ""
                poss_clause = ""
            if prev_is_spouse_zero:
                segments.append((
                    f"{digit}. gyermeke (korábbi kapcsolatból){from_clause}",
                    f"{digit}. gyermekének{poss_clause}",
                ))
            else:
                segments.append((
                    f"{digit}. gyermeke{from_clause}",
                    f"{digit}. gyermekének{poss_clause}",
                ))
    return segments


# ── Chain builders ────────────────────────────────────────────────────────────

def _join_nominative(base: str, segments: list[tuple[str, str]]) -> str:
    """'Cikky 8. gyermekének 1. gyermeke' (last segment nominative)."""
    if not segments:
        return base
    parts = [base]
    for i, (nom, poss) in enumerate(segments):
        parts.append(poss if i < len(segments) - 1 else nom)
    return " ".join(parts)


def _join_possessive(base: str, segments: list[tuple[str, str]]) -> str:
    """'Cikky 8. gyermekének 1. gyermekének' (all segments possessive)."""
    if not segments:
        return base
    return " ".join([base] + [poss for _, poss in segments])


# ── Ancestor description ──────────────────────────────────────────────────────

def _ancestor_desc(path: tuple[int, ...]) -> tuple[str, str]:
    """Return (nominative, possessive) for an ancestor path like (1,) or (1,2)."""
    if path in _ANCESTOR_TERMS:
        return _ANCESTOR_TERMS[path]
    # For depth > 3, recursively reduce: find the longest known prefix,
    # then attach the remainder.
    for depth in range(len(path) - 1, 0, -1):
        prefix = path[:depth]
        if prefix in _ANCESTOR_TERMS:
            _, base_poss = _ANCESTOR_TERMS[prefix]
            rest_nom, rest_poss = _ancestor_desc(path[depth:])
            return f"{base_poss} {rest_nom}", f"{base_poss} {rest_poss}"
    # Absolute fallback (should not normally be reached)
    label = "apja" if path[-1] == 1 else "anyja"
    return label, label.rstrip("a") + "ának"


# ── Sibling description ───────────────────────────────────────────────────────

def _sibling_desc(sibling_path: tuple[int, ...]) -> str:
    """Hungarian description for the sibling suffix path.

    sibling_path[0] = sibling index (1-9).
    sibling_path[1:] = optional child/spouse path from that sibling.
    """
    if not sibling_path:
        return "testvére"
    sib_num = sibling_path[0]
    rest = sibling_path[1:]
    if not rest:
        return f"{sib_num}. testvére"
    rest_segments = _path_segments(rest, first_zero_is_root=False)
    return _join_nominative(f"{sib_num}. testvérének", rest_segments)


# ── Spouse (H) description ────────────────────────────────────────────────────

def _spouse_desc(spouse_path: tuple[int, ...]) -> str:
    """Hungarian description for the numbered-spouse (H) suffix path.

    spouse_path[0] = spouse / partner index (1-9).
    spouse_path[1:] = optional path from that spouse (brought children).
    """
    if not spouse_path:
        return "házastársa"
    num = spouse_path[0]
    rest = spouse_path[1:]
    if not rest:
        return f"{num}. házastársa"
    # A single direct child of the numbered spouse is a brought child
    # (the family member's own/common children stay in their own line).
    if len(rest) == 1 and rest[0] != 0:
        return f"{num}. házastársának {rest[0]}. gyermeke (korábbi kapcsolatból)"
    rest_segments = _path_segments(rest, first_zero_is_root=False)
    return _join_nominative(f"{num}. házastársának", rest_segments)


# ── Public description API ────────────────────────────────────────────────────

def describe_family_code(
    code: str,
    root_names: Optional[dict[str, str]] = None,
) -> str:
    """Return a human-readable Hungarian description for a family code.

    Handles single codes, multi-codes (comma/space separated), range notation
    and external ``#root#`` codes.  Unknown internal root letters are shown
    as-is; external roots use the ``#...#`` string itself as the person's name.
    """
    names: dict[str, str] = {**DEFAULT_ROOT_NAMES, **(root_names or {})}
    code = code.strip()

    # External code (single root path)
    if code.startswith("#"):
        return _describe_single(code, names)

    # Range notation
    if _RANGE_RE.fullmatch(code.upper()):
        return _describe_range(code.upper(), names)

    # Multi-code or single
    parts = _split_multi(code)
    if len(parts) > 1:
        return _describe_multi(parts, names)

    return _describe_single(parts[0], names)


def _describe_single(code: str, names: dict[str, str]) -> str:
    info = parse_extended_code(code)
    root_name = info.external_root if info.is_external else names.get(info.root, info.root)
    first_zero_is_root = not info.is_external
    base_segs = _path_segments(
        info.path_digits, info.path_spouse_nums, first_zero_is_root=first_zero_is_root
    )

    if info.suffix_type is None:
        return _join_nominative(root_name, base_segs)

    base_poss = _join_possessive(root_name, base_segs)

    if info.suffix_type == "ancestor":
        nom, _ = _ancestor_desc(info.ancestor_path)
        return f"{base_poss} {nom}"

    if info.suffix_type == "sibling":
        return f"{base_poss} {_sibling_desc(info.sibling_path)}"

    if info.suffix_type == "spouse":
        return f"{base_poss} {_spouse_desc(info.spouse_path)}"

    if info.suffix_type == "friend":
        base_nom = _join_nominative(root_name, base_segs)
        return f"{base_nom} barátja/ismerőse"

    return _join_nominative(root_name, base_segs)


def _describe_multi(parts: list[str], names: dict[str, str]) -> str:
    """'C81B, C82B' → 'X és Y közös barátja/ismerőse'."""
    base_descs: list[str] = []
    for part in parts:
        info = parse_extended_code(part)
        root_name = info.external_root if info.is_external else names.get(info.root, info.root)
        first_zero_is_root = not info.is_external
        base_segs = _path_segments(
            info.path_digits, info.path_spouse_nums, first_zero_is_root=first_zero_is_root
        )
        base_descs.append(_join_nominative(root_name, base_segs))

    if len(base_descs) == 1:
        return f"{base_descs[0]} barátja/ismerőse"

    joined = ", ".join(base_descs[:-1])
    return f"{joined} és {base_descs[-1]} közös barátja/ismerőse"


def _describe_range(raw: str, names: dict[str, str]) -> str:
    """'C[1-9]B' → 'C1–C9 közös barátja/ismerőse'."""
    m = _RANGE_RE.fullmatch(raw)
    if not m:
        raise ValueError(f"Érvénytelen tartományjelölés: '{raw}'")
    root, start, end = m.group(1), m.group(2), m.group(3)
    if start == end:
        return _describe_single(f"{root}{start}B", names)
    return f"{root}{start}–{root}{end} közös barátja/ismerőse"
