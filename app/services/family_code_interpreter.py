"""Extended family code parser and Hungarian description generator.

Handles the full family code grammar used in this project:
  - Base person codes: C0, C00, C8, C80, C81, C810, C2201, ...
  - Ancestor suffix (F): C0F1, C0F11, C0F21, C00F2, ...
  - Sibling suffix (T): C00T1, C00T10, C00T11, ...
  - Friend/acquaintance suffix (B): C81B
  - Multi-code (shared friend): C81B,C82B  or  C81B C82B
  - Range notation: C[1-9]B, C[81-82]B

The simpler DB-level codes (stored in Person.family_code) use the
validate_family_code() function in family_service.py. This module adds the
full extended grammar on top for human-readable descriptions and display.
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

# Single extended code: root letter + digit path + optional suffix
_SINGLE_RE = re.compile(r"^([A-Z])([0-9]*)(F[12]+|T[1-9][0-9]*|B)?$")

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
    root: str
    path_digits: tuple[int, ...]
    suffix_type: Optional[str]       # "ancestor" | "sibling" | "friend" | None
    ancestor_path: tuple[int, ...]   # digits after F (1=father, 2=mother each step)
    sibling_path: tuple[int, ...]    # digits after T (first = sibling index)
    is_stepchild: bool               # last digit follows a non-initial spouse-zero


# ── Low-level parsing ─────────────────────────────────────────────────────────

def _path_is_valid(path_digits: tuple[int, ...]) -> bool:
    """Validate a descendant/spouse digit path against the spec's 0-semantics.

    The first digit is the root-person marker (0) or an nth-child index (1-9).
    Afterwards a 0 marks the spouse of the preceding person, and a 1-9 after a
    spouse-0 is that spouse's child from a previous relationship (stepchild).
    Two 0s in a row away from the root marker (spouse of a spouse) and a child
    index hung directly off the root marker (e.g. C08) are invalid.
    """
    prev = ""  # "" | "rootmarker" | "person" | "spouse"
    for i, digit in enumerate(path_digits):
        if i == 0:
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


def _detect_stepchild(path_digits: tuple[int, ...]) -> bool:
    """True when the last path digit comes right after a non-initial spouse-zero."""
    n = len(path_digits)
    if n < 2:
        return False
    if path_digits[-1] == 0:
        return False
    if path_digits[-2] != 0:
        return False
    # The zero must not be the very first digit (root-person marker)
    return (n - 2) > 0


def parse_extended_code(code: str) -> ExtendedCodeInfo:
    """Parse an extended family code into its components.

    Accepts upper- and lowercase input; returns the canonical uppercase form
    in ExtendedCodeInfo.raw.  Raises ValueError for malformed codes.
    """
    code = code.strip().upper()
    m = _SINGLE_RE.fullmatch(code)
    if not m:
        raise ValueError(f"Érvénytelen kód: '{code}'")

    root = m.group(1)
    path_str = m.group(2) or ""
    suffix_str = m.group(3) or ""

    path_digits = tuple(int(c) for c in path_str)

    if not _path_is_valid(path_digits):
        raise ValueError(
            f"Érvénytelen kód '{code}': a számkód nem követi a 0-szabályokat "
            "(0 = kiinduló személy vagy házastárs; házastárs után 1-9 = hozott "
            "gyermek; két 0 egymás után nem megengedett)."
        )

    if suffix_str and not path_digits:
        raise ValueError(
            f"Érvénytelen kód '{code}': a jelző (F/T/B) előtt meg kell adni "
            "a személy számkódját (pl. C0F1, nem CF1)."
        )

    suffix_type: Optional[str] = None
    ancestor_path: tuple[int, ...] = ()
    sibling_path: tuple[int, ...] = ()

    if suffix_str.startswith("F"):
        suffix_type = "ancestor"
        ancestor_path = tuple(int(c) for c in suffix_str[1:])
    elif suffix_str.startswith("T"):
        suffix_type = "sibling"
        sibling_path = tuple(int(c) for c in suffix_str[1:])
    elif suffix_str == "B":
        suffix_type = "friend"

    return ExtendedCodeInfo(
        raw=code,
        root=root,
        path_digits=path_digits,
        suffix_type=suffix_type,
        ancestor_path=ancestor_path,
        sibling_path=sibling_path,
        is_stepchild=_detect_stepchild(path_digits),
    )


def validate_extended_code(code: str) -> str:
    """Validate and return the canonical form of an extended family code.

    Also accepts multi-codes and range notation.  Raises ValueError if invalid.
    """
    code = code.strip()
    if not code:
        raise ValueError("Üres kód nem fogadható el.")

    # Range notation?
    if _RANGE_RE.fullmatch(code.upper()):
        expanded = expand_range_code(code)
        if not expanded:
            raise ValueError(f"Érvénytelen tartományjelölés: '{code}'")
        return code.upper()

    # Multi-code or single code
    parts = _split_multi(code)
    for part in parts:
        parse_extended_code(part)   # raises on invalid
    return (" ".join(p.upper() for p in parts) if len(parts) > 1 else parts[0].upper())


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


# ── Path → segment list ───────────────────────────────────────────────────────

def _path_segments(
    path_digits: tuple[int, ...],
    first_zero_is_root: bool = True,
) -> list[tuple[str, str]]:
    """Convert a digit path to (nominative, possessive) term pairs.

    When first_zero_is_root is True (default), a leading 0 is the
    'root person' marker (C0 = Cikky herself) and produces no segment.
    When False (used inside sibling child paths) a leading 0 means spouse.
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
            if prev_is_spouse_zero:
                segments.append((
                    f"{digit}. gyermeke (korábbi kapcsolatból)",
                    f"{digit}. gyermekének",
                ))
            else:
                segments.append((f"{digit}. gyermeke", f"{digit}. gyermekének"))
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


# ── Public description API ────────────────────────────────────────────────────

def describe_family_code(
    code: str,
    root_names: Optional[dict[str, str]] = None,
) -> str:
    """Return a human-readable Hungarian description for a family code.

    Handles single codes, multi-codes (comma/space separated), and range
    notation.  Unknown root letters are shown as-is (e.g. 'G ág').
    """
    names: dict[str, str] = {**DEFAULT_ROOT_NAMES, **(root_names or {})}
    code = code.strip()

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
    root_name = names.get(info.root, info.root)
    base_segs = _path_segments(info.path_digits)

    if info.suffix_type is None:
        return _join_nominative(root_name, base_segs)

    base_poss = _join_possessive(root_name, base_segs)

    if info.suffix_type == "ancestor":
        nom, _ = _ancestor_desc(info.ancestor_path)
        return f"{base_poss} {nom}"

    if info.suffix_type == "sibling":
        return f"{base_poss} {_sibling_desc(info.sibling_path)}"

    if info.suffix_type == "friend":
        base_nom = _join_nominative(root_name, base_segs)
        return f"{base_nom} barátja/ismerőse"

    return _join_nominative(root_name, base_segs)


def _describe_multi(parts: list[str], names: dict[str, str]) -> str:
    """'C81B, C82B' → 'X és Y közös barátja/ismerőse'."""
    base_descs: list[str] = []
    for part in parts:
        info = parse_extended_code(part)
        root_name = names.get(info.root, info.root)
        base_segs = _path_segments(info.path_digits)
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
