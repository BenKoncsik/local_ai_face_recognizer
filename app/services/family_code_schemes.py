"""User-editable family code schemes.

A *scheme* describes how the family codes of this archive are built: which
letters stand for the family roots (C = Cikky, ...), which letters mark the
relationship suffixes (F = ancestors, T = siblings, H = numbered spouses,
B = friends) and which extra notations (multi-codes, ranges, braces, external
#root# identifiers) are allowed.

The previously hard-coded grammar lives on as the read-only built-in example
scheme (:func:`builtin_example_scheme`).  Users can create any number of their
own schemes in the graphical editor, export/import them as JSON files and
activate exactly one; the active scheme drives validation and the Hungarian
descriptions in :mod:`app.services.family_code_interpreter`.

Storage is file based: every user scheme is one JSON file inside
``Documents/localAIFaceRecognizer/settings/family_code_schemes/`` and the
active scheme id is kept in ``_active.json`` next to them, so no Qt machinery
is needed and the files themselves are the export format.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

SCHEME_FORMAT = "facelocal.family-code-scheme.v1"
BUILTIN_SCHEME_ID = "builtin-cgji-example"

_ACTIVE_FILENAME = "_active.json"
_SAFE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9\-]{0,63}$")
_LETTER_RE = re.compile(r"^[A-Z]$")


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class SchemeRoot:
    """One family-tree root person: the letter a code may start with."""

    letter: str
    name: str = ""
    note: str = ""


@dataclass
class FamilyCodeScheme:
    """A complete, user-editable description of a family code grammar.

    A marker letter set to ``""`` disables that suffix kind entirely.  The
    digit rules (0 = root person / spouse, 1-9 = nth child, stepchild after a
    spouse-0) are the fixed core of every scheme and are explained to the user
    in the editor instead of being configurable.
    """

    scheme_id: str
    name: str
    description: str = ""
    roots: list[SchemeRoot] = field(default_factory=list)
    ancestor_letter: str = "F"
    sibling_letter: str = "T"
    spouse_letter: str = "H"
    friend_letter: str = "B"
    allow_unlisted_roots: bool = True
    allow_multi_codes: bool = True
    allow_ranges: bool = True
    allow_braces: bool = True
    allow_external: bool = True
    is_builtin: bool = False

    # -- helpers -----------------------------------------------------------

    def root_names(self) -> dict[str, str]:
        """Letter → display name map (letter itself when no name was given)."""
        return {r.letter: (r.name or r.letter) for r in self.roots if r.letter}

    def root_letters(self) -> set[str]:
        return {r.letter for r in self.roots if r.letter}

    def enabled_marker_letters(self) -> list[str]:
        return [
            letter
            for letter in (
                self.ancestor_letter,
                self.sibling_letter,
                self.spouse_letter,
                self.friend_letter,
            )
            if letter
        ]

    def marker_display(self) -> str:
        """``"F/T/H/B"`` style listing of the enabled marker letters."""
        return "/".join(self.enabled_marker_letters())

    def copy_as_user_scheme(self, new_name: Optional[str] = None) -> "FamilyCodeScheme":
        """Deep copy with a fresh id, never builtin (used by Duplicate/Import)."""
        return replace(
            self,
            scheme_id=new_scheme_id(),
            name=new_name if new_name is not None else self.name,
            roots=[replace(r) for r in self.roots],
            is_builtin=False,
        )

    # -- (de)serialisation ---------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "format": SCHEME_FORMAT,
            "scheme_id": self.scheme_id,
            "name": self.name,
            "description": self.description,
            "roots": [
                {"letter": r.letter, "name": r.name, "note": r.note}
                for r in self.roots
            ],
            "markers": {
                "ancestor": self.ancestor_letter,
                "sibling": self.sibling_letter,
                "spouse": self.spouse_letter,
                "friend": self.friend_letter,
            },
            "options": {
                "allow_unlisted_roots": self.allow_unlisted_roots,
                "allow_multi_codes": self.allow_multi_codes,
                "allow_ranges": self.allow_ranges,
                "allow_braces": self.allow_braces,
                "allow_external": self.allow_external,
            },
        }

    @staticmethod
    def from_dict(data: dict) -> "FamilyCodeScheme":
        """Build a scheme from a JSON payload, normalising letters on the way.

        Raises ValueError for payloads that are not family code schemes at all;
        individual fields are normalised leniently (single uppercase letter or
        disabled) so a hand-edited file still loads.
        """
        if not isinstance(data, dict):
            raise ValueError("A séma fájl tartalma nem JSON objektum.")
        if data.get("format") != SCHEME_FORMAT:
            raise ValueError(
                "Ez a fájl nem családi kód séma (hiányzó vagy ismeretlen 'format' mező)."
            )
        markers = data.get("markers") or {}
        options = data.get("options") or {}
        roots: list[SchemeRoot] = []
        for entry in data.get("roots") or []:
            letter = _normalize_letter(str(entry.get("letter", "")))
            if not letter:
                continue
            roots.append(
                SchemeRoot(
                    letter=letter,
                    name=str(entry.get("name", "")).strip(),
                    note=str(entry.get("note", "")).strip(),
                )
            )
        scheme_id = str(data.get("scheme_id", "")).strip()
        if not _SAFE_ID_RE.fullmatch(scheme_id):
            scheme_id = new_scheme_id()
        return FamilyCodeScheme(
            scheme_id=scheme_id,
            name=str(data.get("name", "")).strip() or "Névtelen séma",
            description=str(data.get("description", "")),
            roots=roots,
            ancestor_letter=_normalize_letter(str(markers.get("ancestor", ""))),
            sibling_letter=_normalize_letter(str(markers.get("sibling", ""))),
            spouse_letter=_normalize_letter(str(markers.get("spouse", ""))),
            friend_letter=_normalize_letter(str(markers.get("friend", ""))),
            allow_unlisted_roots=bool(options.get("allow_unlisted_roots", True)),
            allow_multi_codes=bool(options.get("allow_multi_codes", True)),
            allow_ranges=bool(options.get("allow_ranges", True)),
            allow_braces=bool(options.get("allow_braces", True)),
            allow_external=bool(options.get("allow_external", True)),
            is_builtin=False,
        )


def _normalize_letter(value: str) -> str:
    value = value.strip().upper()
    return value if _LETTER_RE.fullmatch(value) else ""


def new_scheme_id() -> str:
    return uuid.uuid4().hex


def scheme_problems(scheme: FamilyCodeScheme) -> list[str]:
    """Human-readable (Hungarian) list of everything wrong with a scheme.

    An empty list means the scheme is consistent and safe to save/activate.
    """
    problems: list[str] = []
    if not scheme.name.strip():
        problems.append("A séma nevének kitöltése kötelező.")

    seen_roots: set[str] = set()
    for root in scheme.roots:
        if not _LETTER_RE.fullmatch(root.letter or ""):
            problems.append(
                f"A gyökérszemély betűjele egyetlen A-Z betű lehet ('{root.letter}' nem az)."
            )
            continue
        if root.letter in seen_roots:
            problems.append(f"A(z) '{root.letter}' gyökérbetű többször szerepel.")
        seen_roots.add(root.letter)
    if not scheme.roots and not scheme.allow_unlisted_roots:
        problems.append(
            "Adj meg legalább egy gyökérszemélyt, vagy engedélyezd a nem listázott "
            "gyökérbetűket."
        )

    markers = {
        "ős (felmenő)": scheme.ancestor_letter,
        "testvér": scheme.sibling_letter,
        "házastárs sorszám": scheme.spouse_letter,
        "barát/ismerős": scheme.friend_letter,
    }
    seen_markers: dict[str, str] = {}
    for label, letter in markers.items():
        if not letter:
            continue
        if not _LETTER_RE.fullmatch(letter):
            problems.append(
                f"A(z) {label} jelölő egyetlen A-Z betű lehet ('{letter}' nem az)."
            )
            continue
        if letter in seen_markers:
            problems.append(
                f"A(z) '{letter}' betű két jelölőhöz is hozzá van rendelve "
                f"({seen_markers[letter]} és {label})."
            )
        seen_markers[letter] = label
    if scheme.allow_ranges and not scheme.friend_letter:
        problems.append(
            "A tartományjelölés (pl. C[1-9]B) csak akkor használható, ha a "
            "barát/ismerős jelölő engedélyezve van."
        )
    return problems


# ── Built-in example ──────────────────────────────────────────────────────────

def builtin_example_scheme() -> FamilyCodeScheme:
    """The original hard-coded grammar as a read-only example scheme."""
    return FamilyCodeScheme(
        scheme_id=BUILTIN_SCHEME_ID,
        name="Beépített példa (C/G/J/I család)",
        description=(
            "Az alkalmazás eredeti, beépített családi kód rendszere.\n"
            "Négy testvér a család gyökere, mindegyik saját kezdőbetűt kapott: "
            "C = Cikky, G = Gábor, J = Jerne, I = Ildi.\n"
            "A betű után számjegyek írják le a leszármazást (C85 = Cikky 8. "
            "gyermekének 5. gyermeke), a kód végén álló jelölőbetűk pedig a "
            "különleges kapcsolatokat (C0F1 = Cikky apja, C81B = C81 "
            "barátja/ismerőse).\n"
            "Ez a séma nem szerkeszthető — készíts róla másolatot, és írd át a "
            "saját családodra."
        ),
        roots=[
            SchemeRoot("C", "Cikky"),
            SchemeRoot("G", "Gábor"),
            SchemeRoot("J", "Jerne"),
            SchemeRoot("I", "Ildi"),
        ],
        ancestor_letter="F",
        sibling_letter="T",
        spouse_letter="H",
        friend_letter="B",
        allow_unlisted_roots=True,
        allow_multi_codes=True,
        allow_ranges=True,
        allow_braces=True,
        allow_external=True,
        is_builtin=True,
    )


# ── Active scheme (process-wide runtime state) ───────────────────────────────

_active_scheme: Optional[FamilyCodeScheme] = None


def get_active_scheme() -> FamilyCodeScheme:
    """The scheme every validation/description uses when none is passed.

    Falls back to the built-in example when nothing was activated (e.g. in
    tests or before startup wiring ran).
    """
    return _active_scheme if _active_scheme is not None else builtin_example_scheme()


def set_active_scheme(scheme: Optional[FamilyCodeScheme]) -> None:
    """Set (or with ``None``: reset to built-in) the process-wide active scheme."""
    global _active_scheme
    _active_scheme = scheme


# ── Examples for help texts ───────────────────────────────────────────────────

def scheme_example_codes(
    scheme: Optional[FamilyCodeScheme] = None,
) -> list[tuple[str, str]]:
    """(code, Hungarian description) example pairs generated from a scheme.

    Used by tooltips and the editor so the shown examples always match the
    letters the user actually configured.
    """
    from app.services.family_code_interpreter import describe_family_code

    s = scheme if scheme is not None else get_active_scheme()
    root = next(iter(sorted(s.root_letters())), "C")
    candidates = [f"{root}0", f"{root}85", f"{root}80"]
    if s.ancestor_letter:
        candidates.append(f"{root}0{s.ancestor_letter}1")
    if s.sibling_letter:
        candidates.append(f"{root}00{s.sibling_letter}1")
    if s.spouse_letter:
        candidates.append(f"{root}1{s.spouse_letter}2")
    if s.friend_letter:
        candidates.append(f"{root}81{s.friend_letter}")

    examples: list[tuple[str, str]] = []
    for code in candidates:
        try:
            examples.append((code, describe_family_code(code, scheme=s)))
        except ValueError:  # pragma: no cover - defensive against odd schemes
            continue
    return examples


# ── Store ─────────────────────────────────────────────────────────────────────

class FamilyCodeSchemeStore:
    """File-based store for user schemes plus the active scheme selection."""

    def __init__(self, schemes_dir: Optional[Path] = None) -> None:
        if schemes_dir is None:
            from app.paths import ensure_settings_dir

            schemes_dir = ensure_settings_dir() / "family_code_schemes"
        self._dir = Path(schemes_dir)

    @property
    def schemes_dir(self) -> Path:
        return self._dir

    def _ensure_dir(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)

    def _scheme_path(self, scheme_id: str) -> Path:
        return self._dir / f"{scheme_id}.json"

    # -- listing / loading ---------------------------------------------------

    def list_schemes(self) -> list[FamilyCodeScheme]:
        """Built-in example first, then the user schemes sorted by name."""
        user_schemes: list[FamilyCodeScheme] = []
        if self._dir.is_dir():
            for path in sorted(self._dir.glob("*.json")):
                if path.name == _ACTIVE_FILENAME:
                    continue
                scheme = self._load_file(path)
                if scheme is not None:
                    user_schemes.append(scheme)
        user_schemes.sort(key=lambda s: s.name.lower())
        return [builtin_example_scheme()] + user_schemes

    def get_scheme(self, scheme_id: str) -> Optional[FamilyCodeScheme]:
        if scheme_id == BUILTIN_SCHEME_ID:
            return builtin_example_scheme()
        path = self._scheme_path(scheme_id)
        return self._load_file(path) if path.is_file() else None

    def _load_file(self, path: Path) -> Optional[FamilyCodeScheme]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            scheme = FamilyCodeScheme.from_dict(data)
        except (OSError, ValueError) as exc:
            log.warning("Skipping unreadable family code scheme %s: %s", path, exc)
            return None
        # The filename is the source of truth for the id, so a copied file
        # edited by hand cannot shadow another scheme.
        scheme.scheme_id = path.stem
        return scheme

    # -- saving / deleting -----------------------------------------------------

    def save_scheme(self, scheme: FamilyCodeScheme) -> None:
        if scheme.is_builtin or scheme.scheme_id == BUILTIN_SCHEME_ID:
            raise ValueError("A beépített példa séma nem írható felül.")
        if not _SAFE_ID_RE.fullmatch(scheme.scheme_id):
            raise ValueError(f"Érvénytelen séma-azonosító: '{scheme.scheme_id}'")
        self._ensure_dir()
        path = self._scheme_path(scheme.scheme_id)
        path.write_text(
            json.dumps(scheme.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if scheme.scheme_id == self.active_scheme_id():
            set_active_scheme(scheme)

    def delete_scheme(self, scheme_id: str) -> None:
        if scheme_id == BUILTIN_SCHEME_ID:
            raise ValueError("A beépített példa séma nem törölhető.")
        path = self._scheme_path(scheme_id)
        if path.is_file():
            path.unlink()
        if self.active_scheme_id() == scheme_id:
            self.set_active_scheme_id(BUILTIN_SCHEME_ID)

    def duplicate_scheme(
        self, scheme_id: str, new_name: Optional[str] = None
    ) -> FamilyCodeScheme:
        source = self.get_scheme(scheme_id)
        if source is None:
            raise ValueError(f"Nincs ilyen séma: {scheme_id}")
        copy = source.copy_as_user_scheme(
            new_name=new_name if new_name is not None else f"{source.name} (másolat)"
        )
        self.save_scheme(copy)
        return copy

    # -- export / import -------------------------------------------------------

    def export_scheme(self, scheme_id: str, target: Path) -> Path:
        scheme = self.get_scheme(scheme_id)
        if scheme is None:
            raise ValueError(f"Nincs ilyen séma: {scheme_id}")
        target = Path(target)
        target.write_text(
            json.dumps(scheme.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target

    def import_scheme(self, source: Path) -> FamilyCodeScheme:
        """Import a scheme JSON file; collisions get a fresh id / marked name."""
        data = json.loads(Path(source).read_text(encoding="utf-8"))
        scheme = FamilyCodeScheme.from_dict(data)
        if (
            scheme.scheme_id == BUILTIN_SCHEME_ID
            or self._scheme_path(scheme.scheme_id).exists()
        ):
            scheme = scheme.copy_as_user_scheme()
        existing_names = {s.name for s in self.list_schemes()}
        if scheme.name in existing_names:
            scheme.name = f"{scheme.name} (importált)"
        self.save_scheme(scheme)
        return scheme

    # -- active scheme -----------------------------------------------------------

    def active_scheme_id(self) -> str:
        path = self._dir / _ACTIVE_FILENAME
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            scheme_id = str(data.get("active_id", "")).strip()
        except (OSError, ValueError):
            return BUILTIN_SCHEME_ID
        return scheme_id or BUILTIN_SCHEME_ID

    def set_active_scheme_id(self, scheme_id: str) -> FamilyCodeScheme:
        scheme = self.get_scheme(scheme_id)
        if scheme is None:
            raise ValueError(f"Nincs ilyen séma: {scheme_id}")
        self._ensure_dir()
        (self._dir / _ACTIVE_FILENAME).write_text(
            json.dumps({"active_id": scheme_id}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        set_active_scheme(None if scheme_id == BUILTIN_SCHEME_ID else scheme)
        return scheme

    def load_active_into_runtime(self) -> FamilyCodeScheme:
        """Resolve the persisted active scheme and make it the runtime default.

        Falls back to the built-in example when the persisted id no longer
        exists (deleted file, first start).
        """
        scheme_id = self.active_scheme_id()
        scheme = self.get_scheme(scheme_id)
        if scheme is None:
            log.warning(
                "Active family code scheme '%s' not found — using built-in example",
                scheme_id,
            )
            scheme = builtin_example_scheme()
        set_active_scheme(None if scheme.scheme_id == BUILTIN_SCHEME_ID else scheme)
        return scheme
