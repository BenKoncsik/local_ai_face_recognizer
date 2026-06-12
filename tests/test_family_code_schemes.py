"""Tests for user-editable family code schemes (model, store, interpreter)."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from app.services.family_code_interpreter import (
    code_is_relational_marker,
    describe_family_code,
    expand_range_code,
    parse_extended_code,
    validate_extended_code,
    validate_external_family_code,
)
from app.services.family_code_schemes import (
    BUILTIN_SCHEME_ID,
    FamilyCodeScheme,
    FamilyCodeSchemeStore,
    SchemeRoot,
    builtin_example_scheme,
    get_active_scheme,
    scheme_example_codes,
    scheme_problems,
    set_active_scheme,
)


@pytest.fixture(autouse=True)
def _reset_active_scheme():
    # Every test starts and ends with the built-in default as active scheme.
    set_active_scheme(None)
    yield
    set_active_scheme(None)


@pytest.fixture()
def custom_scheme() -> FamilyCodeScheme:
    """A strict scheme with non-default letters and all extras disabled."""
    return FamilyCodeScheme(
        scheme_id="test-custom",
        name="Teszt család",
        roots=[SchemeRoot("K", "Kata"), SchemeRoot("M", "Marci")],
        ancestor_letter="O",
        sibling_letter="S",
        spouse_letter="P",
        friend_letter="X",
        allow_unlisted_roots=False,
        allow_multi_codes=False,
        allow_ranges=False,
        allow_braces=False,
        allow_external=False,
    )


# ── Built-in example ──────────────────────────────────────────────────────────

def test_builtin_scheme_mirrors_legacy_grammar():
    scheme = builtin_example_scheme()
    assert scheme.scheme_id == BUILTIN_SCHEME_ID
    assert scheme.is_builtin
    assert scheme.root_names() == {
        "C": "Cikky",
        "G": "Gábor",
        "J": "Jerne",
        "I": "Ildi",
    }
    assert scheme.marker_display() == "F/T/H/B"
    assert scheme_problems(scheme) == []
    # The default (no explicit scheme, nothing activated) is the builtin.
    assert validate_extended_code("C81B") == "C81B"
    assert describe_family_code("C0") == "Cikky"


# ── Custom marker letters / roots ────────────────────────────────────────────

def test_custom_letters_validate_and_describe(custom_scheme):
    assert validate_extended_code("k81x", scheme=custom_scheme) == "K81X"
    assert validate_extended_code("K0O1", scheme=custom_scheme) == "K0O1"
    assert validate_extended_code("K00S1", scheme=custom_scheme) == "K00S1"
    assert validate_extended_code("M1P2", scheme=custom_scheme) == "M1P2"

    assert describe_family_code("K0", scheme=custom_scheme) == "Kata"
    assert describe_family_code("K0O1", scheme=custom_scheme) == "Kata apja"
    assert describe_family_code("M1", scheme=custom_scheme) == "Marci 1. gyermeke"
    assert describe_family_code("K81X", scheme=custom_scheme).endswith(
        "barátja/ismerőse"
    )

    info = parse_extended_code("K1P2", scheme=custom_scheme)
    assert info.suffix_type == "spouse"
    assert info.spouse_path == (2,)


def test_legacy_letters_rejected_under_custom_scheme(custom_scheme):
    # B is not a marker in this scheme → it is an invalid letter in the code.
    with pytest.raises(ValueError):
        validate_extended_code("K81B", scheme=custom_scheme)
    # C is not a listed root and unlisted roots are disabled.
    with pytest.raises(ValueError):
        validate_extended_code("C8", scheme=custom_scheme)


def test_unlisted_roots_allowed_when_enabled(custom_scheme):
    relaxed = replace(custom_scheme, allow_unlisted_roots=True)
    assert validate_extended_code("Z8", scheme=relaxed) == "Z8"
    # The unknown root shows up as its letter in descriptions.
    assert describe_family_code("Z8", scheme=relaxed) == "Z 8. gyermeke"


# ── Feature toggles ───────────────────────────────────────────────────────────

def test_disabled_extras_are_rejected(custom_scheme):
    with pytest.raises(ValueError):
        validate_extended_code("K81X,K82X", scheme=custom_scheme)  # multi
    with pytest.raises(ValueError):
        validate_extended_code("K[1-9]X", scheme=custom_scheme)  # range
    with pytest.raises(ValueError):
        validate_extended_code("K1{2}3", scheme=custom_scheme)  # braces
    with pytest.raises(ValueError):
        validate_external_family_code("#Zoli#31", scheme=custom_scheme)  # external
    with pytest.raises(ValueError):
        parse_extended_code("#Zoli#31", scheme=custom_scheme)


def test_enabled_extras_work_with_custom_letters(custom_scheme):
    relaxed = replace(
        custom_scheme,
        allow_multi_codes=True,
        allow_ranges=True,
        allow_braces=True,
        allow_external=True,
    )
    assert validate_extended_code("K81X,K82X", scheme=relaxed) == "K81X K82X"
    assert validate_extended_code("K[1-3]X", scheme=relaxed) == "K[1-3]X"
    assert expand_range_code("K[1-3]X", scheme=relaxed) == ["K1X", "K2X", "K3X"]
    assert validate_extended_code("K1{2}3", scheme=relaxed) == "K1{2}3"
    assert validate_external_family_code("#Zoli#31", scheme=relaxed) == "#Zoli#31"


def test_disabled_marker_means_letter_is_invalid(custom_scheme):
    no_friend = replace(custom_scheme, friend_letter="")
    with pytest.raises(ValueError):
        validate_extended_code("K81X", scheme=no_friend)


# ── Active scheme switching ───────────────────────────────────────────────────

def test_active_scheme_drives_default_validation(custom_scheme):
    set_active_scheme(custom_scheme)
    assert get_active_scheme() is custom_scheme
    assert validate_extended_code("K81X") == "K81X"
    with pytest.raises(ValueError):
        validate_extended_code("C81B")  # C root not listed, B not a marker

    set_active_scheme(None)
    assert validate_extended_code("C81B") == "C81B"


# ── Relational marker (uniqueness exemption) ─────────────────────────────────

def test_code_is_relational_marker():
    assert code_is_relational_marker("C81B")
    assert code_is_relational_marker("C[1-9]B")
    assert code_is_relational_marker("C81B C82B")
    assert not code_is_relational_marker("C85")
    assert not code_is_relational_marker("C0F1")


def test_code_is_relational_marker_custom_letters(custom_scheme):
    assert code_is_relational_marker("K81X", scheme=custom_scheme)
    assert not code_is_relational_marker("K81", scheme=custom_scheme)


# ── Scheme consistency check ──────────────────────────────────────────────────

def test_scheme_problems_detects_mistakes(custom_scheme):
    assert scheme_problems(custom_scheme) == []

    unnamed = replace(custom_scheme, name="  ")
    assert any("nev" in p.lower() for p in scheme_problems(unnamed))

    dup_roots = replace(
        custom_scheme,
        roots=[SchemeRoot("K", "Kata"), SchemeRoot("K", "Klára")],
    )
    assert scheme_problems(dup_roots)

    dup_markers = replace(custom_scheme, sibling_letter="O")
    assert scheme_problems(dup_markers)

    range_without_friend = replace(
        custom_scheme, friend_letter="", allow_ranges=True
    )
    assert scheme_problems(range_without_friend)


# ── Examples ──────────────────────────────────────────────────────────────────

def test_scheme_example_codes_use_scheme_letters(custom_scheme):
    examples = dict(scheme_example_codes(custom_scheme))
    assert "K0O1" in examples
    assert "K00S1" in examples
    assert "K81X" in examples
    assert examples["K0"] == "Kata"


# ── Store ─────────────────────────────────────────────────────────────────────

def test_store_save_list_get_delete(tmp_path, custom_scheme):
    store = FamilyCodeSchemeStore(tmp_path / "schemes")

    schemes = store.list_schemes()
    assert [s.scheme_id for s in schemes] == [BUILTIN_SCHEME_ID]

    store.save_scheme(custom_scheme)
    schemes = store.list_schemes()
    assert [s.scheme_id for s in schemes] == [BUILTIN_SCHEME_ID, "test-custom"]

    loaded = store.get_scheme("test-custom")
    assert loaded is not None
    assert loaded.name == "Teszt család"
    assert loaded.friend_letter == "X"
    assert loaded.root_names() == {"K": "Kata", "M": "Marci"}
    assert not loaded.allow_external

    store.delete_scheme("test-custom")
    assert store.get_scheme("test-custom") is None


def test_store_rejects_builtin_mutation(tmp_path):
    store = FamilyCodeSchemeStore(tmp_path / "schemes")
    with pytest.raises(ValueError):
        store.save_scheme(builtin_example_scheme())
    with pytest.raises(ValueError):
        store.delete_scheme(BUILTIN_SCHEME_ID)


def test_store_duplicate_creates_editable_copy(tmp_path):
    store = FamilyCodeSchemeStore(tmp_path / "schemes")
    copy = store.duplicate_scheme(BUILTIN_SCHEME_ID)
    assert copy.scheme_id != BUILTIN_SCHEME_ID
    assert not copy.is_builtin
    assert "másolat" in copy.name
    assert copy.root_names()["C"] == "Cikky"
    # The copy is persisted and editable.
    copy.name = "Saját"
    store.save_scheme(copy)
    assert store.get_scheme(copy.scheme_id).name == "Saját"


def test_store_active_scheme_persistence(tmp_path, custom_scheme):
    store = FamilyCodeSchemeStore(tmp_path / "schemes")
    assert store.active_scheme_id() == BUILTIN_SCHEME_ID

    store.save_scheme(custom_scheme)
    store.set_active_scheme_id("test-custom")
    assert store.active_scheme_id() == "test-custom"
    # Activation also updates the runtime default.
    assert get_active_scheme().scheme_id == "test-custom"
    assert validate_extended_code("K81X") == "K81X"

    # A second store instance (fresh start) resolves the same active scheme.
    set_active_scheme(None)
    store2 = FamilyCodeSchemeStore(tmp_path / "schemes")
    active = store2.load_active_into_runtime()
    assert active.scheme_id == "test-custom"
    assert get_active_scheme().scheme_id == "test-custom"

    # Deleting the active scheme falls back to the builtin.
    store2.delete_scheme("test-custom")
    assert store2.active_scheme_id() == BUILTIN_SCHEME_ID
    assert get_active_scheme().scheme_id == BUILTIN_SCHEME_ID


def test_load_active_falls_back_when_file_missing(tmp_path):
    store = FamilyCodeSchemeStore(tmp_path / "schemes")
    (tmp_path / "schemes").mkdir(parents=True, exist_ok=True)
    (tmp_path / "schemes" / "_active.json").write_text(
        json.dumps({"active_id": "does-not-exist"}), encoding="utf-8"
    )
    active = store.load_active_into_runtime()
    assert active.scheme_id == BUILTIN_SCHEME_ID


def test_store_export_import_roundtrip(tmp_path, custom_scheme):
    store = FamilyCodeSchemeStore(tmp_path / "schemes")
    store.save_scheme(custom_scheme)

    export_path = tmp_path / "out" / "teszt.json"
    export_path.parent.mkdir(parents=True)
    store.export_scheme("test-custom", export_path)
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    assert payload["format"] == "facelocal.family-code-scheme.v1"
    assert payload["markers"]["friend"] == "X"

    # Import into a fresh store: full roundtrip of the grammar settings.
    other = FamilyCodeSchemeStore(tmp_path / "other_schemes")
    imported = other.import_scheme(export_path)
    assert imported.name == "Teszt család"
    assert imported.friend_letter == "X"
    assert not imported.allow_multi_codes
    assert validate_extended_code("K81X", scheme=imported) == "K81X"

    # Importing into the original store collides on id and name → renamed copy.
    second = store.import_scheme(export_path)
    assert second.scheme_id != "test-custom"
    assert second.name == "Teszt család (importált)"


def test_import_rejects_foreign_json(tmp_path):
    store = FamilyCodeSchemeStore(tmp_path / "schemes")
    bogus = tmp_path / "bogus.json"
    bogus.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    with pytest.raises(ValueError):
        store.import_scheme(bogus)


# ── Person info dialog: warn-but-save behaviour ──────────────────────────────

@pytest.fixture()
def person_db(tmp_path):
    from app.db.database import init_db, session_scope
    from app.db.models import Person

    init_db(tmp_path / "persons.db")
    with session_scope() as session:
        person = Person(name="Teszt Elek", is_auto_named=False)
        session.add(person)
        session.flush()
        return person.id


def _auto_answer_message_boxes(monkeypatch, role):
    """Make every QMessageBox auto-click its first button with *role*.

    Returns a list that records one entry per shown box, so tests can assert
    whether a warning appeared at all.
    """
    from PySide6.QtWidgets import QMessageBox

    shown: list[str] = []

    def fake_exec(self):  # noqa: ANN001
        shown.append(self.text())
        for btn in self.buttons():
            if self.buttonRole(btn) == role:
                self._test_clicked = btn
                break
        return 0

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    monkeypatch.setattr(
        QMessageBox, "clickedButton", lambda self: getattr(self, "_test_clicked", None)
    )
    return shown


def test_person_dialog_valid_code_is_canonicalised(qtbot, person_db, monkeypatch):
    from PySide6.QtWidgets import QDialog, QMessageBox
    from app.db.database import session_scope
    from app.db.models import Person
    from app.ui.dialogs.person_info_dialog import PersonInfoDialog

    shown = _auto_answer_message_boxes(monkeypatch, QMessageBox.AcceptRole)
    with session_scope() as session:
        dlg = PersonInfoDialog(session.get(Person, person_db))
    qtbot.addWidget(dlg)

    dlg._family_code.setText("c85")
    dlg.accept()
    assert dlg.result() == QDialog.Accepted
    assert dlg.family_code() == "C85"
    assert shown == []  # a valid code shows no warning


def test_person_dialog_invalid_code_saved_after_override(qtbot, person_db, monkeypatch):
    from PySide6.QtWidgets import QDialog, QMessageBox
    from app.db.database import session_scope
    from app.db.models import Person
    from app.ui.dialogs.person_info_dialog import PersonInfoDialog

    shown = _auto_answer_message_boxes(monkeypatch, QMessageBox.AcceptRole)
    with session_scope() as session:
        dlg = PersonInfoDialog(session.get(Person, person_db))
    qtbot.addWidget(dlg)

    dlg._family_code.setText("rossz-kód!!")
    dlg.accept()
    assert len(shown) == 1  # warned once…
    assert dlg.result() == QDialog.Accepted  # …but the save went through
    assert dlg.family_code() == "rossz-kód!!"  # kept exactly as typed


def test_person_dialog_invalid_code_back_to_edit(qtbot, person_db, monkeypatch):
    from PySide6.QtWidgets import QDialog, QMessageBox
    from app.db.database import session_scope
    from app.db.models import Person
    from app.ui.dialogs.person_info_dialog import PersonInfoDialog

    shown = _auto_answer_message_boxes(monkeypatch, QMessageBox.RejectRole)
    with session_scope() as session:
        dlg = PersonInfoDialog(session.get(Person, person_db))
    qtbot.addWidget(dlg)

    dlg._family_code.setText("rossz-kód!!")
    dlg.accept()
    assert len(shown) == 1
    assert dlg.result() != QDialog.Accepted  # stayed in the editor


def test_person_dialog_help_follows_active_scheme(qtbot, person_db, custom_scheme):
    from app.db.database import session_scope
    from app.db.models import Person
    from app.ui.dialogs.person_info_dialog import PersonInfoDialog

    set_active_scheme(custom_scheme)
    with session_scope() as session:
        dlg = PersonInfoDialog(session.get(Person, person_db))
    qtbot.addWidget(dlg)

    assert "K85" in dlg._family_code.placeholderText()
    assert "Teszt család" in dlg._family_code.toolTip()
    assert "K0O1" in dlg._family_code.toolTip()


# ── Editor dialog smoke test ──────────────────────────────────────────────────

def test_scheme_dialog_smoke(qtbot, tmp_path):
    from app.ui.dialogs.family_code_scheme_dialog import FamilyCodeSchemeDialog

    store = FamilyCodeSchemeStore(tmp_path / "schemes")
    dlg = FamilyCodeSchemeDialog(store=store)
    qtbot.addWidget(dlg)

    # The builtin scheme is selected and read-only.
    assert dlg._current is not None
    assert dlg._current.is_builtin
    assert not dlg._name_edit.isEnabled()
    assert not dlg._save_btn.isEnabled()

    # Live tester describes a code under the builtin scheme.
    dlg._tester_edit.setText("C85")
    assert "Cikky" in dlg._tester_result.text()
    dlg._tester_edit.setText("C09")  # child hung off the bare root marker
    assert "✗" in dlg._tester_result.text()

    # Duplicate creates an editable copy and selects it.
    dlg._on_duplicate()
    assert dlg._current is not None
    assert not dlg._current.is_builtin
    assert dlg._name_edit.isEnabled()

    # Editing a marker letter updates the live examples.
    check, letter_edit, example_lbl, _hint = dlg._marker_rows["friend"]
    assert check.isChecked()
    letter_edit.setText("X")
    dlg._on_marker_letter_edited("X")
    assert "X" in example_lbl.text()

    # Saving persists the change.
    assert dlg._on_save()
    saved = store.get_scheme(dlg._current.scheme_id)
    assert saved.friend_letter == "X"
