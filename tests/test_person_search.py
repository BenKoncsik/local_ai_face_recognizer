"""Tests for the accent-insensitive, case-insensitive person search helpers."""

from __future__ import annotations

import pytest

from app.utils.person_search import (
    PersonEntry,
    filter_entries,
    normalize,
    person_is_unknown,
    search_persons,
)


# ─────────────────────────────────────────────────────────────────────────────
# normalize()
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalize:
    def test_lowercase(self):
        assert normalize("ANNA") == "anna"

    def test_accent_stripped(self):
        assert normalize("Árvíztűrő") == "arvizturо".replace("о", "o")
        # simpler:
        assert normalize("Pánnì") == "panni"
        assert normalize("ÁBÉCÉ") == "abece"

    def test_mixed(self):
        assert normalize("Pánni") == "panni"
        assert normalize("pánni") == "panni"
        assert normalize("PÁNNI") == "panni"

    def test_no_accents_unchanged(self):
        assert normalize("john doe") == "john doe"

    def test_empty(self):
        assert normalize("") == ""


# ─────────────────────────────────────────────────────────────────────────────
# PersonEntry
# ─────────────────────────────────────────────────────────────────────────────

class TestPersonEntry:
    def test_normalized_computed(self):
        e = PersonEntry(person_id=1, name="Péter")
        assert e._normalized == "peter"

    def test_display_text_defaults_to_name(self):
        e = PersonEntry(person_id=1, name="Anna")
        assert e.display_text == "Anna"

    def test_display_text_custom(self):
        e = PersonEntry(person_id=1, name="Anna", display_text="Anna (3 arcs)")
        assert e.display_text == "Anna (3 arcs)"

    def test_search_uses_name_not_display_text(self):
        # display_text might contain "(5 faces)" — search should still match on name
        e = PersonEntry(person_id=1, name="Béla", display_text="Béla (5 arcs)")
        results = search_persons("bela", [e])
        assert results == [e]


# ─────────────────────────────────────────────────────────────────────────────
# search_persons()
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE: list[PersonEntry] = [
    PersonEntry(1, "Anna Kovács"),
    PersonEntry(2, "Béla Nagy"),
    PersonEntry(3, "Panni Kiss"),
    PersonEntry(4, "Pánni Varga"),
    PersonEntry(5, "PANNI Fekete"),
    PersonEntry(6, "Zsuzsa Tóth"),
    PersonEntry(7, "Örökzöld Péter"),
]


class TestSearchPersons:
    def test_empty_query_returns_all(self):
        assert search_persons("", SAMPLE) == SAMPLE

    def test_whitespace_only_returns_all(self):
        assert search_persons("   ", SAMPLE) == SAMPLE

    def test_case_insensitive(self):
        results = search_persons("anna", SAMPLE)
        assert any(e.person_id == 1 for e in results)

    def test_case_insensitive_upper(self):
        results = search_persons("ANNA", SAMPLE)
        assert any(e.person_id == 1 for e in results)

    def test_accent_insensitive_panni(self):
        # "panni" should find "Panni Kiss", "Pánni Varga", "PANNI Fekete"
        results = search_persons("panni", SAMPLE)
        ids = {e.person_id for e in results}
        assert {3, 4, 5}.issubset(ids)

    def test_accent_insensitive_panni(self):
        # "pánni" (with accent) should also find all three
        results = search_persons("pánni", SAMPLE)
        ids = {e.person_id for e in results}
        assert {3, 4, 5}.issubset(ids)

    def test_partial_match(self):
        results = search_persons("ko", SAMPLE)
        assert any(e.person_id == 1 for e in results)  # Kovács

    def test_no_results(self):
        results = search_persons("xyzxyz", SAMPLE)
        assert results == []

    def test_max_results(self):
        many = [PersonEntry(i, f"Person {i}") for i in range(100)]
        results = search_persons("person", many, max_results=10)
        assert len(results) == 10

    def test_prefix_ranked_first(self):
        entries = [
            PersonEntry(1, "Kovács Anna"),  # infix match on "anna"
            PersonEntry(2, "Anna Béla"),    # prefix match on "anna"
        ]
        results = search_persons("anna", entries)
        assert results[0].person_id == 2, "Prefix match should rank first"

    def test_no_cross_contamination_same_name(self):
        # Two persons named "Anna" — both should appear
        entries = [
            PersonEntry(10, "Anna", display_text="Anna (id=10)"),
            PersonEntry(11, "Anna", display_text="Anna (id=11)"),
        ]
        results = search_persons("anna", entries)
        ids = {e.person_id for e in results}
        assert ids == {10, 11}

    def test_accent_o_variants(self):
        entry = PersonEntry(1, "Örökzöld Péter")
        results = search_persons("orokzold", [entry])
        assert results == [entry]

    def test_accent_o_with_accented_query(self):
        entry = PersonEntry(1, "Örökzöld Péter")
        results = search_persons("örökzöld", [entry])
        assert results == [entry]

    def test_not_found_returns_empty(self):
        results = search_persons("Józsi", SAMPLE)
        assert results == []

    def test_subset_not_modified(self):
        # Searching should not modify the original list
        original = list(SAMPLE)
        search_persons("panni", SAMPLE)
        assert SAMPLE == original


# ─────────────────────────────────────────────────────────────────────────────
# filter_entries() — order-preserving filter (face-match mode)
# ─────────────────────────────────────────────────────────────────────────────

class TestFilterEntries:
    def test_empty_query_returns_all_in_order(self):
        assert filter_entries("", SAMPLE) == SAMPLE

    def test_preserves_incoming_order(self):
        # Reverse the input: a relevance re-sort would reorder, filter must not.
        reversed_sample = list(reversed(SAMPLE))
        result = filter_entries("", reversed_sample)
        assert result == reversed_sample

    def test_does_not_rerank_by_prefix(self):
        entries = [
            PersonEntry(1, "Kovács Anna"),  # infix match on "anna"
            PersonEntry(2, "Anna Béla"),    # prefix match on "anna"
        ]
        result = filter_entries("anna", entries)
        # Unlike search_persons, the original order is kept (infix first).
        assert [e.person_id for e in result] == [1, 2]

    def test_filters_accent_insensitive(self):
        result = filter_entries("panni", SAMPLE)
        assert {e.person_id for e in result} == {3, 4, 5}

    def test_no_results(self):
        assert filter_entries("xyzxyz", SAMPLE) == []

    def test_max_results(self):
        many = [PersonEntry(i, f"Person {i}") for i in range(100)]
        assert len(filter_entries("person", many, max_results=10)) == 10


# ─────────────────────────────────────────────────────────────────────────────
# PersonSearchSelect widget — selection reset / best-match preselection
# ─────────────────────────────────────────────────────────────────────────────

class _FakePerson:
    """Minimal stand-in for app.db.models.Person (id + name)."""

    def __init__(
        self,
        pid: int,
        name: str,
        is_auto_named: bool = False,
        is_protected: bool = False,
    ) -> None:
        self.id = pid
        self.name = name
        self.is_auto_named = is_auto_named
        self.is_protected = is_protected


class TestPersonSearchSelectReset:
    @staticmethod
    def _make(qtbot):
        from app.ui.widgets.person_search_select import PersonSearchSelect

        widget = PersonSearchSelect()
        qtbot.addWidget(widget)
        persons = [
            _FakePerson(1, "Apa"),
            _FakePerson(2, "Jerne"),
            _FakePerson(3, "Mama"),
        ]
        return widget, persons

    def test_preselect_best_match_selects_top_when_match_sorted(self, qtbot):
        widget, persons = self._make(qtbot)
        widget._match_checkbox.setChecked(True)
        widget.set_persons(persons)
        # Jerne is the strongest match, Apa the weakest.
        widget.set_match_scores({1: 0.18, 2: 0.49, 3: 0.05})

        assert widget.preselect_best_match() is True
        # Best match (Jerne) is preselected, not the previously edited person.
        assert widget.current_person_id() == 2

    def test_preselect_returns_false_without_match_sort(self, qtbot):
        widget, persons = self._make(qtbot)
        widget._match_checkbox.setChecked(False)
        widget.set_persons(persons)

        # No match-sorting → nothing is auto-selected (Assign stays safe).
        assert widget.preselect_best_match() is False
        assert widget.current_person_id() is None

    def test_clear_selection_drops_previous_target(self, qtbot):
        widget, persons = self._make(qtbot)
        widget.set_persons(persons)
        widget.set_current_by_id(1)  # "Apa" selected on the previous face
        assert widget.current_person_id() == 1

        widget.clear_selection()
        assert widget.current_person_id() is None
        # A subsequent rebuild must not resurrect the stale selection.
        widget.set_persons(persons)
        assert widget.current_person_id() is None

    def test_set_current_by_id_scrolls_selection_into_view(self, qtbot):
        # A long list so the target row would otherwise sit below the fold.
        from app.ui.widgets.person_search_select import PersonSearchSelect

        widget = PersonSearchSelect()
        qtbot.addWidget(widget)
        widget.resize(220, 200)
        widget.show()
        qtbot.waitExposed(widget)
        many = [_FakePerson(i, f"P{i:03d}") for i in range(1, 41)]
        widget.set_persons(many)

        # Selecting a row far down the list must reveal it, not keep the old
        # scroll position — the row's visual rect lands inside the viewport.
        widget.set_current_by_id(40)
        assert widget.current_person_id() == 40
        item = widget._list.currentItem()
        rect = widget._list.visualItemRect(item)
        viewport = widget._list.viewport().rect()
        assert viewport.intersects(rect)

    def test_set_current_by_id_missing_target_scrolls_to_top(self, qtbot):
        widget, persons = self._make(qtbot)
        widget.set_persons(persons)
        # An id not present in the list clears the selection and resets scroll;
        # no exception, current selection is None.
        widget.set_current_by_id(999)
        assert widget.current_person_id() is None


# ─────────────────────────────────────────────────────────────────────────────
# PersonSearchSelect widget — match-score ordering (merge dialog default)
# ─────────────────────────────────────────────────────────────────────────────

class TestPersonSearchSelectMatchSort:
    @staticmethod
    def _visible_ids(widget) -> list[int]:
        from app.ui.widgets.person_search_select import _ROLE_ID

        return [
            widget._list.item(row).data(_ROLE_ID)
            for row in range(widget._list.count())
        ]

    @staticmethod
    def _make(qtbot):
        from app.ui.widgets.person_search_select import PersonSearchSelect

        widget = PersonSearchSelect()
        qtbot.addWidget(widget)
        persons = [
            _FakePerson(1, "Andris"),
            _FakePerson(2, "Béla"),
            _FakePerson(3, "Cili"),
            _FakePerson(4, "Dani"),
        ]
        widget.set_persons(persons)
        return widget

    def test_default_sort_orders_list_by_score_desc(self, qtbot):
        widget = self._make(qtbot)
        # Deliberately not in name order: Cili highest, then Andris, Dani, Béla.
        widget.set_match_scores(
            {1: 0.71, 2: 0.10, 3: 0.93, 4: 0.55}, default_sort=True
        )
        # default_sort turns the checkbox on without the caller toggling it.
        assert widget._match_checkbox.isChecked() is True
        assert self._visible_ids(widget) == [3, 1, 4, 2]

    def test_default_sort_does_not_persist_global_preference(self, qtbot):
        # Forcing the checkbox on for this selector must not write the global
        # QSettings preference (it is set with signals blocked).
        from app.ui.widgets.person_search_select import _MATCH_SORT_QSETTINGS_KEY
        from app.app_settings import app_qsettings

        app_qsettings().setValue(_MATCH_SORT_QSETTINGS_KEY, False)
        widget = self._make(qtbot)
        widget.set_match_scores({1: 0.9, 2: 0.1}, default_sort=True)
        assert (
            app_qsettings().value(_MATCH_SORT_QSETTINGS_KEY, False, type=bool)
            is False
        )

    def test_search_preserves_score_order(self, qtbot):
        widget = self._make(qtbot)
        widget.set_match_scores(
            {1: 0.71, 2: 0.10, 3: 0.93, 4: 0.55}, default_sort=True
        )
        # All four names contain "i"; the filtered results keep score order.
        widget._search.setText("i")
        ids = self._visible_ids(widget)
        assert ids == [3, 1, 4], "score order must survive filtering"

    def test_no_default_sort_keeps_name_order(self, qtbot):
        widget = self._make(qtbot)
        # Without default_sort the checkbox stays off → alphabetical order.
        widget._match_checkbox.setChecked(False)
        widget.set_match_scores({1: 0.71, 2: 0.10, 3: 0.93, 4: 0.55})
        assert widget._match_checkbox.isChecked() is False
        assert self._visible_ids(widget) == [1, 2, 3, 4]

    def test_percentage_shown_when_match_sorted(self, qtbot):
        widget = self._make(qtbot)
        widget.set_match_scores({1: 0.9, 2: 0.1}, default_sort=True)
        # Top row shows the match percentage next to the name.
        assert "90%" in widget._list.item(0).text()


# ─────────────────────────────────────────────────────────────────────────────
# person_is_unknown() — shared "is this a placeholder identity?" predicate
# ─────────────────────────────────────────────────────────────────────────────

class TestPersonIsUnknown:
    def test_named_person_is_not_unknown(self):
        assert person_is_unknown(_FakePerson(1, "Anna")) is False

    def test_auto_named_is_unknown(self):
        assert person_is_unknown(_FakePerson(1, "Unknown 3", is_auto_named=True)) is True

    def test_protected_is_unknown(self):
        assert person_is_unknown(_FakePerson(1, "Ismeretlen", is_protected=True)) is True

    def test_missing_attributes_default_false(self):
        class _Bare:
            id = 1
            name = "X"

        assert person_is_unknown(_Bare()) is False


# ─────────────────────────────────────────────────────────────────────────────
# search_persons(hide_unknown=…) — base list hides unknown, search reveals it
# ─────────────────────────────────────────────────────────────────────────────

class TestSearchPersonsHideUnknown:
    @staticmethod
    def _entries() -> list[PersonEntry]:
        return [
            PersonEntry(1, "Anna"),
            PersonEntry(2, "Unknown 7", is_unknown=True),
            PersonEntry(3, "Béla"),
        ]

    def test_base_list_hides_unknown(self):
        result = search_persons("", self._entries(), hide_unknown=True)
        assert [e.person_id for e in result] == [1, 3]

    def test_base_list_keeps_unknown_when_flag_off(self):
        result = search_persons("", self._entries(), hide_unknown=False)
        assert [e.person_id for e in result] == [1, 2, 3]

    def test_search_reveals_unknown(self):
        # Even with hide_unknown, a query that matches the name surfaces it.
        result = search_persons("unknown", self._entries(), hide_unknown=True)
        assert [e.person_id for e in result] == [2]

    def test_search_for_named_unaffected(self):
        result = search_persons("anna", self._entries(), hide_unknown=True)
        assert [e.person_id for e in result] == [1]


# ─────────────────────────────────────────────────────────────────────────────
# PersonSearchSelect widget — unknown filtering end to end
# ─────────────────────────────────────────────────────────────────────────────

class TestPersonSearchSelectUnknown:
    @staticmethod
    def _visible_ids(widget) -> list[int]:
        from app.ui.widgets.person_search_select import _ROLE_ID

        return [
            widget._list.item(row).data(_ROLE_ID)
            for row in range(widget._list.count())
        ]

    @staticmethod
    def _make(qtbot):
        from app.ui.widgets.person_search_select import PersonSearchSelect

        widget = PersonSearchSelect()
        qtbot.addWidget(widget)
        persons = [
            _FakePerson(1, "Anna"),
            _FakePerson(2, "Unknown 7", is_auto_named=True),
            _FakePerson(3, "Béla"),
        ]
        widget.set_persons(persons)
        return widget

    def test_base_list_hides_unknown_by_default(self, qtbot):
        widget = self._make(qtbot)
        assert self._visible_ids(widget) == [1, 3]

    def test_search_reveals_unknown(self, qtbot):
        widget = self._make(qtbot)
        widget._search.setText("unknown")
        assert self._visible_ids(widget) == [2]

    def test_opt_out_shows_unknown_in_base(self, qtbot):
        widget = self._make(qtbot)
        widget.set_hide_unknown_in_base(False)
        # Alphabetical order: Anna, Béla, Unknown 7.
        assert self._visible_ids(widget) == [1, 3, 2]

    def test_match_sort_keeps_unknown_visible(self, qtbot):
        widget = self._make(qtbot)
        # Match-sorting is a relevance ranking, so unknown clusters stay listed.
        widget.set_match_scores({1: 0.2, 2: 0.9, 3: 0.1}, default_sort=True)
        assert self._visible_ids(widget) == [2, 1, 3]
