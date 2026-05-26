"""Tests for the accent-insensitive, case-insensitive person search helpers."""

from __future__ import annotations

import pytest

from app.utils.person_search import PersonEntry, normalize, search_persons


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
