"""Tests for fuzzy person-name matching (pure logic)."""

from __future__ import annotations

from app.services.name_matching import (
    name_key,
    name_similarity,
    names_match,
    normalize_name,
    phonetic_key,
)


class TestNormalizeName:
    def test_lowercase_and_accents(self):
        assert normalize_name("PÁNNI") == "panni"
        assert normalize_name("Kovács Anna") == "kovacs anna"

    def test_collapses_whitespace(self):
        assert normalize_name("  John   Doe ") == "john doe"

    def test_strips_punctuation_and_honorifics(self):
        assert normalize_name("Dr. Kovács Anna") == "kovacs anna"
        assert normalize_name("ifj. Nagy Béla") == "nagy bela"

    def test_none_and_empty(self):
        assert normalize_name(None) == ""
        assert normalize_name("") == ""
        assert normalize_name("   ") == ""


class TestNameKey:
    def test_order_independent(self):
        assert name_key("Kovács Anna") == name_key("Anna Kovács")

    def test_distinct_names_differ(self):
        assert name_key("Kovács Anna") != name_key("Nagy Béla")


class TestPhoneticKey:
    def test_homophones_collapse(self):
        assert phonetic_key("Filip") == phonetic_key("Philip")

    def test_doubled_letters_collapse(self):
        assert phonetic_key("Anna") == phonetic_key("Ana")


class TestNameSimilarity:
    def test_exact_after_normalization(self):
        assert name_similarity("Anna", "anna") == 1.0
        assert name_similarity("Pánni", "panni") == 1.0

    def test_extra_spaces(self):
        assert name_similarity("  John   Doe ", "John Doe") == 1.0

    def test_token_order(self):
        assert name_similarity("Kovács Anna", "Anna Kovács") >= 0.95

    def test_close_names_high(self):
        assert name_similarity("Katalin", "Katalín") >= 0.85

    def test_unrelated_low(self):
        assert name_similarity("Anna", "Béla") < 0.5

    def test_empty_is_zero(self):
        assert name_similarity("", "Anna") == 0.0
        assert name_similarity(None, "Anna") == 0.0

    def test_nickname_match(self):
        # "Pista" is a common Hungarian nickname for "István".
        score = name_similarity(
            "Pista", "István Nagy", nicknames_b=("Pista",)
        )
        assert score >= 0.9

    def test_names_match_threshold(self):
        assert names_match("Anna", "anna")
        assert not names_match("Anna", "Béla")
