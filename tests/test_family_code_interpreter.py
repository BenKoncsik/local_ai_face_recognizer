"""Tests for the extended family code interpreter.

All expected descriptions are in Hungarian.  The test cases match the
examples given in the project specification exactly.
"""

from __future__ import annotations

import pytest

from app.services.family_code_interpreter import (
    DEFAULT_ROOT_NAMES,
    describe_family_code,
    expand_range_code,
    parse_extended_code,
    validate_extended_code,
    validate_external_family_code,
)


# ── parse_extended_code ───────────────────────────────────────────────────────

class TestParseExtendedCode:
    def test_root_person(self):
        info = parse_extended_code("C0")
        assert info.root == "C"
        assert info.path_digits == (0,)
        assert info.suffix_type is None
        assert not info.is_stepchild

    def test_spouse(self):
        info = parse_extended_code("C00")
        assert info.path_digits == (0, 0)
        assert info.suffix_type is None
        assert not info.is_stepchild

    def test_child(self):
        info = parse_extended_code("C8")
        assert info.path_digits == (8,)
        assert info.suffix_type is None

    def test_child_spouse(self):
        info = parse_extended_code("C80")
        assert info.path_digits == (8, 0)

    def test_grandchild(self):
        info = parse_extended_code("C81")
        assert info.path_digits == (8, 1)

    def test_great_grandchild(self):
        info = parse_extended_code("C811")
        assert info.path_digits == (8, 1, 1)

    def test_stepchild_detected(self):
        info = parse_extended_code("C2201")
        assert info.path_digits == (2, 2, 0, 1)
        assert info.is_stepchild

    def test_own_child_not_stepchild(self):
        info = parse_extended_code("C221")
        assert info.path_digits == (2, 2, 1)
        assert not info.is_stepchild

    def test_ancestor_father(self):
        info = parse_extended_code("C0F1")
        assert info.suffix_type == "ancestor"
        assert info.ancestor_path == (1,)

    def test_ancestor_mother(self):
        info = parse_extended_code("C0F2")
        assert info.suffix_type == "ancestor"
        assert info.ancestor_path == (2,)

    def test_ancestor_paternal_grandfather(self):
        info = parse_extended_code("C0F11")
        assert info.ancestor_path == (1, 1)

    def test_ancestor_paternal_grandmother(self):
        info = parse_extended_code("C0F12")
        assert info.ancestor_path == (1, 2)

    def test_ancestor_maternal_grandfather(self):
        info = parse_extended_code("C0F21")
        assert info.ancestor_path == (2, 1)

    def test_ancestor_maternal_grandmother(self):
        info = parse_extended_code("C0F22")
        assert info.ancestor_path == (2, 2)

    def test_sibling(self):
        info = parse_extended_code("C00T1")
        assert info.suffix_type == "sibling"
        assert info.sibling_path == (1,)

    def test_sibling_child(self):
        info = parse_extended_code("C00T11")
        assert info.sibling_path == (1, 1)

    def test_sibling_spouse(self):
        info = parse_extended_code("C00T10")
        assert info.sibling_path == (1, 0)

    def test_friend(self):
        info = parse_extended_code("C81B")
        assert info.suffix_type == "friend"

    def test_case_insensitive(self):
        info = parse_extended_code("c81b")
        assert info.raw == "C81B"
        assert info.suffix_type == "friend"

    def test_invalid_no_path_before_suffix(self):
        with pytest.raises(ValueError):
            parse_extended_code("CF1")

    def test_invalid_b_with_digits_after(self):
        with pytest.raises(ValueError):
            parse_extended_code("C81B1")

    def test_invalid_t_with_zero_sibling(self):
        with pytest.raises(ValueError):
            parse_extended_code("C00T0")

    def test_invalid_f_with_digit_3(self):
        with pytest.raises(ValueError):
            parse_extended_code("C0F3")

    def test_invalid_child_off_root_marker(self):
        # C08: a child index cannot hang directly off the bare root marker.
        with pytest.raises(ValueError):
            parse_extended_code("C08")

    def test_invalid_spouse_of_spouse(self):
        # C800: two 0s in a row away from the root marker.
        with pytest.raises(ValueError):
            parse_extended_code("C800")

    def test_valid_spouse_of_root(self):
        info = parse_extended_code("C00")
        assert info.path_digits == (0, 0)

    def test_valid_stepchild_via_spouse(self):
        # C805 = C8's spouse (C80)'s 5th child from a previous relationship.
        info = parse_extended_code("C805")
        assert info.path_digits == (8, 0, 5)
        assert info.is_stepchild

    def test_invalid_empty(self):
        with pytest.raises(ValueError):
            validate_extended_code("")


# ── describe_family_code – base paths ────────────────────────────────────────

class TestDescribeBasePaths:
    def test_root_person_cikky(self):
        assert describe_family_code("C0") == "Cikky"

    def test_spouse_of_cikky(self):
        assert describe_family_code("C00") == "Cikky házastársa"

    def test_8th_child(self):
        assert describe_family_code("C8") == "Cikky 8. gyermeke"

    def test_8th_child_spouse(self):
        assert describe_family_code("C80") == "Cikky 8. gyermekének házastársa"

    def test_8th_child_1st_grandchild(self):
        assert describe_family_code("C81") == "Cikky 8. gyermekének 1. gyermeke"

    def test_grandchild_spouse(self):
        assert describe_family_code("C810") == (
            "Cikky 8. gyermekének 1. gyermekének házastársa"
        )

    def test_great_grandchild(self):
        assert describe_family_code("C811") == (
            "Cikky 8. gyermekének 1. gyermekének 1. gyermeke"
        )

    def test_own_child_not_stepchild(self):
        assert describe_family_code("C221") == (
            "Cikky 2. gyermekének 2. gyermekének 1. gyermeke"
        )

    def test_stepchild_annotation(self):
        assert describe_family_code("C2201") == (
            "Cikky 2. gyermekének 2. gyermekének házastársának "
            "1. gyermeke (korábbi kapcsolatból)"
        )


# ── describe_family_code – ancestor (F) suffix ───────────────────────────────

class TestDescribeAncestors:
    def test_father(self):
        assert describe_family_code("C0F1") == "Cikky apja"

    def test_mother(self):
        assert describe_family_code("C0F2") == "Cikky anyja"

    def test_paternal_grandfather(self):
        assert describe_family_code("C0F11") == "Cikky apai nagyapja"

    def test_paternal_grandmother(self):
        assert describe_family_code("C0F12") == "Cikky apai nagyanyja"

    def test_maternal_grandfather(self):
        assert describe_family_code("C0F21") == "Cikky anyai nagyapja"

    def test_maternal_grandmother(self):
        assert describe_family_code("C0F22") == "Cikky anyai nagyanyja"

    def test_spouse_father(self):
        assert describe_family_code("C00F1") == "Cikky házastársának apja"

    def test_spouse_mother(self):
        assert describe_family_code("C00F2") == "Cikky házastársának anyja"

    def test_child_spouse_father(self):
        assert describe_family_code("C80F1") == (
            "Cikky 8. gyermekének házastársának apja"
        )

    def test_grandchild_spouse_mother(self):
        assert describe_family_code("C810F2") == (
            "Cikky 8. gyermekének 1. gyermekének házastársának anyja"
        )

    def test_deep_ancestor_depth3(self):
        # C0F111 = Cikky apai dédapja (3 generáció felfelé apai ágon)
        assert describe_family_code("C0F111") == "Cikky apai dédapja"

    def test_deep_ancestor_depth4(self):
        # C0F1111 = Cikky apai dédapjának apja (4. szint, rekurzív fallback)
        result = describe_family_code("C0F1111")
        assert "apai dédapjának" in result
        assert result.endswith("apja")


# ── describe_family_code – sibling (T) suffix ────────────────────────────────

class TestDescribeSiblings:
    def test_first_sibling(self):
        assert describe_family_code("C00T1") == "Cikky házastársának 1. testvére"

    def test_second_sibling(self):
        assert describe_family_code("C00T2") == "Cikky házastársának 2. testvére"

    def test_sibling_spouse(self):
        assert describe_family_code("C00T10") == (
            "Cikky házastársának 1. testvérének házastársa"
        )

    def test_sibling_child(self):
        assert describe_family_code("C00T11") == (
            "Cikky házastársának 1. testvérének 1. gyermeke"
        )


# ── describe_family_code – friend (B) suffix ─────────────────────────────────

class TestDescribeFriends:
    def test_single_friend(self):
        assert describe_family_code("C81B") == (
            "Cikky 8. gyermekének 1. gyermeke barátja/ismerőse"
        )

    def test_multi_comma(self):
        result = describe_family_code("C81B,C82B")
        assert "C81B" not in result          # codes not leaking into description
        assert "közös barátja/ismerőse" in result
        assert "8. gyermekének 1. gyermeke" in result
        assert "8. gyermekének 2. gyermeke" in result

    def test_multi_space(self):
        result = describe_family_code("C81B C82B")
        assert "közös barátja/ismerőse" in result

    def test_multi_comma_space(self):
        result = describe_family_code("C81B, C82B")
        assert "közös barátja/ismerőse" in result


# ── Range notation ────────────────────────────────────────────────────────────

class TestRangeNotation:
    def test_expand_range(self):
        assert expand_range_code("C[1-9]B") == [
            "C1B", "C2B", "C3B", "C4B", "C5B", "C6B", "C7B", "C8B", "C9B"
        ]

    def test_expand_short_range(self):
        assert expand_range_code("C[81-82]B") == ["C81B", "C82B"]

    def test_expand_backwards_returns_empty(self):
        assert expand_range_code("C[9-1]B") == []

    def test_describe_range(self):
        result = describe_family_code("C[1-9]B")
        assert result == "C1–C9 közös barátja/ismerőse"

    def test_describe_two_element_range(self):
        result = describe_family_code("C[81-82]B")
        assert result == "C81–C82 közös barátja/ismerőse"

    def test_describe_single_element_range(self):
        result = describe_family_code("C[3-3]B")
        assert "barátja/ismerőse" in result
        assert "közös" not in result


# ── validate_extended_code ────────────────────────────────────────────────────

class TestValidateExtendedCode:
    def test_valid_simple_codes(self):
        for code in ["C0", "C00", "C8", "C80", "C81", "C810", "C811", "C2201"]:
            assert validate_extended_code(code) == code

    def test_valid_ancestor_codes(self):
        for code in ["C0F1", "C0F2", "C0F11", "C0F12", "C0F21", "C00F1"]:
            assert validate_extended_code(code) == code

    def test_valid_sibling_codes(self):
        for code in ["C00T1", "C00T2", "C00T10", "C00T11"]:
            assert validate_extended_code(code) == code

    def test_valid_friend_codes(self):
        assert validate_extended_code("C81B") == "C81B"

    def test_valid_range(self):
        assert validate_extended_code("C[1-9]B") == "C[1-9]B"

    def test_valid_multi_code(self):
        result = validate_extended_code("C81B,C82B")
        assert "C81B" in result
        assert "C82B" in result

    def test_invalid_b_with_number(self):
        with pytest.raises(ValueError):
            validate_extended_code("C81B1")

    def test_invalid_empty(self):
        with pytest.raises(ValueError):
            validate_extended_code("")

    def test_lowercase_normalised(self):
        assert validate_extended_code("c0f1") == "C0F1"


# ── Default root persons (C, G, J, I) ─────────────────────────────────────────

class TestDefaultRootPersons:
    def test_all_four_roots_registered(self):
        assert set("CGJI").issubset(DEFAULT_ROOT_NAMES)

    def test_cikky(self):
        assert describe_family_code("C0") == "Cikky"

    def test_gabor(self):
        assert describe_family_code("G0") == "Gábor"

    def test_jerne(self):
        assert describe_family_code("J0") == "Jerne"

    def test_ildi(self):
        assert describe_family_code("I0") == "Ildi"

    def test_root_descendant(self):
        assert describe_family_code("G1") == "Gábor 1. gyermeke"

    def test_root_ancestor(self):
        assert describe_family_code("I0F1") == "Ildi apja"

    def test_root_spouse(self):
        assert describe_family_code("J00") == "Jerne házastársa"


# ── Custom root names ─────────────────────────────────────────────────────────

class TestCustomRootNames:
    def test_override_root_name(self):
        names = {"C": "Cikky", "G": "György"}
        assert describe_family_code("G0", root_names=names) == "György"
        assert describe_family_code("G0F1", root_names=names) == "György apja"

    def test_unknown_root_shows_letter(self):
        result = describe_family_code("X0")
        assert result == "X"

    def test_i_and_j_roots(self):
        names = {"I": "Irén", "J": "János"}
        assert describe_family_code("I0", root_names=names) == "Irén"
        assert describe_family_code("J0F2", root_names=names) == "János anyja"


# ── Multiple spouses (H) suffix ───────────────────────────────────────────────

class TestMultiSpouseSuffix:
    def test_parse_first_spouse(self):
        info = parse_extended_code("G1H1")
        assert info.suffix_type == "spouse"
        assert info.spouse_path == (1,)
        assert info.path_digits == (1,)

    def test_parse_second_spouse(self):
        info = parse_extended_code("G1H2")
        assert info.suffix_type == "spouse"
        assert info.spouse_path == (2,)

    def test_parse_spouse_brought_child(self):
        info = parse_extended_code("G1H21")
        assert info.suffix_type == "spouse"
        assert info.spouse_path == (2, 1)

    def test_describe_first_spouse(self):
        names = {"G": "Gábor"}
        assert describe_family_code("G1H1", names) == "Gábor 1. gyermekének 1. házastársa"

    def test_describe_second_spouse(self):
        names = {"G": "Gábor"}
        assert describe_family_code("G1H2", names) == "Gábor 1. gyermekének 2. házastársa"

    def test_describe_spouse_brought_child(self):
        names = {"G": "Gábor"}
        assert describe_family_code("G1H21", names) == (
            "Gábor 1. gyermekének 2. házastársának 1. gyermeke (korábbi kapcsolatból)"
        )

    def test_root_spouse_h(self):
        names = {"G": "Gábor"}
        assert describe_family_code("G0H2", names) == "Gábor 2. házastársa"

    def test_zero_spouse_still_valid(self):
        # Backward compatibility: the single-spouse 0 form keeps working.
        assert describe_family_code("G10") == "Gábor 1. gyermekének házastársa"

    def test_validate_h_codes(self):
        for code in ["G1H1", "G1H2", "G1H21", "G0H3"]:
            assert validate_extended_code(code) == code

    def test_invalid_h_zero_spouse_index(self):
        with pytest.raises(ValueError):
            parse_extended_code("G1H0")


# ── Parent-disambiguation braces {n} ──────────────────────────────────────────

class TestParentBraces:
    def test_parse_brace(self):
        info = parse_extended_code("G1{2}3")
        assert info.path_digits == (1, 3)
        assert info.path_spouse_nums == (None, 2)

    def test_parse_brace_first_spouse(self):
        info = parse_extended_code("G1{1}1")
        assert info.path_digits == (1, 1)
        assert info.path_spouse_nums == (None, 1)

    def test_describe_brace_second_spouse(self):
        names = {"G": "Gábor"}
        assert describe_family_code("G1{2}3", names) == (
            "Gábor 1. gyermekének 3. gyermeke a 2. házastárstól"
        )

    def test_describe_brace_first_spouse(self):
        names = {"G": "Gábor"}
        # "első" starts with a vowel → "az 1." per the spec.
        assert describe_family_code("G1{1}1", names) == (
            "Gábor 1. gyermekének 1. gyermeke az 1. házastárstól"
        )

    def test_brace_canonical_form(self):
        assert validate_extended_code("G1{2}3") == "G1{2}3"

    def test_brace_must_precede_child(self):
        # A brace cannot dangle at the end.
        with pytest.raises(ValueError):
            parse_extended_code("G1{2}")

    def test_brace_not_before_spouse_zero(self):
        with pytest.raises(ValueError):
            parse_extended_code("G1{2}0")


# ── External family identifiers (#root#path) ──────────────────────────────────

class TestExternalCodes:
    def test_parse_root(self):
        info = parse_extended_code("#Zoli#")
        assert info.is_external
        assert info.external_root == "Zoli"
        assert info.path_digits == ()

    def test_parse_spouse(self):
        info = parse_extended_code("#Zoli#0")
        assert info.is_external
        assert info.path_digits == (0,)

    def test_parse_child(self):
        info = parse_extended_code("#Zoli#3")
        assert info.path_digits == (3,)

    def test_parse_grandchild(self):
        info = parse_extended_code("#Zoli#31")
        assert info.path_digits == (3, 1)

    def test_root_case_preserved(self):
        info = parse_extended_code("#KovacsZoli#")
        assert info.external_root == "KovacsZoli"
        assert info.raw == "#KovacsZoli#"

    def test_numeric_root(self):
        info = parse_extended_code("#47#1")
        assert info.external_root == "47"
        assert info.path_digits == (1,)

    def test_describe_root(self):
        assert describe_family_code("#Zoli#") == "Zoli"

    def test_describe_spouse(self):
        assert describe_family_code("#Zoli#0") == "Zoli házastársa"

    def test_describe_child(self):
        assert describe_family_code("#Zoli#3") == "Zoli 3. gyermeke"

    def test_describe_grandchild(self):
        assert describe_family_code("#Zoli#31") == "Zoli 3. gyermekének 1. gyermeke"

    def test_describe_short_root(self):
        assert describe_family_code("#Z#31") == "Z 3. gyermekének 1. gyermeke"

    def test_describe_external_ancestor(self):
        # The root person is implicit, so a suffix attaches directly.
        assert describe_family_code("#Zoli#F1") == "Zoli apja"

    def test_validate_external_field(self):
        assert validate_external_family_code("#Zoli#") == "#Zoli#"
        assert validate_external_family_code("#Zoli#31") == "#Zoli#31"

    def test_validate_external_empty(self):
        assert validate_external_family_code("") is None
        assert validate_external_family_code(None) is None

    def test_validate_external_rejects_internal(self):
        with pytest.raises(ValueError):
            validate_external_family_code("C81B")

    def test_validate_external_rejects_empty_root(self):
        with pytest.raises(ValueError):
            validate_external_family_code("##")

    def test_main_field_rejects_external(self):
        with pytest.raises(ValueError):
            validate_extended_code("#Zoli#")
