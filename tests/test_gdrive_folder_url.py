"""Tests for the Drive folder URL/ID parser."""

from __future__ import annotations

import pytest

from app.gdrive.folder_url import parse_folder_input


class TestParseFolderInput:
    @pytest.mark.parametrize("raw,expected", [
        # Standard "shared with you" URL
        (
            "https://drive.google.com/drive/folders/1aBcDeFgHiJkLmNoPqRsTuVwXyZ?usp=sharing",
            "1aBcDeFgHiJkLmNoPqRsTuVwXyZ",
        ),
        # URL with /u/0/ user prefix
        (
            "https://drive.google.com/drive/u/0/folders/abcdef1234567890",
            "abcdef1234567890",
        ),
        # Legacy ?id= URL
        (
            "https://drive.google.com/open?id=ZyXwVuTsRqPoNmLkJiHgFeDcBa987654",
            "ZyXwVuTsRqPoNmLkJiHgFeDcBa987654",
        ),
        # Bare ID
        (
            "1a2b3c4d5e6f7g8h9i0j",
            "1a2b3c4d5e6f7g8h9i0j",
        ),
        # Bare ID with surrounding whitespace
        (
            "   abcdef1234567890   ",
            "abcdef1234567890",
        ),
    ])
    def test_recognised_inputs(self, raw: str, expected: str) -> None:
        assert parse_folder_input(raw) == expected

    @pytest.mark.parametrize("raw", [
        "",
        "  ",
        "not-a-url",                            # too short for a bare id
        "ftp://example.com/folder",
        "https://example.com/something",        # no folder id
        "https://drive.google.com/drive/",      # no id
    ])
    def test_invalid_inputs_return_none(self, raw: str) -> None:
        assert parse_folder_input(raw) is None
