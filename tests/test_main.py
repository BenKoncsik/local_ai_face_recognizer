"""Tests for the application entry point (argument parsing, no GUI launch)."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from app.main import parse_args


def test_parse_args_defaults():
    with patch.object(sys, "argv", ["face-local"]):
        args = parse_args()
    assert args.config is None
    assert args.debug is False
    assert args.db is None


def test_parse_args_debug_and_db():
    with patch.object(
        sys,
        "argv",
        ["face-local", "--debug", "--db", "/tmp/x.db", "--config", "cfg.yaml"],
    ):
        args = parse_args()
    assert args.debug is True
    assert args.db == "/tmp/x.db"
    assert args.config == "cfg.yaml"


def test_main_module_importable():
    import app.main as main_mod

    assert callable(main_mod.main)
    assert callable(main_mod.parse_args)
