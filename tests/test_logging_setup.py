"""Tests for structured logging setup."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import QObject, Signal

from app.logging_setup import DATE_FORMAT, LOG_FORMAT, QLogHandler, setup_logging


@pytest.fixture(autouse=True)
def _reset_root_logger() -> None:
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    root.setLevel(logging.WARNING)
    yield
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()


class TestSetupLogging:
    def test_adds_stderr_handler(self) -> None:
        setup_logging(level=logging.DEBUG)
        root = logging.getLogger()
        assert root.level == logging.DEBUG
        assert any(isinstance(h, logging.StreamHandler) for h in root.handlers)

    def test_optional_file_handler(self, tmp_path: Path) -> None:
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)
            handler.close()
        log_file = tmp_path / "logs" / "app.log"
        setup_logging(log_file=str(log_file))
        logging.getLogger("test.logger").info("hello file")
        for handler in logging.getLogger().handlers:
            handler.flush()
            handler.close()
        assert log_file.exists()
        assert "hello file" in log_file.read_text(encoding="utf-8")

    def test_second_call_is_idempotent(self) -> None:
        setup_logging()
        count = len(logging.getLogger().handlers)
        setup_logging()
        assert len(logging.getLogger().handlers) == count

    def test_quiets_noisy_third_party_loggers(self) -> None:
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        setup_logging()
        assert logging.getLogger("PIL").getEffectiveLevel() == logging.WARNING
        assert logging.getLogger("urllib3").getEffectiveLevel() == logging.WARNING


class _SignalHost(QObject):
    log_emitted = Signal(str, int)


class TestQLogHandler:
    def test_emits_formatted_message(self, qtbot) -> None:
        host = _SignalHost()
        handler = QLogHandler(host.log_emitted)
        handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(handler)

        with qtbot.waitSignal(host.log_emitted, timeout=1000) as blocker:
            logging.getLogger("ui.test").warning("panel updated")

        message, levelno = blocker.args
        assert "panel updated" in message
        assert levelno == logging.WARNING
        assert LOG_FORMAT
        assert DATE_FORMAT

    def test_format_matches_setup_logging(self) -> None:
        handler = QLogHandler(MagicMock())
        record = logging.LogRecord(
            name="app.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )
        formatted = handler.format(record)
        assert "hello" in formatted
        assert "INFO" in formatted
