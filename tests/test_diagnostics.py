"""Tests for runtime TFLite diagnostics."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.diagnostics import TFLiteDiagnostics, check_tflite_backend, print_diagnostics


class TestTFLiteDiagnostics:
    def test_ok_true_when_active_backend_present(self) -> None:
        diag = TFLiteDiagnostics(
            python_version="3.11",
            platform="darwin",
            machine="arm64",
            backends=[],
            active_backend="ai_edge_litert",
            model_exists=True,
            model_path="/tmp/model.tflite",
            interpreter_ok=True,
            interpreter_error=None,
        )
        assert diag.ok is True

    def test_ok_false_without_backend(self) -> None:
        diag = TFLiteDiagnostics(
            python_version="3.11",
            platform="darwin",
            machine="arm64",
            backends=[],
            active_backend=None,
            model_exists=False,
            model_path="/tmp/model.tflite",
            interpreter_ok=False,
            interpreter_error=None,
        )
        assert diag.ok is False


class TestCheckTfliteBackend:
    def test_reports_missing_model(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        missing = tmp_path / "missing.tflite"
        diag = check_tflite_backend(str(missing))
        assert diag.model_path == str(missing)
        assert diag.model_exists is False
        assert len(diag.backends) == 3

    def test_uses_first_available_backend_for_interpreter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = tmp_path / "model.tflite"
        model.write_bytes(b"fake")

        interp = MagicMock()
        fake_cls = MagicMock(return_value=interp)

        import builtins

        real_import = builtins.__import__

        def _import(name, *args, **kwargs):
            if name == "ai_edge_litert.interpreter":
                mod = MagicMock()
                mod.Interpreter = fake_cls
                return mod
            if name == "ai_edge_litert":
                pkg = MagicMock()
                pkg.__version__ = "1.2.3"
                return pkg
            if name == "tflite_runtime.interpreter":
                raise ImportError("skip runtime")
            if name == "tflite_runtime":
                raise ImportError("skip runtime")
            if name == "tensorflow":
                raise ImportError("skip tf")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _import)
        diag = check_tflite_backend(str(model))
        assert diag.active_backend == "ai_edge_litert"
        assert diag.interpreter_ok is True
        fake_cls.assert_called_once_with(model_path=str(model))
        interp.allocate_tensors.assert_called_once()


class TestPrintDiagnostics:
    def test_prints_summary(self, capsys: pytest.CaptureFixture[str]) -> None:
        diag = TFLiteDiagnostics(
            python_version="3.11",
            platform="darwin",
            machine="arm64",
            backends=[],
            active_backend=None,
            model_exists=False,
            model_path="/tmp/model.tflite",
            interpreter_ok=False,
            interpreter_error=None,
        )
        print_diagnostics(diag)
        out = capsys.readouterr().out
        assert "TFLite diagnostics" in out
        assert "ACTION REQUIRED" in out
