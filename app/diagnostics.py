"""Runtime diagnostics for the face-local application.

Run from the command line::

    python -m app.diagnostics

Or call :func:`check_tflite_backend` programmatically (e.g. from the settings
dialog) to surface actionable errors before the pipeline starts.
"""

from __future__ import annotations

import logging
import platform
import sys
from pathlib import Path
from typing import NamedTuple

log = logging.getLogger(__name__)


class BackendResult(NamedTuple):
    name: str
    available: bool
    detail: str


class TFLiteDiagnostics(NamedTuple):
    python_version: str
    platform: str
    machine: str
    backends: list[BackendResult]
    active_backend: str | None
    model_exists: bool
    model_path: str
    interpreter_ok: bool
    interpreter_error: str | None

    @property
    def ok(self) -> bool:
        return self.active_backend is not None


def check_tflite_backend(model_path: str | None = None) -> TFLiteDiagnostics:
    """Check which TFLite backends are available and whether the model loads."""
    from app.paths import resource_path

    resolved_model = Path(model_path) if model_path else resource_path("models/mobilefacenet.tflite")

    backends: list[BackendResult] = []

    # --- ai_edge_litert ---
    try:
        from ai_edge_litert.interpreter import Interpreter as _I  # type: ignore[import]
        import ai_edge_litert as _ael
        backends.append(BackendResult("ai_edge_litert", True, getattr(_ael, "__version__", "unknown")))
        _active_cls = _I
        active_backend = "ai_edge_litert"
    except ImportError as e:
        backends.append(BackendResult("ai_edge_litert", False, str(e)))
        _active_cls = None
        active_backend = None

    # --- tflite_runtime ---
    try:
        import tflite_runtime.interpreter as _tflite  # type: ignore[import]
        import tflite_runtime as _tr
        backends.append(BackendResult("tflite_runtime", True, getattr(_tr, "__version__", "unknown")))
        if _active_cls is None:
            _active_cls = _tflite.Interpreter
            active_backend = "tflite_runtime"
    except ImportError as e:
        backends.append(BackendResult("tflite_runtime", False, str(e)))

    # --- tensorflow ---
    try:
        import tensorflow as _tf  # type: ignore[import]
        backends.append(BackendResult("tensorflow", True, getattr(_tf, "__version__", "unknown")))
        if _active_cls is None:
            _active_cls = _tf.lite.Interpreter
            active_backend = "tensorflow"
    except ImportError as e:
        backends.append(BackendResult("tensorflow", False, str(e)))

    # --- Model file and interpreter test ---
    model_exists = resolved_model.exists()
    interpreter_ok = False
    interpreter_error: str | None = None

    if model_exists and _active_cls is not None:
        try:
            interp = _active_cls(model_path=str(resolved_model))
            interp.allocate_tensors()
            interpreter_ok = True
        except Exception as exc:
            interpreter_error = str(exc)

    return TFLiteDiagnostics(
        python_version=sys.version,
        platform=sys.platform,
        machine=platform.machine(),
        backends=backends,
        active_backend=active_backend,
        model_exists=model_exists,
        model_path=str(resolved_model),
        interpreter_ok=interpreter_ok,
        interpreter_error=interpreter_error,
    )


def print_diagnostics(diag: TFLiteDiagnostics | None = None) -> None:
    """Print a formatted diagnostics report to stdout."""
    if diag is None:
        diag = check_tflite_backend()

    print("=" * 60)
    print("Face-Local — TFLite diagnostics")
    print("=" * 60)
    print(f"Python:   {diag.python_version}")
    print(f"Platform: {diag.platform} / {diag.machine}")
    print()
    print("TFLite backends:")
    for b in diag.backends:
        status = "OK " if b.available else "---"
        marker = " <-- active" if b.name == diag.active_backend else ""
        print(f"  [{status}] {b.name}: {b.detail}{marker}")
    print()
    print(f"Active backend: {diag.active_backend or 'NONE (no backend found!)'}")
    print(f"Model file:     {diag.model_path}")
    print(f"Model exists:   {'yes' if diag.model_exists else 'NO'}")
    if diag.model_exists and diag.active_backend:
        print(f"Interpreter:    {'OK' if diag.interpreter_ok else 'FAILED: ' + (diag.interpreter_error or '')}")
    print()

    if not diag.ok:
        print("ACTION REQUIRED: No TFLite backend installed.")
        print("  Run:  pip install ai-edge-litert")
        print("  Supports Python 3.9-3.12 on Linux x64/ARM64, macOS, Windows x64.")
        print("  Alternative: pip install tensorflow")
    elif not diag.model_exists:
        print("ACTION REQUIRED: TFLite model file not found.")
        print(f"  Expected at: {diag.model_path}")
        print("  Download mobilefacenet.tflite or run build_and_run.ps1/sh")
    elif not diag.interpreter_ok:
        print(f"ACTION REQUIRED: Interpreter failed: {diag.interpreter_error}")
    else:
        print("OK — TFLite backend and model are ready.")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    diag = check_tflite_backend()
    print_diagnostics(diag)
    sys.exit(0 if diag.ok and diag.interpreter_ok else 1)
