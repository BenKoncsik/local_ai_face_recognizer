"""Hardware-accelerator resolution and a tiny runtime status registry.

This module has two jobs:

1. **Provider resolution** — build the ordered execution-provider list for
   ``onnxruntime`` sessions in a *platform-safe* way.  On NVIDIA machines CUDA
   keeps priority, so the Windows / Linux behaviour is unchanged; on Apple
   Silicon the CoreML Execution Provider is inserted ahead of CPU so supported
   ops run on the Neural Engine / GPU.  ``CPUExecutionProvider`` is always the
   final fallback, so a missing or unusable accelerator can never break a
   session — at worst everything runs on CPU exactly as before.

2. **Status registry** — each model backend reports which accelerator it
   actually ended up on (the two components are ``detection`` and
   ``embedding``).  The Task Manager's *Performance* tab reads this to show a
   truthful accelerator row.

   NOTE: macOS exposes **no public per-process ANE utilisation API** (Activity
   Monitor has no ANE meter either), so this reports *which* accelerator is
   active — not a live percentage.  A live ANE meter would require
   ``sudo powermetrics``, which is not viable for a normal GUI app.
"""

from __future__ import annotations

import logging
import platform
import threading
from typing import Dict, List

log = logging.getLogger(__name__)

# Accelerator kinds.
ACC_CPU = "cpu"
ACC_CUDA = "cuda"
ACC_COREML = "coreml"   # Apple Neural Engine / GPU, reached via CoreML.
ACC_CORAL = "coral"     # Google Edge TPU.

_HUMAN = {
    ACC_CPU: "CPU",
    ACC_CUDA: "CUDA GPU",
    ACC_COREML: "Neural Engine (CoreML)",
    ACC_CORAL: "Coral Edge TPU",
}

# Strongest-first ordering used when summarising several components.
_STRENGTH = (ACC_CUDA, ACC_COREML, ACC_CORAL, ACC_CPU)

_lock = threading.Lock()
_active: Dict[str, Dict[str, str]] = {}   # component -> {"acc": ..., "detail": ...}


# ----------------------------------------------------------------------
# Platform probes
# ----------------------------------------------------------------------

def is_macos() -> bool:
    return platform.system() == "Darwin"


def is_apple_silicon() -> bool:
    return is_macos() and platform.machine() == "arm64"


def _ort_available_providers() -> List[str]:
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception:  # noqa: BLE001 — onnxruntime is an optional dependency
        return []


def coreml_ep_available() -> bool:
    """True when onnxruntime's CoreML EP can be used on this (macOS) machine."""
    return is_macos() and "CoreMLExecutionProvider" in _ort_available_providers()


# ----------------------------------------------------------------------
# Provider resolution
# ----------------------------------------------------------------------

def onnx_providers(allow_cuda: bool = True) -> List[str]:
    """Ordered onnxruntime provider list with CPU always last.

    CUDA keeps priority wherever it is present (so the established Windows /
    Linux GPU path is untouched); CoreML is only ever added on macOS.  Plain
    provider-name strings are returned so callers can hand them straight to
    insightface / ``get_model`` without change.
    """
    avail = _ort_available_providers()
    provs: List[str] = []
    if allow_cuda and "CUDAExecutionProvider" in avail:
        provs.append("CUDAExecutionProvider")
    if is_macos() and "CoreMLExecutionProvider" in avail:
        provs.append("CoreMLExecutionProvider")
    provs.append("CPUExecutionProvider")
    return provs


def classify_providers(active_providers: List[str]) -> str:
    """Map an onnxruntime *active* provider list to an accelerator constant.

    onnxruntime lists ``CPUExecutionProvider`` last as the fallback; when an
    accelerator partitioned some nodes its provider appears before CPU, so the
    presence of CUDA/CoreML anywhere in the list means it is doing real work.
    """
    if not active_providers:
        return ACC_CPU
    if "CUDAExecutionProvider" in active_providers:
        return ACC_CUDA
    if "CoreMLExecutionProvider" in active_providers:
        return ACC_COREML
    return ACC_CPU


# ----------------------------------------------------------------------
# Status registry
# ----------------------------------------------------------------------

def report(component: str, accelerator: str, detail: str = "") -> None:
    """Record that *component* is running on *accelerator* (a status, not a metric)."""
    with _lock:
        _active[component] = {"acc": accelerator, "detail": detail}
    log.info(
        "Accelerator: %s → %s%s",
        component, accelerator, f" ({detail})" if detail else "",
    )


def report_onnx(component: str, active_providers: List[str], detail: str = "") -> None:
    """Convenience wrapper: classify an onnxruntime provider list, then report."""
    report(component, classify_providers(active_providers), detail)


def active() -> Dict[str, Dict[str, str]]:
    """Snapshot of every component's current accelerator."""
    with _lock:
        return {k: dict(v) for k, v in _active.items()}


def reset() -> None:
    """Forget every reported component (used by tests and on a hard reload)."""
    with _lock:
        _active.clear()


def human(acc: str) -> str:
    return _HUMAN.get(acc, acc)


def summary() -> str:
    """One-line accelerator label for the Performance tab.

    Shows the strongest accelerator any component reported, or ``—`` before any
    model has been built this session.
    """
    snapshot = active()
    if not snapshot:
        return "—"
    accs = {v["acc"] for v in snapshot.values()}
    for kind in _STRENGTH:
        if kind in accs:
            return human(kind)
    return human(ACC_CPU)


def detail_lines() -> List[str]:
    """Per-component accelerator descriptions for a tooltip."""
    lines: List[str] = []
    for component, info in sorted(active().items()):
        label = human(info["acc"])
        extra = f" — {info['detail']}" if info.get("detail") else ""
        lines.append(f"{component}: {label}{extra}")
    return lines
