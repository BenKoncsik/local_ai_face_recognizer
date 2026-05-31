"""Detector factory — Coral auto-detection and CPU fallback."""

from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
import sys

from app.config import DetectionConfig
from app.detectors.base import FaceDetector

log = logging.getLogger(__name__)


def _find_edgetpu_lib() -> str:
    candidates = (
        ["/usr/local/lib/libedgetpu.1.dylib", "/opt/homebrew/lib/libedgetpu.1.dylib"]
        if platform.system() == "Darwin"
        else ["/usr/lib/libedgetpu.so.1", "/usr/local/lib/libedgetpu.so.1"]
    )
    for p in candidates:
        if os.path.exists(p):
            return p
    return "libedgetpu.1.dylib" if platform.system() == "Darwin" else "libedgetpu.so.1"


_EDGETPU_LIB = _find_edgetpu_lib()

_CORAL_USB_PATTERNS = (
    r"Global Unichip",
    r"\bCoral\b",
    r"\b1a6e\b",
    r"\b18d1\b",
)


_PROBE_SCRIPT = r"""
import sys, os
LIB = os.environ["FL_EDGETPU_LIB"]
MODEL = os.environ.get("FL_CORAL_MODEL") or ""
try:
    from ai_edge_litert.interpreter import Interpreter, load_delegate
except ImportError:
    try:
        from pycoral.utils.edgetpu import list_edge_tpus
        sys.exit(0 if list_edge_tpus() else 10)
    except Exception:
        sys.exit(11)

try:
    delegate = load_delegate(LIB)
except Exception as exc:
    sys.stderr.write(f"delegate-load-failed: {exc}\n")
    sys.exit(20)

if not MODEL:
    sys.exit(0)

try:
    import numpy as np
    interp = Interpreter(model_path=MODEL, experimental_delegates=[delegate])
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    dummy = np.zeros(inp["shape"], dtype=inp["dtype"])
    interp.set_tensor(inp["index"], dummy)
    interp.invoke()
    sys.exit(0)
except Exception as exc:
    msg = str(exc)
    sys.stderr.write(f"inference-failed: {msg}\n")
    if "EdgeTpuDelegateForCustomOp" in msg or "custom op" in msg.lower():
        sys.exit(30)
    sys.exit(31)
"""


def _macos_coral_usb_visible() -> bool | None:
    """Best-effort check whether macOS currently enumerates a Coral USB device."""
    if platform.system() != "Darwin":
        return None

    commands = (
        ["ioreg", "-p", "IOUSB", "-w", "0", "-l"],
        ["system_profiler", "SPUSBDataType"],
    )

    for cmd in commands:
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=8,
            )
        except Exception:  # noqa: BLE001
            continue

        haystack = "\n".join(part for part in (proc.stdout, proc.stderr) if part)
        if not haystack.strip():
            continue

        if any(re.search(pattern, haystack, flags=re.IGNORECASE) for pattern in _CORAL_USB_PATTERNS):
            return True

        # If the command succeeded and produced USB inventory, but none of the
        # Coral identifiers are present, we can treat it as a negative result.
        if proc.returncode == 0:
            return False

    return None


def probe_coral(model_path: str | None = None) -> bool:
    """Probe the EdgeTPU delegate in an isolated subprocess.

    ai-edge-litert's XNNPACK lazy delegate can corrupt the process heap during
    ``allocate_tensors()`` on macOS ARM64, triggering SIGTRAP in libmalloc
    which cannot be caught by Python. Running the probe in a child process
    means a crash there leaves the main app alive and we simply fall back to
    CPU detection.
    """
    usb_visible = _macos_coral_usb_visible()
    if usb_visible is False:
        log.warning(
            "Coral probe: macOS does not currently enumerate a Coral USB device "
            "(not present in IOUSB/system_profiler). Check the cable, hub, and "
            "USB permissions, then re-plug the accelerator."
        )
        return False

    env = dict(os.environ)
    env["FL_EDGETPU_LIB"] = _EDGETPU_LIB
    if model_path:
        env["FL_CORAL_MODEL"] = model_path

    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE_SCRIPT],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        log.warning("Coral probe: subprocess timed out — assuming TPU unusable")
        return False
    except Exception as exc:
        log.warning("Coral probe: subprocess failed to launch: %s", exc)
        return False

    if proc.returncode == 0:
        log.info("Coral probe: OK (subprocess) — device is usable")
        return True

    # Non-zero → subprocess exited or crashed. A C-level abort returns
    # negative/-SIGTRAP on POSIX; log whatever it told us.
    stderr = (proc.stderr or "").strip()
    if proc.returncode < 0:
        log.warning(
            "Coral probe: subprocess crashed (signal %d) — TPU unusable. "
            "This usually means libedgetpu loaded but the device is not responding. "
            "Unplug & re-plug the Coral USB accelerator.",
            -proc.returncode,
        )
    elif proc.returncode == 30:
        log.warning(
            "Coral probe: library loaded but device not responding "
            "(EdgeTpuDelegateForCustomOp). Re-plug the USB accelerator. %s",
            stderr,
        )
    elif proc.returncode == 20:
        log.warning("Coral probe: delegate load failed. %s", stderr)
    else:
        log.warning("Coral probe: failed (exit %d). %s", proc.returncode, stderr)
    return False


def create_detector(config: DetectionConfig) -> FaceDetector:
    """Create the best available face detector."""
    if config.coral_model_path:
        if probe_coral(config.coral_model_path):
            try:
                from app.detectors.coral_detector import CoralDetector

                detector = CoralDetector(model_path=config.coral_model_path)
                log.info("Using Coral Edge TPU detector (backend: %s)", detector.backend_name)
                return detector
            except FileNotFoundError as exc:
                log.warning("Coral model file missing: %s — falling back to CPU", exc)
            except ImportError as exc:
                log.warning("Coral backend unavailable: %s — falling back to CPU", exc)
            except Exception as exc:  # noqa: BLE001
                log.warning("Coral init failed: %s — falling back to CPU", exc)
        else:
            log.warning(
                "coral_model_path is set but EdgeTPU probe failed. "
                "See the previous warning for the detected cause; falling back to CPU."
            )
    else:
        log.info(
            "detection.coral_model_path not set — using CPU detector. "
            "Set this in config.yaml to enable Coral acceleration."
        )

    # Prefer the landmark-capable YuNet detector (needed for "aligned" crops).
    if config.use_yunet:
        try:
            from app.detectors.yunet_detector import YuNetDetector

            detector = YuNetDetector(model_path=config.yunet_model_path)
            log.info("Using CPU detector (backend: %s)", detector.backend_name)
            return detector
        except FileNotFoundError as exc:
            log.warning(
                "YuNet model unavailable (%s) — falling back to Caffe/Haar CPU "
                "detector. 'aligned' crop mode will degrade to 'square'.", exc
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("YuNet init failed: %s — falling back to Caffe/Haar CPU", exc)

    from app.detectors.cpu_detector import CpuDetector

    detector = CpuDetector(model_path=config.cpu_model_path)
    log.info("Using CPU detector (backend: %s)", detector.backend_name)
    return detector
