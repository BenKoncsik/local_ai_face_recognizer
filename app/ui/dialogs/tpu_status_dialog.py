"""TPU status & repair dialog."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from typing import Dict, Any, List

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QProgressBar,
)

from app.ui.i18n import t

def _find_edgetpu_lib() -> str:
    import os
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


# ── Probe ─────────────────────────────────────────────────────────────────────

_PROBE_SCRIPT = r"""
import json, os, sys
out = {
    "ai_edge_litert": False, "ai_edge_litert_ver": None,
    "pycoral": False, "pycoral_ver": None,
    "libedgetpu": False, "delegate_ok": False, "inference_ok": False,
    "devices_pycoral": [], "error": None,
}
LIB = os.environ["FL_EDGETPU_LIB"]
MODEL = os.environ.get("FL_CORAL_MODEL") or ""

try:
    import ai_edge_litert as ael
    out["ai_edge_litert"] = True
    out["ai_edge_litert_ver"] = getattr(ael, "__version__", "installed")
except ImportError:
    pass

try:
    import pycoral
    out["pycoral"] = True
    out["pycoral_ver"] = getattr(pycoral, "__version__", "installed")
except ImportError:
    pass

delegate = None
if out["ai_edge_litert"]:
    try:
        from ai_edge_litert.interpreter import Interpreter, load_delegate
        delegate = load_delegate(LIB)
        out["libedgetpu"] = True
        out["delegate_ok"] = True
    except FileNotFoundError:
        out["error"] = f"libedgetpu shared library not found ({LIB})"
    except Exception as exc:
        out["error"] = str(exc)

if delegate is not None and MODEL and os.path.exists(MODEL):
    try:
        import numpy as np
        from ai_edge_litert.interpreter import Interpreter
        interp = Interpreter(model_path=MODEL, experimental_delegates=[delegate])
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        dummy = np.zeros(inp["shape"], dtype=inp["dtype"])
        interp.set_tensor(inp["index"], dummy)
        interp.invoke()
        out["inference_ok"] = True
    except Exception as exc:
        msg = str(exc)
        if "EdgeTpuDelegateForCustomOp" in msg or "custom op" in msg.lower():
            out["error"] = (
                "Library loads, but device is not responding. "
                "Unplug & re-plug the Coral USB accelerator, then try again. "
                "On macOS, check System Settings -> Privacy & Security -> USB."
            )
            out["delegate_ok"] = False
        else:
            out["error"] = f"Inference test failed: {msg}"
            out["delegate_ok"] = False

if out["pycoral"] and not out["delegate_ok"]:
    try:
        from pycoral.utils.edgetpu import list_edge_tpus
        out["devices_pycoral"] = [str(d) for d in list_edge_tpus()]
        out["libedgetpu"] = True
        out["delegate_ok"] = bool(out["devices_pycoral"])
    except Exception as exc:
        if not out["error"]:
            out["error"] = str(exc)

sys.stdout.write(json.dumps(out))
"""


def _crashed_result(signal_num: int) -> Dict[str, Any]:
    return {
        "ai_edge_litert": True,
        "ai_edge_litert_ver": "installed",
        "pycoral": False,
        "pycoral_ver": None,
        "libedgetpu": True,
        "delegate_ok": False,
        "inference_ok": False,
        "devices_pycoral": [],
        "error": (
            f"TPU probe subprocess crashed (signal {signal_num}). "
            "This usually means libedgetpu loaded but the device is not responding. "
            "Unplug and re-plug the Coral USB accelerator, then retry."
        ),
    }


def probe_tpu() -> Dict[str, Any]:
    """Probe TPU state in an isolated subprocess — see factory.probe_coral."""
    env = dict(os.environ)
    env["FL_EDGETPU_LIB"] = _EDGETPU_LIB

    try:
        from app.config import load_config
        cfg = load_config()
        if cfg.detection.coral_model_path:
            env["FL_CORAL_MODEL"] = cfg.detection.coral_model_path
    except Exception:
        pass

    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE_SCRIPT],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return {
            "ai_edge_litert": False, "ai_edge_litert_ver": None,
            "pycoral": False, "pycoral_ver": None,
            "libedgetpu": False, "delegate_ok": False, "inference_ok": False,
            "devices_pycoral": [], "error": "TPU probe timed out.",
        }

    if proc.returncode < 0:
        return _crashed_result(-proc.returncode)

    try:
        return json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return {
            "ai_edge_litert": False, "ai_edge_litert_ver": None,
            "pycoral": False, "pycoral_ver": None,
            "libedgetpu": False, "delegate_ok": False, "inference_ok": False,
            "devices_pycoral": [],
            "error": f"Probe produced no JSON. stderr={proc.stderr.strip()[:400]}",
        }


_LIBEDGETPU_ZIP = (
    "https://github.com/google-coral/libedgetpu/releases/download/"
    "release-grouper/edgetpu_runtime_20221024.zip"
)


def _fix_commands() -> List[str]:
    """Return the exact shell commands needed to install libedgetpu."""
    if platform.system() == "Darwin":
        import os
        already = os.path.exists("/usr/local/lib/libedgetpu.1.dylib")
        if already:
            # Library installed but not in Homebrew search path — just symlink
            return [
                "# libedgetpu megtalálható /usr/local/lib/-ben, de az Apple Silicon",
                "# csak /opt/homebrew/lib/-ben keresi. Szimlink megoldja:",
                "sudo ln -sf /usr/local/lib/libedgetpu.1.dylib"
                " /opt/homebrew/lib/libedgetpu.1.dylib",
            ]
        return [
            f'curl -fsSL "{_LIBEDGETPU_ZIP}" -o /tmp/edgetpu_rt.zip',
            "unzip -qo /tmp/edgetpu_rt.zip -d /tmp/edgetpu_rt",
            "sudo bash /tmp/edgetpu_rt/edgetpu_runtime/install.sh",
            "# Ha települt, de még nem látja — szimlink kell:",
            "sudo ln -sf /usr/local/lib/libedgetpu.1.dylib"
            " /opt/homebrew/lib/libedgetpu.1.dylib",
        ]
    else:
        return [
            "echo 'deb https://packages.cloud.google.com/apt coral-edgetpu-stable main'"
            " | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list",
            "curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg"
            " | sudo apt-key add -",
            "sudo apt-get update && sudo apt-get install -y libedgetpu1-std",
        ]


# ── Background installer ──────────────────────────────────────────────────────

class _InstallerThread(QThread):
    output = Signal(str)
    finished_ok = Signal(bool)

    def __init__(self, commands: List[str]) -> None:
        super().__init__()
        self._commands = commands

    def run(self) -> None:
        ok = True
        for cmd in self._commands:
            self.output.emit(f"$ {cmd}")
            try:
                proc = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=180
                )
                if proc.stdout.strip():
                    self.output.emit(proc.stdout.strip())
                if proc.returncode != 0:
                    self.output.emit(f"[HIBA / ERROR] {proc.stderr.strip()}")
                    ok = False
            except subprocess.TimeoutExpired:
                self.output.emit("[HIBA / ERROR] Időtúllépés / Command timed out")
                ok = False
            except Exception as exc:
                self.output.emit(f"[HIBA / ERROR] {exc}")
                ok = False
        self.finished_ok.emit(ok)


# ── Dialog ────────────────────────────────────────────────────────────────────

class TpuStatusDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("tpu_title"))
        self.setMinimumWidth(560)
        self._thread: _InstallerThread | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        info = probe_tpu()
        tpu_ok = info["delegate_ok"] and info.get("inference_ok", False)

        # ── Summary ───────────────────────────────────────────────────────
        summary = QLabel(t("tpu_ok_label") if tpu_ok else t("tpu_warn_label"))
        font = summary.font()
        font.setPointSize(13)
        font.setBold(True)
        summary.setFont(font)
        summary.setAlignment(Qt.AlignCenter)
        summary.setStyleSheet(f"color: {'#4caf50' if tpu_ok else '#f57c00'};")
        layout.addWidget(summary)

        # ── Status details ────────────────────────────────────────────────
        lines: List[str] = []
        if info["ai_edge_litert"]:
            lines.append(f"✓ ai-edge-litert {info['ai_edge_litert_ver']}")
        else:
            lines.append(f"✗ {t('tpu_ai_edge_missing')}")

        if info["libedgetpu"]:
            lines.append(f"✓ {t('tpu_libedge_ok')}")
        else:
            lines.append(f"✗ {t('tpu_libedge_miss')}")

        if info["pycoral"]:
            lines.append(f"✓ pycoral {info['pycoral_ver']} (legacy)")

        lines.append("")
        lines.append(t("tpu_devices"))
        if info["devices_pycoral"]:
            for d in info["devices_pycoral"]:
                lines.append(f"  • {d}")
        elif info["delegate_ok"]:
            lines.append(f"  • {t('tpu_delegate_loaded')}")
        else:
            lines.append(f"  {t('tpu_none')}")

        lines.append("")
        if info.get("inference_ok"):
            lines.append(t("tpu_inference_ok"))
        elif info["libedgetpu"]:
            lines.append(t("tpu_inference_fail"))
            lines.append("")
            lines.append(t("tpu_phantom_tip"))

        if info["error"]:
            lines.append("")
            lines.append(t("tpu_error", msg=info["error"]))

        status_box = QTextEdit()
        status_box.setReadOnly(True)
        status_box.setMaximumHeight(160)
        status_box.setPlainText("\n".join(lines))
        layout.addWidget(status_box)

        # ── Fix section (only when broken) ────────────────────────────────
        if not tpu_ok:
            cmds = _fix_commands()

            # Copyable script block
            script_group = QGroupBox(t("tpu_manual_commands"))
            script_layout = QVBoxLayout(script_group)

            script_box = QTextEdit()
            mono = QFont("Menlo" if platform.system() == "Darwin" else "Monospace")
            mono.setPointSize(11)
            script_box.setFont(mono)
            script_box.setReadOnly(True)
            script_box.setMaximumHeight(90)
            script_box.setPlainText("\n".join(cmds))
            script_box.setStyleSheet("background:#1e1e1e; color:#d4d4d4;")
            script_layout.addWidget(script_box)

            copy_btn = QPushButton(f"📋 {t('tpu_copy_commands')}")
            copy_btn.clicked.connect(
                lambda: QApplication.clipboard().setText("\n".join(cmds))
            )
            script_layout.addWidget(copy_btn)
            layout.addWidget(script_group)

            # Auto-fix button + output
            fix_group = QGroupBox(t("tpu_auto_fix"))
            fix_layout = QVBoxLayout(fix_group)

            self._fix_output = QTextEdit()
            self._fix_output.setReadOnly(True)
            self._fix_output.setMaximumHeight(120)
            self._fix_output.setVisible(False)
            fix_layout.addWidget(self._fix_output)

            self._progress = QProgressBar()
            self._progress.setRange(0, 0)
            self._progress.setVisible(False)
            fix_layout.addWidget(self._progress)

            self._fix_btn = QPushButton(f"🔧 {t('tpu_run_auto_fix')}")
            self._fix_btn.clicked.connect(lambda: self._on_fix(cmds))
            fix_layout.addWidget(self._fix_btn)

            layout.addWidget(fix_group)

        # ── Close button ──────────────────────────────────────────────────
        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _on_fix(self, cmds: List[str]) -> None:
        self._fix_btn.setEnabled(False)
        self._fix_output.setVisible(True)
        self._progress.setVisible(True)

        self._thread = _InstallerThread(cmds)
        self._thread.output.connect(self._fix_output.append)
        self._thread.finished_ok.connect(self._on_fix_done)
        self._thread.start()

    def _on_fix_done(self, ok: bool) -> None:
        self._progress.setVisible(False)
        self._fix_btn.setEnabled(True)
        if ok:
            self._fix_output.append(t("tpu_fix_done"))
        else:
            self._fix_output.append(t("tpu_fix_failed"))
