"""Background dependency installer.

Checks for optional heavy AI packages (ai-edge-litert, insightface) and
installs them via pip if missing.  Works in both script and frozen-app modes.

Frozen-app note
---------------
In a PyInstaller bundle sys.executable is the frozen .exe.  We search PATH
for a real Python interpreter to call pip with.  After a successful install
we add the user site-packages directory to sys.path so the *current* process
can import the newly installed package without a restart (works because the
embedder and InsightFace are lazily imported at first scan).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

log = logging.getLogger(__name__)

# Packages we try to install, in priority order.
# Each entry: (import_name, pip_spec, display_name, size_hint)
CANDIDATE_PACKAGES: list[tuple[str, str, str, str]] = [
    (
        "ai_edge_litert",
        "ai-edge-litert>=1.0",
        "ai-edge-litert",
        "~15 MB",
    ),
    # onnxruntime is a shared dependency of both the SFace embedder (via OpenCV)
    # and InsightFace.  Install it first as a standalone lightweight package so
    # the verification gate has its runtime even if the full insightface install
    # later fails or takes longer.
    (
        "onnxruntime",
        "onnxruntime",
        "onnxruntime",
        "~7 MB",
    ),
    # InsightFace SCRFD: the verification co-detector that recovers profile /
    # turned faces the frontal-biased primary detector misses.  The pip install
    # covers both the library and its onnxruntime dependency.  The buffalo_l
    # model pack (~200 MB) is downloaded automatically by InsightFace on first
    # use (i.e. at the start of the first scan after install).
    (
        "insightface",
        "insightface",
        "insightface",
        "~10 MB + 200 MB model",
    ),
]


class PackageCheckResult(NamedTuple):
    import_name: str
    pip_spec: str
    display_name: str
    available: bool


def check_packages() -> list[PackageCheckResult]:
    """Return availability status for each candidate package."""
    results = []
    for import_name, pip_spec, display_name, _ in CANDIDATE_PACKAGES:
        available = _is_importable(import_name)
        results.append(PackageCheckResult(import_name, pip_spec, display_name, available))
        log.debug("Dep check: %s -> %s", display_name, "ok" if available else "missing")
    return results


def missing_packages() -> list[PackageCheckResult]:
    return [r for r in check_packages() if not r.available]


def find_python_executable() -> str | None:
    """Return a usable Python >= 3.11 executable path.

    In a frozen bundle sys.executable is the .exe wrapper — we look for a
    real interpreter on PATH instead.  In a normal script run we use
    sys.executable directly.
    """
    if not getattr(sys, "frozen", False):
        return sys.executable

    candidates = [
        "python3.13", "python3.12", "python3.11",
        "python3", "python",
    ]
    for name in candidates:
        exe = shutil.which(name)
        if not exe:
            continue
        try:
            result = subprocess.run(
                [exe, "-c",
                 "import sys; print('ok' if sys.version_info >= (3,11) else 'old')"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip() == "ok":
                log.debug("Found Python for pip: %s", exe)
                return exe
        except Exception:
            continue
    log.warning("No usable Python found on PATH for background install.")
    return None


def install_package(pip_spec: str, python_exe: str) -> bool:
    """Run pip install for *pip_spec* using *python_exe*.

    Uses --user when in a frozen app or when not inside a virtual environment,
    so admin rights are never required.
    """
    in_venv = sys.prefix != sys.base_prefix
    frozen = getattr(sys, "frozen", False)

    cmd = [python_exe, "-m", "pip", "install", "--quiet"]
    if frozen or not in_venv:
        cmd.append("--user")
    cmd.extend(pip_spec.split())

    log.info("Installing: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            log.info("Installed %s successfully.", pip_spec)
            return True
        log.warning("pip failed for %s:\n%s", pip_spec, result.stderr[-1000:])
        return False
    except subprocess.TimeoutExpired:
        log.warning("pip install timed out for %s", pip_spec)
        return False
    except Exception as exc:
        log.warning("pip install error for %s: %s", pip_spec, exc)
        return False


def refresh_sys_path() -> None:
    """After --user install, make the new package importable in this process."""
    try:
        import site

        user_site = site.getusersitepackages()
        if user_site and user_site not in sys.path:
            sys.path.insert(0, user_site)
            log.debug("Added user site to sys.path: %s", user_site)
    except Exception as exc:
        log.debug("Could not refresh sys.path: %s", exc)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _is_importable(module_name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(module_name) is not None
