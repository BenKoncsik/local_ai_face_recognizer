from __future__ import annotations

from pathlib import Path


def test_release_artifact_download_inline_python_compiles() -> None:
    workflow = Path(".github/workflows/build-release.yml")
    text = workflow.read_text(encoding="utf-8")

    step_start = text.index("      - name: Download release artifacts")
    heredoc_start = text.index("          python3 - <<'PY'\n", step_start)
    code_start = heredoc_start + len("          python3 - <<'PY'\n")
    code_end = text.index("          PY\n", code_start)
    code = "\n".join(
        line[10:] if line.startswith("          ") else line
        for line in text[code_start:code_end].splitlines()
    )

    assert "def download_file(" in code
    assert "http.client.IncompleteRead" in code
    compile(code, "build-release-download-inline.py", "exec")
