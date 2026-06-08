#!/usr/bin/env bash
# Run the full test suite and generate a compact pass/fail/warning report.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
REPORT_DIR="${TEST_REPORT_DIR:-$REPO_ROOT/test-reports}"
LOG_FILE="$REPORT_DIR/latest.log"
JUNIT_FILE="$REPORT_DIR/latest.junit.xml"
REPORT_FILE="$REPORT_DIR/latest.md"
COVERAGE_JSON="$REPORT_DIR/coverage.json"
COVERAGE_XML="$REPORT_DIR/coverage.xml"

cd "$REPO_ROOT"

if [ -x "$VENV_PYTHON" ]; then
    PYTHON="$VENV_PYTHON"
else
    PYTHON="${PYTHON:-python3}"
fi

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "ERROR: Python interpreter not found: $PYTHON" >&2
    echo "Install Python 3.11+ or create the local virtualenv first." >&2
    exit 127
fi

if ! "$PYTHON" -m pytest --version >/dev/null 2>&1; then
    echo "ERROR: pytest is not installed for $PYTHON" >&2
    echo "Run: $PYTHON -m pip install -e '.[dev]'" >&2
    exit 127
fi

if [ "$#" -gt 0 ]; then
    PYTEST_ARGS=("$@")
else
    PYTEST_ARGS=("tests")
fi

mkdir -p "$REPORT_DIR"
PYTEST_COMMAND=(
    "$PYTHON" -m pytest
    --tb=short
    -ra
    "${PYTEST_ARGS[@]}"
    "--junitxml=$JUNIT_FILE"
    --cov=app
    --cov-report=term-missing:skip-covered
    "--cov-report=json:$COVERAGE_JSON"
    "--cov-report=xml:$COVERAGE_XML"
)

echo "==> Running tests"
echo "    Python: $("$PYTHON" -c 'import sys; print(sys.executable)')"
echo "    Command: ${PYTEST_COMMAND[*]}"
echo "    Log: $LOG_FILE"
echo "    Report: $REPORT_FILE"
echo ""

rm -f "$LOG_FILE" "$JUNIT_FILE" "$REPORT_FILE" "$COVERAGE_JSON" "$COVERAGE_XML"

"${PYTEST_COMMAND[@]}" 2>&1 | tee "$LOG_FILE"
PYTEST_STATUS=${PIPESTATUS[0]}

PYTEST_STATUS="$PYTEST_STATUS" \
PYTEST_COMMAND="${PYTEST_COMMAND[*]}" \
LOG_FILE="$LOG_FILE" \
JUNIT_FILE="$JUNIT_FILE" \
REPORT_FILE="$REPORT_FILE" \
COVERAGE_JSON="$COVERAGE_JSON" \
COVERAGE_XML="$COVERAGE_XML" \
"$PYTHON" <<'PY'
from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
import json
import os
import re
import xml.etree.ElementTree as ET


def first_line(value: str | None) -> str:
    if not value:
        return "No detailed reason was reported by pytest."
    return value.strip().splitlines()[0].strip() or "No detailed reason was reported by pytest."


def testcase_id(case: ET.Element) -> str:
    file_name = case.attrib.get("file", "")
    class_name = case.attrib.get("classname", "")
    name = case.attrib.get("name", "")
    if file_name:
        return f"{file_name}::{name}"
    if class_name:
        return f"{class_name}::{name}"
    return name


def coverage_summary(path: Path) -> dict[str, float | int | None]:
    empty = {
        "percent": None,
        "covered_lines": None,
        "total_lines": None,
        "missing_lines": None,
    }
    if not path.exists() or not path.stat().st_size:
        return empty
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return empty
    totals = payload.get("totals") if isinstance(payload, dict) else None
    if not isinstance(totals, dict):
        return empty
    total_lines = totals.get("num_statements")
    covered_lines = totals.get("covered_lines")
    missing_lines = totals.get("missing_lines")
    percent = totals.get("percent_covered")
    return {
        "percent": float(percent) if isinstance(percent, int | float) else None,
        "covered_lines": int(covered_lines) if isinstance(covered_lines, int) else None,
        "total_lines": int(total_lines) if isinstance(total_lines, int) else None,
        "missing_lines": int(missing_lines) if isinstance(missing_lines, int) else None,
    }


pytest_status = int(os.environ["PYTEST_STATUS"])
pytest_command = os.environ["PYTEST_COMMAND"]
log_path = Path(os.environ["LOG_FILE"])
junit_path = Path(os.environ["JUNIT_FILE"])
report_path = Path(os.environ["REPORT_FILE"])
coverage_json_path = Path(os.environ["COVERAGE_JSON"])
coverage_xml_path = Path(os.environ["COVERAGE_XML"])

log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
log_lines = log_text.splitlines()

summary_line = ""
for line in log_lines:
    if re.match(r"^=+ .* =+$", line):
        summary_line = line

warning_total = 0
if summary_line:
    warning_match = re.search(r"(\d+) warnings?", summary_line)
    if warning_match:
        warning_total = int(warning_match.group(1))

warning_modules: list[tuple[str, int]] = []
warning_details: Counter[tuple[str, str, str, str]] = Counter()
in_warning_summary = False
module_re = re.compile(r"^([^=\s].*?): (\d+) warnings?$")
detail_re = re.compile(r"^\s+(.+?):(\d+):\s+([^:]*Warning):\s+(.*)$")

for line in log_lines:
    if " warnings summary " in line:
        in_warning_summary = True
        continue
    if in_warning_summary and line.startswith("-- Docs:"):
        in_warning_summary = False
        continue
    if not in_warning_summary:
        continue
    module_match = module_re.match(line)
    if module_match:
        warning_modules.append((module_match.group(1), int(module_match.group(2))))
        continue
    detail_match = detail_re.match(line)
    if detail_match:
        source, line_no, category, message = detail_match.groups()
        warning_details[(source, line_no, category, message)] += 1

tests: dict[str, list[tuple[str, str]]] = {
    "passed": [],
    "failed": [],
    "errors": [],
    "skipped": [],
}

if junit_path.exists():
    root = ET.parse(junit_path).getroot()
    for case in root.iter("testcase"):
        node_id = testcase_id(case)
        failure = case.find("failure")
        error = case.find("error")
        skipped = case.find("skipped")
        if failure is not None:
            tests["failed"].append((node_id, first_line(failure.attrib.get("message") or failure.text)))
        elif error is not None:
            tests["errors"].append((node_id, first_line(error.attrib.get("message") or error.text)))
        elif skipped is not None:
            tests["skipped"].append((node_id, first_line(skipped.attrib.get("message") or skipped.text)))
        else:
            tests["passed"].append((node_id, "All assertions and fixtures completed successfully."))

count_lines = [
    ("Passed", len(tests["passed"])),
    ("Failed", len(tests["failed"])),
    ("Errors", len(tests["errors"])),
    ("Skipped", len(tests["skipped"])),
    ("Warnings", warning_total),
]
total_tests = sum(len(items) for items in tests.values())
pass_rate = (len(tests["passed"]) / total_tests * 100.0) if total_tests else 0.0
coverage = coverage_summary(coverage_json_path)

status_text = "PASS" if pytest_status == 0 else f"FAIL (exit code {pytest_status})"
generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

lines: list[str] = [
    "# Test Report",
    "",
    f"- Generated: {generated_at}",
    f"- Result: {status_text}",
    f"- Command: `{pytest_command}`",
    f"- Raw log: `{log_path}`",
    f"- JUnit XML: `{junit_path}`",
    f"- Coverage JSON: `{coverage_json_path}`",
    f"- Coverage XML: `{coverage_xml_path}`",
    f"- Total tests: {total_tests}",
    f"- Pass rate: {pass_rate:.2f}",
]
if coverage["percent"] is not None:
    lines.append(f"- Coverage: {coverage['percent']:.2f}")
    lines.append(f"- Coverage covered lines: {coverage['covered_lines']}")
    lines.append(f"- Coverage total lines: {coverage['total_lines']}")
    lines.append(f"- Coverage missing lines: {coverage['missing_lines']}")
if summary_line:
    lines.append(f"- Pytest summary: `{summary_line}`")

lines.extend(["", "## Counts", ""])
for label, count in count_lines:
    lines.append(f"- {label}: {count}")

lines.extend(["", "## Failures And Errors", ""])
if tests["failed"] or tests["errors"]:
    for node_id, reason in tests["failed"]:
        lines.append(f"- FAIL `{node_id}`")
        lines.append(f"  - Why: {reason}")
    for node_id, reason in tests["errors"]:
        lines.append(f"- ERROR `{node_id}`")
        lines.append(f"  - Why: {reason}")
else:
    lines.append("- None.")
    lines.append("- Why: every collected test completed without assertion errors or unhandled exceptions.")

lines.extend(["", "## Warnings", ""])
if warning_total:
    lines.append(f"- Total warnings reported by pytest: {warning_total}")
    if warning_modules:
        lines.append("")
        lines.append("### Warning Counts By Test Module")
        lines.append("")
        for module, count in sorted(warning_modules, key=lambda item: item[1], reverse=True):
            lines.append(f"- `{module}`: {count}")
    if warning_details:
        lines.append("")
        lines.append("### Warning Reasons")
        lines.append("")
        for (source, line_no, category, message), count in warning_details.most_common():
            suffix = f" ({count} grouped entries)" if count > 1 else ""
            lines.append(f"- `{category}` at `{source}:{line_no}`{suffix}")
            lines.append(f"  - Why: {message}")
else:
    lines.append("- None.")

lines.extend(["", "## Passed Tests", ""])
if tests["passed"]:
    for node_id, reason in tests["passed"]:
        lines.append(f"- PASS `{node_id}`")
        lines.append(f"  - Why: {reason}")
else:
    lines.append("- No passed test list was available in the generated JUnit XML.")

if tests["skipped"]:
    lines.extend(["", "## Skipped Tests", ""])
    for node_id, reason in tests["skipped"]:
        lines.append(f"- SKIP `{node_id}`")
        lines.append(f"  - Why: {reason}")

report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

echo ""
echo "==> Test evaluation"

SUMMARY_LINE="$(grep -E '^=+ .* =+$' "$LOG_FILE" | tail -n 1 || true)"
if [ -n "$SUMMARY_LINE" ]; then
    echo "    $SUMMARY_LINE"
fi

if [ "$PYTEST_STATUS" -eq 0 ]; then
    echo "    Result: PASS"
else
    echo "    Result: FAIL (exit code $PYTEST_STATUS)"
fi
echo "    Report: $REPORT_FILE"
echo "    Log: $LOG_FILE"

exit "$PYTEST_STATUS"
