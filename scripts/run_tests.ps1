param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PytestArgs
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
$ReportDir = if ($env:TEST_REPORT_DIR) { $env:TEST_REPORT_DIR } else { Join-Path $RepoRoot "test-reports" }
$LogFile = Join-Path $ReportDir "latest.log"
$JunitFile = Join-Path $ReportDir "latest.junit.xml"
$ReportFile = Join-Path $ReportDir "latest.md"
$CoverageJson = Join-Path $ReportDir "coverage.json"
$CoverageXml = Join-Path $ReportDir "coverage.xml"

Set-Location $RepoRoot

$Python = if (Test-Path (Join-Path $RepoRoot ".venv\Scripts\python.exe")) {
    Join-Path $RepoRoot ".venv\Scripts\python.exe"
} else {
    "python"
}

try {
    & $Python -m pytest --version *> $null
} catch {
    Write-Error "pytest is not installed for $Python. Run: $Python -m pip install -e '.[dev]'"
    exit 127
}

if (-not $PytestArgs -or $PytestArgs.Count -eq 0) {
    $PytestArgs = @("tests")
}

New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null
Remove-Item $LogFile, $JunitFile, $ReportFile, $CoverageJson, $CoverageXml -ErrorAction SilentlyContinue

$Command = @(
    $Python,
    "-m",
    "pytest",
    "--tb=short",
    "-ra"
) + $PytestArgs + @(
    "--junitxml=$JunitFile",
    "--cov=app",
    "--cov-report=term-missing:skip-covered",
    "--cov-report=json:$CoverageJson",
    "--cov-report=xml:$CoverageXml"
)

Write-Host "==> Running tests"
Write-Host "    Python: $(& $Python -c 'import sys; print(sys.executable)')"
Write-Host "    Command: $($Command -join ' ')"
Write-Host "    Log: $LogFile"
Write-Host "    Report: $ReportFile"
Write-Host ""

$Output = & $Python -m pytest --tb=short -ra @PytestArgs "--junitxml=$JunitFile" "--cov=app" "--cov-report=term-missing:skip-covered" "--cov-report=json:$CoverageJson" "--cov-report=xml:$CoverageXml" 2>&1
$PytestStatus = $LASTEXITCODE
$Output | Tee-Object -FilePath $LogFile

$env:PYTEST_STATUS = "$PytestStatus"
$env:PYTEST_COMMAND = ($Command -join " ")
$env:LOG_FILE = "$LogFile"
$env:JUNIT_FILE = "$JunitFile"
$env:REPORT_FILE = "$ReportFile"
$env:COVERAGE_JSON = "$CoverageJson"
$env:COVERAGE_XML = "$CoverageXml"

$ReportScript = @'
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

tests: dict[str, list[tuple[str, str]]] = {"passed": [], "failed": [], "errors": [], "skipped": []}

if junit_path.exists() and junit_path.stat().st_size:
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

total_tests = sum(len(items) for items in tests.values())
pass_rate = (len(tests["passed"]) / total_tests * 100.0) if total_tests else 0.0
coverage = coverage_summary(coverage_json_path)
status_text = "PASS" if pytest_status == 0 else f"FAIL (exit code {pytest_status})"
generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

lines = [
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

lines.extend([
    "",
    "## Counts",
    "",
    f"- Passed: {len(tests['passed'])}",
    f"- Failed: {len(tests['failed'])}",
    f"- Errors: {len(tests['errors'])}",
    f"- Skipped: {len(tests['skipped'])}",
    f"- Warnings: {warning_total}",
    "",
    "## Failures And Errors",
    "",
])

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
        lines.extend(["", "### Warning Counts By Test Module", ""])
        for module, count in sorted(warning_modules, key=lambda item: item[1], reverse=True):
            lines.append(f"- `{module}`: {count}")
    if warning_details:
        lines.extend(["", "### Warning Reasons", ""])
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
'@

$ReportScript | & $Python -

Write-Host ""
Write-Host "==> Test evaluation"
$SummaryLine = Select-String -Path $LogFile -Pattern '^=+ .* =+$' | Select-Object -Last 1
if ($SummaryLine) {
    Write-Host "    $($SummaryLine.Line)"
}
if ($PytestStatus -eq 0) {
    Write-Host "    Result: PASS"
} else {
    Write-Host "    Result: FAIL (exit code $PytestStatus)"
}
Write-Host "    Report: $ReportFile"
Write-Host "    Log: $LogFile"

exit $PytestStatus
