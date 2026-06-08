#!/usr/bin/env python3
"""Generate a static GitHub Pages dashboard from test report artifacts."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def read_json(path: Path) -> Any:
    if not path.exists() or not path.stat().st_size:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def parse_report(report_text: str) -> dict[str, Any]:
    counts = {
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "warnings": 0,
    }
    result = "UNKNOWN"
    summary = ""
    total_tests: int | None = None
    pass_rate: float | None = None
    coverage: dict[str, float | int | None] = {
        "percent": None,
        "covered_lines": None,
        "total_lines": None,
        "missing_lines": None,
    }

    for line in report_text.splitlines():
        if line.startswith("- Result:"):
            result = line.split(":", 1)[1].strip()
        elif line.startswith("- Pytest summary:"):
            summary = line.split(":", 1)[1].strip().strip("`")
        elif line.startswith("- Total tests:"):
            try:
                total_tests = int(line.split(":", 1)[1].strip())
            except ValueError:
                total_tests = None
        elif line.startswith("- Pass rate:"):
            try:
                pass_rate = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass_rate = None
        elif line.startswith("- Coverage:"):
            try:
                coverage["percent"] = float(line.split(":", 1)[1].strip())
            except ValueError:
                coverage["percent"] = None
        elif line.startswith("- Coverage covered lines:"):
            try:
                coverage["covered_lines"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                coverage["covered_lines"] = None
        elif line.startswith("- Coverage total lines:"):
            try:
                coverage["total_lines"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                coverage["total_lines"] = None
        elif line.startswith("- Coverage missing lines:"):
            try:
                coverage["missing_lines"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                coverage["missing_lines"] = None
        else:
            match = re.match(r"- (Passed|Failed|Errors|Skipped|Warnings):\s+(\d+)", line)
            if match:
                counts[match.group(1).lower()] = int(match.group(2))

    if total_tests is None:
        total_tests = counts["passed"] + counts["failed"] + counts["errors"] + counts["skipped"]
    if pass_rate is None:
        pass_rate = counts["passed"] / total_tests * 100.0 if total_tests else 0.0

    return {
        "result": result,
        "summary": summary,
        "counts": counts,
        "total_tests": total_tests,
        "pass_rate": pass_rate,
        "coverage": coverage,
    }


def parse_coverage_json(path: Path) -> dict[str, float | int | None]:
    empty = {
        "percent": None,
        "covered_lines": None,
        "total_lines": None,
        "missing_lines": None,
    }
    payload = read_json(path)
    totals = payload.get("totals") if isinstance(payload, dict) else None
    if not isinstance(totals, dict):
        return empty

    percent = totals.get("percent_covered")
    covered_lines = totals.get("covered_lines")
    total_lines = totals.get("num_statements")
    missing_lines = totals.get("missing_lines")
    return {
        "percent": float(percent) if isinstance(percent, int | float) else None,
        "covered_lines": int(covered_lines) if isinstance(covered_lines, int) else None,
        "total_lines": int(total_lines) if isinstance(total_lines, int) else None,
        "missing_lines": int(missing_lines) if isinstance(missing_lines, int) else None,
    }


def load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict) and isinstance(payload.get("runs"), list):
        runs = [item for item in payload["runs"] if isinstance(item, dict)]
    elif isinstance(payload, list):
        runs = [item for item in payload if isinstance(item, dict)]
    else:
        return []

    for run in runs:
        counts = run.get("counts")
        if not isinstance(counts, dict):
            continue
        total_tests = sum(
            int(counts.get(key, 0) or 0)
            for key in ("passed", "failed", "errors", "skipped")
        )
        run.setdefault("total_tests", total_tests)
        if "pass_rate" not in run:
            run["pass_rate"] = (
                int(counts.get("passed", 0) or 0) / total_tests * 100.0
                if total_tests
                else 0.0
            )
    return runs


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def build_run_metadata(
    parsed: dict[str, Any],
    coverage: dict[str, float | int | None],
    run_slug: str,
    platform: str,
) -> dict[str, Any]:
    repository = env("GITHUB_REPOSITORY", "local")
    server_url = env("GITHUB_SERVER_URL", "https://github.com")
    run_id = env("GITHUB_RUN_ID", f"local-{run_slug}")
    run_attempt = env("GITHUB_RUN_ATTEMPT", "1")
    run_number = env("GITHUB_RUN_NUMBER", "0")
    sha = env("GITHUB_SHA", "")
    ref_name = env("GITHUB_REF_NAME", "")

    return {
        "key": f"{run_id}-{run_attempt}-{platform}",
        "run_id": run_id,
        "run_attempt": run_attempt,
        "run_number": run_number,
        "workflow": env("GITHUB_WORKFLOW", "Local test run"),
        "event": env("GITHUB_EVENT_NAME", "local"),
        "actor": env("GITHUB_ACTOR", ""),
        "platform": platform,
        "repository": repository,
        "branch": ref_name,
        "sha": sha,
        "short_sha": sha[:7],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "result": parsed["result"],
        "summary": parsed["summary"],
        "counts": parsed["counts"],
        "total_tests": parsed["total_tests"],
        "pass_rate": parsed["pass_rate"],
        "coverage_percent": coverage["percent"],
        "coverage_covered_lines": coverage["covered_lines"],
        "coverage_total_lines": coverage["total_lines"],
        "coverage_missing_lines": coverage["missing_lines"],
        "run_url": f"{server_url}/{repository}/actions/runs/{run_id}" if run_id.isdigit() else "",
        "commit_url": f"{server_url}/{repository}/commit/{sha}" if sha else "",
        "report_url": f"runs/{run_slug}/report.html",
        "markdown_url": f"runs/{run_slug}/latest.md",
        "log_url": f"runs/{run_slug}/latest.log",
        "junit_url": f"runs/{run_slug}/latest.junit.xml",
        "coverage_json_url": f"runs/{run_slug}/coverage.json",
        "coverage_xml_url": f"runs/{run_slug}/coverage.xml",
    }


def status_class(result: str) -> str:
    return "pass" if result == "PASS" else "fail" if result.startswith("FAIL") else "unknown"


def as_float(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def format_percent(value: Any) -> str:
    number = as_float(value)
    return "n/a" if number is None else f"{number:.2f}%"


def format_delta(value: float | None) -> tuple[str, str]:
    if value is None:
        return ("no previous run", "neutral")
    if abs(value) < 0.005:
        return ("+0.00 pts", "neutral")
    sign = "+" if value > 0 else "-"
    css_class = "positive" if value > 0 else "negative"
    return (f"{sign}{abs(value):.2f} pts", css_class)


def previous_same_platform(history: list[dict[str, Any]], index: int) -> dict[str, Any] | None:
    platform = history[index].get("platform")
    for candidate in history[index + 1 :]:
        if candidate.get("platform") == platform:
            return candidate
    return None


def metric_delta(history: list[dict[str, Any]], index: int, key: str) -> float | None:
    current = as_float(history[index].get(key))
    previous = previous_same_platform(history, index)
    previous_value = as_float(previous.get(key)) if previous else None
    if current is None or previous_value is None:
        return None
    return current - previous_value


def latest_by_platform(history: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    seen: set[str] = set()
    latest: list[tuple[int, dict[str, Any]]] = []
    for index, run in enumerate(history):
        platform = str(run.get("platform", "unknown"))
        if platform in seen:
            continue
        seen.add(platform)
        latest.append((index, run))
    return latest


def render_metric_chart(history: list[dict[str, Any]]) -> str:
    points = [
        run
        for run in reversed(history[:30])
        if as_float(run.get("coverage_percent")) is not None
        or as_float(run.get("pass_rate")) is not None
    ]
    if not points:
        return '<p class="empty">No coverage history has been recorded yet.</p>'

    width = 760
    height = 230
    pad_l = 42
    pad_r = 18
    pad_t = 16
    pad_b = 36
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    def xy(index: int, value: float) -> tuple[float, float]:
        x = pad_l + (plot_w * index / max(len(points) - 1, 1))
        y = pad_t + plot_h - (plot_h * max(0.0, min(100.0, value)) / 100.0)
        return x, y

    def line(key: str) -> str:
        coords = []
        for index, run in enumerate(points):
            value = as_float(run.get(key))
            if value is None:
                continue
            x, y = xy(index, value)
            coords.append(f"{x:.1f},{y:.1f}")
        return " ".join(coords)

    def dots(key: str, css_class: str) -> str:
        circles = []
        for index, run in enumerate(points):
            value = as_float(run.get(key))
            if value is None:
                continue
            x, y = xy(index, value)
            circles.append(f'<circle class="{css_class}" cx="{x:.1f}" cy="{y:.1f}" r="4" />')
        return "".join(circles)

    labels = []
    for tick in (0, 50, 100):
        y = pad_t + plot_h - (plot_h * tick / 100.0)
        labels.append(
            f'<line class="grid" x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" />'
            f'<text class="axis" x="8" y="{y + 4:.1f}">{tick}%</text>'
        )

    x_labels = []
    if len(points) == 1:
        label_indices = [0]
    else:
        label_indices = sorted({0, len(points) // 2, len(points) - 1})
    for index in label_indices:
        run = points[index]
        x, _ = xy(index, 0)
        label = f"#{run.get('run_number', '?')} {run.get('platform', '')}"
        x_labels.append(
            f'<text class="axis x-label" x="{x:.1f}" y="{height - 8}" text-anchor="middle">'
            f"{html.escape(label)}</text>"
        )

    return f"""
    <div class="chart-wrap">
      <svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Coverage and pass-rate history">
        {''.join(labels)}
        <line class="axis-line" x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height - pad_b}" />
        <line class="axis-line" x1="{pad_l}" y1="{height - pad_b}" x2="{width - pad_r}" y2="{height - pad_b}" />
        <polyline class="coverage-line" points="{line('coverage_percent')}" />
        <polyline class="pass-line" points="{line('pass_rate')}" />
        {dots('coverage_percent', 'coverage-dot')}
        {dots('pass_rate', 'pass-dot')}
        {''.join(x_labels)}
      </svg>
      <div class="legend">
        <span><i class="swatch coverage"></i>Coverage</span>
        <span><i class="swatch pass"></i>Pass rate</span>
      </div>
    </div>
    """


def render_report_html(report_text: str, metadata: dict[str, Any]) -> str:
    title = f"Test Report #{metadata['run_number']}"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin: 0; font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #172026; background: #f5f7fa; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 32px 20px; }}
    a {{ color: #0b5cad; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #fff; border: 1px solid #d7dee8; border-radius: 6px; padding: 16px; overflow: auto; }}
  </style>
</head>
<body>
  <main>
    <p><a href="../../index.html">Back to dashboard</a></p>
    <pre>{html.escape(report_text)}</pre>
  </main>
</body>
</html>
"""


def render_index(history: list[dict[str, Any]]) -> str:
    latest_platforms = latest_by_platform(history)
    platform_cards = []
    for index, run in latest_platforms:
        counts = run.get("counts", {})
        result = str(run.get("result", "UNKNOWN"))
        coverage_delta, coverage_delta_class = format_delta(
            metric_delta(history, index, "coverage_percent")
        )
        pass_delta, pass_delta_class = format_delta(metric_delta(history, index, "pass_rate"))
        platform_cards.append(
            f"""
      <article class="panel platform-panel">
        <div class="panel-head">
          <div>
            <div class="label">{html.escape(str(run.get('platform', 'unknown')))}</div>
            <h2>{html.escape(str(run.get('platform', 'unknown')))}</h2>
          </div>
          <span class="badge {status_class(result)}">{html.escape(result)}</span>
        </div>
        <div class="metrics">
          <div><div class="label">Coverage</div><div class="metric">{format_percent(run.get('coverage_percent'))}</div><div class="delta {coverage_delta_class}">{html.escape(coverage_delta)}</div></div>
          <div><div class="label">Pass rate</div><div class="metric">{format_percent(run.get('pass_rate'))}</div><div class="delta {pass_delta_class}">{html.escape(pass_delta)}</div></div>
          <div><div class="label">Passed</div><div class="metric">{counts.get('passed', 0)}</div></div>
          <div><div class="label">Failed</div><div class="metric">{counts.get('failed', 0) + counts.get('errors', 0)}</div></div>
          <div><div class="label">Skipped</div><div class="metric">{counts.get('skipped', 0)}</div></div>
        </div>
        <p class="summary-line">{html.escape(str(run.get('summary') or 'No pytest summary recorded.'))}</p>
        <p class="links"><a href="{html.escape(run.get('report_url', '#'))}">report</a> <a href="{html.escape(run.get('log_url', '#'))}">log</a> <a href="{html.escape(run.get('coverage_json_url', '#'))}">coverage json</a></p>
      </article>
            """
        )

    rows = []
    for index, run in enumerate(history):
        result = str(run.get("result", "UNKNOWN"))
        counts_for_run = run.get("counts", {})
        run_label = f"#{run.get('run_number', '?')}"
        if run.get("run_attempt") and run.get("run_attempt") != "1":
            run_label += f".{run['run_attempt']}"
        run_link = html.escape(run.get("run_url") or run.get("report_url", "#"))
        commit = html.escape(run.get("short_sha") or "")
        commit_link = html.escape(run.get("commit_url") or "#")
        coverage_delta, coverage_delta_class = format_delta(
            metric_delta(history, index, "coverage_percent")
        )
        pass_delta, pass_delta_class = format_delta(metric_delta(history, index, "pass_rate"))
        rows.append(
            "<tr>"
            f"<td><span class=\"badge {status_class(result)}\">{html.escape(result)}</span></td>"
            f"<td><a href=\"{run_link}\">{html.escape(run_label)}</a></td>"
            f"<td>{html.escape(run.get('platform', ''))}</td>"
            f"<td>{html.escape(run.get('branch', ''))}</td>"
            f"<td><a href=\"{commit_link}\">{commit}</a></td>"
            f"<td>{format_percent(run.get('coverage_percent'))}<br><span class=\"delta {coverage_delta_class}\">{html.escape(coverage_delta)}</span></td>"
            f"<td>{format_percent(run.get('pass_rate'))}<br><span class=\"delta {pass_delta_class}\">{html.escape(pass_delta)}</span></td>"
            f"<td>{run.get('total_tests', sum(counts_for_run.get(k, 0) for k in ('passed', 'failed', 'errors', 'skipped')))}</td>"
            f"<td>{counts_for_run.get('passed', 0)}</td>"
            f"<td>{counts_for_run.get('failed', 0)}</td>"
            f"<td>{counts_for_run.get('errors', 0)}</td>"
            f"<td>{counts_for_run.get('skipped', 0)}</td>"
            f"<td>{counts_for_run.get('warnings', 0)}</td>"
            f"<td>{html.escape(run.get('generated_at', ''))}</td>"
            f"<td><a href=\"{html.escape(run.get('report_url', '#'))}\">report</a> "
            f"<a href=\"{html.escape(run.get('log_url', '#'))}\">log</a></td>"
            "</tr>"
        )

    updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    chart = render_metric_chart(history)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Face-Local Test Dashboard</title>
  <style>
    :root {{ color-scheme: light; }}
    body {{ margin: 0; font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #172026; background: #f5f7fa; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 32px 20px; }}
    h1 {{ margin: 0 0 6px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 0; font-size: 18px; letter-spacing: 0; }}
    .section-title {{ margin: 28px 0 10px; font-size: 18px; }}
    a {{ color: #0b5cad; }}
    .topline {{ color: #5f6b7a; margin: 0 0 24px; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }}
    .panel {{ background: #fff; border: 1px solid #d7dee8; border-radius: 6px; padding: 14px; }}
    .panel-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }}
    .metrics {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; margin-top: 14px; }}
    .metric {{ font-size: 24px; font-weight: 700; }}
    .label {{ color: #5f6b7a; font-size: 12px; text-transform: uppercase; }}
    .delta {{ color: #5f6b7a; font-size: 12px; }}
    .delta.positive {{ color: #116329; }}
    .delta.negative {{ color: #9f1c11; }}
    .summary-line {{ min-height: 20px; color: #415063; }}
    .links {{ margin-bottom: 0; }}
    .badge {{ display: inline-flex; align-items: center; border-radius: 999px; padding: 3px 9px; font-weight: 700; font-size: 12px; }}
    .badge.pass {{ background: #ddf7e8; color: #116329; }}
    .badge.fail {{ background: #ffe1df; color: #9f1c11; }}
    .badge.unknown {{ background: #e9eef5; color: #415063; }}
    .chart-wrap {{ background: #fff; border: 1px solid #d7dee8; border-radius: 6px; padding: 14px; }}
    .chart {{ width: 100%; height: auto; display: block; }}
    .grid {{ stroke: #e6ebf2; stroke-width: 1; }}
    .axis-line {{ stroke: #c8d2df; stroke-width: 1; }}
    .axis {{ fill: #5f6b7a; font-size: 11px; }}
    .coverage-line {{ fill: none; stroke: #0b5cad; stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }}
    .pass-line {{ fill: none; stroke: #0f7b45; stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }}
    .coverage-dot {{ fill: #0b5cad; stroke: #fff; stroke-width: 2; }}
    .pass-dot {{ fill: #0f7b45; stroke: #fff; stroke-width: 2; }}
    .legend {{ display: flex; gap: 18px; margin-top: 8px; color: #415063; }}
    .swatch {{ display: inline-block; width: 18px; height: 3px; margin-right: 6px; vertical-align: middle; }}
    .swatch.coverage {{ background: #0b5cad; }}
    .swatch.pass {{ background: #0f7b45; }}
    .empty {{ color: #5f6b7a; margin: 0; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d7dee8; border-radius: 6px; overflow: hidden; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid #e6ebf2; text-align: left; white-space: nowrap; vertical-align: top; }}
    th {{ background: #edf2f7; color: #415063; font-size: 12px; text-transform: uppercase; }}
    tr:last-child td {{ border-bottom: 0; }}
    .table-wrap {{ overflow-x: auto; }}
    @media (max-width: 860px) {{ .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
  </style>
</head>
<body>
  <main>
    <h1>Face-Local Test Dashboard</h1>
    <p class="topline">Updated {html.escape(updated_at)}</p>

    <section class="summary">
      {''.join(platform_cards) if platform_cards else '<div class="panel">No runs recorded yet.</div>'}
    </section>

    <h2 class="section-title">Coverage And Pass Rate</h2>
    {chart}

    <h2 class="section-title">Run History</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Result</th><th>Run</th><th>Platform</th><th>Branch</th><th>Commit</th>
            <th>Coverage</th><th>Pass rate</th><th>Total</th><th>Pass</th><th>Fail</th><th>Error</th><th>Skip</th><th>Warn</th><th>Generated</th><th>Files</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows) if rows else '<tr><td colspan="15">No runs recorded yet.</td></tr>'}
        </tbody>
      </table>
    </div>
  </main>
</body>
</html>
"""


def copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", default="test-reports")
    parser.add_argument("--site-dir", default="test-reports/site")
    parser.add_argument("--max-history", type=int, default=100)
    parser.add_argument("--platform", default=os.environ.get("TEST_PLATFORM", "local"))
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    site_dir = Path(args.site_dir)
    site_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / "latest.md"
    log_path = report_dir / "latest.log"
    junit_path = report_dir / "latest.junit.xml"
    coverage_json_path = report_dir / "coverage.json"
    coverage_xml_path = report_dir / "coverage.xml"
    report_text = read_text(report_path)
    parsed = parse_report(report_text)
    coverage = parse_coverage_json(coverage_json_path)
    if coverage["percent"] is None:
        coverage = parsed["coverage"]

    sha = env("GITHUB_SHA", "local")
    run_number = env("GITHUB_RUN_NUMBER", datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"))
    run_attempt = env("GITHUB_RUN_ATTEMPT", "1")
    safe_platform = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.platform.strip()).strip("-") or "local"
    run_slug = f"{run_number}-{run_attempt}-{safe_platform}-{sha[:7]}"
    run_dir = site_dir / "runs" / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    if not report_text:
        report_text = "# Test Report\n\nNo `latest.md` report was generated.\n"
        report_path.write_text(report_text, encoding="utf-8")

    copy_if_exists(report_path, run_dir / "latest.md")
    copy_if_exists(log_path, run_dir / "latest.log")
    copy_if_exists(junit_path, run_dir / "latest.junit.xml")
    copy_if_exists(coverage_json_path, run_dir / "coverage.json")
    copy_if_exists(coverage_xml_path, run_dir / "coverage.xml")

    metadata = build_run_metadata(parsed, coverage, run_slug, args.platform)
    (run_dir / "report.html").write_text(render_report_html(report_text, metadata), encoding="utf-8")

    history_path = site_dir / "history.json"
    history = load_history(history_path)
    history = [run for run in history if run.get("key") != metadata["key"]]
    history.insert(0, metadata)
    history = history[: args.max_history]

    history_path.write_text(json.dumps({"runs": history}, indent=2) + "\n", encoding="utf-8")
    (site_dir / "index.html").write_text(render_index(history), encoding="utf-8")
    (site_dir / ".nojekyll").write_text("", encoding="utf-8")

    print(f"Dashboard written to {site_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
