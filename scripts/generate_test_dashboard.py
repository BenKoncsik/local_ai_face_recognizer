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

    for line in report_text.splitlines():
        if line.startswith("- Result:"):
            result = line.split(":", 1)[1].strip()
        elif line.startswith("- Pytest summary:"):
            summary = line.split(":", 1)[1].strip().strip("`")
        else:
            match = re.match(r"- (Passed|Failed|Errors|Skipped|Warnings):\s+(\d+)", line)
            if match:
                counts[match.group(1).lower()] = int(match.group(2))

    return {"result": result, "summary": summary, "counts": counts}


def load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict) and isinstance(payload.get("runs"), list):
        return [item for item in payload["runs"] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def build_run_metadata(parsed: dict[str, Any], run_slug: str) -> dict[str, Any]:
    repository = env("GITHUB_REPOSITORY", "local")
    server_url = env("GITHUB_SERVER_URL", "https://github.com")
    run_id = env("GITHUB_RUN_ID", f"local-{run_slug}")
    run_attempt = env("GITHUB_RUN_ATTEMPT", "1")
    run_number = env("GITHUB_RUN_NUMBER", "0")
    sha = env("GITHUB_SHA", "")
    ref_name = env("GITHUB_REF_NAME", "")

    return {
        "key": f"{run_id}-{run_attempt}",
        "run_id": run_id,
        "run_attempt": run_attempt,
        "run_number": run_number,
        "workflow": env("GITHUB_WORKFLOW", "Local test run"),
        "event": env("GITHUB_EVENT_NAME", "local"),
        "actor": env("GITHUB_ACTOR", ""),
        "repository": repository,
        "branch": ref_name,
        "sha": sha,
        "short_sha": sha[:7],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "result": parsed["result"],
        "summary": parsed["summary"],
        "counts": parsed["counts"],
        "run_url": f"{server_url}/{repository}/actions/runs/{run_id}" if run_id.isdigit() else "",
        "commit_url": f"{server_url}/{repository}/commit/{sha}" if sha else "",
        "report_url": f"runs/{run_slug}/report.html",
        "markdown_url": f"runs/{run_slug}/latest.md",
        "log_url": f"runs/{run_slug}/latest.log",
        "junit_url": f"runs/{run_slug}/latest.junit.xml",
    }


def status_class(result: str) -> str:
    return "pass" if result == "PASS" else "fail" if result.startswith("FAIL") else "unknown"


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
    latest = history[0] if history else {}
    counts = latest.get("counts", {})
    rows = []
    for run in history:
        result = str(run.get("result", "UNKNOWN"))
        counts_for_run = run.get("counts", {})
        run_label = f"#{run.get('run_number', '?')}"
        if run.get("run_attempt") and run.get("run_attempt") != "1":
            run_label += f".{run['run_attempt']}"
        run_link = html.escape(run.get("run_url") or run.get("report_url", "#"))
        commit = html.escape(run.get("short_sha") or "")
        commit_link = html.escape(run.get("commit_url") or "#")
        rows.append(
            "<tr>"
            f"<td><span class=\"badge {status_class(result)}\">{html.escape(result)}</span></td>"
            f"<td><a href=\"{run_link}\">{html.escape(run_label)}</a></td>"
            f"<td>{html.escape(run.get('branch', ''))}</td>"
            f"<td><a href=\"{commit_link}\">{commit}</a></td>"
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

    latest_result = str(latest.get("result", "UNKNOWN"))
    latest_summary = str(latest.get("summary", "No test run has been recorded yet."))
    latest_report = str(latest.get("report_url", "#"))
    updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

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
    h2 {{ margin: 28px 0 10px; font-size: 18px; letter-spacing: 0; }}
    a {{ color: #0b5cad; }}
    .topline {{ color: #5f6b7a; margin: 0 0 24px; }}
    .summary {{ display: grid; grid-template-columns: 1.4fr repeat(5, minmax(96px, 1fr)); gap: 10px; }}
    .panel {{ background: #fff; border: 1px solid #d7dee8; border-radius: 6px; padding: 14px; }}
    .metric {{ font-size: 24px; font-weight: 700; }}
    .label {{ color: #5f6b7a; font-size: 12px; text-transform: uppercase; }}
    .badge {{ display: inline-flex; align-items: center; border-radius: 999px; padding: 3px 9px; font-weight: 700; font-size: 12px; }}
    .badge.pass {{ background: #ddf7e8; color: #116329; }}
    .badge.fail {{ background: #ffe1df; color: #9f1c11; }}
    .badge.unknown {{ background: #e9eef5; color: #415063; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d7dee8; border-radius: 6px; overflow: hidden; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid #e6ebf2; text-align: left; white-space: nowrap; }}
    th {{ background: #edf2f7; color: #415063; font-size: 12px; text-transform: uppercase; }}
    tr:last-child td {{ border-bottom: 0; }}
    .table-wrap {{ overflow-x: auto; }}
    @media (max-width: 860px) {{ .summary {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
  </style>
</head>
<body>
  <main>
    <h1>Face-Local Test Dashboard</h1>
    <p class="topline">Updated {html.escape(updated_at)}</p>

    <section class="summary">
      <div class="panel">
        <div class="label">Latest result</div>
        <p><span class="badge {status_class(latest_result)}">{html.escape(latest_result)}</span></p>
        <p>{html.escape(latest_summary)}</p>
        <p><a href="{html.escape(latest_report)}">Open latest report</a></p>
      </div>
      <div class="panel"><div class="label">Passed</div><div class="metric">{counts.get('passed', 0)}</div></div>
      <div class="panel"><div class="label">Failed</div><div class="metric">{counts.get('failed', 0)}</div></div>
      <div class="panel"><div class="label">Errors</div><div class="metric">{counts.get('errors', 0)}</div></div>
      <div class="panel"><div class="label">Skipped</div><div class="metric">{counts.get('skipped', 0)}</div></div>
      <div class="panel"><div class="label">Warnings</div><div class="metric">{counts.get('warnings', 0)}</div></div>
    </section>

    <h2>Run History</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Result</th><th>Run</th><th>Branch</th><th>Commit</th>
            <th>Pass</th><th>Fail</th><th>Error</th><th>Skip</th><th>Warn</th><th>Generated</th><th>Files</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows) if rows else '<tr><td colspan="11">No runs recorded yet.</td></tr>'}
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
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    site_dir = Path(args.site_dir)
    site_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / "latest.md"
    log_path = report_dir / "latest.log"
    junit_path = report_dir / "latest.junit.xml"
    report_text = read_text(report_path)
    parsed = parse_report(report_text)

    sha = env("GITHUB_SHA", "local")
    run_number = env("GITHUB_RUN_NUMBER", datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"))
    run_attempt = env("GITHUB_RUN_ATTEMPT", "1")
    run_slug = f"{run_number}-{run_attempt}-{sha[:7]}"
    run_dir = site_dir / "runs" / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    if not report_text:
        report_text = "# Test Report\n\nNo `latest.md` report was generated.\n"
        report_path.write_text(report_text, encoding="utf-8")

    copy_if_exists(report_path, run_dir / "latest.md")
    copy_if_exists(log_path, run_dir / "latest.log")
    copy_if_exists(junit_path, run_dir / "latest.junit.xml")

    metadata = build_run_metadata(parsed, run_slug)
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
