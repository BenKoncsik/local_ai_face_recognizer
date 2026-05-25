#!/usr/bin/env python3
"""Create/update GitHub releases and upload release assets."""

from __future__ import annotations

import argparse
import glob
import json
import mimetypes
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

API_BASE = "https://api.github.com"
ISSUE_REF_RE = re.compile(r"(?<![\w/-])#(\d+)\b")
VERSION_TAG_RE = re.compile(r"^v\d+\.\d+\.\d+$")


@dataclass(frozen=True)
class CommitNote:
    sha: str
    subject: str
    body: str = ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    upload_parser = subparsers.add_parser("upload", help="Upload assets to a GitHub release.")
    add_shared_release_args(upload_parser)
    upload_parser.add_argument("patterns", nargs="+", help="Glob patterns for files to upload.")

    notes_parser = subparsers.add_parser("update-notes", help="Update GitHub release metadata.")
    add_shared_release_args(notes_parser)
    notes_parser.add_argument("--notes", required=True, help="Release notes body.")

    build_notes_parser = subparsers.add_parser("build-notes", help="Build release notes from Git history.")
    build_notes_parser.add_argument("--repo", required=True, help="Repository in OWNER/REPO format.")
    build_notes_parser.add_argument("--tag", required=True, help="Release tag.")
    build_notes_parser.add_argument("--target", required=True, help="Target commit SHA.")
    build_notes_parser.add_argument("--run-url", required=True, help="GitHub Actions run URL.")
    build_notes_parser.add_argument(
        "--runner-note",
        action="append",
        default=[],
        help="Runner note to include, for example 'macOS runner: Runner name'.",
    )
    build_notes_parser.add_argument(
        "--build-result",
        action="append",
        default=[],
        help="Build result to include, for example 'macOS build: success'.",
    )

    args = parser.parse_args()
    if args.command == "build-notes":
        notes = build_release_notes(
            repo=args.repo,
            tag=args.tag,
            target=args.target,
            run_url=args.run_url,
            runner_notes=args.runner_note,
            build_results=args.build_result,
        )
        print(notes)
        return 0

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        print("GITHUB_TOKEN or GH_TOKEN must be set.", file=sys.stderr)
        return 1

    client = GitHubReleaseClient(token=token, repo=args.repo)

    if args.command == "upload":
        release = client.ensure_release(
            tag=args.tag,
            target=args.target,
            name=args.name,
        )
        files = resolve_patterns(args.patterns)
        if not files:
            print("No release assets matched the provided patterns.", file=sys.stderr)
            return 1
        for file_path in files:
            client.upload_asset(release=release, file_path=file_path)
        return 0

    if args.command == "update-notes":
        release = client.ensure_release(
            tag=args.tag,
            target=args.target,
            name=args.name,
        )
        client.update_release(
            release_id=release["id"],
            name=args.name,
            notes=args.notes,
            prerelease=False,
        )
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


def add_shared_release_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", required=True, help="Repository in OWNER/REPO format.")
    parser.add_argument("--tag", required=True, help="Release tag.")
    parser.add_argument("--target", required=True, help="Target commit SHA.")
    parser.add_argument("--name", required=True, help="Release name.")


def resolve_patterns(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        for match in sorted(glob.glob(pattern)):
            path = Path(match)
            if path.is_file():
                files.append(path)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for file_path in files:
        resolved = file_path.resolve()
        if resolved not in seen:
            deduped.append(file_path)
            seen.add(resolved)
    return deduped


def build_release_notes(
    repo: str,
    tag: str,
    target: str,
    run_url: str,
    runner_notes: list[str] | None = None,
    build_results: list[str] | None = None,
) -> str:
    previous_tag = find_previous_release_tag(tag=tag, target=target)
    commits = collect_commit_notes(target=target, previous_tag=previous_tag)

    lines = [
        "Automated cross-platform build from `main`.",
        "",
        "## Release summary",
        f"- Commit: {target}",
        f"- Run: {run_url}",
    ]
    if previous_tag:
        lines.append(f"- Changes since: {previous_tag}")
    else:
        lines.append("- Changes since: first release")

    for note in runner_notes or []:
        lines.append(f"- {note}")
    for result in build_results or []:
        lines.append(f"- {result}")

    lines.extend(["", "## Elkészült jegyek"])
    issue_numbers = extract_issue_numbers_from_commits(commits)
    if issue_numbers:
        for issue_number in issue_numbers:
            lines.append(f"- {format_issue_ref(repo, issue_number)}")
    else:
        lines.append("- Nincs commit üzenetben hivatkozott jegy.")

    lines.extend(["", "## Commit kommentek"])
    if commits:
        for commit in commits:
            lines.append(format_commit_note(repo=repo, commit=commit))
    else:
        lines.append("- Nincs új commit az előző release óta.")

    return "\n".join(lines) + "\n"


def find_previous_release_tag(tag: str, target: str) -> str | None:
    try:
        tags = run_git("tag", "--merged", target, "--list", "v[0-9]*", "--sort=-v:refname")
    except RuntimeError:
        return None

    for candidate in tags.splitlines():
        candidate = candidate.strip()
        if candidate and candidate != tag and VERSION_TAG_RE.fullmatch(candidate):
            return candidate
    return None


def collect_commit_notes(target: str, previous_tag: str | None) -> list[CommitNote]:
    revision = f"{previous_tag}..{target}" if previous_tag else target
    try:
        raw = run_git("log", "--format=%H%x1f%s%x1f%b%x1e", revision)
    except RuntimeError:
        return []

    commits: list[CommitNote] = []
    for record in raw.rstrip("\x1e\n").split("\x1e"):
        record = record.strip()
        if not record:
            continue
        parts = record.split("\x1f", 2)
        if len(parts) < 2:
            continue
        sha = parts[0].strip()
        subject = normalize_commit_text(parts[1])
        body = normalize_commit_text(parts[2]) if len(parts) == 3 else ""
        commits.append(CommitNote(sha=sha, subject=subject, body=body))
    return commits


def extract_issue_numbers_from_commits(commits: list[CommitNote]) -> list[str]:
    seen: set[str] = set()
    numbers: list[str] = []
    for commit in commits:
        for issue_number in ISSUE_REF_RE.findall(f"{commit.subject}\n{commit.body}"):
            if issue_number not in seen:
                seen.add(issue_number)
                numbers.append(issue_number)
    return numbers


def format_commit_note(repo: str, commit: CommitNote) -> str:
    subject = commit.subject or "(no commit message)"
    message = subject
    if commit.body:
        message = f"{subject} - {commit.body}"
    issues = ISSUE_REF_RE.findall(f"{commit.subject}\n{commit.body}")
    issue_suffix = ""
    if issues:
        linked_issues = ", ".join(format_issue_ref(repo, issue) for issue in dict.fromkeys(issues))
        issue_suffix = f" (jegy: {linked_issues})"
    return f"- `{commit.sha[:7]}` {message}{issue_suffix}"


def format_issue_ref(repo: str, issue_number: str) -> str:
    return f"[#{issue_number}](https://github.com/{repo}/issues/{issue_number})"


def normalize_commit_text(text: str) -> str:
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def run_git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"git {' '.join(args)} failed")
    return completed.stdout


class GitHubReleaseClient:
    def __init__(self, token: str, repo: str) -> None:
        self._repo = repo
        self._headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "face-local-release-script",
        }

    def ensure_release(self, tag: str, target: str, name: str) -> dict[str, Any]:
        existing = self.get_release_by_tag(tag)
        if existing is not None:
            return existing

        payload = {
            "tag_name": tag,
            "target_commitish": target,
            "name": name,
            "body": "",
            "draft": False,
            "prerelease": False,
        }
        try:
            return self._request_json(
                "POST",
                f"/repos/{self._repo}/releases",
                payload=payload,
            )
        except urllib.error.HTTPError as exc:
            if exc.code == 422:
                existing = self.get_release_by_tag(tag)
                if existing is not None:
                    return existing
            raise

    def get_release_by_tag(self, tag: str) -> dict[str, Any] | None:
        encoded_tag = urllib.parse.quote(tag, safe="")
        try:
            return self._request_json(
                "GET",
                f"/repos/{self._repo}/releases/tags/{encoded_tag}",
            )
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise

    def update_release(self, release_id: int, name: str, notes: str, prerelease: bool) -> dict[str, Any]:
        payload = {
            "name": name,
            "body": notes,
            "prerelease": prerelease,
            "draft": False,
        }
        return self._request_json(
            "PATCH",
            f"/repos/{self._repo}/releases/{release_id}",
            payload=payload,
        )

    def upload_asset(self, release: dict[str, Any], file_path: Path) -> None:
        asset_name = file_path.name
        for asset in release.get("assets", []):
            if asset.get("name") == asset_name:
                self._request_json(
                    "DELETE",
                    f"/repos/{self._repo}/releases/assets/{asset['id']}",
                    allow_empty=True,
                )
                break

        upload_url = release["upload_url"].split("{", 1)[0]
        content_type = mimetypes.guess_type(asset_name)[0] or "application/octet-stream"
        with file_path.open("rb") as handle:
            data = handle.read()

        request = urllib.request.Request(
            url=f"{upload_url}?name={urllib.parse.quote(asset_name)}",
            data=data,
            method="POST",
            headers={
                **self._headers,
                "Content-Type": content_type,
                "Content-Length": str(len(data)),
            },
        )
        with urllib.request.urlopen(request) as response:
            response.read()

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        allow_empty: bool = False,
    ) -> dict[str, Any]:
        data = None
        headers = dict(self._headers)
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(
            url=f"{API_BASE}{path}",
            data=data,
            method=method,
            headers=headers,
        )
        with urllib.request.urlopen(request) as response:
            raw = response.read()
        if not raw:
            if allow_empty:
                return {}
            return {}
        return json.loads(raw.decode("utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
