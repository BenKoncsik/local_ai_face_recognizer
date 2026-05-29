#!/usr/bin/env python3
"""Move GitHub issues referenced by the current push to a ProjectV2 status."""

from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

GRAPHQL_URL = "https://api.github.com/graphql"
ZERO_SHA_RE = re.compile(r"^0+$")
ISSUE_RE = re.compile(r"(?<![\w/])#(\d+)\b")


class GitHubError(RuntimeError):
    pass


def warn(message: str) -> None:
    print(f"::warning::{message}")


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def truthy_env(name: str) -> bool:
    return env(name).lower() in {"1", "true", "yes", "on"}


def graphql(query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
    token = env("GITHUB_TOKEN") or env("GH_TOKEN")
    if not token:
        raise GitHubError("GITHUB_TOKEN or GH_TOKEN is required.")

    payload = json.dumps({"query": query, "variables": variables or {}}).encode()
    request = urllib.request.Request(
        GRAPHQL_URL,
        data=payload,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise GitHubError(f"GitHub GraphQL request failed: HTTP {exc.code}: {detail}") from exc

    if body.get("errors"):
        raise GitHubError(json.dumps(body["errors"], indent=2))
    return body["data"]


def normalize_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def load_event() -> dict[str, Any]:
    event_path = env("GITHUB_EVENT_PATH")
    if not event_path:
        return {}
    path = Path(event_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def collect_messages_from_event(event: dict[str, Any]) -> list[str]:
    messages: list[str] = []
    for commit in event.get("commits") or []:
        message = commit.get("message")
        if message:
            messages.append(str(message))
    head_message = (event.get("head_commit") or {}).get("message")
    if head_message:
        messages.append(str(head_message))
    return messages


def git_log_messages(before: str, after: str) -> list[str]:
    if not after:
        return []

    if before and not ZERO_SHA_RE.fullmatch(before):
        revision = f"{before}..{after}"
        command = ["git", "log", "--format=%B%x1e", revision]
    else:
        command = ["git", "log", "--format=%B%x1e", "-n", "1", after]

    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        warn(f"Could not read commit messages with git log: {exc}")
        return []

    return [message.strip() for message in output.split("\x1e") if message.strip()]


def referenced_issue_numbers() -> list[int]:
    event = load_event()
    messages = collect_messages_from_event(event)
    if not messages:
        before = str(event.get("before") or env("GITHUB_EVENT_BEFORE"))
        after = str(event.get("after") or env("GITHUB_SHA"))
        messages = git_log_messages(before, after)

    numbers = sorted({int(match) for message in messages for match in ISSUE_RE.findall(message)})
    if numbers:
        print(f"Referenced issues in this workflow run: {', '.join(f'#{n}' for n in numbers)}")
    else:
        print("No issue references found in this workflow run.")
    return numbers


def resolve_owner_type(owner: str) -> str:
    data = graphql(
        """
        query($owner: String!) {
          repositoryOwner(login: $owner) {
            __typename
            login
          }
        }
        """,
        {"owner": owner},
    )
    repository_owner = data.get("repositoryOwner")
    if repository_owner:
        owner_type = repository_owner.get("__typename")
        if owner_type in {"Organization", "User"}:
            return owner_type
    raise GitHubError(f"Owner '{owner}' was not found.")


def query_owner_projects(owner: str, owner_type: str) -> list[dict[str, Any]]:
    field = "organization" if owner_type == "Organization" else "user"
    data = graphql(
        f"""
        query($owner: String!) {{
          {field}(login: $owner) {{
            projectsV2(first: 100) {{
              nodes {{
                id
                number
                title
                closed
              }}
            }}
          }}
        }}
        """,
        {"owner": owner},
    )
    return data[field]["projectsV2"]["nodes"]


def query_project_by_number(owner: str, owner_type: str, number: int) -> dict[str, Any]:
    field = "organization" if owner_type == "Organization" else "user"
    data = graphql(
        f"""
        query($owner: String!, $number: Int!) {{
          {field}(login: $owner) {{
            projectV2(number: $number) {{
              id
              number
              title
              closed
            }}
          }}
        }}
        """,
        {"owner": owner, "number": number},
    )
    project = data[field]["projectV2"]
    if not project:
        raise GitHubError(f"Project #{number} was not found for {owner}.")
    return project


def resolve_project(project_owner: str, repo: str) -> dict[str, Any]:
    owner_type = resolve_owner_type(project_owner)
    project_number = env("PROJECT_NUMBER")
    if project_number:
        return query_project_by_number(project_owner, owner_type, int(project_number))

    configured_title = env("PROJECT_TITLE")
    title_candidates = [configured_title] if configured_title else [repo.replace("_", " ").replace("-", " ")]
    normalized_candidates = {normalize_title(title) for title in title_candidates if title}
    projects = [project for project in query_owner_projects(project_owner, owner_type) if not project.get("closed")]

    for project in projects:
        if normalize_title(project["title"]) in normalized_candidates:
            return project

    available = ", ".join(f"#{project['number']} {project['title']}" for project in projects) or "none"
    raise GitHubError(
        "Could not find the target project. "
        f"Set PROJECT_NUMBER or PROJECT_TITLE. Available open projects: {available}"
    )


def resolve_status_field(project_id: str) -> tuple[str, str]:
    status_field_name = env("PROJECT_STATUS_FIELD", "Status")
    ready_option_name = env("PROJECT_READY_OPTION", "Ready")
    data = graphql(
        """
        query($projectId: ID!) {
          node(id: $projectId) {
            ... on ProjectV2 {
              fields(first: 100) {
                nodes {
                  ... on ProjectV2SingleSelectField {
                    id
                    name
                    options {
                      id
                      name
                    }
                  }
                }
              }
            }
          }
        }
        """,
        {"projectId": project_id},
    )

    fields = data["node"]["fields"]["nodes"]
    status_field = next(
        (field for field in fields if field and normalize_title(field["name"]) == normalize_title(status_field_name)),
        None,
    )
    if not status_field:
        raise GitHubError(f"Project field '{status_field_name}' was not found.")

    ready_option = next(
        (
            option
            for option in status_field["options"]
            if normalize_title(option["name"]) == normalize_title(ready_option_name)
        ),
        None,
    )
    if not ready_option:
        options = ", ".join(option["name"] for option in status_field["options"])
        raise GitHubError(
            f"Project status option '{ready_option_name}' was not found. Available options: {options}"
        )

    return status_field["id"], ready_option["id"]


def query_issue(owner: str, repo: str, number: int) -> dict[str, Any] | None:
    data = graphql(
        """
        query($owner: String!, $repo: String!, $number: Int!) {
          repository(owner: $owner, name: $repo) {
            issue(number: $number) {
              id
              number
              title
              url
              projectItems(first: 100) {
                nodes {
                  id
                  project {
                    id
                  }
                }
              }
            }
          }
        }
        """,
        {"owner": owner, "repo": repo, "number": number},
    )
    return data["repository"]["issue"]


def add_issue_to_project(project_id: str, issue_id: str) -> str:
    data = graphql(
        """
        mutation($projectId: ID!, $contentId: ID!) {
          addProjectV2ItemById(input: {projectId: $projectId, contentId: $contentId}) {
            item {
              id
            }
          }
        }
        """,
        {"projectId": project_id, "contentId": issue_id},
    )
    return data["addProjectV2ItemById"]["item"]["id"]


def set_project_status(project_id: str, item_id: str, field_id: str, option_id: str) -> None:
    graphql(
        """
        mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $optionId: String!) {
          updateProjectV2ItemFieldValue(
            input: {
              projectId: $projectId
              itemId: $itemId
              fieldId: $fieldId
              value: {singleSelectOptionId: $optionId}
            }
          ) {
            projectV2Item {
              id
            }
          }
        }
        """,
        {
            "projectId": project_id,
            "itemId": item_id,
            "fieldId": field_id,
            "optionId": option_id,
        },
    )


def main() -> int:
    token = env("GITHUB_TOKEN") or env("GH_TOKEN")
    if not token:
        warn("PROJECTS_TOKEN is not configured; skipping project status update.")
        return 0

    repository = env("GITHUB_REPOSITORY")
    if "/" not in repository:
        raise GitHubError("GITHUB_REPOSITORY must be set to owner/repo.")
    repo_owner, repo = repository.split("/", 1)
    project_owner = env("PROJECT_OWNER", repo_owner)

    numbers = referenced_issue_numbers()
    if not numbers:
        return 0

    project = resolve_project(project_owner, repo)
    print(f"Target project: {project_owner}/#{project['number']} {project['title']}")
    field_id, option_id = resolve_status_field(project["id"])

    moved: list[str] = []
    skipped: list[str] = []
    for number in numbers:
        issue = query_issue(repo_owner, repo, number)
        if not issue:
            skipped.append(f"#{number} (not an issue in this repository)")
            continue

        item = next(
            (
                project_item
                for project_item in issue["projectItems"]["nodes"]
                if project_item["project"]["id"] == project["id"]
            ),
            None,
        )
        item_id = item["id"] if item else add_issue_to_project(project["id"], issue["id"])
        set_project_status(project["id"], item_id, field_id, option_id)
        moved.append(f"#{issue['number']}")

    if moved:
        print(f"Moved to {env('PROJECT_READY_OPTION', 'Ready')}: {', '.join(moved)}")
    if skipped:
        warn(f"Skipped: {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GitHubError as exc:
        if truthy_env("PROJECT_ISSUE_MOVE_BEST_EFFORT"):
            target_status = env("PROJECT_READY_OPTION", "Ready")
            warn(f"Could not move referenced issues to {target_status}; continuing. {exc}")
            raise SystemExit(0)
        print(f"::error::{exc}")
        raise SystemExit(1)
