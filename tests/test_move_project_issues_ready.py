from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "move_project_issues_ready.py"
SPEC = importlib.util.spec_from_file_location("move_project_issues_ready", SCRIPT_PATH)
assert SPEC and SPEC.loader
move_project_issues_ready = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(move_project_issues_ready)


def test_resolve_owner_type_uses_repository_owner_for_organizations(monkeypatch) -> None:
    queries: list[str] = []

    def fake_graphql(query: str, variables: dict[str, object]) -> dict[str, object]:
        queries.append(query)
        assert variables == {"owner": "HunKonTech"}
        return {
            "repositoryOwner": {
                "__typename": "Organization",
                "login": "HunKonTech",
            }
        }

    monkeypatch.setattr(move_project_issues_ready, "graphql", fake_graphql)

    assert move_project_issues_ready.resolve_owner_type("HunKonTech") == "Organization"
    assert "repositoryOwner(login: $owner)" in queries[0]
    assert "user(login: $owner)" not in queries[0]


def test_resolve_owner_type_uses_repository_owner_for_users(monkeypatch) -> None:
    def fake_graphql(query: str, variables: dict[str, object]) -> dict[str, object]:
        return {
            "repositoryOwner": {
                "__typename": "User",
                "login": variables["owner"],
            }
        }

    monkeypatch.setattr(move_project_issues_ready, "graphql", fake_graphql)

    assert move_project_issues_ready.resolve_owner_type("octocat") == "User"


def test_resolve_status_options_returns_requested_option_ids(monkeypatch) -> None:
    def fake_graphql(query: str, variables: dict[str, object]) -> dict[str, object]:
        assert variables == {"projectId": "project-id"}
        return {
            "node": {
                "fields": {
                    "nodes": [
                        {
                            "id": "status-field-id",
                            "name": "Status",
                            "options": [
                                {"id": "committed-id", "name": "Commitolt"},
                                {"id": "review-id", "name": "In review"},
                                {"id": "done-id", "name": "Done"},
                            ],
                        }
                    ]
                }
            }
        }

    monkeypatch.setattr(move_project_issues_ready, "graphql", fake_graphql)

    field_id, option_ids = move_project_issues_ready.resolve_status_options(
        "project-id",
        ["Commitolt", "In review"],
    )

    assert field_id == "status-field-id"
    assert option_ids == {
        "Commitolt": "committed-id",
        "In review": "review-id",
    }


def test_move_project_issues_by_status_moves_only_issue_items_with_source_status(monkeypatch) -> None:
    items = [
        {
            "id": "item-1",
            "content": {"__typename": "Issue", "number": 12},
            "fieldValues": {
                "nodes": [
                    {
                        "field": {"id": "status-field-id"},
                        "optionId": "review-id",
                    }
                ]
            },
        },
        {
            "id": "item-2",
            "content": {"__typename": "Issue", "number": 13},
            "fieldValues": {
                "nodes": [
                    {
                        "field": {"id": "status-field-id"},
                        "optionId": "committed-id",
                    }
                ]
            },
        },
        {
            "id": "item-3",
            "content": {"__typename": "PullRequest", "number": 14},
            "fieldValues": {
                "nodes": [
                    {
                        "field": {"id": "status-field-id"},
                        "optionId": "review-id",
                    }
                ]
            },
        },
    ]
    updates: list[tuple[str, str]] = []

    monkeypatch.setattr(move_project_issues_ready, "query_project_items", lambda project_id: items)
    monkeypatch.setattr(
        move_project_issues_ready,
        "set_project_status",
        lambda project_id, item_id, field_id, option_id: updates.append((item_id, option_id)),
    )

    moved = move_project_issues_ready.move_project_issues_by_status(
        "project-id",
        "status-field-id",
        "review-id",
        "done-id",
    )

    assert moved == ["#12"]
    assert updates == [("item-1", "done-id")]
