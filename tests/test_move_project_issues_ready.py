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
