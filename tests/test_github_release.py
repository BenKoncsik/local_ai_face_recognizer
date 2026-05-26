from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import github_release


def test_build_release_notes_includes_issue_links_and_commit_comments(monkeypatch) -> None:
    commits = [
        github_release.CommitNote(
            sha="177fffa1234567890",
            subject="Fix export crash #35",
            body="Also closes #33.",
        ),
        github_release.CommitNote(
            sha="abb5b4a1234567890",
            subject="Polish release workflow #35",
        ),
    ]

    monkeypatch.setattr(github_release, "find_previous_release_tag", lambda repo, tag, target: "v1.2.2")
    monkeypatch.setattr(github_release, "collect_commit_notes", lambda target, previous_tag: commits)

    notes = github_release.build_release_notes(
        repo="example/face-local",
        tag="v1.2.3",
        target="abc123",
        run_url="https://github.com/example/face-local/actions/runs/123",
        runner_notes=["macOS runner: Runner: Mac"],
        build_results=["macOS build: success"],
    )

    assert "## Elkészült jegyek" in notes
    assert "[#35](https://github.com/example/face-local/issues/35)" in notes
    assert "[#33](https://github.com/example/face-local/issues/33)" in notes
    assert "## Commit kommentek" in notes
    assert "`177fffa` Fix export crash #35 - Also closes #33." in notes
    assert "`abb5b4a` Polish release workflow #35" in notes
    assert "- Changes since: v1.2.2" in notes
    assert "- macOS build: success" in notes


def test_build_release_notes_handles_no_issue_references(monkeypatch) -> None:
    commits = [
        github_release.CommitNote(
            sha="90c6b661234567890",
            subject="Update README",
        ),
    ]

    monkeypatch.setattr(github_release, "find_previous_release_tag", lambda repo, tag, target: None)
    monkeypatch.setattr(github_release, "collect_commit_notes", lambda target, previous_tag: commits)

    notes = github_release.build_release_notes(
        repo="example/face-local",
        tag="v1.2.3",
        target="abc123",
        run_url="https://github.com/example/face-local/actions/runs/123",
    )

    assert "- Changes since: first release" in notes
    assert "- Nincs commit üzenetben hivatkozott jegy." in notes
    assert "`90c6b66` Update README" in notes


def test_find_previous_release_tag_prefers_latest_github_release(monkeypatch) -> None:
    monkeypatch.setattr(
        github_release,
        "github_api_get_json",
        lambda path: [
            {"tag_name": "v1.2.3", "draft": False, "prerelease": False},
            {"tag_name": "v1.2.2", "draft": False, "prerelease": False},
            {"tag_name": "v1.2.1", "draft": False, "prerelease": False},
        ],
    )
    monkeypatch.setattr(github_release, "is_ref_ancestor", lambda ref, target: ref == "v1.2.2")

    previous_tag = github_release.find_previous_release_tag(
        repo="example/face-local",
        tag="v1.2.3",
        target="abc123",
    )

    assert previous_tag == "v1.2.2"


def test_collect_commit_notes_reads_all_commits_since_previous_release(monkeypatch) -> None:
    calls = []

    def fake_run_git(*args: str) -> str:
        calls.append(args)
        return (
            "1111111111111111111111111111111111111111\x1fFirst pushed commit #1\x1f\x1e"
            "2222222222222222222222222222222222222222\x1fSecond pushed commit #2\x1fBody\x1e"
        )

    monkeypatch.setattr(github_release, "run_git", fake_run_git)

    commits = github_release.collect_commit_notes(target="targetsha", previous_tag="v1.2.2")

    assert calls == [("log", "--format=%H%x1f%s%x1f%b%x1e", "v1.2.2..targetsha")]
    assert [commit.subject for commit in commits] == ["First pushed commit #1", "Second pushed commit #2"]
