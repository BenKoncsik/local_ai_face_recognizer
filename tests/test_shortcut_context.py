"""Tests for page/context-scoped keyboard shortcuts.

The same key combination must be allowed on two different pages, while staying
unique within a single page. Runtime dispatch must only fire the shortcut whose
context matches the active page (global shortcuts fire everywhere).
"""

from __future__ import annotations

from app.services.shortcut_service import ShortcutService


def _svc() -> ShortcutService:
    # A fresh service starts from the in-code defaults (plus whatever is stored
    # locally). Tests mutate ``current_key`` in memory only, never persisting.
    return ShortcutService()


def test_context_derived_from_category():
    svc = _svc()
    assert svc.get("bbox.delete").context == "image"
    assert svc.get("collage.node_delete").context == "collage"
    assert svc.get("person.merge").context == "faces"
    assert svc.get("general.settings").context == "general"
    assert svc.get("general.settings").is_global


def test_same_key_allowed_across_pages():
    svc = _svc()
    svc.get("bbox.delete").current_key = "Del"
    svc.get("collage.node_delete").current_key = "Del"
    # Editing the image-page Del must not collide with the collage-page Del.
    assert svc.find_conflict("Del", exclude_id="bbox.delete") is None
    assert svc.find_conflict("Del", exclude_id="collage.node_delete") is None


def test_conflict_within_same_page():
    svc = _svc()
    svc.get("bbox.delete").current_key = "Del"
    svc.get("bbox.edit").current_key = "Del"
    conflict = svc.find_conflict("Del", exclude_id="bbox.delete")
    assert conflict is not None and conflict.id == "bbox.edit"


def test_global_shortcut_conflicts_across_pages():
    svc = _svc()
    # A "general" shortcut is live on every page, so a page shortcut may not
    # reuse its key.
    general_key = svc.get("general.search_focus").current_key
    svc.get("image.info").current_key = general_key
    conflict = svc.find_conflict(general_key, exclude_id="image.info")
    assert conflict is not None and conflict.id == "general.search_focus"


def test_active_context_gates_firing():
    svc = _svc()
    fired: list[str] = []
    svc.get("bbox.delete").current_key = "Del"
    svc.get("collage.node_delete").current_key = "Del"
    svc.register("bbox.delete", lambda: fired.append("image"))
    svc.register("collage.node_delete", lambda: fired.append("collage"))

    svc.set_active_context("collage")
    assert svc.dispatch("Del") is True
    assert fired == ["collage"]

    fired.clear()
    svc.set_active_context("image")
    assert svc.dispatch("Del") is True
    assert fired == ["image"]

    # A page with no matching shortcuts: nothing fires.
    fired.clear()
    svc.set_active_context("other")
    assert svc.dispatch("Del") is False
    assert fired == []


def test_no_scoping_fallback_is_context_agnostic():
    svc = _svc()
    # Before any page is selected, find_conflict with no context falls back to a
    # global search (preserves the legacy behaviour for callers without a page).
    svc.get("bbox.delete").current_key = "Del"
    svc.get("collage.node_delete").current_key = "Del"
    conflict = svc.find_conflict("Del", context=None)
    assert conflict is not None
