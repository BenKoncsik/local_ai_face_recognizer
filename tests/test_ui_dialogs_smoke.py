"""Parametrized smoke tests for every app.ui.dialogs.* module.

Each dialog (or its simplest constructible widget class) is imported and
instantiated with minimal mocked / empty fixtures so the UI layer can be
exercised without manual clicking.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Callable

import pytest
from PySide6.QtWidgets import QWidget
from sqlalchemy.orm import selectinload

from app.config import AppConfig
from app.db.database import init_db, session_scope
from app.db.models import Person, Place, TaggedObject
from app.services.duplicate_unknown_face_finder import OverlappingUnknownFaceMatch
from app.services.face_diagnostics_service import FaceDiagnostics
from app.services.update_service import ReleaseInfo
from app.services.family_code_schemes import FamilyCodeSchemeStore


@pytest.fixture()
def db(tmp_path):
    init_db(tmp_path / "ui_dialogs_smoke.db")


@pytest.fixture()
def app_config() -> AppConfig:
    return AppConfig()


def _noop(*_args, **_kwargs) -> None:
    pass


def _add_person(name: str, *, auto: bool = False) -> int:
    with session_scope() as session:
        person = Person(name=name, is_auto_named=auto)
        session.add(person)
        session.flush()
        return person.id


def _add_place(name: str) -> Place:
    with session_scope() as session:
        place = Place(name=name)
        session.add(place)
        session.flush()
        session.refresh(place)
        return place


def _add_tagged_object(name: str) -> int:
    with session_scope() as session:
        obj = TaggedObject(name=name)
        session.add(obj)
        session.flush()
        return obj.id


def _minimal_face_diagnostics() -> FaceDiagnostics:
    return FaceDiagnostics(
        face_id=1,
        has_embedding=False,
        current_person_id=None,
        current_person_name=None,
        assignment_source=None,
        assignment_confidence=None,
        quality_score=None,
        is_low_quality=None,
        bbox=(0, 0, 10, 10),
        adaptive_threshold=0.5,
        base_threshold=0.4,
        margin_required=0.05,
        verdict="smoke test",
    )


def _minimal_release() -> ReleaseInfo:
    return ReleaseInfo(
        version="9.9.9",
        tag="v9.9.9",
        url="https://example.com",
        asset_name="face-local.zip",
        asset_url="https://example.com/asset.zip",
        asset_size=1024,
    )


DialogFactory = Callable[[AppConfig], QWidget]

_DIALOG_FACTORIES: dict[str, DialogFactory] = {}


def _register(module: str, factory: DialogFactory) -> None:
    _DIALOG_FACTORIES[module] = factory


_register(
    "ai_visualization_window",
    lambda _cfg: __import__(
        "app.ui.dialogs.ai_visualization_window",
        fromlist=["AIVisualizationWindow"],
    ).AIVisualizationWindow(),
)

_register(
    "auto_assignments_tab",
    lambda _cfg: __import__(
        "app.ui.dialogs.auto_assignments_tab",
        fromlist=["AutoAssignmentsTab"],
    ).AutoAssignmentsTab(),
)

_register(
    "auto_merge_review_dialog",
    lambda _cfg: __import__(
        "app.ui.dialogs.auto_merge_review_dialog",
        fromlist=["AutoMergeReviewDialog"],
    ).AutoMergeReviewDialog(),
)

_register(
    "collage_node_dialog",
    lambda _cfg: __import__(
        "app.ui.dialogs.collage_node_dialog",
        fromlist=["CollageNodeDialog"],
    ).CollageNodeDialog(1, ["<b>smoke</b> test"]),
)

_register(
    "export_dialog",
    lambda _cfg: __import__(
        "app.ui.dialogs.export_dialog",
        fromlist=["ExportDialog"],
    ).ExportDialog(),
)

_register(
    "face_diagnostics_dialog",
    lambda _cfg: __import__(
        "app.ui.dialogs.face_diagnostics_dialog",
        fromlist=["FaceDiagnosticsDialog"],
    ).FaceDiagnosticsDialog(_minimal_face_diagnostics()),
)

def _make_family_code_scheme_dialog(cfg: AppConfig) -> QWidget:
    from pathlib import Path

    store_dir = Path(cfg.storage.db_path).parent / "schemes"
    return __import__(
        "app.ui.dialogs.family_code_scheme_dialog",
        fromlist=["FamilyCodeSchemeDialog"],
    ).FamilyCodeSchemeDialog(store=FamilyCodeSchemeStore(store_dir))


_register("family_code_scheme_dialog", _make_family_code_scheme_dialog)

_register(
    "gdrive_settings_tab",
    lambda _cfg: __import__(
        "app.ui.dialogs.gdrive_settings_tab",
        fromlist=["GDriveSettingsTab"],
    ).GDriveSettingsTab(),
)

_register(
    "group_manager_dialog",
    lambda _cfg: __import__(
        "app.ui.dialogs.group_manager_dialog",
        fromlist=["GroupManagerDialog"],
    ).GroupManagerDialog(),
)

_register(
    "identity_repair_dialog",
    lambda _cfg: __import__(
        "app.ui.dialogs.identity_repair_dialog",
        fromlist=["IdentityRepairDialog"],
    ).IdentityRepairDialog([]),
)

_register(
    "ignored_faces_dialog",
    lambda cfg: __import__(
        "app.ui.dialogs.ignored_faces_dialog",
        fromlist=["IgnoredFacesDialog"],
    ).IgnoredFacesDialog(cfg),
)

_register(
    "image_library_dialog",
    lambda _cfg: __import__(
        "app.ui.dialogs.image_library_dialog",
        fromlist=["ImageLibraryMissingDialog"],
    ).ImageLibraryMissingDialog("/tmp/missing-library-root"),
)


def _make_manual_face_dialog(cfg: AppConfig) -> QWidget:
    mod = __import__(
        "app.ui.dialogs.manual_face_dialog",
        fromlist=["NoFaceImagesDialog"],
    )
    return mod.NoFaceImagesDialog(cfg)


_register("manual_face_dialog", _make_manual_face_dialog)

_register(
    "merge_decision_graph_dialog",
    lambda _cfg: __import__(
        "app.ui.dialogs.merge_decision_graph_dialog",
        fromlist=["MergeDecisionGraphDialog"],
    ).MergeDecisionGraphDialog(None),
)


def _make_merge_dialog(_cfg: AppConfig) -> QWidget:
    mod = __import__("app.ui.dialogs.merge_dialog", fromlist=["MergeDialog"])
    alice_id = _add_person("Alice")
    bob_id = _add_person("Bob")
    with session_scope() as session:
        source = session.get(Person, alice_id)
        persons = (
            session.query(Person)
            .options(selectinload(Person.faces))
            .filter(Person.id.in_([alice_id, bob_id]))
            .all()
        )
        return mod.MergeDialog(source, persons)


_register("merge_dialog", _make_merge_dialog)


def _make_move_faces_dialog(_cfg: AppConfig) -> QWidget:
    mod = __import__(
        "app.ui.dialogs.move_faces_dialog",
        fromlist=["MoveFacesDialog"],
    )
    with session_scope() as session:
        persons = session.query(Person).all()
    return mod.MoveFacesDialog(1, persons)


_register("move_faces_dialog", _make_move_faces_dialog)

_register(
    "object_info_dialog",
    lambda _cfg: __import__(
        "app.ui.dialogs.object_info_dialog",
        fromlist=["ObjectInfoDialog"],
    ).ObjectInfoDialog(_add_tagged_object("smoke-object")),
)

_register(
    "object_merge_dialog",
    lambda _cfg: __import__(
        "app.ui.dialogs.object_merge_dialog",
        fromlist=["ObjectMergeDialog"],
    ).ObjectMergeDialog([]),
)

_register(
    "object_picker_dialog",
    lambda _cfg: __import__(
        "app.ui.dialogs.object_picker_dialog",
        fromlist=["ObjectPickerDialog"],
    ).ObjectPickerDialog(),
)

_register(
    "overlapping_unknown_faces_dialog",
    lambda _cfg: __import__(
        "app.ui.dialogs.overlapping_unknown_faces_dialog",
        fromlist=["OverlappingUnknownFacesDialog"],
    ).OverlappingUnknownFacesDialog([], images_examined=0),
)


def _make_person_info_dialog(_cfg: AppConfig) -> QWidget:
    mod = __import__(
        "app.ui.dialogs.person_info_dialog",
        fromlist=["PersonInfoDialog"],
    )
    person_id = _add_person("Smoke Person")
    with session_scope() as session:
        person = session.get(Person, person_id)
        return mod.PersonInfoDialog(person)


_register("person_info_dialog", _make_person_info_dialog)

_register(
    "place_edit_dialog",
    lambda _cfg: __import__(
        "app.ui.dialogs.place_edit_dialog",
        fromlist=["PlaceEditDialog"],
    ).PlaceEditDialog(),
)


def _make_place_merge_dialog(_cfg: AppConfig) -> QWidget:
    mod = __import__(
        "app.ui.dialogs.place_merge_dialog",
        fromlist=["PlaceMergeDialog"],
    )
    place = _add_place("Balaton")
    return mod.PlaceMergeDialog([place], target_id=place.id)


_register("place_merge_dialog", _make_place_merge_dialog)

_register(
    "rerecognition_history_dialog",
    lambda _cfg: __import__(
        "app.ui.dialogs.rerecognition_history_dialog",
        fromlist=["ReRecognitionHistoryDialog"],
    ).ReRecognitionHistoryDialog(),
)

_register(
    "rerecognition_review_dialog",
    lambda _cfg: __import__(
        "app.ui.dialogs.rerecognition_review_dialog",
        fromlist=["ReRecognitionReviewDialog"],
    ).ReRecognitionReviewDialog([], batch_id="smoke-batch", persons=[]),
)

_register(
    "rename_dialog",
    lambda _cfg: __import__(
        "app.ui.dialogs.rename_dialog",
        fromlist=["RenameDialog"],
    ).RenameDialog("Old Name"),
)


def _make_scan_modes_dialog(_cfg: AppConfig) -> QWidget:
    mod = __import__(
        "app.ui.dialogs.scan_modes_dialog",
        fromlist=["ScanModesDialog"],
    )
    return mod.ScanModesDialog(
        on_incremental=_noop,
        on_full_rescan=_noop,
        on_face_rescan_fast=_noop,
        on_face_rescan_accurate=_noop,
        on_reset_unknown_persons=_noop,
        on_find_overlapping_unknown_faces=_noop,
    )


_register("scan_modes_dialog", _make_scan_modes_dialog)

_register(
    "settings_dialog",
    lambda cfg: __import__(
        "app.ui.dialogs.settings_dialog",
        fromlist=["SettingsDialog"],
    ).SettingsDialog(cfg.storage.db_path, app_config=cfg),
)

_register(
    "shortcuts_settings_tab",
    lambda _cfg: __import__(
        "app.ui.dialogs.shortcuts_settings_tab",
        fromlist=["ShortcutsSettingsTab"],
    ).ShortcutsSettingsTab(),
)

_register(
    "suggestion_dialog",
    lambda cfg: __import__(
        "app.ui.dialogs.suggestion_dialog",
        fromlist=["SuggestionDialog"],
    ).SuggestionDialog(cfg),
)


def _make_suggestion_viewer(_cfg: AppConfig) -> QWidget:
    mod = __import__(
        "app.ui.dialogs.suggestion_viewer",
        fromlist=["FaceGalleryDialog"],
    )
    person_id = _add_person("Gallery Person")
    return mod.FaceGalleryDialog(person_id, "Gallery Person")


_register("suggestion_viewer", _make_suggestion_viewer)

_register(
    "task_manager_dialog",
    lambda _cfg: __import__(
        "app.ui.dialogs.task_manager_dialog",
        fromlist=["TaskManagerDialog"],
    ).TaskManagerDialog(),
)

_register(
    "tpu_status_dialog",
    lambda _cfg: __import__(
        "app.ui.dialogs.tpu_status_dialog",
        fromlist=["TpuStatusDialog"],
    ).TpuStatusDialog(),
)

_register(
    "update_dialog",
    lambda _cfg: __import__(
        "app.ui.dialogs.update_dialog",
        fromlist=["UpdateDialog"],
    ).UpdateDialog(_minimal_release()),
)


@pytest.mark.parametrize(
    "module_name",
    sorted(_DIALOG_FACTORIES.keys()),
    ids=sorted(_DIALOG_FACTORIES.keys()),
)
def test_dialog_module_smoke(module_name: str, qtbot, db, app_config, monkeypatch):
    if module_name == "settings_dialog":
        # Background probe threads + a long test run can destabilise Qt/WebEngine.
        def _track_without_start(thread):
            thread.start = lambda: None  # type: ignore[method-assign]
            return thread

        monkeypatch.setattr(
            "app.ui.dialogs.settings_dialog._track_thread",
            _track_without_start,
        )

    factory = _DIALOG_FACTORIES[module_name]
    widget = factory(app_config)
    qtbot.addWidget(widget)
    assert widget is not None
    widget.close()


def test_overlapping_match_dataclass_importable():
    """Sanity check for the overlapping-dialog helper type."""
    match = OverlappingUnknownFaceMatch(
        image_id=1,
        image_path="/tmp/a.jpg",
        image_relative_path=None,
        unknown_face_id=2,
        known_face_id=3,
        known_person_name="Anna",
        overlap=0.9,
        unknown_bbox=(0, 0, 10, 10),
        known_bbox=(0, 0, 20, 20),
    )
    assert match.display_path == "/tmp/a.jpg"


def test_merge_decision_dto_importable():
    from app.services.merge_suggestion_service import MergeDecisionDTO

    dto = MergeDecisionDTO(
        decision_id=1,
        candidate_name="Unknown 1",
        target_name="Anna",
        candidate_crop_path=None,
        target_crop_path=None,
        confidence=0.9,
        decision="accepted",
        source="manual",
        decided_at=datetime.now(UTC).replace(tzinfo=None),
    )
    assert dto.decision == "accepted"
