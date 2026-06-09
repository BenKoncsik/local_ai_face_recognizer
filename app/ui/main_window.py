"""Main application window."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QCloseEvent, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QSystemTrayIcon,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from app.config import AppConfig, save_db_path
from app.db.database import ensure_unknown_person, init_db, session_scope
from app.db.models import Face, Image, Person
from app.jobs.job_types import JobState
from app.logging_setup import QLogHandler
from app.paths import app_icon_path
from app.services.duplicate_unknown_face_finder import DuplicateUnknownFaceFinder
from app.services.identity_service import BulkReassignResult, IdentityService
from app.services.recognition_service import RecognitionService
from app.ui.dialogs.export_dialog import ExportDialog
from app.ui.dialogs.manual_face_dialog import NoFaceImagesDialog
from app.ui.dialogs.merge_dialog import MergeDialog
from app.ui.dialogs.move_faces_dialog import MoveFacesDialog
from app.ui.dialogs.overlapping_unknown_faces_dialog import (
    OverlappingUnknownFacesDialog,
)
from app.ui.dialogs.person_info_dialog import PersonInfoDialog
from app.ui.dialogs.rename_dialog import RenameDialog
from app.ui.dialogs.scan_modes_dialog import ScanModesDialog
from app.ui.dialogs.settings_dialog import SettingsDialog
from app.ui.dialogs.suggestion_dialog import SuggestionDialog
from app.ui.dialogs.update_dialog import UpdateDialog
from app.ui.i18n import t
from app.ui.panels.cluster_panel import ClusterPanel
from app.ui.panels.collage_panel import CollagePanel
from app.ui.panels.family_search_panel import FamilySearchPanel
from app.ui.panels.image_browser_panel import ImageBrowserPanel
from app.ui.panels.locations_panel import LocationsPanel
from app.ui.panels.log_panel import LogPanel
from app.ui.panels.objects_panel import ObjectsPanel
from app.ui.panels.persons_panel import PersonsPanel
from app.ui.panels.preview_panel import PreviewPanel
from app.ui.panels.sidebar_panel import SidebarPanel
from app.ui.widgets.flow_layout import FlowContainer
from app.workers.match_job_worker import MatchJobWorker
from app.workers.pipeline_worker import PipelineWorker

log = logging.getLogger(__name__)


def _last_dir(key: str, default: str = "") -> str:
    """Read a remembered directory path from persistent settings."""
    from app.app_settings import app_qsettings

    return app_qsettings().value(key, default, type=str)


def _save_dir(key: str, path: str) -> None:
    """Persist a directory path so the next file dialog reopens there."""
    from app.app_settings import app_qsettings

    app_qsettings().setValue(key, path)


class MainWindow(QMainWindow):
    """Primary application window."""

    log_signal = Signal(str, int)
    _update_ready = Signal(object)   # ReleaseInfo

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self._config = config
        self._worker: Optional[PipelineWorker] = None
        self._match_worker: Optional[MatchJobWorker] = None
        self._current_person_id: Optional[int] = None
        self._current_face_id: Optional[int] = None
        self._recorder = None  # ScreenRecorderService, lazily created on start
        self._recording_log = None  # RecordingTimelineLog while recording
        self._recording_metadata = None  # RecordingMetadataWriter while recording
        self._last_audio_validation = None  # AudioValidation of the last stop
        # Active image/person context shown in the image-browser tab (for the
        # recording timeline log).
        self._browser_image_name: Optional[str] = None
        self._browser_person_name: Optional[str] = None
        self._load_recording_prefs()
        self._connect_display_events()
        self._pending_suggestion_count: int = 0
        self._open_suggestion_count: int = 0
        self._match_active: bool = False
        self._match_paused: bool = False
        self._db_path: str = str(config.db_path_resolved)
        self._pending_release = None

        # Google Drive project session — None when local mode is active
        from app.gdrive.project_session import GDriveProjectSession
        self._gdrive_session: Optional[GDriveProjectSession] = None
        self._gdrive_open_thread: Optional[QThread] = None
        self._gdrive_close_thread: Optional[QThread] = None
        self._gdrive_closing: bool = False  # True while shutdown in progress

        self._apply_window_icon()

        init_db(config.db_path_resolved)
        ensure_unknown_person()

        self._build_ui()
        self._connect_log_handler()
        self._refresh_persons()
        self._retranslate()
        self._restore_last_folder()
        self._setup_tray()
        self._image_browser.refresh()
        self._check_image_library_on_startup()
        self._setup_gdrive_chip()

        self.resize(1280, 780)
        self._update_ready.connect(self._on_update_found)
        self._start_update_check()
        self._setup_shortcuts()
        self._start_match_worker()

    # ------------------------------------------------------------------
    # Keyboard shortcuts
    # ------------------------------------------------------------------

    def _setup_shortcuts(self) -> None:
        """Register global shortcut handlers and bind them to this window.

        Shortcuts are dispatched via per-window QShortcut objects owned by this
        MainWindow (see ShortcutService).  We deliberately do NOT install an
        application-wide event filter: that crashes PySide6 once a
        QWebEngineView (the Places map) is alive.
        """
        from app.services.shortcut_service import get_shortcut_service
        svc = get_shortcut_service()
        svc.register("general.settings",  self._on_settings)
        svc.register("general.log_panel", self._toggle_log_panel)
        svc.set_host(self)
        # Scope shortcuts to the active page: switching tabs swaps which
        # page-specific shortcuts are live, so the same key can mean different
        # things on different pages. Global ("general") shortcuts stay active
        # everywhere.
        self._tabs.currentChanged.connect(self._on_tab_context_changed)
        self._on_tab_context_changed(self._tabs.currentIndex())

    # Maps each outer tab to the shortcut context (page) it activates. Tabs not
    # listed here fall back to "other" — only global shortcuts stay live there.
    _TAB_CONTEXT = {
        0: "faces",    # Arcfelismerés
        1: "image",    # Képböngésző
        4: "faces",    # Személyek
        6: "collage",  # Kollázs
    }

    def _on_tab_context_changed(self, index: int) -> None:
        from app.services.shortcut_service import get_shortcut_service
        context = self._TAB_CONTEXT.get(index, "other")
        get_shortcut_service().set_active_context(context)

    def _toggle_log_panel(self) -> None:
        if hasattr(self, "_log_dock"):
            self._log_dock.setVisible(not self._log_dock.isVisible())

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._build_toolbar()
        self._build_central()
        self._build_log_dock()
        self._build_status_bar()

    def _apply_window_icon(self) -> None:
        icon_path = app_icon_path()
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

    def _build_toolbar(self) -> None:
        tb = QToolBar(t("main_toolbar"))
        tb.setMovable(False)
        self.addToolBar(tb)

        self._folder_btn = QPushButton()
        self._folder_btn.clicked.connect(self._on_select_folder)
        tb.addWidget(self._folder_btn)

        self._folder_label = QLabel()
        self._folder_label.setStyleSheet("color: #A6ADC8;")
        self._folder_label.setMaximumWidth(380)
        tb.addWidget(self._folder_label)

        tb.addSeparator()

        self._scan_modes_btn = QPushButton()
        self._scan_modes_btn.setEnabled(True)
        self._scan_modes_btn.clicked.connect(self._on_open_scan_modes)
        tb.addWidget(self._scan_modes_btn)

        self._stop_btn = QPushButton()
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        tb.addWidget(self._stop_btn)

        tb.addSeparator()

        self._export_btn = QPushButton()
        self._export_btn.clicked.connect(self._on_open_export)
        tb.addWidget(self._export_btn)

        tb.addSeparator()

        self._no_face_btn = QPushButton()
        self._no_face_btn.clicked.connect(self._on_no_face_images)
        tb.addWidget(self._no_face_btn)

        tb.addSeparator()

        self._suggestions_btn = QPushButton()
        self._suggestions_btn.setToolTip(t("suggestions_tip"))
        self._suggestions_btn.clicked.connect(self._on_show_suggestions)
        tb.addWidget(self._suggestions_btn)

        tb.addSeparator()

        self._settings_btn = QPushButton()
        self._settings_btn.clicked.connect(self._on_settings)
        tb.addWidget(self._settings_btn)

        tb.addSeparator()

        from app.ui.widgets.recording_controls import RecordingControls
        self._recording_controls = RecordingControls()
        self._recording_controls.start_requested.connect(self._on_record_start)
        self._recording_controls.pause_toggle_requested.connect(
            self._on_record_pause_toggle
        )
        self._recording_controls.stop_requested.connect(self._on_record_stop)
        tb.addWidget(self._recording_controls)

        tb.addSeparator()

        self._gdrive_btn = QPushButton()
        self._gdrive_btn.clicked.connect(self._on_toggle_drive_project)
        self._gdrive_btn.setVisible(False)  # shown when prefs.is_ready
        tb.addWidget(self._gdrive_btn)

    def _build_central(self) -> None:
        # Outer tabs: Arcok | Kollázs
        self._tabs = QTabWidget()
        self._tabs.setTabPosition(QTabWidget.North)
        self._tabs.setDocumentMode(True)

        # --- Tab 0: Arcfelismerés (existing layout) ---
        face_widget = QWidget()
        face_layout = QVBoxLayout(face_widget)
        face_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        # A wide, visible handle that is easy to grab, and panels that can be
        # dragged down to their real minimums without snapping shut.
        splitter.setHandleWidth(8)
        splitter.setChildrenCollapsible(False)
        splitter.setOpaqueResize(True)
        self._face_splitter = splitter

        self._sidebar = SidebarPanel()
        self._sidebar.person_selected.connect(self._on_person_selected)
        self._sidebar.set_recluster_callback(self._on_recluster)
        self._sidebar.setMinimumWidth(260)
        self._sidebar.setMaximumWidth(400)
        splitter.addWidget(self._sidebar)

        centre = QWidget()
        centre_layout = QVBoxLayout(centre)
        centre_layout.setContentsMargins(0, 0, 0, 0)

        self._cluster_panel = ClusterPanel()
        self._cluster_panel.face_selected.connect(self._on_face_selected)
        self._cluster_panel.face_right_clicked.connect(self._on_cluster_face_right_clicked)
        self._cluster_panel.selection_changed.connect(self._on_face_selection_changed)
        centre_layout.addWidget(self._cluster_panel)

        actions = self._build_action_row()
        centre_layout.addWidget(actions)
        # The centre column must be able to shrink so the preview splitter
        # can be dragged wider; only the sidebar keeps a hard minimum.
        centre.setMinimumWidth(160)
        splitter.addWidget(centre)

        self._preview_panel = PreviewPanel()
        self._preview_panel.setMinimumWidth(200)
        self._preview_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._preview_panel.face_selected.connect(self._on_preview_face_selected)
        self._preview_panel.face_assign_requested.connect(self._on_preview_face_assign)
        self._preview_panel.face_delete_requested.connect(self._on_preview_face_delete)
        self._preview_panel.face_create_requested.connect(self._on_preview_face_create)
        self._preview_panel.face_bbox_update_requested.connect(
            self._on_preview_face_bbox_update
        )
        self._preview_panel.face_diagnostics_requested.connect(
            self._on_preview_face_diagnostics
        )
        self._preview_panel.face_set_thumbnail_requested.connect(
            self._on_face_set_thumbnail
        )
        self._preview_panel.face_clear_thumbnail_requested.connect(
            self._on_face_clear_thumbnail
        )
        self._preview_panel.object_create_requested.connect(
            self._on_preview_object_create
        )
        splitter.addWidget(self._preview_panel)

        # Sidebar keeps its size; the centre grid and the preview share the
        # remaining width and follow the handle as it is dragged.
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([320, 480, 480])
        splitter.setStyleSheet(
            "QSplitter::handle:horizontal { background: #45475a; "
            "margin: 2px 0; border-radius: 2px; }"
            "QSplitter::handle:horizontal:hover { background: #89b4fa; }"
        )

        face_layout.addWidget(splitter)
        self._tabs.addTab(face_widget, t("tab_face_recognition"))

        # --- Tab 1: Képböngésző ---
        self._image_browser = ImageBrowserPanel(config=self._config)
        self._image_browser.person_data_changed.connect(self._refresh_persons)
        self._image_browser.image_displayed.connect(self._on_browser_image_displayed)
        self._image_browser.active_person_changed.connect(
            self._on_browser_person_changed
        )
        self._image_browser.object_open_requested.connect(self._open_object_sheet)
        self._tabs.addTab(self._image_browser, t("tab_image_browser"))

        # --- Tab 2: Családi kereső ---
        self._family_search = FamilySearchPanel()
        self._family_search.image_open_requested.connect(self._open_image_from_family_search)
        self._tabs.addTab(self._family_search, t("tab_family_search"))

        # --- Tab 3: Helyek ---
        self._locations_panel = LocationsPanel()
        self._tabs.addTab(self._locations_panel, t("tab_locations"))

        # --- Tab 4: Személyek ---
        self._persons_panel = PersonsPanel()
        self._persons_panel.person_data_changed.connect(self._refresh_persons)
        self._tabs.addTab(self._persons_panel, t("tab_persons"))

        # --- Tab 5: Objektumok ---
        self._objects_panel = ObjectsPanel()
        self._objects_panel.object_data_changed.connect(
            self._refresh_preview_object_markers
        )
        self._tabs.addTab(self._objects_panel, t("tab_objects"))

        # --- Tab 6: Kollázs nézet ---
        self._collage_panel = CollagePanel()
        self._tabs.addTab(self._collage_panel, t("tab_collage"))

        self.setCentralWidget(self._tabs)

    def _build_action_row(self) -> QWidget:
        # A wrapping flow layout: when the centre column is too narrow to fit
        # every button on one line, the buttons reflow onto additional rows
        # instead of clipping or forcing the column wide. This keeps the
        # preview splitter freely draggable at any width.
        container = FlowContainer(h_spacing=4, v_spacing=4)
        layout = container.layout()

        self._rename_btn = QPushButton()
        self._rename_btn.setEnabled(False)
        self._rename_btn.clicked.connect(self._on_rename)
        layout.addWidget(self._rename_btn)

        self._merge_btn = QPushButton()
        self._merge_btn.setEnabled(False)
        self._merge_btn.clicked.connect(self._on_merge)
        layout.addWidget(self._merge_btn)

        self._delete_person_btn = QPushButton()
        self._delete_person_btn.setEnabled(False)
        self._delete_person_btn.setStyleSheet(
            "QPushButton { color: #F38BA8; border-color: #6B3040; }"
            "QPushButton:hover { background-color: #3D2030; border-color: #F38BA8; }"
            "QPushButton:disabled { color: #6C7086; border-color: #313244; }"
        )
        self._delete_person_btn.clicked.connect(self._on_delete_person)
        layout.addWidget(self._delete_person_btn)

        self._remove_face_btn = QPushButton()
        self._remove_face_btn.setEnabled(False)
        self._remove_face_btn.clicked.connect(self._on_remove_face)
        layout.addWidget(self._remove_face_btn)

        self._reassign_btn = QPushButton()
        self._reassign_btn.setEnabled(False)
        self._reassign_btn.clicked.connect(self._on_reassign_face)
        layout.addWidget(self._reassign_btn)

        self._move_faces_btn = QPushButton()
        self._move_faces_btn.setVisible(False)
        self._move_faces_btn.clicked.connect(self._on_move_faces_batch)
        layout.addWidget(self._move_faces_btn)

        self._sel_count_lbl = QLabel()
        self._sel_count_lbl.setStyleSheet("color: #888;")
        layout.addWidget(self._sel_count_lbl)

        self._person_info_btn = QPushButton()
        self._person_info_btn.setEnabled(False)
        self._person_info_btn.setToolTip(t("person_info_tip"))
        self._person_info_btn.clicked.connect(self._on_person_info)
        layout.addWidget(self._person_info_btn)

        return container

    def _build_log_dock(self) -> None:
        self._log_panel = LogPanel()
        self._log_dock = QDockWidget(self)
        self._log_dock.setWidget(self._log_panel)
        self._log_dock.setAllowedAreas(Qt.BottomDockWidgetArea)
        self.addDockWidget(Qt.BottomDockWidgetArea, self._log_dock)
        self._log_dock.setMinimumHeight(120)

    def _build_status_bar(self) -> None:
        status = QStatusBar()
        self.setStatusBar(status)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        self._progress_bar.setMaximumWidth(300)
        status.addPermanentWidget(self._progress_bar)

        self._update_notify_btn = QPushButton()
        self._update_notify_btn.setVisible(False)
        self._update_notify_btn.setStyleSheet(
            "QPushButton { color: #F9E2AF; font-weight: bold; padding: 1px 6px; "
            "background-color: #3D3020; border: 1px solid #F9E2AF; border-radius: 3px; }"
            "QPushButton:hover { background-color: #5A4530; }"
        )
        self._update_notify_btn.clicked.connect(self._on_update_notify_clicked)
        status.addPermanentWidget(self._update_notify_btn)

        # Drive status chip — permanent widget on the right side of the bar.
        self._gdrive_chip_btn = QPushButton()
        self._gdrive_chip_btn.setFlat(True)
        self._gdrive_chip_btn.setVisible(False)
        self._gdrive_chip_btn.setStyleSheet(
            "QPushButton { color: #89DCEB; font-size: 11px; padding: 1px 6px; "
            "border: 1px solid #313244; border-radius: 3px; background: #1E1E2E; }"
            "QPushButton:hover { background: #313244; }"
        )
        self._gdrive_chip_btn.clicked.connect(self._on_gdrive_chip_clicked)
        status.addPermanentWidget(self._gdrive_chip_btn)

        # Background name-matching status chip (left of the message label).
        self._match_chip_btn = QPushButton()
        self._match_chip_btn.setFlat(True)
        self._match_chip_btn.setVisible(False)
        self._match_chip_btn.setStyleSheet(
            "QPushButton { color: #A6E3A1; font-size: 11px; padding: 1px 6px; "
            "border: 1px solid #313244; border-radius: 3px; background: #1E1E2E; }"
            "QPushButton:hover { background: #313244; }"
        )
        self._match_chip_btn.setToolTip(t("match_chip_tip"))
        self._match_chip_btn.clicked.connect(self._on_match_chip_clicked)
        status.addPermanentWidget(self._match_chip_btn)

        self._status_label = QLabel()
        status.addWidget(self._status_label)

    # ------------------------------------------------------------------
    # Background name-matching worker
    # ------------------------------------------------------------------

    def _start_match_worker(self) -> None:
        """Create and start the background merge-suggestion worker."""
        self._match_worker = MatchJobWorker(self._config, parent=self)
        self._match_worker.status_changed.connect(self._on_match_status)
        self._match_worker.suggestions_found.connect(self._on_match_suggestions_found)
        self._match_worker.idle.connect(self._on_match_idle)
        self._match_worker.start()
        # Prime the badge with whatever is already persisted from a prior run.
        try:
            with session_scope() as session:
                from app.services.merge_suggestion_service import MergeSuggestionService
                self._open_suggestion_count = MergeSuggestionService(
                    session, self._config.matching
                ).count_open()
        except Exception:  # noqa: BLE001
            self._open_suggestion_count = 0
        self._refresh_match_chip()

    def _enqueue_full_match(self) -> None:
        if self._match_worker is not None:
            self._match_worker.enqueue_full_scan()

    def _enqueue_scoped_match(self, person_ids) -> None:
        """High-priority match for the persons in the current view."""
        if self._match_worker is None:
            return
        auto_ids = []
        try:
            with session_scope() as session:
                for pid in person_ids:
                    p = session.get(Person, pid)
                    if p is not None and p.is_auto_named:
                        auto_ids.append(pid)
        except Exception:  # noqa: BLE001
            return
        if auto_ids:
            self._match_worker.enqueue_scoped(auto_ids, label=t("suggestions_candidate"))

    @Slot(object)
    def _on_match_status(self, status) -> None:
        self._match_chip_btn.setVisible(True)
        if status.state == JobState.RUNNING:
            self._match_active = True
            # A fresh RUNNING update means the worker is making progress, so it
            # is no longer paused even if the user toggled pause earlier.
            self._match_paused = False
            self._match_chip_btn.setText(
                t("match_chip_running", label=status.label,
                  processed=status.processed, total=status.total,
                  pct=status.percent(), found=status.found)
            )
        elif status.state == JobState.PAUSED:
            self._match_paused = True
            self._match_chip_btn.setText(t("match_chip_paused"))
        elif status.state == JobState.FAILED:
            self._match_active = False
            self._match_paused = False
            self._match_chip_btn.setText(t("match_chip_failed"))
        elif status.state == JobState.CANCELLED:
            self._match_active = False
            self._match_paused = False
            self._match_chip_btn.setText(t("match_chip_cancelled"))
        elif status.state == JobState.DONE:
            self._match_active = False
            self._match_paused = False
            self._refresh_match_chip()

    @Slot()
    def _on_match_chip_clicked(self) -> None:
        """When a job is active show pause/resume/cancel; else open suggestions."""
        if not self._match_active:
            self._on_show_suggestions()
            return
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        if self._match_paused:
            menu.addAction(t("match_menu_resume"), self._on_match_resume)
        else:
            menu.addAction(t("match_menu_pause"), self._on_match_pause)
        menu.addAction(t("match_menu_cancel"), self._on_match_cancel)
        menu.addSeparator()
        menu.addAction(t("match_menu_review"), self._on_show_suggestions)
        menu.exec(self._match_chip_btn.mapToGlobal(
            self._match_chip_btn.rect().topLeft()
        ))

    def _on_match_pause(self) -> None:
        if self._match_worker is not None:
            self._match_worker.pause()
            self._match_paused = True
            self._match_chip_btn.setText(t("match_chip_paused"))

    def _on_match_resume(self) -> None:
        if self._match_worker is not None:
            self._match_worker.resume()
            self._match_paused = False

    def _on_match_cancel(self) -> None:
        if self._match_worker is not None:
            self._match_worker.cancel_current()
            self._match_active = False
            self._match_paused = False

    @Slot(int)
    def _on_match_suggestions_found(self, open_count: int) -> None:
        self._open_suggestion_count = open_count
        self._refresh_match_chip()

    @Slot()
    def _on_match_idle(self) -> None:
        self._refresh_match_chip()

    def _refresh_match_chip(self) -> None:
        self._match_chip_btn.setVisible(self._open_suggestion_count > 0)
        self._match_chip_btn.setText(
            t("match_chip_done", n=self._open_suggestion_count)
        )

    def _shutdown_match_worker(self) -> None:
        """Graceful stop of the background worker (no half-written state)."""
        if self._match_worker is not None:
            self._match_worker.shutdown()
            if not self._match_worker.wait(10_000):
                log.warning("MatchJobWorker did not stop within 10 s — terminating")
                self._match_worker.terminate()
                self._match_worker.wait(3_000)
            self._match_worker = None

    # ------------------------------------------------------------------
    # Retranslate — call after language change
    # ------------------------------------------------------------------

    def _retranslate(self) -> None:
        self.setWindowTitle(t("window_title"))
        self._folder_btn.setText(t("select_folder"))
        if not hasattr(self, "_root_folder"):
            self._folder_label.setText(f"  {t('no_folder')}")
        self._scan_modes_btn.setText(t("scanModes.openButton"))
        self._stop_btn.setText(t("stop"))
        self._export_btn.setText(t("tb_export"))
        self._no_face_btn.setText(t("view_no_face"))
        self._suggestions_btn.setText(t("suggestions_btn"))
        self._suggestions_btn.setToolTip(t("suggestions_tip"))
        self._settings_btn.setText(t("settings"))
        if hasattr(self, "_recording_controls"):
            self._recording_controls.retranslate()
        self._update_gdrive_toolbar_btn()
        self._rename_btn.setText(t("rename_person"))
        self._merge_btn.setText(t("merge_into"))
        self._delete_person_btn.setText(t("delete_person"))
        self._remove_face_btn.setText(t("remove_face"))
        self._reassign_btn.setText(t("reassign_face"))
        self._move_faces_btn.setText(t("persons_move_faces_btn"))
        self._person_info_btn.setText(t("person_info"))
        self._person_info_btn.setToolTip(t("person_info_tip"))
        self._tabs.setTabText(0, t("tab_face_recognition"))
        self._tabs.setTabText(1, t("tab_image_browser"))
        self._tabs.setTabText(2, t("tab_family_search"))
        self._tabs.setTabText(3, t("tab_locations"))
        self._tabs.setTabText(4, t("tab_persons"))
        self._tabs.setTabText(5, t("tab_objects"))
        self._tabs.setTabText(6, t("tab_collage"))
        self._log_dock.setWindowTitle(t("activity_log"))
        self._status_label.setText(t("ready"))
        if hasattr(self, "_image_browser"):
            self._image_browser.retranslate()
        if hasattr(self, "_family_search"):
            self._family_search.retranslate()
        if hasattr(self, "_locations_panel"):
            self._locations_panel.retranslate()
        if hasattr(self, "_persons_panel"):
            self._persons_panel.retranslate()
        if hasattr(self, "_objects_panel"):
            self._objects_panel.retranslate()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _connect_log_handler(self) -> None:
        handler = QLogHandler(signal=self.log_signal)
        handler.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(handler)
        self.log_signal.connect(self._log_panel.append_log)

    # ------------------------------------------------------------------
    # Toolbar slots
    # ------------------------------------------------------------------

    @Slot()
    def _on_select_folder(self) -> None:
        start_dir = _last_dir("paths/last_folder", str(Path.home()))
        folder = QFileDialog.getExistingDirectory(
            self, t("select_folder"), start_dir
        )
        if folder:
            self._root_folder = folder
            _save_dir("paths/last_folder", folder)
            self._set_folder_label(folder)
            self._scan_modes_btn.setEnabled(True)
            log.info("Root folder selected: %s", folder)

    def _set_folder_label(self, path: str) -> None:
        """Show *path* in the toolbar, elided so it never overflows the toolbar."""
        from PySide6.QtGui import QFontMetrics

        self._folder_label.setToolTip(path)
        metrics = QFontMetrics(self._folder_label.font())
        elided = metrics.elidedText(path, Qt.ElideMiddle, 360)
        self._folder_label.setText(f"  {elided}")

    def _restore_last_folder(self) -> None:
        """Re-select the folder used in the previous session, if it still exists."""
        last = _last_dir("paths/last_folder", "")
        if last and Path(last).is_dir():
            self._root_folder = last
            self._set_folder_label(last)
            self._scan_modes_btn.setEnabled(True)
            log.info("Restored last folder: %s", last)

    @Slot()
    def _on_force_rescan(self) -> None:
        if not hasattr(self, "_root_folder"):
            QMessageBox.warning(self, t("no_folder_title"), t("no_folder_msg"))
            return
        if self._worker and self._worker.isRunning():
            QMessageBox.information(self, t("busy_title"), t("busy_msg"))
            return

        with session_scope() as session:
            from app.db.models import Face, Image
            n_images = session.query(Image).count()

        reply = QMessageBox.question(
            self,
            t("force_rescan_title"),
            t("force_rescan_msg", n=n_images),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        with session_scope() as session:
            from app.db.models import Face, Image, Person
            deleted = (
                session.query(Face)
                .filter(Face.detector_backend != "manual")
                .delete(synchronize_session="fetch")
            )
            session.query(Image).update({"detection_done": False, "embedding_done": False})
            # Remove auto-named persons that lost all their faces in this wipe.
            orphans = (
                session.query(Person)
                .filter(Person.is_auto_named == True)  # noqa: E712
                .filter(~Person.faces.any())
                .all()
            )
            for p in orphans:
                session.delete(p)
            if orphans:
                log.info("Force rescan: removed %d orphaned auto-named person(s).", len(orphans))

        self._current_person_id = None
        self._current_face_id = None
        self._cluster_panel.clear()
        self._preview_panel.clear()
        self._refresh_persons()
        log.info("Force rescan: reset all %d images, deleted %d auto-detected face(s).", n_images, deleted)
        self._on_scan()

    def _prepare_redetect(self) -> Optional[int]:
        """Reset detection flags for all images and delete auto-detected faces.

        Returns the number of images reset, or ``None`` if the caller should
        abort (e.g. worker already running).
        """
        if self._gdrive_session is None and not hasattr(self, "_root_folder"):
            QMessageBox.warning(self, t("no_folder_title"), t("no_folder_msg"))
            return None
        if self._worker and self._worker.isRunning():
            QMessageBox.information(self, t("busy_title"), t("busy_msg"))
            return None

        with session_scope() as session:
            from app.db.models import Face, Image
            n_images = session.query(Image).count()

        return n_images

    @Slot()
    def _on_redetect_fast(self) -> None:
        n_images = self._prepare_redetect()
        if n_images is None:
            return

        reply = QMessageBox.question(
            self,
            t("redetect_title"),
            t("redetect_msg", n=n_images),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._reset_detection_state()
        log.info("Re-detect (fast): reset %d images.", n_images)
        self._start_pipeline(high_accuracy=False)

    @Slot()
    def _on_redetect_accurate(self) -> None:
        n_images = self._prepare_redetect()
        if n_images is None:
            return

        reply = QMessageBox.question(
            self,
            t("redetect_accurate_title"),
            t("redetect_accurate_msg", n=n_images),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._reset_detection_state()
        log.info("Re-detect (accurate): reset %d images.", n_images)
        self._start_pipeline(high_accuracy=True)

    def _reset_detection_state(self) -> None:
        """Delete unnamed auto-detected faces and reset detection flags on all images.

        Named faces (person_id IS NOT NULL) are intentionally preserved so that
        person profiles remain intact and recognition can still run after re-detection.
        """
        with session_scope() as session:
            from app.db.models import Face, Image
            deleted = (
                session.query(Face)
                .filter(Face.detector_backend != "manual")
                .filter(Face.person_id.is_(None))  # keep named faces as training examples
                .delete(synchronize_session="fetch")
            )
            session.query(Image).update({"detection_done": False, "embedding_done": False})
        log.debug(
            "Reset detection: %d unnamed auto face(s) deleted (named faces preserved).",
            deleted,
        )
        self._current_person_id = None
        self._current_face_id = None
        self._cluster_panel.clear()
        self._preview_panel.clear()
        self._refresh_persons()

    def _start_pipeline(self, high_accuracy: bool = False) -> None:
        """Launch the pipeline worker with the given mode."""
        drive_active = self._gdrive_session is not None
        if not drive_active and not hasattr(self, "_root_folder"):
            return
        self._set_scanning_state(True)

        # Drive mode: pass client + folder info; local mode: pass root folder.
        if drive_active and self._gdrive_session is not None:
            from app.paths import drive_mirror_dir
            folders = self._gdrive_session.folders
            root_id = folders.root_id if folders else ""
            self._worker = PipelineWorker(
                root_folder="",          # unused in Drive mode
                config=self._config,
                parent=self,
                high_accuracy=high_accuracy,
                db_path_override=self._db_path,
                drive_client=self._gdrive_session._client,
                drive_root_folder_id=root_id,
                drive_mirror_dir=drive_mirror_dir(root_id),
            )
        else:
            self._worker = PipelineWorker(
                root_folder=self._root_folder,
                config=self._config,
                parent=self,
                high_accuracy=high_accuracy,
                db_path_override=self._db_path,
            )
        self._pending_suggestion_count = 0
        self._worker.progress.connect(self._on_progress)
        self._worker.log_message.connect(self._log_panel.append_plain)
        self._worker.suggestions_ready.connect(self._on_suggestions_ready)
        self._worker.finished.connect(self._on_pipeline_finished)
        self._worker.error.connect(self._on_pipeline_error)
        self._worker.start()

    @Slot()
    def _on_open_scan_modes(self) -> None:
        dlg = ScanModesDialog(
            on_incremental=self._on_scan,
            on_full_rescan=self._on_force_rescan,
            on_face_rescan_fast=self._on_redetect_fast,
            on_face_rescan_accurate=self._on_redetect_accurate,
            on_reset_unknown_persons=self._on_reset_unknown_persons,
            on_find_overlapping_unknown_faces=self._on_find_overlapping_unknown_faces,
            on_identity_repair_scan=self._on_identity_repair_scan,
            on_cleanup_empty_unknown_persons=self._on_cleanup_empty_unknown_persons,
            parent=self,
        )
        dlg.exec()

    @Slot()
    def _on_reset_unknown_persons(self) -> None:
        from app.gdrive import preferences as _gprefs

        prefs = _gprefs.load()
        if prefs.enabled and self._gdrive_session is None:
            QMessageBox.information(
                self, t("gdrive_chip_opening"), t("gdrive_scan_no_session")
            )
            return
        if self._gdrive_session is None and not hasattr(self, "_root_folder"):
            QMessageBox.warning(self, t("no_folder_title"), t("no_folder_msg"))
            return
        if self._worker and self._worker.isRunning():
            QMessageBox.information(self, t("busy_title"), t("busy_msg"))
            return

        reply = QMessageBox.question(
            self,
            t("reset_unknowns_title"),
            t("reset_unknowns_msg"),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        # Stop matching first: its queued jobs may still hold snapshots of the
        # Unknown persons that are about to be deleted.
        self._shutdown_match_worker()
        try:
            with session_scope() as session:
                from app.services.unknown_person_reset_service import UnknownPersonResetService

                result = UnknownPersonResetService(session).reset()
        finally:
            self._start_match_worker()

        self._current_person_id = None
        self._current_face_id = None
        self._cluster_panel.clear()
        self._preview_panel.clear()
        self._refresh_persons()
        self._image_browser._reload_current_face_data()
        log.info(
            "Unknown identity reset: deleted %d person(s), unassigned %d face(s).",
            result.deleted_persons,
            result.unassigned_faces,
        )
        self._status_label.setText(
            t(
                "reset_unknowns_status",
                persons=result.deleted_persons,
                faces=result.unassigned_faces,
            )
        )
        self._start_pipeline(high_accuracy=False)

    @Slot()
    def _on_cleanup_empty_unknown_persons(self) -> None:
        if self._worker and self._worker.isRunning():
            QMessageBox.information(self, t("busy_title"), t("busy_msg"))
            return

        reply = QMessageBox.question(
            self,
            t("cleanup_empty_unknowns_title"),
            t("cleanup_empty_unknowns_msg"),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            with session_scope() as session:
                deleted = IdentityService(session).cleanup_empty_unknown_persons()
        except Exception as exc:  # noqa: BLE001
            log.exception("Cleanup of empty Unknown persons failed")
            QMessageBox.critical(self, t("error"), str(exc))
            return

        if deleted:
            self._current_person_id = None
            self._current_face_id = None
            self._refresh_persons()
            self._image_browser._reload_current_face_data()
            self._status_label.setText(
                t("cleanup_empty_unknowns_status", persons=deleted)
            )
        else:
            QMessageBox.information(
                self,
                t("cleanup_empty_unknowns_title"),
                t("cleanup_empty_unknowns_none"),
            )

    @Slot()
    def _on_scan(self) -> None:
        from app.gdrive import preferences as _gprefs
        _prefs = _gprefs.load()

        # Drive mode is ON but the session hasn't connected yet.
        if _prefs.enabled and self._gdrive_session is None:
            QMessageBox.information(
                self, t("gdrive_chip_opening"), t("gdrive_scan_no_session")
            )
            return

        if self._gdrive_session is None and not hasattr(self, "_root_folder"):
            QMessageBox.warning(self, t("no_folder_title"), t("no_folder_msg"))
            return

        if self._worker and self._worker.isRunning():
            QMessageBox.information(self, t("busy_title"), t("busy_msg"))
            return

        self._start_pipeline(high_accuracy=False)

    @Slot()
    def _on_stop(self) -> None:
        if self._worker:
            self._worker.abort()
            self._stop_btn.setEnabled(False)


    @Slot()
    def _on_no_face_images(self) -> None:
        dlg = NoFaceImagesDialog(config=self._config, parent=self)
        dlg.changed.connect(self._refresh_persons)
        dlg.exec()

    @Slot()
    def _on_show_suggestions(self) -> None:
        dlg = SuggestionDialog(self._config, parent=self)
        dlg.data_changed.connect(self._refresh_persons)
        dlg.data_changed.connect(self._image_browser._reload_current_face_data)
        dlg.exec()
        self._refresh_persons()
        self._image_browser._reload_current_face_data()
        # The user may have resolved suggestions — refresh the status badge.
        try:
            with session_scope() as session:
                from app.services.merge_suggestion_service import MergeSuggestionService
                self._open_suggestion_count = MergeSuggestionService(
                    session, self._config.matching
                ).count_open()
        except Exception:  # noqa: BLE001
            pass
        self._refresh_match_chip()

    @Slot()
    def _on_find_overlapping_unknown_faces(self) -> None:
        if self._worker and self._worker.isRunning():
            QMessageBox.information(self, t("busy_title"), t("busy_msg"))
            return

        threshold = self._config.detection.duplicate_unknown_iou_threshold
        containment = self._config.detection.duplicate_unknown_containment_threshold
        try:
            with session_scope() as session:
                finder = DuplicateUnknownFaceFinder(
                    session,
                    iou_threshold=threshold,
                    containment_threshold=containment,
                )
                matches = finder.find()
                images_examined = finder.images_examined
        except Exception as exc:  # noqa: BLE001
            log.exception("Overlapping unknown face search failed")
            QMessageBox.critical(self, t("error"), t("overlap_search_error", error=exc))
            return

        log.info(
            "Átfedő arckeretek keresése: %d kép vizsgálva, %d találat.",
            images_examined,
            len(matches),
        )
        self._status_label.setText(
            t("overlap_status_found", images=images_examined, matches=len(matches))
        )

        if not matches:
            QMessageBox.information(
                self,
                t("overlap_no_matches_title"),
                t("overlap_no_matches_msg", images=images_examined),
            )
            return

        dlg = OverlappingUnknownFacesDialog(
            matches=matches,
            images_examined=images_examined,
            parent=self,
        )
        dlg.open_face_requested.connect(self._open_overlap_match_face)
        dlg.exec()
        if not dlg.delete_requested():
            return

        selected_ids = dlg.selected_unknown_face_ids()
        if not selected_ids:
            QMessageBox.information(
                self,
                t("overlap_no_selection_title"),
                t("overlap_no_selection_msg"),
            )
            return

        reply = QMessageBox.question(
            self,
            t("overlap_confirm_title"),
            t("overlap_confirm_msg", n=len(selected_ids)),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        preview_image_id = self._preview_panel.current_image_id
        try:
            with session_scope() as session:
                finder = DuplicateUnknownFaceFinder(
                    session,
                    iou_threshold=threshold,
                    containment_threshold=containment,
                )
                result = finder.delete_unknown_faces(selected_ids)
        except Exception as exc:  # noqa: BLE001
            log.exception("Overlapping unknown face cleanup failed")
            QMessageBox.critical(self, t("error"), t("overlap_delete_error", error=exc))
            return

        log.info(
            "Átfedő arckeretek törlése: kijelölt=%d, törölt=%d, kihagyott=%d.",
            result.requested,
            result.deleted,
            len(result.missing_or_changed),
        )
        self._status_label.setText(t("overlap_deleted_status", n=result.deleted))
        if result.missing_or_changed:
            QMessageBox.warning(
                self,
                t("warning"),
                t(
                    "overlap_delete_skipped_msg",
                    n=len(result.missing_or_changed),
                ),
            )

        self._refresh_after_overlapping_unknown_delete(
            deleted_image_ids=set(result.image_ids),
            previous_preview_image_id=preview_image_id,
            deleted_face_ids=set(selected_ids),
        )

    def _on_identity_repair_scan(self) -> None:
        if self._worker and self._worker.isRunning():
            QMessageBox.information(self, t("busy_title"), t("busy_msg"))
            return

        from app.app_settings import app_qsettings
        from app.services.identity_repair_service import IdentityRepairService
        from app.ui.dialogs.identity_repair_dialog import IdentityRepairDialog

        exclude_low_quality = app_qsettings().value(
            "face_quality/exclude_low_quality", True, type=bool
        )
        try:
            with session_scope() as session:
                svc = IdentityRepairService(
                    session,
                    config=self._config.identity_repair,
                    exclude_low_quality=exclude_low_quality,
                )
                candidates = svc.scan()
        except Exception as exc:  # noqa: BLE001
            log.exception("Identity repair scan failed")
            QMessageBox.critical(self, t("error"), t("repair_error", error=exc))
            return

        if not candidates:
            QMessageBox.information(
                self, t("repair_no_matches_title"), t("repair_no_matches_msg")
            )
            return

        dlg = IdentityRepairDialog(candidates=candidates, parent=self)
        dlg.exec()
        if not dlg.merge_requested():
            return

        pairs = dlg.selected_pairs()
        if not pairs:
            return

        reply = QMessageBox.question(
            self,
            t("repair_confirm_title"),
            t("repair_confirm_msg", n=len(pairs)),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            with session_scope() as session:
                svc = IdentityRepairService(
                    session,
                    config=self._config.identity_repair,
                    exclude_low_quality=exclude_low_quality,
                )
                result = svc.apply(pairs)
                # Consolidating fragmented identities can leave face-less
                # Unknown placeholders behind — sweep them in the same session.
                IdentityService(session).cleanup_empty_unknown_persons()
        except Exception as exc:  # noqa: BLE001
            log.exception("Identity repair apply failed")
            QMessageBox.critical(self, t("error"), t("repair_error", error=exc))
            return

        self._status_label.setText(
            t(
                "repair_done_status",
                groups=result.groups_consolidated,
                merged=result.persons_merged_away,
            )
        )
        self._refresh_persons()
        if self._current_person_id is not None:
            self._on_person_selected(self._current_person_id)
        self._image_browser._reload_current_face_data()

    @Slot(int)
    def _open_overlap_match_face(self, known_face_id: int) -> None:
        self._tabs.setCurrentIndex(0)
        with session_scope() as session:
            face = session.get(Face, known_face_id)
            if face is None:
                return
            person_id = face.person_id

        if person_id is not None:
            self._on_person_selected(person_id)
        self._current_face_id = known_face_id
        self._show_face_in_preview(known_face_id)
        self._remove_face_btn.setEnabled(True)
        self._reassign_btn.setEnabled(True)

    def _refresh_after_overlapping_unknown_delete(
        self,
        deleted_image_ids: set[int],
        previous_preview_image_id: Optional[int],
        deleted_face_ids: set[int],
    ) -> None:
        face_to_restore = (
            self._current_face_id
            if self._current_face_id not in deleted_face_ids
            else None
        )
        self._current_face_id = face_to_restore

        self._refresh_persons()
        if self._current_person_id is not None:
            self._on_person_selected(self._current_person_id)
            self._current_face_id = face_to_restore
        self._image_browser._reload_current_face_data()

        if (
            previous_preview_image_id is None
            or previous_preview_image_id not in deleted_image_ids
        ):
            if face_to_restore is not None:
                self._show_face_in_preview(face_to_restore)
            return

        replacement_face_id: Optional[int] = None
        with session_scope() as session:
            image = session.get(Image, previous_preview_image_id)
            if image is not None:
                named = [f for f in image.faces if f.person_id is not None and not f.is_excluded]
                visible = [f for f in image.faces if not f.is_excluded]
                replacement = (named or visible or [None])[0]
                replacement_face_id = replacement.id if replacement is not None else None

        if replacement_face_id is not None:
            self._current_face_id = replacement_face_id
            self._show_face_in_preview(replacement_face_id)
        else:
            self._preview_panel.clear()

    # ------------------------------------------------------------------
    # Google Drive workflow
    # ------------------------------------------------------------------

    def _setup_gdrive_chip(self) -> None:
        """Initialise the Drive chip visibility and start the refresh timer."""
        self._update_gdrive_toolbar_btn()
        self._update_gdrive_chip()

        # Poll session status every 15 seconds to keep the chip fresh.
        self._gdrive_status_timer = QTimer(self)
        self._gdrive_status_timer.setInterval(15_000)
        self._gdrive_status_timer.timeout.connect(self._update_gdrive_chip)
        self._gdrive_status_timer.start()

        # Auto-open Drive session on startup if mode is enabled and ready.
        from app.gdrive import preferences as _gprefs
        _p = _gprefs.load()
        if _p.enabled and _p.is_ready and self._gdrive_session is None:
            self._status_label.setText(t("gdrive_mode_enabled_opening"))
            QTimer.singleShot(400, self._on_open_drive_project)

    def _update_gdrive_toolbar_btn(self) -> None:
        """Show/hide and label the Drive toolbar button based on current state."""
        from app.gdrive import preferences
        prefs = preferences.load()
        is_open = self._gdrive_session is not None

        if is_open:
            self._gdrive_btn.setText(t("gdrive_close_project_btn"))
            self._gdrive_btn.setToolTip(t("gdrive_close_project_tip"))
            self._gdrive_btn.setVisible(True)
        elif prefs.is_ready:
            self._gdrive_btn.setText(t("gdrive_open_project_btn"))
            self._gdrive_btn.setToolTip(t("gdrive_open_project_tip"))
            self._gdrive_btn.setVisible(True)
        else:
            self._gdrive_btn.setVisible(False)

    def _update_gdrive_chip(self) -> None:
        """Update the status bar Drive chip from the current session state."""
        from app.gdrive import preferences
        prefs = preferences.load()

        if self._gdrive_session is None:
            if not prefs.is_ready:
                self._gdrive_chip_btn.setVisible(False)
                return
            self._gdrive_chip_btn.setText(t("gdrive_chip_idle"))
            self._gdrive_chip_btn.setToolTip(t("gdrive_open_project_tip"))
            self._gdrive_chip_btn.setVisible(True)
            return

        # Session is open — show sync status.
        status = self._gdrive_session.status
        if status.current_op:
            label = t("gdrive_chip_syncing")
        elif not status.last_sync_succeeded:
            label = t("gdrive_chip_error")
        else:
            folders = self._gdrive_session.folders
            name = folders.root_name if folders else ""
            label = t("gdrive_chip_open", name=name) if name else t("gdrive_chip_open", name="Drive")
        self._gdrive_chip_btn.setText(label)
        self._gdrive_chip_btn.setToolTip(t("gdrive_close_project_tip"))
        self._gdrive_chip_btn.setVisible(True)

    @Slot()
    def _on_gdrive_chip_clicked(self) -> None:
        """Clicking the chip opens the project if closed, closes if open."""
        self._on_toggle_drive_project()

    @Slot()
    def _on_toggle_drive_project(self) -> None:
        """Open or close the Drive project depending on current state."""
        if self._gdrive_session is not None:
            self._on_close_drive_project()
        else:
            self._on_open_drive_project()

    @Slot()
    def _on_open_drive_project(self) -> None:
        """Start opening the Drive project session in a background thread."""
        from app.gdrive import preferences
        from app.gdrive.connectivity import GDriveOfflineError, is_online

        prefs = preferences.load()
        if not prefs.is_ready:
            QMessageBox.information(
                self, t("gdrive_not_configured_title"), t("gdrive_not_configured_msg")
            )
            return

        if not is_online():
            QMessageBox.warning(self, t("gdrive_error_title"), t("gdrive_offline_error"))
            return

        if self._gdrive_open_thread and self._gdrive_open_thread.isRunning():
            return  # already opening

        self._gdrive_btn.setEnabled(False)
        self._gdrive_chip_btn.setText(t("gdrive_chip_opening"))
        self._gdrive_chip_btn.setVisible(True)

        account_email = prefs.account_email
        folder_id = prefs.folder_id

        class _OpenThread(QThread):
            succeeded = Signal(object, str)   # (session, local_db_path)
            failed = Signal(str)

            def run(self_inner) -> None:  # noqa: N805
                try:
                    from app.gdrive.drive_client import build_drive_client
                    from app.gdrive.project_session import GDriveProjectSession
                    client = build_drive_client(account_email)
                    session = GDriveProjectSession(client, folder_id)
                    local_db = session.open()
                    self_inner.succeeded.emit(session, str(local_db))
                except Exception as exc:  # noqa: BLE001
                    log.exception("Drive project open failed")
                    self_inner.failed.emit(str(exc))

        thread = _OpenThread(self)
        thread.succeeded.connect(self._on_drive_open_succeeded)
        thread.failed.connect(self._on_drive_open_failed)
        thread.finished.connect(lambda: self._gdrive_btn.setEnabled(True))
        self._gdrive_open_thread = thread
        thread.start()

    @Slot(object, str)
    def _on_drive_open_succeeded(self, session, local_db_path: str) -> None:
        """Called on the main thread after Drive session opens successfully."""
        self._gdrive_session = session

        # Switch the database to the Drive-downloaded local copy.
        init_db(local_db_path)
        ensure_unknown_person()
        self._db_path = local_db_path
        self._current_person_id = None
        self._current_face_id = None
        self._cluster_panel.clear()
        self._preview_panel.clear()

        folders = session.folders
        name = folders.root_name if folders else "Drive"

        # Tell the image browser to use Drive-mode lazy loading.
        from app.paths import drive_mirror_dir
        root_id = folders.root_id if folders else ""
        self._image_browser.set_drive_mode(
            client=session._client,
            mirror_dir=drive_mirror_dir(root_id),
            project_name=name,
        )

        self._refresh_persons()
        self._image_browser.refresh()

        log.info("Drive project opened: %s (local DB: %s)", name, local_db_path)
        self._status_label.setText(t("gdrive_project_opened", name=name))
        self._update_gdrive_toolbar_btn()
        self._update_gdrive_chip()

    @Slot(str)
    def _on_drive_open_failed(self, error: str) -> None:
        log.error("Drive project open failed: %s", error)
        QMessageBox.critical(
            self, t("gdrive_error_title"), t("gdrive_open_failed", error=error)
        )
        self._gdrive_chip_btn.setText(t("gdrive_chip_idle"))
        self._update_gdrive_toolbar_btn()
        self._update_gdrive_chip()

    @Slot()
    def _on_close_drive_project(self) -> None:
        """Ask for confirmation and then close the Drive session."""
        if self._gdrive_session is None:
            return

        reply = QMessageBox.question(
            self,
            t("gdrive_confirm_close_title"),
            t("gdrive_confirm_close_msg"),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._start_drive_close(after_quit=False)

    def _start_drive_close(self, *, after_quit: bool = False) -> None:
        """Kick off Drive session shutdown in a background thread.

        Args:
            after_quit: When ``True``, call ``QApplication.quit()`` after the
                        thread finishes so the app window actually closes.
        """
        if self._gdrive_session is None:
            if after_quit:
                QApplication.quit()
            return

        session = self._gdrive_session
        self._gdrive_session = None  # mark as closing
        self._gdrive_chip_btn.setText(t("gdrive_chip_closing"))
        self._gdrive_btn.setEnabled(False)

        class _CloseThread(QThread):
            done = Signal()

            def run(self_inner) -> None:  # noqa: N805
                try:
                    session.close(upload_pending=True)
                except Exception as exc:  # noqa: BLE001
                    log.warning("Drive session close error: %s", exc)
                self_inner.done.emit()

        thread = _CloseThread(self)
        if after_quit:
            thread.done.connect(QApplication.quit)
        else:
            thread.done.connect(self._on_drive_close_done)
        self._gdrive_close_thread = thread
        thread.start()

    @Slot()
    def _on_drive_close_done(self) -> None:
        """Called after the Drive session has been fully closed."""
        log.info("Drive session closed.")
        self._gdrive_closing = False

        # Switch back to the local database and refresh all views.
        local_db = str(self._config.db_path_resolved)
        self._db_path = local_db
        init_db(local_db)
        ensure_unknown_person()
        self._current_person_id = None
        self._current_face_id = None
        self._cluster_panel.clear()
        self._preview_panel.clear()
        self._image_browser.clear_drive_mode()
        self._refresh_persons()
        self._image_browser.refresh()

        self._update_gdrive_toolbar_btn()
        self._update_gdrive_chip()
        self._gdrive_btn.setEnabled(True)
        self._status_label.setText(t("ready"))

    @Slot()
    def _on_drive_prefs_changed(self) -> None:
        """Refresh Drive UI and auto-open/close session when mode toggle changes."""
        from app.gdrive import preferences as _gprefs
        prefs = _gprefs.load()

        if prefs.enabled and prefs.is_ready and self._gdrive_session is None:
            # User just turned Drive ON — open the session automatically.
            self._status_label.setText(t("gdrive_mode_enabled_opening"))
            self._on_open_drive_project()
        elif not prefs.enabled and self._gdrive_session is not None:
            # User just turned Drive OFF — close the session gracefully.
            self._start_drive_close(after_quit=False)

        self._update_gdrive_toolbar_btn()
        self._update_gdrive_chip()

    # ------------------------------------------------------------------
    # Settings slot
    # ------------------------------------------------------------------

    @Slot()
    def _on_settings(self) -> None:
        dlg = SettingsDialog(current_db_path=self._db_path, parent=self)
        # Wire up the Drive prefs-changed signal so the chip/button refresh.
        if hasattr(dlg, "_gdrive_tab"):
            dlg._gdrive_tab.prefs_changed.connect(self._on_drive_prefs_changed)
        if dlg.exec() != SettingsDialog.Accepted:
            return

        # Database change
        new_db = dlg.selected_db_path()
        if new_db and new_db != self._db_path:
            self._db_path = new_db
            self._config.storage.db_path = new_db
            save_db_path(new_db)
            init_db(new_db)  # also re-initialises the ImageLibraryService
            ensure_unknown_person()
            self._check_image_library_on_startup()
            self._current_person_id = None
            self._current_face_id = None
            self._cluster_panel.clear()
            self._preview_panel.clear()
            self._refresh_persons()
            self._image_browser.refresh()
            self._locations_panel.refresh()
            self._persons_panel.refresh()
            self._objects_panel.refresh()
            QMessageBox.information(self, t("settings_title"), t("db_switched"))
            log.info("Database switched to: %s", new_db)

        # Language change
        if dlg.language_changed():
            self._retranslate()
            self._refresh_persons()

        # Pick up any recording-setting changes for the next recording.
        self._load_recording_prefs()

    # ------------------------------------------------------------------
    # Screen recording
    # ------------------------------------------------------------------

    def _on_record_start(self) -> None:
        from app.services.screen_recorder_service import (
            CaptureDevices,
            RecorderState,
            RecordingDisplayMode,
            RecordingOptions,
            ScreenRecorderService,
            effective_fps,
            probe_devices,
            resolve_capture_region,
            resolve_ffmpeg,
            selected_displays,
        )

        rec_cfg = self._config.recording
        ffmpeg = resolve_ffmpeg(rec_cfg.ffmpeg_path)
        if not ffmpeg:
            QMessageBox.warning(
                self, t("rec_ffmpeg_missing_title"), t("rec_ffmpeg_missing_body")
            )
            return

        # Privacy notice — show unless the user opted out.
        if not self._confirm_recording_privacy():
            return

        # Pick output folder — default to the last-used / configured dir.
        start_dir = rec_cfg.output_dir or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(
            self, t("rec_choose_dir"), start_dir
        )
        if not chosen:
            return
        # Remember the choice (config + persisted settings).
        rec_cfg.output_dir = chosen
        self._save_recording_output_dir(chosen)

        from datetime import datetime
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = Path(chosen) / f"recording_{stamp}"

        devices: CaptureDevices = probe_devices(
            ffmpeg,
            want_system_audio=rec_cfg.capture_system_audio,
            mic_name=rec_cfg.audio_input_device,
            system_audio_name=rec_cfg.system_audio_device,
        )
        if not rec_cfg.capture_microphone:
            devices.microphone = None
            devices.microphone_name = None

        # Resolve which monitors / window to capture, and cap the fps for
        # multi-monitor captures.
        from app.ui.display_utils import active_window_bounds, enumerate_displays
        mode = RecordingDisplayMode.from_value(rec_cfg.display_mode)
        displays = enumerate_displays()
        # On macOS the avfoundation "Capture screen N" device index is not the
        # monitor ordinal (capture devices come after the cameras), so map each
        # monitor to its real avfoundation index before resolving the region.
        if sys.platform.startswith("darwin") and displays:
            from app.services.screen_recorder_service import probe_screen_indices
            screen_idx = probe_screen_indices(ffmpeg)
            for i, disp in enumerate(displays):
                if i < len(screen_idx):
                    try:
                        disp.av_index = int(screen_idx[i])
                    except (TypeError, ValueError):
                        disp.av_index = None
        region = resolve_capture_region(
            mode,
            displays,
            rec_cfg.selected_display_ids,
            active_window_bounds(self),
        )
        fps = effective_fps(
            rec_cfg.fps,
            mode,
            displays,
            rec_cfg.selected_display_ids,
            auto_reduce=rec_cfg.auto_reduce_fps,
            multi_monitor_cap=rec_cfg.multi_monitor_fps_cap,
        )
        if sys.platform.startswith("darwin") and mode in (
            RecordingDisplayMode.ALL_DISPLAYS,
            RecordingDisplayMode.ACTIVE_WINDOW,
        ):
            log.info(
                "macOS avfoundation cannot capture a window or merge displays; "
                "recording the primary screen for mode %s", mode.value
            )
        options = RecordingOptions(
            fps=fps,
            quality=rec_cfg.quality,
            segment_seconds=rec_cfg.segment_seconds,
            capture_cursor=rec_cfg.capture_cursor,
            mic_volume=rec_cfg.mic_volume,
            system_volume=rec_cfg.system_volume,
            mute_microphone=rec_cfg.mute_microphone,
            mute_system_audio=rec_cfg.mute_system_audio,
            meter_audio=True,  # drive the live VU meter
        )

        self._recorder = ScreenRecorderService(ffmpeg, parent=self)
        self._recorder.state_changed.connect(self._on_recorder_state)
        self._recorder.elapsed_changed.connect(self._on_recorder_elapsed)
        self._recorder.error.connect(self._on_recorder_error)
        self._recorder.audio_level.connect(self._on_recorder_audio_level)
        self._recorder.audio_validated.connect(self._on_recorder_audio_validated)

        from app.services.recording_timeline_log import RecordingTimelineLog
        from app.services.subtitle_service import subtitle_path_for_video
        session_dir.mkdir(parents=True, exist_ok=True)
        self._recording_log = RecordingTimelineLog(
            subtitle_path_for_video(session_dir / "recording.mp4"),
            person_prefix=t("rec_person_prefix"),
        )

        # Metadata — written immediately as "crashed" until a clean finalize.
        captured = selected_displays(mode, displays, rec_cfg.selected_display_ids)
        self._recording_metadata = self._make_recording_metadata(
            session_dir, mode, fps, captured, devices
        )

        self._recorder.start(
            session_dir,
            devices,
            options,
            concat_on_stop=rec_cfg.concat_on_stop,
            region=region,
        )
        if self._recorder.state is RecorderState.RECORDING:
            self._notify_recording_context()

    def _confirm_recording_privacy(self) -> bool:
        """Show the privacy notice; return ``False`` if the user cancels."""
        from app.app_settings import app_qsettings
        qs = app_qsettings()
        if qs.value("recording/privacy_ack", False, type=bool):
            return True
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle(t("rec_privacy_title"))
        box.setText(t("rec_privacy_body"))
        box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        dont_ask = QCheckBox(t("rec_privacy_dont_ask"))
        box.setCheckBox(dont_ask)
        if box.exec() != QMessageBox.Ok:
            return False
        if dont_ask.isChecked():
            qs.setValue("recording/privacy_ack", True)
        return True

    def _make_recording_metadata(
        self, session_dir, mode, fps, captured_displays, devices
    ):
        """Build + write the initial (pessimistic) recording metadata."""
        from datetime import datetime, timezone

        from app.services.recording_metadata import (
            METADATA_FILENAME,
            RecordingMetadata,
            RecordingMetadataWriter,
        )
        from app.services.screen_recorder_service import displays_bounding_box

        box = displays_bounding_box(captured_displays)
        meta = RecordingMetadata(
            started_at=datetime.now(timezone.utc).isoformat(),
            platform=sys.platform,
            fps=fps,
            width=box[2] if box else None,
            height=box[3] if box else None,
            display_mode=mode.value,
            monitors=[
                {
                    "id": d.id,
                    "name": d.name,
                    "width": d.width,
                    "height": d.height,
                    "is_primary": d.is_primary,
                    "x": d.x,
                    "y": d.y,
                }
                for d in captured_displays
            ],
            layout={"x": box[0], "y": box[1], "width": box[2], "height": box[3]}
            if box
            else None,
            captured_microphone=devices.microphone is not None,
            captured_system_audio=devices.system_audio is not None,
        )
        writer = RecordingMetadataWriter(session_dir / METADATA_FILENAME, meta)
        writer.begin()
        return writer

    def _on_record_pause_toggle(self) -> None:
        from app.services.screen_recorder_service import RecorderState
        if self._recorder is None:
            return
        if self._recorder.state is RecorderState.RECORDING:
            self._recorder.pause()
        elif self._recorder.state is RecorderState.PAUSED:
            self._recorder.resume()
            self._notify_recording_context()

    def _on_record_stop(self) -> None:
        if self._recorder is None:
            return
        self._last_audio_validation = None
        elapsed = self._recorder.elapsed_seconds
        # stop() validates the final file and emits ``audio_validated``
        # synchronously, so ``_last_audio_validation`` is populated below.
        final = self._recorder.stop()
        if self._recording_log is not None:
            self._recording_log.finalize(elapsed)
            self._recording_log = None
        self._finalize_recording_metadata(elapsed, crashed=False)
        out_dir = self._recorder.output_dir
        self._recorder = None
        target = final or out_dir
        if target is not None:
            body = t("rec_saved_body", path=str(target))
            validation = self._last_audio_validation
            if validation is not None and not validation.has_audio:
                # Make a silent recording impossible to miss.  The remedy is
                # platform-specific (BlackHole/TCC on macOS vs. Stereo Mix /
                # Windows mic-privacy on Windows), so pick the right guidance.
                warn_key = (
                    "rec_no_audio_warning_windows"
                    if sys.platform.startswith("win")
                    else "rec_no_audio_warning"
                )
                QMessageBox.warning(
                    self,
                    t("rec_saved_title"),
                    f"{body}\n\n⚠ {t(warn_key)}",
                )
            else:
                QMessageBox.information(self, t("rec_saved_title"), body)

    def _on_recorder_state(self, state) -> None:
        self._recording_controls.set_state(state)

    def _on_recorder_elapsed(self, seconds: int) -> None:
        self._recording_controls.set_elapsed(seconds)

    def _on_recorder_audio_level(self, db: float) -> None:
        self._recording_controls.set_audio_level(db)

    def _on_recorder_audio_validated(self, validation) -> None:
        """Remember the final-mux audio check so stop() can warn if silent."""
        self._last_audio_validation = validation
        if not validation.has_audio:
            log.warning("recording finished with no usable audio track")
            # Persist into the (still-open) crash-safe metadata for post-mortem.
            if self._recording_metadata is not None:
                self._recording_metadata.add_error(validation.summary())

    def _on_recorder_error(self, message: str) -> None:
        QMessageBox.critical(self, t("rec_error_title"), message)
        elapsed = self._recorder.elapsed_seconds if self._recorder is not None else 0
        if self._recording_log is not None and self._recorder is not None:
            self._recording_log.finalize(elapsed)
        self._recording_log = None
        if self._recording_metadata is not None:
            self._recording_metadata.add_error(message)
        self._finalize_recording_metadata(elapsed, crashed=True)

    def _finalize_recording_metadata(self, elapsed: float, *, crashed: bool) -> None:
        """Write the closing metadata record (clean stop or crash)."""
        if self._recording_metadata is None:
            return
        try:
            from datetime import datetime, timezone
            self._recording_metadata.finalize(
                datetime.now(timezone.utc).isoformat(),
                float(elapsed),
                crashed=crashed,
            )
        except Exception:  # noqa: BLE001 — never let metadata break shutdown
            log.debug("could not finalize recording metadata", exc_info=True)
        self._recording_metadata = None

    def _connect_display_events(self) -> None:
        """Watch for monitor hot-plug / layout changes (best-effort, no crash).

        A monitor disappearing or resizing mid-recording can break the active
        ffmpeg capture; that surfaces via the recorder's ``error`` signal (which
        finalizes the log + metadata as crashed).  Here we only log the event so
        it is diagnosable and never let a display-layer hiccup raise.
        """
        try:
            app = QApplication.instance()
            if app is None:
                return
            app.screenAdded.connect(self._on_screen_changed)
            app.screenRemoved.connect(self._on_screen_changed)
        except Exception:  # noqa: BLE001
            log.debug("could not connect display events", exc_info=True)

    def _on_screen_changed(self, _screen=None) -> None:
        try:
            from app.services.screen_recorder_service import RecorderState
            recording = (
                self._recorder is not None
                and self._recorder.state in (
                    RecorderState.RECORDING,
                    RecorderState.PAUSED,
                )
            )
            log.info(
                "display configuration changed (recording=%s); "
                "active capture is unaffected unless a captured monitor was lost",
                recording,
            )
        except Exception:  # noqa: BLE001
            log.debug("screen-change handler failed", exc_info=True)

    def _notify_recording_context(
        self,
        image_name: Optional[str] = None,
        track_person: bool = True,
    ) -> None:
        """Record which image/person is active in the timeline log.

        When *image_name* is given it is used directly (e.g. from the image
        browser); otherwise the face-tab preview's current image is resolved.
        *track_person* controls whether the selected person is logged.
        """
        from app.services.screen_recorder_service import RecorderState
        if self._recording_log is None or self._recorder is None:
            return
        if self._recorder.state not in (
            RecorderState.RECORDING,
            RecorderState.PAUSED,
        ):
            return
        person_name: Optional[str] = None
        image_id = (
            None if image_name is not None else self._preview_panel.current_image_id
        )
        try:
            with session_scope() as session:
                if image_id is not None:
                    img = session.get(Image, image_id)
                    if img is not None:
                        image_name = Path(img.file_path).name
                if track_person and self._current_person_id is not None:
                    p = session.get(Person, self._current_person_id)
                    if p is not None:
                        person_name = p.name
        except Exception:  # noqa: BLE001 — logging must never break the UI
            log.debug("recording context lookup failed", exc_info=True)
        self._recording_log.note_active(
            image_name, person_name, self._recorder.elapsed_seconds
        )

    @Slot(int, str)
    def _on_browser_image_displayed(self, _image_id: int, file_path: str) -> None:
        """Image-browser tab opened an image — update the recording timeline.

        A new image clears any face selection (see ``_load_image``), so the
        active person resets to ``None`` until the user clicks a face.
        """
        self._browser_image_name = Path(file_path).name
        self._browser_person_name = None
        self._notify_browser_context()

    @Slot(object)
    def _on_browser_person_changed(self, person_name) -> None:
        """A face was selected / (re)assigned in the image-browser tab."""
        self._browser_person_name = person_name or None
        self._notify_browser_context()

    def _notify_browser_context(self) -> None:
        """Push the image-browser tab's active image + person to the log."""
        if self._recording_log is None or self._recorder is None:
            return
        from app.services.screen_recorder_service import RecorderState
        if self._recorder.state not in (
            RecorderState.RECORDING,
            RecorderState.PAUSED,
        ):
            return
        self._recording_log.note_active(
            self._browser_image_name,
            self._browser_person_name,
            self._recorder.elapsed_seconds,
        )

    def _save_recording_output_dir(self, path: str) -> None:
        try:
            from app.app_settings import app_qsettings
            app_qsettings().setValue("recording/output_dir", path)
        except Exception:  # noqa: BLE001
            log.debug("could not persist recording output dir", exc_info=True)

    def _load_recording_prefs(self) -> None:
        """Overlay persisted recording settings (QSettings) onto the config."""
        try:
            from app.app_settings import app_qsettings
            qs = app_qsettings()
            rec = self._config.recording
            saved_dir = qs.value("recording/output_dir", None, type=str)
            if saved_dir:
                rec.output_dir = saved_dir
            ffmpeg_path = qs.value("recording/ffmpeg_path", None, type=str)
            if ffmpeg_path:
                rec.ffmpeg_path = ffmpeg_path
            rec.quality = qs.value("recording/quality", rec.quality, type=str)
            rec.fps = qs.value("recording/fps", rec.fps, type=int)
            rec.segment_seconds = qs.value(
                "recording/segment_seconds", rec.segment_seconds, type=int
            )
            rec.capture_cursor = qs.value(
                "recording/capture_cursor", rec.capture_cursor, type=bool
            )
            rec.capture_microphone = qs.value(
                "recording/capture_microphone", rec.capture_microphone, type=bool
            )
            rec.capture_system_audio = qs.value(
                "recording/capture_system_audio", rec.capture_system_audio, type=bool
            )
            # Empty string → keep the auto-pick (None), not a literal "".
            rec.audio_input_device = (
                qs.value("recording/audio_input_device", "", type=str) or None
            )
            rec.system_audio_device = (
                qs.value("recording/system_audio_device", "", type=str) or None
            )
            rec.mic_volume = qs.value(
                "recording/mic_volume", rec.mic_volume, type=float
            )
            rec.system_volume = qs.value(
                "recording/system_volume", rec.system_volume, type=float
            )
            rec.mute_microphone = qs.value(
                "recording/mute_microphone", rec.mute_microphone, type=bool
            )
            rec.mute_system_audio = qs.value(
                "recording/mute_system_audio", rec.mute_system_audio, type=bool
            )
            rec.concat_on_stop = qs.value(
                "recording/concat_on_stop", rec.concat_on_stop, type=bool
            )
            rec.display_mode = qs.value(
                "recording/display_mode", rec.display_mode, type=str
            )
            rec.auto_reduce_fps = qs.value(
                "recording/auto_reduce_fps", rec.auto_reduce_fps, type=bool
            )
            saved_ids = qs.value("recording/selected_display_ids", None)
            if saved_ids is not None:
                # QSettings may round-trip a single string instead of a list.
                if isinstance(saved_ids, str):
                    saved_ids = [saved_ids] if saved_ids else []
                rec.selected_display_ids = [str(x) for x in saved_ids]
        except Exception:  # noqa: BLE001
            log.debug("could not load recording prefs", exc_info=True)

    # ------------------------------------------------------------------
    # Pipeline slots
    # ------------------------------------------------------------------

    @Slot(int, int, str, str)
    def _on_progress(self, current: int, total: int, stage: str, detail: str) -> None:
        if total > 0:
            self._progress_bar.setValue(int(current / total * 100))
        self._status_label.setText(f"{stage}: {detail}")

    @Slot(int)
    def _on_suggestions_ready(self, count: int) -> None:
        self._pending_suggestion_count = count

    @Slot(bool, str)
    def _on_pipeline_finished(self, success: bool, summary: str) -> None:
        self._set_scanning_state(False)
        self._status_label.setText(summary)
        self._progress_bar.setValue(100)
        self._refresh_persons()
        self._image_browser.refresh()
        self._locations_panel.refresh()
        self._persons_panel.refresh()
        self._objects_panel.refresh()
        if not success:
            QMessageBox.warning(self, t("warning"), summary)
            return

        # Kick off a background full archive match so suggestions appear
        # incrementally without blocking the UI.
        self._enqueue_full_match()

        if self._pending_suggestion_count > 0:
            reply = QMessageBox.question(
                self,
                t("suggestions_found_title"),
                t("suggestions_found_msg", n=self._pending_suggestion_count),
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self._on_show_suggestions()
        self._pending_suggestion_count = 0

    @Slot(str)
    def _on_pipeline_error(self, message: str) -> None:
        self._set_scanning_state(False)
        QMessageBox.critical(self, t("error"), message)

    # ------------------------------------------------------------------
    # Person / face interaction
    # ------------------------------------------------------------------

    @Slot(int)
    def _on_person_selected(self, person_id: int) -> None:
        self._current_person_id = person_id
        self._current_face_id = None

        is_protected = False
        with session_scope() as session:
            svc = IdentityService(session)
            person = session.get(Person, person_id)
            faces = svc.get_faces_for_person(person_id)
            if person is None:
                return
            from app.services.face_crop_service import ensure_unique_face_crops
            ensure_unique_face_crops(
                session,
                faces,
                self._config.crops_dir_resolved,
                self._config.scan.thumbnail_size,
            )
            is_protected = person.is_protected
            for f in faces:
                _ = f.image  # noqa: F841
            self._cluster_panel.show_person(person.name, faces)
            self._preview_panel.clear()

        self._rename_btn.setEnabled(not is_protected)
        self._merge_btn.setEnabled(True)
        self._delete_person_btn.setEnabled(not is_protected)
        self._remove_face_btn.setEnabled(False)
        self._reassign_btn.setEnabled(False)
        self._person_info_btn.setEnabled(not is_protected)

        # Prioritise matching the currently viewed (unknown) person.
        self._enqueue_scoped_match([person_id])
        self._notify_recording_context()

    @Slot(int)
    def _on_face_selected(self, face_id: int) -> None:
        self._current_face_id = face_id

        self._show_face_in_preview(face_id)

        self._remove_face_btn.setEnabled(True)
        self._reassign_btn.setEnabled(True)
        self._notify_recording_context()

    def _show_face_in_preview(self, face_id: int) -> None:
        """Reload the preview image and select *face_id*."""
        with session_scope() as session:
            face = session.get(Face, face_id)
            if face is None:
                log.warning("_show_face_in_preview: face_id=%d not found in DB", face_id)
                return
            _ = face.image
            if face.image:
                for f in face.image.faces:
                    _ = f.person
                log.debug(
                    "_show_face_in_preview: face_id=%d image_id=%d annotations=%d",
                    face_id, face.image.id, len([f for f in face.image.faces if not f.is_excluded]),
                )
            self._preview_panel.show_face(face)
        self._refresh_preview_object_markers()

    def _refresh_preview_object_markers(self) -> None:
        """Load object occurrences for the current preview image and render them."""
        image_id = self._preview_panel.current_image_id
        if image_id is None:
            self._preview_panel.set_object_occurrences([])
            return
        from app.db.models import TaggedObject
        from app.services.object_service import ObjectService
        markers = []
        try:
            with session_scope() as session:
                for occ in ObjectService(session).get_occurrences_for_image(image_id):
                    if occ.point_x is None or occ.point_y is None:
                        continue
                    obj = session.get(TaggedObject, occ.object_id)
                    name = obj.name if obj is not None else None
                    markers.append((occ.occurrence_id, occ.point_x, occ.point_y, name))
        except Exception:
            log.exception("Failed to load object markers for image %s", image_id)
            markers = []
        self._preview_panel.set_object_occurrences(markers)

    @Slot(int, int, int)
    def _on_preview_object_create(self, image_id: int, x: int, y: int) -> None:
        """Open the object picker for a clicked point and record the occurrence."""
        from app.services.object_service import ObjectService
        from app.ui.dialogs.object_picker_dialog import ObjectPickerDialog

        dlg = ObjectPickerDialog(self)
        if dlg.exec() != QDialog.Accepted or dlg.chosen_object_id is None:
            return
        try:
            with session_scope() as session:
                ObjectService(session).add_occurrence(
                    dlg.chosen_object_id, image_id, x, y, note=dlg.occurrence_note
                )
        except Exception:
            log.exception("Failed to add object occurrence")
            return
        self._refresh_preview_object_markers()
        if hasattr(self, "_objects_panel"):
            self._objects_panel.refresh()
        self.statusBar().showMessage(t("object_tagged_ok"), 3000)

    @Slot(int)
    def _open_object_sheet(self, object_id: int) -> None:
        """Switch to the Objects tab and show the given object's data sheet."""
        if not hasattr(self, "_objects_panel"):
            return
        self._tabs.setCurrentWidget(self._objects_panel)
        self._objects_panel.open_object(object_id)

    @Slot(int, int, int)
    def _on_cluster_face_right_clicked(self, face_id: int, gx: int, gy: int) -> None:
        """Show the preview context menu when a cluster thumbnail is right-clicked."""
        self._current_face_id = face_id
        self._remove_face_btn.setEnabled(True)
        self._reassign_btn.setEnabled(True)
        # Ensure the preview panel has the image loaded so "Edit bbox" works.
        self._show_face_in_preview(face_id)
        person_name = self._cluster_panel.get_face_person_name(face_id)
        self._preview_panel.show_face_context_menu(face_id, gx, gy, person_name=person_name)

    @Slot(int)
    def _on_preview_face_selected(self, face_id: int) -> None:
        """Handle a face click directly in the preview image (not from the thumbnail grid)."""
        self._current_face_id = face_id

        with session_scope() as session:
            face = session.get(Face, face_id)
            if face and face.person_id:
                self._current_person_id = face.person_id

        self._remove_face_btn.setEnabled(True)
        self._reassign_btn.setEnabled(True)
        self._notify_recording_context()

    @Slot(int)
    def _on_preview_face_assign(self, face_id: int) -> None:
        """Handle 'Személyhez adás' from the preview panel."""
        self._on_preview_face_selected(face_id)
        self._on_reassign_face()

    @Slot(int)
    def _on_preview_face_diagnostics(self, face_id: int) -> None:
        """Show the 'why this identity?' developer diagnostics for a face."""
        from app.app_settings import app_qsettings
        from app.services.face_diagnostics_service import FaceDiagnosticsService
        from app.ui.dialogs.face_diagnostics_dialog import FaceDiagnosticsDialog

        exclude_low_quality = app_qsettings().value(
            "face_quality/exclude_low_quality", True, type=bool
        )
        try:
            with session_scope() as session:
                svc = FaceDiagnosticsService(
                    session,
                    config=self._config.recognition,
                    exclude_low_quality=exclude_low_quality,
                )
                diag = svc.explain(face_id)
        except Exception as exc:  # noqa: BLE001
            log.exception("Face diagnostics failed")
            QMessageBox.critical(self, t("error"), t("diag_error", error=exc))
            return

        if diag is None:
            return
        FaceDiagnosticsDialog(diag, parent=self).exec()

    @Slot(int)
    def _on_face_set_thumbnail(self, face_id: int) -> None:
        """Set a face crop as the manual thumbnail for its person."""
        with session_scope() as session:
            face = session.get(Face, face_id)
            if face is None or face.person_id is None:
                return
            try:
                IdentityService(session).set_person_thumbnail(
                    face.person_id, face_id
                )
            except ValueError as exc:
                QMessageBox.warning(
                    self,
                    t("thumbnail_set_error", error="", default="Thumbnail"),
                    t("thumbnail_set_error", error=str(exc)),
                )
                return
        self._refresh_persons()

    @Slot(int)
    def _on_face_clear_thumbnail(self, face_id: int) -> None:
        """Reset a person's thumbnail to automatic selection."""
        with session_scope() as session:
            face = session.get(Face, face_id)
            if face is None or face.person_id is None:
                return
            IdentityService(session).clear_manual_person_thumbnail(face.person_id)
        self._refresh_persons()

    def _on_preview_face_delete(self, face_id: int) -> None:
        """Handle 'Arc törlése' from the preview panel context menu (hard delete)."""
        reply = QMessageBox.question(
            self,
            t("remove_face_title"),
            t("remove_face_msg"),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        with session_scope() as session:
            face = session.get(Face, face_id)
            if face is None:
                return
            session.delete(face)

        log.info("Face %d deleted from preview panel.", face_id)
        self._preview_panel.clear()
        if self._current_person_id is not None:
            self._on_person_selected(self._current_person_id)
        self._image_browser._reload_current_face_data()

    @Slot(int, int, int, int, int)
    def _on_preview_face_create(
        self, image_id: int, x: int, y: int, w: int, h: int
    ) -> None:
        """Save a manually marked face from the face-recognition preview."""
        from app.db.models import Image
        from app.detectors.base import Detection
        from app.utils.image_utils import load_image_bgr_normalized as load_image_bgr, save_face_crop

        log.debug(
            "face create: image_id=%d bbox=(%d,%d,%d,%d)", image_id, x, y, w, h
        )

        # Load image for clamping bbox to actual dimensions.
        # If the file is temporarily inaccessible we still create the face
        # (unclamped) so the annotation is not silently lost.
        img_path: Optional[str] = None
        img_bgr = None
        with session_scope() as session:
            image = session.get(Image, image_id)
            if image is None:
                log.warning("face create: image_id=%d not found in DB", image_id)
                return
            img_path = image.file_path

        img_bgr = load_image_bgr(img_path) if img_path else None
        if img_bgr is None:
            log.warning(
                "face create: cannot load image file %r — creating face with unclamped bbox",
                img_path,
            )

        # Clamp bbox to image dimensions when the image is available.
        if img_bgr is not None:
            detection = Detection(x=x, y=y, w=w, h=h, confidence=1.0).clamp(
                img_bgr.shape[1], img_bgr.shape[0]
            )
        else:
            detection = Detection(x=x, y=y, w=w, h=h, confidence=1.0)

        # Commit face row first (atomically) so it always lands in the DB.
        new_face_id: Optional[int] = None
        with session_scope() as session:
            face = Face(
                image_id=image_id,
                bbox_x=detection.x,
                bbox_y=detection.y,
                bbox_w=detection.w,
                bbox_h=detection.h,
                confidence=1.0,
                detector_backend="manual",
                crop_path=None,
            )
            session.add(face)
            session.flush()
            new_face_id = face.id
            log.debug("bbox added to state: face_id=%d", new_face_id)

            # Save crop in the same transaction so face_id is available for naming.
            if img_bgr is not None:
                crops_dir = self._config.crops_dir_resolved
                crops_dir.mkdir(parents=True, exist_ok=True)
                try:
                    crop_path = save_face_crop(
                        img_bgr=img_bgr,
                        detection=detection,
                        crops_dir=crops_dir,
                        image_id=image_id,
                        thumbnail_size=self._config.scan.thumbnail_size,
                        face_index=new_face_id,
                    )
                    if crop_path is not None:
                        face.crop_path = str(crop_path)
                except Exception:
                    log.exception(
                        "face create: crop save failed for face_id=%d — face will have no thumbnail",
                        new_face_id,
                    )

        if new_face_id is None:
            log.warning("face create: face_id is None after flush — aborting")
            return

        log.info(
            "Manual face added from preview: image_id=%d face_id=%d bbox=(%d,%d,%d,%d)",
            image_id, new_face_id, detection.x, detection.y, detection.w, detection.h,
        )
        self._current_face_id = new_face_id
        self._show_face_in_preview(new_face_id)
        self._remove_face_btn.setEnabled(True)
        self._reassign_btn.setEnabled(True)
        self._refresh_persons()
        self._image_browser._reload_current_face_data()

    @Slot(int, int, int, int, int)
    def _on_preview_face_bbox_update(
        self, face_id: int, x: int, y: int, w: int, h: int
    ) -> None:
        """Update a face bounding box redrawn in the face-recognition preview."""
        old_crop_path: Optional[str] = None
        new_crop_path: Optional[str] = None

        with session_scope() as session:
            from app.detectors.base import Detection
            from app.services.face_crop_service import (
                crop_path_is_shared,
                face_debug_state,
                save_crop_for_face,
            )
            from app.utils.image_utils import load_image_bgr_normalized

            face = session.get(Face, face_id)
            if face is None or face.image is None:
                return
            img_bgr = load_image_bgr_normalized(face.image.file_path)
            if img_bgr is None:
                return

            detection = Detection(x=x, y=y, w=w, h=h, confidence=1.0).clamp(
                img_bgr.shape[1], img_bgr.shape[0]
            )
            face.bbox_x = detection.x
            face.bbox_y = detection.y
            face.bbox_w = detection.w
            face.bbox_h = detection.h
            if face.image:
                _ = face.image.file_path
            if face.person:
                _ = face.person.name

            crops_dir = self._config.crops_dir_resolved
            crops_dir.mkdir(parents=True, exist_ok=True)

            old_crop_path = face.crop_path
            shared_before = crop_path_is_shared(session, face)
            if shared_before:
                log.warning(
                    "Shared crop path before bbox update: %s",
                    face_debug_state(face, old_crop_path),
                )
            crop_path = save_crop_for_face(
                face,
                crops_dir=crops_dir,
                thumbnail_size=self._config.scan.thumbnail_size,
                img_bgr=img_bgr,
            )
            if crop_path is not None:
                new_crop_path = str(crop_path)

        # Invalidate Qt's pixmap cache so the updated thumbnail is shown immediately.
        if old_crop_path or new_crop_path:
            from PySide6.QtGui import QPixmapCache
            if old_crop_path:
                QPixmapCache.remove(old_crop_path)
            if new_crop_path:
                QPixmapCache.remove(new_crop_path)

        log.info(
            "Face %d bbox updated from preview: bbox=(%d,%d,%d,%d) crop=%s",
            face_id, x, y, w, h, new_crop_path,
        )
        self._current_face_id = face_id
        self._show_face_in_preview(face_id)
        if self._current_person_id is not None:
            self._on_person_selected(self._current_person_id)
            self._show_face_in_preview(face_id)
        self._image_browser._reload_current_face_data()

    # ------------------------------------------------------------------
    # Identity actions
    # ------------------------------------------------------------------

    @Slot()
    def _on_rename(self) -> None:
        if self._current_person_id is None:
            return

        with session_scope() as session:
            person = session.get(Person, self._current_person_id)
            if person is None:
                return
            if person.is_protected:
                QMessageBox.warning(self, t("protected_rename_title"), t("protected_rename_msg"))
                return
            dlg = RenameDialog(person.name, parent=self)
            if dlg.exec() != RenameDialog.Accepted:
                return
            new_name = dlg.new_name()
            if not new_name:
                QMessageBox.warning(self, t("empty_name_title"), t("empty_name_msg"))
                return
            IdentityService(session).rename_person(self._current_person_id, new_name)

        self._refresh_persons()
        self._image_browser._reload_current_face_data()

    @Slot()
    def _on_merge(self) -> None:
        if self._current_person_id is None:
            return

        with session_scope() as session:
            persons = session.query(Person).order_by(Person.name).all()
            for p in persons:
                _ = p.faces  # noqa: F841
            source = session.get(Person, self._current_person_id)
            if source is None or len(persons) < 2:
                return

            # Compute face-match scores for the source person so the
            # MergeDialog's person selector can show probabilistic name
            # suggestions (same behaviour as the image-browser inline editor).
            recognition_cfg = getattr(self._config, "recognition", None) if self._config else None
            match_scores = {}
            try:
                from app.services.recognition_service import RecognitionService
                svc = RecognitionService(session, recognition_cfg)
                # First try the normal path: build recognition profiles and use
                # the source's profile centroid when available.
                profiles = svc.build_profiles()
                prof = profiles.get(source.id)
                if prof is not None:
                    match_scores = svc.score_persons(prof.centroid)

                # Fallback: if no profile/scores (e.g. source is unknown/auto-named)
                # compute a centroid from the source person's face embeddings and
                # score that against per-person centroids built from every
                # person's available embeddings (includes auto-named/unknown).
                if not match_scores:
                    try:
                        import numpy as np

                        # Build normalized centroid for the source person
                        src_vecs = []
                        for f in source.faces:
                            fv = f.get_embedding()
                            if fv is None:
                                continue
                            fn = float(np.linalg.norm(fv))
                            if fn > 0:
                                src_vecs.append(fv / fn)
                        if src_vecs:
                            src_centroid = np.mean(src_vecs, axis=0)
                            scn = float(np.linalg.norm(src_centroid))
                            if scn > 0:
                                src_centroid = src_centroid / scn

                                # Score against every person by their centroid
                                fb: dict[int, float] = {}
                                for p in persons:
                                    vecs = []
                                    for f in p.faces:
                                        fv = f.get_embedding()
                                        if fv is None:
                                            continue
                                        fn = float(np.linalg.norm(fv))
                                        if fn > 0:
                                            vecs.append(fv / fn)
                                    if not vecs:
                                        continue
                                    centroid = np.mean(vecs, axis=0)
                                    cn = float(np.linalg.norm(centroid))
                                    if cn <= 0:
                                        continue
                                    centroid = centroid / cn
                                    score = float(np.dot(src_centroid, centroid))
                                    fb[p.id] = score
                                match_scores = fb
                    except Exception:
                        log.exception("Merge dialog fallback scoring failed; continuing without scores")
            except Exception:
                log.exception("Failed to compute match scores for merge dialog; continuing without scores")

            dlg = MergeDialog(source, persons, parent=self, match_scores=match_scores)
            if dlg.exec() != MergeDialog.Accepted:
                return
            target_id = dlg.target_person_id()
            if target_id is None:
                return

            try:
                IdentityService(session).merge_persons(
                    source_id=self._current_person_id, target_id=target_id
                )
            except ValueError as exc:
                QMessageBox.warning(self, t("merge_error_title"), str(exc))
                return

        self._current_person_id = None
        self._cluster_panel.clear()
        self._preview_panel.clear()
        self._refresh_persons()
        self._image_browser._reload_current_face_data()

    @Slot()
    def _on_delete_person(self) -> None:
        if self._current_person_id is None:
            return

        with session_scope() as session:
            person = session.get(Person, self._current_person_id)
            if person is None:
                return
            if person.is_protected:
                QMessageBox.warning(self, t("protected_delete_title"), t("protected_delete_msg"))
                return
            name = person.name

        reply = QMessageBox.question(
            self,
            t("delete_person_title"),
            t("delete_person_confirm", name=name),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        with session_scope() as session:
            IdentityService(session).delete_person(self._current_person_id)

        self._current_person_id = None
        self._current_face_id = None
        self._cluster_panel.clear()
        self._preview_panel.clear()
        self._delete_person_btn.setEnabled(False)
        self._rename_btn.setEnabled(False)
        self._merge_btn.setEnabled(False)
        self._person_info_btn.setEnabled(False)
        self._refresh_persons()
        self._image_browser._reload_current_face_data()
        log.info("Person '%s' deleted.", name)

    @Slot()
    def _on_remove_face(self) -> None:
        if self._current_face_id is None or self._current_person_id is None:
            return

        reply = QMessageBox.question(
            self,
            t("remove_face_title"),
            t("remove_face_msg"),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        with session_scope() as session:
            IdentityService(session).remove_face_from_cluster(self._current_face_id)

        self._on_person_selected(self._current_person_id)

    @Slot()
    def _on_reassign_face(self) -> None:
        if self._current_face_id is None:
            return

        with session_scope() as session:
            persons = session.query(Person).order_by(Person.name).all()
            for p in persons:
                _ = p.faces  # noqa: F841

            class _FakePerson:
                name = f"Face #{self._current_face_id}"
                id = -1
                faces: list = []

            # Optional face-match ordering: score the face being reassigned
            # against the known people so the user can sort candidates by
            # similarity.  A missing embedding yields an empty mapping, which
            # the selector treats as "no match data" (default ordering).
            match_scores: dict[int, float] = {}
            face_for_scoring = session.get(Face, self._current_face_id)
            if face_for_scoring is not None:
                emb = face_for_scoring.get_embedding()
                if emb is not None:
                    try:
                        match_scores = RecognitionService(
                            session, self._config.recognition
                        ).score_persons(emb) or {}
                    except Exception:  # noqa: BLE001
                        log.exception("Face-match scoring failed; using default order")
                        match_scores = {}

                    # Fallback: if no scores were produced (e.g. recognition
                    # profiles exclude auto-named people), build per-person
                    # centroids from available face embeddings so unknown/auto
                    # persons are also considered.
                    if not match_scores:
                        try:
                            import numpy as np

                            norm = float(np.linalg.norm(emb))
                            if norm > 0:
                                nemb = emb / norm
                                fb: dict[int, float] = {}
                                for p in persons:
                                    vecs = []
                                    for f in p.faces:
                                        fv = f.get_embedding()
                                        if fv is None:
                                            continue
                                        fn = float(np.linalg.norm(fv))
                                        if fn > 0:
                                            vecs.append(fv / fn)
                                    if not vecs:
                                        continue
                                    centroid = np.mean(vecs, axis=0)
                                    cn = float(np.linalg.norm(centroid))
                                    if cn <= 0:
                                        continue
                                    centroid = centroid / cn
                                    score = float(np.dot(nemb, centroid))
                                    fb[p.id] = score
                                match_scores = fb
                        except Exception:
                            log.exception("Fallback face-match scoring failed; continuing without scores")

            dlg = MergeDialog(
                _FakePerson(), persons, parent=self, match_scores=match_scores
            )
            dlg.setWindowTitle(t("reassign_title"))
            if dlg.exec() != MergeDialog.Accepted:
                return
            target_id = dlg.target_person_id()
            if target_id is None:
                return
            face = session.get(Face, self._current_face_id)
            log.debug(
                "Reassign face: face_id=%d crop=%s bbox=(%s,%s,%s,%s) "
                "from person_id=%s → target_person_id=%d",
                self._current_face_id,
                face.crop_path if face else "?",
                face.bbox_x if face else "?", face.bbox_y if face else "?",
                face.bbox_w if face else "?", face.bbox_h if face else "?",
                self._current_person_id, target_id,
            )
            IdentityService(session).reassign_face(self._current_face_id, target_id)
            log.info(
                "Reassign done: face_id=%d → person_id=%d",
                self._current_face_id, target_id,
            )

        if self._current_person_id:
            self._on_person_selected(self._current_person_id)
        self._show_face_in_preview(self._current_face_id)
        self._image_browser._reload_current_face_data()

    # ------------------------------------------------------------------
    # Batch face move (multi-select in the cluster grid)
    # ------------------------------------------------------------------

    @Slot(int)
    def _on_face_selection_changed(self, count: int) -> None:
        if count <= 0:
            self._sel_count_lbl.setText("")
        else:
            self._sel_count_lbl.setText(t("faces_selected_n", n=count))
        self._move_faces_btn.setVisible(count > 0)

    @Slot()
    def _on_move_faces_batch(self) -> None:
        face_ids = self._cluster_panel.selected_face_ids()
        if not face_ids:
            return

        src_name = ""
        with session_scope() as session:
            if self._current_person_id is not None:
                src = session.get(Person, self._current_person_id)
                src_name = src.name if src is not None else ""
            persons = session.query(Person).order_by(Person.name).all()
            dlg = MoveFacesDialog(
                face_count=len(face_ids),
                persons=persons,
                exclude_person_id=self._current_person_id,
                parent=self,
            )
        if dlg.exec() != MoveFacesDialog.Accepted:
            return

        target_id = dlg.selected_person_id()
        new_name = dlg.new_person_name()

        if new_name:
            dst_name = new_name
        elif target_id is not None:
            if target_id == self._current_person_id:
                QMessageBox.information(
                    self, t("move_faces_title"), t("move_faces_same_person")
                )
                return
            with session_scope() as session:
                target = session.get(Person, target_id)
                dst_name = target.name if target is not None else ""
        else:
            return

        reply = QMessageBox.question(
            self,
            t("move_faces_confirm_title"),
            t("move_faces_confirm_msg", n=len(face_ids), src=src_name, dst=dst_name),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            with session_scope() as session:
                svc = IdentityService(session)
                if new_name:
                    person = Person(name=new_name, is_auto_named=False)
                    session.add(person)
                    session.flush()
                    target_id = person.id
                result = svc.reassign_faces_bulk(face_ids, target_id)
        except Exception as exc:  # noqa: BLE001
            log.exception("Bulk face move failed")
            QMessageBox.critical(
                self, t("move_faces_error_title"), t("persons_save_error", error=str(exc))
            )
            return

        self._cluster_panel.clear_selection()
        if self._current_person_id is not None:
            self._on_person_selected(self._current_person_id)
        self._image_browser._reload_current_face_data()

        # Offer an undo via a dialog button.
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle(t("move_faces_title"))
        box.setText(t("move_faces_done", n=result.moved_count, dst=dst_name))
        undo_btn = box.addButton(t("move_faces_undo"), QMessageBox.AcceptRole)
        box.addButton(QMessageBox.Ok)
        box.exec()
        if box.clickedButton() is undo_btn:
            self._undo_face_move(result)

    def _undo_face_move(self, result: BulkReassignResult) -> None:
        try:
            with session_scope() as session:
                IdentityService(session).restore_face_assignments(
                    result.snapshots, result.removed_persons
                )
        except Exception as exc:  # noqa: BLE001
            log.exception("Undo of bulk face move failed")
            QMessageBox.critical(
                self,
                t("move_faces_error_title"),
                t("move_faces_undo_error", error=str(exc)),
            )
            return
        if self._current_person_id is not None:
            self._on_person_selected(self._current_person_id)
        self._image_browser._reload_current_face_data()

    @Slot()
    def _on_person_info(self) -> None:
        if self._current_person_id is None:
            return

        with session_scope() as session:
            person = session.get(Person, self._current_person_id)
            if person is None:
                return
            dlg = PersonInfoDialog(person, parent=self)
            if dlg.exec() != PersonInfoDialog.Accepted:
                return
            person.gender = dlg.gender()
            person.family_code = dlg.family_code() or None
            person.external_family_code = dlg.external_family_code() or None
            person.last_name = dlg.last_name() or None
            person.first_name = dlg.first_name() or None
            person.second_name = dlg.second_name() or None
            person.nickname = dlg.nickname() or None
            person.married_name = dlg.married_name() or None
            person.birth_place = dlg.birth_place() or None
            person.birth_date = dlg.birth_date() or None
            person.death_date = dlg.death_date() or None
            person.death_place = dlg.death_place() or None
            person.notes = dlg.notes() or None
            session.flush()
            try:
                from app.services.family_service import FamilyService
                FamilyService(session).link_derived_parents(person.id)
            except Exception:
                log.exception("Could not link derived parents for person %d", person.id)

        log.info(
            "Személyadatok mentve: %s %s",
            dlg.last_name(), dlg.first_name()
        )
        # Reflect the saved structured data in the sidebar and the Persons tab
        # table (the dialog only wrote to the DB).
        self._refresh_persons()

    @Slot()
    def _on_recluster(self) -> None:
        reply = QMessageBox.question(
            self,
            t("recluster_title"),
            t("recluster_msg"),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._status_label.setText(t("reclustering"))
        QApplication.processEvents()

        with session_scope() as session:
            assignments, _stats = RecognitionService(
                session, self._config.recognition
            ).recognize_pending()
            n = len(assignments)

        self._status_label.setText(t("recluster_done", n=n))
        self._refresh_persons()

    # ------------------------------------------------------------------
    # Collage import / export
    # ------------------------------------------------------------------

    @Slot()
    def _on_import_collage(self) -> None:
        """Import one or more Picasa collage files (.cxf / .cfx)."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            t("open_collage_file"),
            _last_dir("paths/last_collage", str(Path.home())),
            t("picasa_collage_filter"),
        )
        if not files:
            return
        _save_dir("paths/last_collage", str(Path(files[0]).parent))

        # Optionally ask for an extra search root to resolve Windows paths
        search_root = QFileDialog.getExistingDirectory(
            self,
            t("extra_search_root"),
            _last_dir("paths/last_search_root", str(Path.home())),
        )
        if search_root:
            _save_dir("paths/last_search_root", search_root)
        search_roots = [search_root] if search_root else []

        imported, errors = 0, []
        for fpath in files:
            try:
                with session_scope() as session:
                    from app.services.collage_service import CollageService
                    svc = CollageService(session)
                    collage = svc.import_collage(fpath, search_roots=search_roots)
                    log.info(
                        "Kollázs importálva: %s  (%d elem, %d hiányzó)",
                        collage.album_title or fpath,
                        len(collage.nodes),
                        sum(1 for n in collage.nodes if n.src_missing),
                    )
                imported += 1
            except Exception as exc:
                log.error("Kollázs import hiba (%s): %s", fpath, exc)
                errors.append(f"{Path(fpath).name}: {exc}")

        # Switch to collage tab and refresh
        self._collage_panel.refresh_collage_list()
        self._tabs.setCurrentIndex(5)

        msg = t("collages_imported", n=imported)
        if errors:
            msg += f"\n\n{t('import_errors')}\n" + "\n".join(errors)
            QMessageBox.warning(self, t("collage_import_title"), msg)
        else:
            self._status_label.setText(msg)
            log.info(msg)

    @Slot()
    def _on_export_collage_html(self) -> None:
        """Export all collages to a static HTML gallery."""
        target = QFileDialog.getExistingDirectory(
            self,
            t("html_export_folder"),
            _last_dir("paths/last_export", str(Path.home())),
        )
        if not target:
            return
        _save_dir("paths/last_export", target)

        try:
            with session_scope() as session:
                from app.services.export_service import ExportService
                out = ExportService(session).export_collage_html(target)
            QMessageBox.information(
                self, t("collage_html_export"),
                t("static_site_ready", path=out)
            )
        except Exception as exc:
            log.exception("Collage HTML export failed")
            QMessageBox.critical(self, t("export_error"), str(exc))

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    @Slot()
    def _on_open_export(self) -> None:
        person_name: Optional[str] = None
        if self._current_person_id is not None:
            with session_scope() as session:
                p = session.get(Person, self._current_person_id)
                if p:
                    person_name = p.name
        dlg = ExportDialog(
            current_person_id=self._current_person_id,
            current_person_name=person_name,
            current_image_id=self._preview_panel.current_image_id,
            on_collage_import=self._on_import_collage,
            on_collage_html_export=self._on_export_collage_html,
            on_project_export=self._on_export_project_package,
            on_project_import=self._on_import_project_package,
            parent=self,
        )
        dlg.exec()

    # ------------------------------------------------------------------
    # Full project package (.facepack) export / import
    # ------------------------------------------------------------------

    def _build_package_service(self) -> "ProjectPackageService":
        from app.services.image_library_service import get_image_library_optional
        from app.services.project_package_service import ProjectPackageService

        lib = get_image_library_optional()
        return ProjectPackageService(
            db_path=self._db_path,
            crops_dir=self._config.crops_dir_resolved,
            library_root=lib.library_root if lib else None,
            local_config_path=lib.local_config_path() if lib else None,
        )

    @Slot()
    def _on_export_project_package(self) -> None:
        from app.services.project_package_service import (
            PACKAGE_EXTENSION,
            ProjectPackageError,
        )

        default_name = f"face_local_project{PACKAGE_EXTENSION}"
        start_dir = _last_dir("paths/last_package", str(Path.home()))
        path, _ = QFileDialog.getSaveFileName(
            self,
            t("pkg_export_dialog"),
            str(Path(start_dir) / default_name),
            t("pkg_filter"),
        )
        if not path:
            return
        _save_dir("paths/last_package", str(Path(path).parent))

        try:
            from app import __version__ as app_version
        except Exception:  # noqa: BLE001
            app_version = ""

        progress = QProgressDialog(
            t("pkg_progress_export"), None, 0, 100, self
        )
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setCancelButton(None)
        progress.setValue(0)

        def on_progress(cur: int, total: int, _msg: str) -> None:
            pct = int(cur / total * 100) if total else 0
            progress.setValue(min(pct, 100))
            QApplication.processEvents()

        try:
            svc = self._build_package_service()
            result = svc.export_package(
                path, app_version=str(app_version), progress_cb=on_progress
            )
        except ProjectPackageError as exc:
            progress.close()
            log.exception("Project package export failed")
            QMessageBox.critical(self, t("pkg_error_title"), str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            progress.close()
            log.exception("Project package export failed")
            QMessageBox.critical(self, t("pkg_error_title"), str(exc))
            return
        finally:
            progress.close()

        msg = t(
            "pkg_export_ok",
            images=result.image_count,
            crops=result.crop_count,
            path=result.package_path,
        )
        if result.warning_count:
            msg += t("pkg_export_warn", missing=result.warning_count)
        QMessageBox.information(self, t("pkg_export_done"), msg)

    @Slot()
    def _on_import_project_package(self) -> None:
        from app.services.project_package_service import ProjectPackageError, ProjectPackageService

        path, _ = QFileDialog.getOpenFileName(
            self,
            t("pkg_import_dialog"),
            _last_dir("paths/last_package", str(Path.home())),
            t("pkg_filter"),
        )
        if not path:
            return
        _save_dir("paths/last_package", str(Path(path).parent))

        validation = ProjectPackageService.validate_package(path)
        if not validation.ok:
            QMessageBox.critical(
                self, t("pkg_error_title"), "\n".join(validation.errors)
            )
            return

        dest = QFileDialog.getExistingDirectory(
            self, t("pkg_import_dest"), str(Path.home())
        )
        if not dest:
            return
        # Extract into a dedicated, non-empty-safe subfolder of the chosen dir.
        dest_dir = Path(dest) / Path(path).stem
        if dest_dir.exists() and any(dest_dir.iterdir()):
            from datetime import datetime
            dest_dir = Path(dest) / f"{Path(path).stem}_{datetime.now():%Y%m%d_%H%M%S}"

        reply = QMessageBox.question(
            self,
            t("pkg_import_title"),
            t("pkg_import_confirm", dest=dest_dir),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        progress = QProgressDialog(
            t("pkg_progress_import"), None, 0, 100, self
        )
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setCancelButton(None)
        progress.setValue(0)

        def on_progress(cur: int, total: int, _msg: str) -> None:
            pct = int(cur / total * 100) if total else 0
            progress.setValue(min(pct, 100))
            QApplication.processEvents()

        try:
            result = ProjectPackageService.import_package(
                path, dest_dir, progress_cb=on_progress
            )
        except ProjectPackageError as exc:
            progress.close()
            log.exception("Project package import failed")
            QMessageBox.critical(self, t("pkg_error_title"), str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            progress.close()
            log.exception("Project package import failed")
            QMessageBox.critical(self, t("pkg_error_title"), str(exc))
            return
        finally:
            progress.close()

        reply = QMessageBox.question(
            self,
            t("pkg_import_title"),
            t("pkg_import_ok", dest=result.project_dir),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._activate_imported_project(result)
        else:
            QMessageBox.information(
                self, t("pkg_import_title"),
                t("pkg_import_later", dest=result.project_dir),
            )

    def _activate_imported_project(self, result) -> None:
        """Open the freshly imported project as the active database."""
        from app.services.image_library_service import get_image_library_optional
        from app.services.project_package_service import ProjectPackageService

        new_db = str(result.db_path)
        self._db_path = new_db
        self._config.storage.db_path = new_db
        self._config.storage.crops_dir = str(result.crops_dir)
        save_db_path(new_db)
        init_db(new_db)  # also re-initialises the ImageLibraryService
        ensure_unknown_person()

        # Point the library root at the extracted images and remap crop paths,
        # then migrate any legacy absolute paths to portable relative paths.
        lib = get_image_library_optional()
        if lib is not None:
            try:
                lib.set_library_root(result.images_dir)
            except OSError:
                log.warning("Imported images dir missing: %s", result.images_dir)
        with session_scope() as session:
            ProjectPackageService.remap_crop_paths(session, result.crops_dir)
            if lib is not None and lib.is_available():
                try:
                    lib.migrate_to_relative_paths(
                        session, result.images_dir, validate_files=False
                    )
                except Exception:  # noqa: BLE001
                    log.exception("Relative-path migration after import failed")

        self._current_person_id = None
        self._current_face_id = None
        self._cluster_panel.clear()
        self._preview_panel.clear()
        self._refresh_persons()
        self._image_browser.refresh()
        if hasattr(self, "_locations_panel"):
            self._locations_panel.refresh()
        if hasattr(self, "_persons_panel"):
            self._persons_panel.refresh()
        if hasattr(self, "_objects_panel"):
            self._objects_panel.refresh()
        QMessageBox.information(
            self, t("pkg_import_title"), t("pkg_import_opened")
        )
        log.info("Activated imported project: %s", new_db)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_scanning_state(self, scanning: bool) -> None:
        self._scan_modes_btn.setEnabled(not scanning)
        self._stop_btn.setEnabled(scanning)
        self._progress_bar.setVisible(scanning)
        if scanning:
            self._progress_bar.setValue(0)

    # ------------------------------------------------------------------
    # System tray
    # ------------------------------------------------------------------

    def _setup_tray(self) -> None:
        self._tray = QSystemTrayIcon(self)
        icon = self.windowIcon()
        if icon.isNull():
            icon = QIcon.fromTheme(
                "dialog-information",
                self.style().standardIcon(self.style().StandardPixmap.SP_ComputerIcon),
            )
        self._tray.setIcon(icon)
        self._tray.setToolTip("Face-Local")
        if QSystemTrayIcon.isSystemTrayAvailable():
            self._tray.show()

    def _notify(self, title: str, message: str) -> None:
        """Send a system tray notification if enabled in settings."""
        from app.app_settings import app_qsettings
        enabled = app_qsettings().value("updates/notify", True, type=bool)
        if not enabled:
            return
        if QSystemTrayIcon.isSystemTrayAvailable() and QSystemTrayIcon.supportsMessages():
            self._tray.showMessage(title, message, QSystemTrayIcon.MessageIcon.Information, 6000)
        else:
            # Fallback: status bar
            self._status_label.setText(f"{title}: {message}")

    # ------------------------------------------------------------------
    # Update check
    # ------------------------------------------------------------------

    def _start_update_check(self) -> None:
        """Background update check at startup.

        On macOS, when SparkleHelper is bundled inside the .app, delegates to
        Sparkle for a native background check and returns early.  Sparkle will
        show its own update sheet if a new version is found; the Qt dialog is
        not used in that path.

        On all other platforms (and on macOS outside a frozen bundle, e.g. when
        running from source), falls back to the GitHub-API-based check.
        """
        import sys
        if sys.platform == "darwin":
            from app.updater import start_background_update_check
            if start_background_update_check():
                # SparkleHelper is running; it handles everything from here.
                return

        from app import __version__
        from app.services.update_service import fetch_latest_release, is_newer

        signal = self._update_ready

        class _CheckThread(QThread):
            def run(self_inner) -> None:  # noqa: N805
                release = fetch_latest_release()
                if release and is_newer(release.version, __version__):
                    signal.emit(release)

        self._check_thread = _CheckThread(self)
        self._check_thread.start()

    @Slot(object)
    def _on_update_found(self, release) -> None:
        self._pending_release = release
        self._update_notify_btn.setText(
            t("status_update_available", version=release.version)
        )
        self._update_notify_btn.setVisible(True)
        self._status_label.setText(t("update_status_found", version=release.version))
        self._notify(
            t("update_notification_title"),
            t("update_notification_msg", version=release.version),
        )

    @Slot()
    def _on_update_notify_clicked(self) -> None:
        if self._pending_release:
            dlg = UpdateDialog(self._pending_release, parent=self, auto_start=True)
            dlg.exec()

    def _refresh_persons(self) -> None:
        from sqlalchemy.orm import selectinload
        with session_scope() as session:
            # Eager-load faces and each face's image in two batched queries.
            # The old code lazy-loaded them, which meant a SELECT per person
            # for .faces plus a SELECT per face for .image (the explicit
            # prewarm loop below).  With many persons/faces that N+1 fan-out
            # blocked the UI thread for seconds on every refresh — e.g. after
            # each single face assignment, which is what froze the browser.
            persons: List[Person] = (
                session.query(Person)
                .order_by(Person.name)
                .options(selectinload(Person.faces).selectinload(Face.image))
                .all()
            )
            from app.services.face_crop_service import ensure_unique_face_crops
            all_faces = [
                f
                for p in persons
                for f in p.faces
                if not f.is_excluded
            ]
            ensure_unique_face_crops(
                session,
                all_faces,
                self._config.crops_dir_resolved,
                self._config.scan.thumbnail_size,
            )
            self._sidebar.populate(persons)
        if hasattr(self, "_family_search"):
            self._family_search.refresh()
        # Keep the standalone Persons tab table in sync with edits made
        # anywhere (image browser, sidebar dialog, batch ops).  It reloads
        # immediately when visible, otherwise lazily on next show.
        if hasattr(self, "_persons_panel"):
            self._persons_panel.mark_stale()
        log.debug("Sidebar refreshed: %d person(s)", len(persons))

    @Slot(int)
    def _open_image_from_family_search(self, image_id: int) -> None:
        self._tabs.setCurrentIndex(1)
        self._image_browser.open_image_by_id(image_id)

    # ------------------------------------------------------------------
    # Image library startup check
    # ------------------------------------------------------------------

    def _check_image_library_on_startup(self) -> None:
        """Show a dialog if the library root is configured but not found."""
        from app.services.image_library_service import get_image_library_optional

        svc = get_image_library_optional()
        if svc is None:
            return

        root = svc.library_root
        if root is None:
            return  # Not configured — no warning needed.

        if svc.is_available():
            log.debug("Image library root available: %s", root)
            return

        # Root is configured but not reachable — prompt the user.
        from app.ui.dialogs.image_library_dialog import ImageLibraryMissingDialog

        dlg = ImageLibraryMissingDialog(missing_path=str(root), parent=self)
        if dlg.exec() == ImageLibraryMissingDialog.Accepted:
            new_root = dlg.new_root()
            if new_root:
                try:
                    svc.set_library_root(new_root)
                    log.info("Image library root updated to: %s", new_root)
                except (NotADirectoryError, RuntimeError) as exc:
                    from app.ui.i18n import t
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.warning(self, t("error"), str(exc))

    # ------------------------------------------------------------------
    # Window close
    # ------------------------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        """Intercept close to shut down an open Drive session gracefully."""
        # Finalize any in-progress recording so segments + timeline are saved.
        if self._recorder is not None:
            from app.services.screen_recorder_service import RecorderState
            if self._recorder.state in (
                RecorderState.RECORDING,
                RecorderState.PAUSED,
            ):
                self._on_record_stop()
        if self._gdrive_session is not None and not self._gdrive_closing:
            # Don't actually close yet — let the Drive thread finish, then quit.
            event.ignore()
            self._gdrive_closing = True
            self._gdrive_chip_btn.setText(t("gdrive_chip_closing"))
            self._status_label.setText(t("gdrive_closing_wait"))
            self._start_drive_close(after_quit=True)
            return
        self._shutdown_match_worker()
        event.accept()
