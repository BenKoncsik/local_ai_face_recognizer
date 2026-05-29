"""Main application window."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QEvent, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QCloseEvent, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
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
from app.logging_setup import QLogHandler
from app.paths import app_icon_path
from app.services.duplicate_unknown_face_finder import DuplicateUnknownFaceFinder
from app.services.identity_service import IdentityService
from app.services.recognition_service import RecognitionService
from app.ui.dialogs.export_dialog import ExportDialog
from app.ui.dialogs.manual_face_dialog import NoFaceImagesDialog
from app.ui.dialogs.merge_dialog import MergeDialog
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
from app.ui.panels.preview_panel import PreviewPanel
from app.ui.panels.sidebar_panel import SidebarPanel
from app.workers.pipeline_worker import PipelineWorker

log = logging.getLogger(__name__)


def _last_dir(key: str, default: str = "") -> str:
    """Read a remembered directory path from persistent settings."""
    from PySide6.QtCore import QSettings

    return QSettings("FaceLocal", "FaceLocal").value(key, default, type=str)


def _save_dir(key: str, path: str) -> None:
    """Persist a directory path so the next file dialog reopens there."""
    from PySide6.QtCore import QSettings

    QSettings("FaceLocal", "FaceLocal").setValue(key, path)


class MainWindow(QMainWindow):
    """Primary application window."""

    log_signal = Signal(str, int)
    _update_ready = Signal(object)   # ReleaseInfo

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self._config = config
        self._worker: Optional[PipelineWorker] = None
        self._current_person_id: Optional[int] = None
        self._current_face_id: Optional[int] = None
        self._pending_suggestion_count: int = 0
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

    # ------------------------------------------------------------------
    # Keyboard shortcuts
    # ------------------------------------------------------------------

    def _setup_shortcuts(self) -> None:
        """Install app-level event filter and register global shortcut handlers."""
        from app.services.shortcut_service import get_shortcut_service
        svc = get_shortcut_service()
        svc.register("general.settings",  self._on_settings)
        svc.register("general.log_panel", self._toggle_log_panel)
        QApplication.instance().installEventFilter(self)

    def _toggle_log_panel(self) -> None:
        if hasattr(self, "_log_dock"):
            self._log_dock.setVisible(not self._log_dock.isVisible())

    def eventFilter(self, obj, event) -> bool:
        if event.type() != QEvent.Type.KeyPress:
            return False
        # Only dispatch when our window (not a dialog) is active
        if QApplication.activeWindow() is not self:
            return False
        from app.services.shortcut_service import get_shortcut_service, is_input_widget, normalize_key
        svc = get_shortcut_service()
        if not svc.is_enabled():
            return False
        focused = QApplication.focusWidget()
        if is_input_widget(focused):
            return False
        key_str = normalize_key(event)
        return svc.dispatch(key_str)

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
        centre_layout.addWidget(self._cluster_panel)

        actions = self._build_action_row()
        centre_layout.addLayout(actions)
        splitter.addWidget(centre)

        self._preview_panel = PreviewPanel()
        self._preview_panel.setMinimumWidth(280)
        self._preview_panel.face_selected.connect(self._on_preview_face_selected)
        self._preview_panel.face_assign_requested.connect(self._on_preview_face_assign)
        self._preview_panel.face_delete_requested.connect(self._on_preview_face_delete)
        self._preview_panel.face_create_requested.connect(self._on_preview_face_create)
        self._preview_panel.face_bbox_update_requested.connect(
            self._on_preview_face_bbox_update
        )
        splitter.addWidget(self._preview_panel)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 3)
        splitter.setSizes([320, 300, 660])

        face_layout.addWidget(splitter)
        self._tabs.addTab(face_widget, t("tab_face_recognition"))

        # --- Tab 1: Képböngésző ---
        self._image_browser = ImageBrowserPanel(config=self._config)
        self._image_browser.person_data_changed.connect(self._refresh_persons)
        self._tabs.addTab(self._image_browser, t("tab_image_browser"))

        # --- Tab 2: Családi kereső ---
        self._family_search = FamilySearchPanel()
        self._family_search.image_open_requested.connect(self._open_image_from_family_search)
        self._tabs.addTab(self._family_search, t("tab_family_search"))

        # --- Tab 3: Helyek ---
        self._locations_panel = LocationsPanel()
        self._tabs.addTab(self._locations_panel, t("tab_locations"))

        # --- Tab 4: Kollázs nézet ---
        self._collage_panel = CollagePanel()
        self._tabs.addTab(self._collage_panel, t("tab_collage"))

        self.setCentralWidget(self._tabs)

    def _build_action_row(self) -> QHBoxLayout:
        layout = QHBoxLayout()

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

        self._person_info_btn = QPushButton()
        self._person_info_btn.setEnabled(False)
        self._person_info_btn.setToolTip(t("person_info_tip"))
        self._person_info_btn.clicked.connect(self._on_person_info)
        layout.addWidget(self._person_info_btn)

        layout.addStretch()
        return layout

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

        self._status_label = QLabel()
        status.addWidget(self._status_label)

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
        self._update_gdrive_toolbar_btn()
        self._rename_btn.setText(t("rename_person"))
        self._merge_btn.setText(t("merge_into"))
        self._delete_person_btn.setText(t("delete_person"))
        self._remove_face_btn.setText(t("remove_face"))
        self._reassign_btn.setText(t("reassign_face"))
        self._person_info_btn.setText(t("person_info"))
        self._person_info_btn.setToolTip(t("person_info_tip"))
        self._tabs.setTabText(0, t("tab_face_recognition"))
        self._tabs.setTabText(1, t("tab_image_browser"))
        self._tabs.setTabText(2, t("tab_family_search"))
        self._tabs.setTabText(3, t("tab_locations"))
        self._tabs.setTabText(4, t("tab_collage"))
        self._log_dock.setWindowTitle(t("activity_log"))
        self._status_label.setText(t("ready"))
        if hasattr(self, "_image_browser"):
            self._image_browser.retranslate()
        if hasattr(self, "_family_search"):
            self._family_search.retranslate()
        if hasattr(self, "_locations_panel"):
            self._locations_panel.retranslate()

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
            on_find_overlapping_unknown_faces=self._on_find_overlapping_unknown_faces,
            parent=self,
        )
        dlg.exec()

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
        dlg = SuggestionDialog(self._config.suggestions, parent=self)
        dlg.data_changed.connect(self._refresh_persons)
        dlg.data_changed.connect(self._image_browser._reload_current_face_data)
        dlg.exec()
        self._refresh_persons()
        self._image_browser._reload_current_face_data()

    @Slot()
    def _on_find_overlapping_unknown_faces(self) -> None:
        if self._worker and self._worker.isRunning():
            QMessageBox.information(self, t("busy_title"), t("busy_msg"))
            return

        threshold = self._config.detection.duplicate_unknown_iou_threshold
        try:
            with session_scope() as session:
                finder = DuplicateUnknownFaceFinder(session, iou_threshold=threshold)
                matches = finder.find()
                images_examined = finder.images_examined
        except Exception as exc:  # noqa: BLE001
            log.exception("Overlapping unknown face search failed")
            QMessageBox.critical(self, t("error"), t("overlap_search_error", error=exc))
            return

        log.info(
            "Átfedő kérdőjeles keretek keresése: %d kép vizsgálva, %d találat.",
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
                finder = DuplicateUnknownFaceFinder(session, iou_threshold=threshold)
                result = finder.delete_unknown_faces(selected_ids)
        except Exception as exc:  # noqa: BLE001
            log.exception("Overlapping unknown face cleanup failed")
            QMessageBox.critical(self, t("error"), t("overlap_delete_error", error=exc))
            return

        log.info(
            "Átfedő kérdőjeles keretek törlése: kijelölt=%d, törölt=%d, kihagyott=%d.",
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
            QMessageBox.information(self, t("settings_title"), t("db_switched"))
            log.info("Database switched to: %s", new_db)

        # Language change
        if dlg.language_changed():
            self._retranslate()
            self._refresh_persons()

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
        if not success:
            QMessageBox.warning(self, t("warning"), summary)
            return

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

    @Slot(int)
    def _on_face_selected(self, face_id: int) -> None:
        self._current_face_id = face_id

        self._show_face_in_preview(face_id)

        self._remove_face_btn.setEnabled(True)
        self._reassign_btn.setEnabled(True)

    def _show_face_in_preview(self, face_id: int) -> None:
        """Reload the preview image and select *face_id*."""
        with session_scope() as session:
            face = session.get(Face, face_id)
            if face:
                _ = face.image
                if face.image:
                    for f in face.image.faces:
                        _ = f.person
                self._preview_panel.show_face(face)

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

    @Slot(int)
    def _on_preview_face_assign(self, face_id: int) -> None:
        """Handle 'Személyhez adás' from the preview panel."""
        self._on_preview_face_selected(face_id)
        self._on_reassign_face()

    @Slot(int)
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
        new_face_id: Optional[int] = None
        with session_scope() as session:
            from app.db.models import Image
            from app.detectors.base import Detection
            from app.utils.image_utils import load_image_bgr, save_face_crop

            image = session.get(Image, image_id)
            if image is None:
                return
            img_bgr = load_image_bgr(image.file_path)
            if img_bgr is None:
                return

            detection = Detection(x=x, y=y, w=w, h=h, confidence=1.0).clamp(
                img_bgr.shape[1], img_bgr.shape[0]
            )
            crops_dir = self._config.crops_dir_resolved
            crops_dir.mkdir(parents=True, exist_ok=True)

            # Insert face first to obtain its stable DB primary key, then name
            # the crop after it to guarantee a globally-unique filename.
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

        if new_face_id is None:
            return

        log.info(
            "Manual face added from preview: image_id=%d face_id=%d bbox=(%d,%d,%d,%d)",
            image_id, new_face_id, x, y, w, h,
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
            from app.utils.image_utils import load_image_bgr

            face = session.get(Face, face_id)
            if face is None or face.image is None:
                return
            img_bgr = load_image_bgr(face.image.file_path)
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

            dlg = MergeDialog(source, persons, parent=self)
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

            dlg = MergeDialog(_FakePerson(), persons, parent=self)
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

        log.info(
            "Személyadatok mentve: %s %s",
            dlg.last_name(), dlg.first_name()
        )

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
            n = len(
                RecognitionService(
                    session, self._config.recognition
                ).recognize_pending()
            )

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
        self._tabs.setCurrentIndex(4)

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
            on_collage_import=self._on_import_collage,
            on_collage_html_export=self._on_export_collage_html,
            parent=self,
        )
        dlg.exec()

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
        from PySide6.QtCore import QSettings
        enabled = QSettings("FaceLocal", "FaceLocal").value("updates/notify", True, type=bool)
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
        """Background thread — check GitHub releases without blocking the UI."""
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
            dlg = UpdateDialog(self._pending_release, parent=self)
            dlg.exec()

    def _refresh_persons(self) -> None:
        with session_scope() as session:
            persons: List[Person] = (
                session.query(Person).order_by(Person.name).all()
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
            for p in persons:
                for f in p.faces:
                    _ = f.image  # noqa: F841
            self._sidebar.populate(persons)
        if hasattr(self, "_family_search"):
            self._family_search.refresh()
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
        if self._gdrive_session is not None and not self._gdrive_closing:
            # Don't actually close yet — let the Drive thread finish, then quit.
            event.ignore()
            self._gdrive_closing = True
            self._gdrive_chip_btn.setText(t("gdrive_chip_closing"))
            self._status_label.setText(t("gdrive_closing_wait"))
            self._start_drive_close(after_quit=True)
            return
        event.accept()
