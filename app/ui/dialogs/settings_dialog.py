"""Settings dialog — language, database management, TPU status and updates."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSettings, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app import __version__
from app.services.image_library_service import get_image_library_optional
from app.ui.i18n import SUPPORTED, current_language, set_language, t


def _qsettings() -> QSettings:
    from app.app_settings import app_qsettings
    return app_qsettings()


# Strong refs to probe/stats threads so they outlive a quickly closed dialog
# (a parented QThread would be destroyed while still running → hard crash).
_ACTIVE_THREADS: set = set()


def _track_thread(thread: QThread) -> QThread:
    _ACTIVE_THREADS.add(thread)
    thread.finished.connect(lambda th=thread: _ACTIVE_THREADS.discard(th))
    return thread


class _TpuProbeThread(QThread):
    """Runs probe_tpu() in background so the dialog doesn't freeze."""

    result_ready = Signal(dict)

    def run(self) -> None:
        from app.ui.dialogs.tpu_status_dialog import probe_tpu
        self.result_ready.emit(probe_tpu())


class _GeneralStatsThread(QThread):
    """Collects the General tab's slow stats off the UI thread.

    The Drive cache size walks the whole cache directory and the library
    stats run DB counts — both froze the dialog while it opened.
    """

    result_ready = Signal(float, int, int)  # cache_mb, n_relative, n_absolute

    def run(self) -> None:
        cache_mb = 0.0
        try:
            from app.gdrive.cache import get_cache_manager
            cache_mb = get_cache_manager().current_size_bytes() / (1024 * 1024)
        except Exception:  # noqa: BLE001 — stats only, never break Settings
            pass
        n_rel = n_abs = 0
        try:
            from app.db.database import session_scope
            from app.db.models import Image
            with session_scope() as session:
                n_rel = session.query(Image).filter(
                    Image.relative_path.isnot(None)
                ).count()
                n_abs = session.query(Image).filter(
                    Image.relative_path.is_(None)
                ).count()
        except Exception:  # noqa: BLE001
            pass
        self.result_ready.emit(cache_mb, n_rel, n_abs)


class _AudioDevicesThread(QThread):
    """Probes ffmpeg audio devices in the background (subprocess call)."""

    result_ready = Signal(list)

    def __init__(self, ffmpeg_setting: str, parent=None) -> None:
        super().__init__(parent)
        self._ffmpeg_setting = ffmpeg_setting

    def run(self) -> None:
        try:
            from app.services.screen_recorder_service import (
                list_audio_devices,
                resolve_ffmpeg,
            )
            ffmpeg = resolve_ffmpeg(self._ffmpeg_setting or None)
            self.result_ready.emit(list(list_audio_devices(ffmpeg)))
        except Exception:  # noqa: BLE001 — empty list leaves "Automatic"
            self.result_ready.emit([])


class _PackageInstallThread(QThread):
    """Installs one package via pip in the background."""

    done = Signal(bool, str)   # (success, display_name)

    def __init__(self, pip_spec: str, display_name: str, parent=None) -> None:
        super().__init__(parent)
        self._pip_spec = pip_spec
        self._display_name = display_name

    def run(self) -> None:
        from app.services.dependency_installer import (
            find_python_executable,
            install_package,
            refresh_sys_path,
        )
        python_exe = find_python_executable()
        if not python_exe:
            self.done.emit(False, self._display_name)
            return
        ok = install_package(self._pip_spec, python_exe)
        if ok:
            refresh_sys_path()
        self.done.emit(ok, self._display_name)


_MOBILEFACENET_MIRRORS = [
    "https://github.com/MCarlomagno/FaceRecognitionAuth/raw/refs/heads/master/assets/mobilefacenet.tflite",
    "https://github.com/pb-julian/liteface/raw/main/tflite_models/mobilefacenet.tflite",
    "https://github.com/shubham0204/FaceRecognition_With_FaceNet_Android/raw/master/app/src/main/assets/mobile_face_net.tflite",
]


class _ModelDownloadThread(QThread):
    """Downloads mobilefacenet.tflite from the first working mirror."""

    progress = Signal(int)      # 0-100
    done = Signal(bool, str)    # (success, dest_path)

    def __init__(self, dest_path: str, parent=None) -> None:
        super().__init__(parent)
        self._dest = dest_path

    def run(self) -> None:
        import urllib.request
        from pathlib import Path

        dest = Path(self._dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(dest) + ".tmp")

        for url in _MOBILEFACENET_MIRRORS:
            try:
                tmp.unlink(missing_ok=True)
                req = urllib.request.Request(url, headers={"User-Agent": "face-local/1.0"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    total = int(resp.headers.get("Content-Length", 0))
                    downloaded = 0
                    with tmp.open("wb") as f:
                        while True:
                            chunk = resp.read(65536)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total:
                                self.progress.emit(int(downloaded / total * 100))
                tmp.rename(dest)
                self.done.emit(True, str(dest))
                return
            except Exception:  # noqa: BLE001
                tmp.unlink(missing_ok=True)
                continue

        self.done.emit(False, str(dest))


class SettingsDialog(QDialog):
    """Settings dialog: language, database management, and TPU status."""

    def __init__(self, current_db_path: str, parent=None, app_config=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("settings_title"))
        self.setMinimumWidth(540)
        self.setMinimumHeight(300)
        self.resize(560, 640)
        # Open the settings dialog maximised (full screen working area).
        self.setWindowState(self.windowState() | Qt.WindowMaximized)
        self._current_db_path = current_db_path
        if app_config is None:
            from app.config import load_config
            app_config = load_config()
        self._app_config = app_config
        self._new_db_path: Optional[str] = None
        self._language_changed = False
        self._probe_thread: Optional[_TpuProbeThread] = None
        self._stats_thread: Optional[_GeneralStatsThread] = None
        self._audio_thread: Optional[_AudioDevicesThread] = None
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        tabs = QTabWidget()
        self._tabs_widget = tabs
        tabs.addTab(self._build_tab_general(), t("settings_tab_general"))
        tabs.addTab(self._build_tab_pairing(), t("settings_tab_pairing"))
        tabs.addTab(self._build_tab_face_quality(), t("settings_tab_quality"))
        tabs.addTab(self._build_tab_tasks(), t("settings_tab_tasks"))
        tabs.addTab(self._build_tab_ai_model(), t("settings_tab_ai_model"))
        tabs.addTab(self._build_tab_packages(), t("settings_tab_packages"))
        # Heavy tabs (shortcut table, ffmpeg probing, Drive prefs) are built
        # lazily on first visit so the dialog itself opens instantly.
        self._lazy_tab_builders = {}
        for label, builder in (
            (t("settings_tab_shortcuts"), self._build_tab_shortcuts),
            (t("settings_tab_recording"), self._build_tab_recording),
            (t("settings_tab_gdrive"), self._build_tab_gdrive),
        ):
            placeholder = QWidget()
            QVBoxLayout(placeholder).setContentsMargins(0, 0, 0, 0)
            idx = tabs.addTab(placeholder, label)
            self._lazy_tab_builders[idx] = builder
        tabs.addTab(self._build_tab_debug(), t("settings_tab_debug"))
        tabs.currentChanged.connect(self._materialize_tab)
        outer.addWidget(tabs, stretch=1)

        # ── Buttons — always visible outside the tabs ─────────────────────
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        outer.addWidget(btns)

        self._start_tpu_probe()
        self._start_general_stats()

    def _materialize_tab(self, index: int) -> None:
        """Build a lazily constructed tab's real content on first visit."""
        builder = self._lazy_tab_builders.pop(index, None)
        if builder is None:
            return
        placeholder = self._tabs_widget.widget(index)
        placeholder.layout().addWidget(builder())

    def _start_general_stats(self) -> None:
        """Fetch Drive-cache size + library path counts off the UI thread."""
        self._gdrive_cache_size_label.setText("…")
        self._lib_stats_label.setText("…")
        self._stats_thread = _track_thread(_GeneralStatsThread())
        self._stats_thread.result_ready.connect(self._on_general_stats_ready)
        self._stats_thread.start()

    def _on_general_stats_ready(self, cache_mb: float, n_rel: int, n_abs: int) -> None:
        self._gdrive_cache_size_label.setText(
            t("gdrive_cache_size_value", mb=cache_mb)
        )
        parts = []
        if n_rel:
            parts.append(t("img_lib_n_relative", n=n_rel))
        if n_abs:
            parts.append(t("img_lib_n_absolute", n=n_abs))
        self._lib_stats_label.setText("  ".join(parts))

    def _build_tab_general(self) -> QWidget:
        """Build the first tab containing all existing settings."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(14)
        layout.setContentsMargins(2, 2, 2, 2)

        # ── Language ──────────────────────────────────────────────────────
        lang_group = QGroupBox(t("lang_label").rstrip(":"))
        lang_layout = QFormLayout(lang_group)

        self._lang_combo = QComboBox()
        for code, name in SUPPORTED.items():
            self._lang_combo.addItem(name, userData=code)
        idx = self._lang_combo.findData(current_language())
        if idx >= 0:
            self._lang_combo.setCurrentIndex(idx)
        lang_layout.addRow(t("lang_label"), self._lang_combo)
        layout.addWidget(lang_group)

        # ── Database ──────────────────────────────────────────────────────
        db_group = QGroupBox(t("db_group"))
        db_layout = QVBoxLayout(db_group)

        cur_row = QHBoxLayout()
        cur_row.addWidget(QLabel(t("current_db")))
        self._db_label = QLineEdit(self._current_db_path)
        self._db_label.setReadOnly(True)
        self._db_label.setStyleSheet("color: #aaa;")
        cur_row.addWidget(self._db_label)
        db_layout.addLayout(cur_row)

        btn_row = QHBoxLayout()
        new_btn = QPushButton(t("new_db"))
        new_btn.clicked.connect(self._on_new_db)
        btn_row.addWidget(new_btn)

        open_btn = QPushButton(t("open_db"))
        open_btn.clicked.connect(self._on_open_db)
        btn_row.addWidget(open_btn)
        btn_row.addStretch()
        db_layout.addLayout(btn_row)
        layout.addWidget(db_group)

        # ── Image Library ─────────────────────────────────────────────────
        lib_group = QGroupBox(t("img_lib_group"))
        lib_layout = QVBoxLayout(lib_group)

        root_row = QHBoxLayout()
        root_row.addWidget(QLabel(t("img_lib_root_label")))
        self._lib_root_label = QLineEdit()
        self._lib_root_label.setReadOnly(True)
        self._lib_root_label.setStyleSheet("color: #aaa;")
        root_row.addWidget(self._lib_root_label, 1)
        lib_layout.addLayout(root_row)

        lib_btn_row = QHBoxLayout()
        change_lib_btn = QPushButton(t("img_lib_change_btn"))
        change_lib_btn.clicked.connect(self._on_change_library_root)
        lib_btn_row.addWidget(change_lib_btn)

        self._migrate_btn = QPushButton(t("img_lib_migrate_btn"))
        self._migrate_btn.setToolTip(t("img_lib_migrate_tip"))
        self._migrate_btn.clicked.connect(self._on_migrate_paths)
        lib_btn_row.addWidget(self._migrate_btn)
        lib_btn_row.addStretch()
        lib_layout.addLayout(lib_btn_row)

        self._lib_stats_label = QLabel()
        self._lib_stats_label.setStyleSheet("color: #888; font-size: 11px;")
        lib_layout.addWidget(self._lib_stats_label)

        layout.addWidget(lib_group)
        self._refresh_library_status()

        # ── Updates ───────────────────────────────────────────────────────
        upd_group = QGroupBox(t("updates_group"))
        upd_layout = QVBoxLayout(upd_group)

        self._update_status_label = QLabel(t("current_version", version=__version__))
        self._update_status_label.setTextFormat(Qt.TextFormat.RichText)
        upd_layout.addWidget(self._update_status_label)

        self._notify_check = QCheckBox(t("notify_updates"))
        notify_enabled = _qsettings().value("updates/notify", True, type=bool)
        self._notify_check.setChecked(notify_enabled)
        upd_layout.addWidget(self._notify_check)

        upd_btn_row = QHBoxLayout()
        self._check_upd_btn = QPushButton(f"🔄  {t('check_updates_btn')}")
        self._check_upd_btn.clicked.connect(self._on_check_update)
        upd_btn_row.addWidget(self._check_upd_btn)
        upd_btn_row.addStretch()
        upd_layout.addLayout(upd_btn_row)
        layout.addWidget(upd_group)

        # ── TPU status ────────────────────────────────────────────────────
        tpu_group = QGroupBox(t("tpu_title"))
        tpu_layout = QVBoxLayout(tpu_group)

        self._tpu_summary_label = QLabel()
        tpu_layout.addWidget(self._tpu_summary_label)

        tpu_btn_row = QHBoxLayout()
        check_btn = QPushButton(t("tpu_status"))
        check_btn.clicked.connect(self._on_tpu_check)
        tpu_btn_row.addWidget(check_btn)

        self._tpu_fix_btn = QPushButton(f"🔧 {t('fix')}")
        self._tpu_fix_btn.clicked.connect(self._on_tpu_fix)
        self._tpu_fix_btn.setVisible(False)
        tpu_btn_row.addWidget(self._tpu_fix_btn)

        tpu_btn_row.addStretch()
        tpu_layout.addLayout(tpu_btn_row)
        layout.addWidget(tpu_group)

        # ── Google Drive cache ────────────────────────────────────────────
        gdrive_group = QGroupBox(t("gdrive_cache_group"))
        gdrive_layout = QVBoxLayout(gdrive_group)

        from app.paths import gdrive_cache_dir
        _cache_dir = gdrive_cache_dir()
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel(t("gdrive_cache_dir_label")))
        self._gdrive_cache_dir_label = QLineEdit(str(_cache_dir))
        self._gdrive_cache_dir_label.setReadOnly(True)
        self._gdrive_cache_dir_label.setStyleSheet("color: #aaa;")
        dir_row.addWidget(self._gdrive_cache_dir_label, 1)
        gdrive_layout.addLayout(dir_row)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel(t("gdrive_cache_size_label")))
        # Filled in asynchronously — walking the cache directory is slow.
        self._gdrive_cache_size_label = QLabel("…")
        self._gdrive_cache_size_label.setStyleSheet("color: #aaa; font-size: 11px;")
        size_row.addWidget(self._gdrive_cache_size_label)
        size_row.addStretch()
        gdrive_layout.addLayout(size_row)

        note_label = QLabel(t("gdrive_cache_note"))
        note_label.setWordWrap(True)
        note_label.setStyleSheet("color: #888; font-size: 11px;")
        gdrive_layout.addWidget(note_label)

        layout.addWidget(gdrive_group)

        layout.addStretch()
        scroll.setWidget(inner)
        return scroll

    def _build_tab_pairing(self) -> QWidget:
        """Build the second tab for image-pairing settings."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(14)

        deol_group = QGroupBox(t("deoldified_group"))
        deol_layout = QVBoxLayout(deol_group)

        self._deoldified_check = QCheckBox(t("deoldified_toggle"))
        self._deoldified_check.setChecked(
            _qsettings().value("deoldified/auto_pair", False, type=bool)
        )
        self._deoldified_check.setToolTip(t("deoldified_toggle_tip"))
        deol_layout.addWidget(self._deoldified_check)

        self._deoldified_sync_check = QCheckBox(t("deoldified_sync_toggle"))
        self._deoldified_sync_check.setChecked(
            _qsettings().value("deoldified/auto_sync", True, type=bool)
        )
        self._deoldified_sync_check.setToolTip(t("deoldified_sync_toggle_tip"))
        deol_layout.addWidget(self._deoldified_sync_check)

        layout.addWidget(deol_group)

        geo_group = QGroupBox(t("geocoding_group"))
        geo_layout = QVBoxLayout(geo_group)
        self._geocoding_check = QCheckBox(t("geocoding_enable"))
        self._geocoding_check.setChecked(
            _qsettings().value("geocoding/enabled", False, type=bool)
        )
        self._geocoding_check.setToolTip(t("geocoding_enable_tip"))
        geo_layout.addWidget(self._geocoding_check)
        layout.addWidget(geo_group)

        layout.addStretch()
        return widget

    def _build_tab_recording(self) -> QWidget:
        """Build the screen-recording settings tab."""
        qs = _qsettings()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(14)

        # ── Output folder ─────────────────────────────────────────────────
        out_group = QGroupBox(t("settings_tab_recording"))
        out_form = QFormLayout(out_group)

        dir_row = QHBoxLayout()
        self._rec_dir_edit = QLineEdit(
            qs.value("recording/output_dir", "", type=str) or ""
        )
        self._rec_dir_edit.setPlaceholderText(t("rec_set_unset"))
        dir_row.addWidget(self._rec_dir_edit)
        browse_btn = QPushButton(t("rec_set_browse"))
        browse_btn.clicked.connect(self._on_browse_recording_dir)
        dir_row.addWidget(browse_btn)
        out_form.addRow(t("rec_set_output_dir"), dir_row)

        self._rec_quality_combo = QComboBox()
        for key, label_key in (
            ("low", "rec_quality_low"),
            ("normal", "rec_quality_normal"),
            ("better", "rec_quality_better"),
        ):
            self._rec_quality_combo.addItem(t(label_key), userData=key)
        cur_quality = qs.value("recording/quality", "normal", type=str)
        qidx = self._rec_quality_combo.findData(cur_quality)
        if qidx >= 0:
            self._rec_quality_combo.setCurrentIndex(qidx)
        out_form.addRow(t("rec_set_quality"), self._rec_quality_combo)

        self._rec_fps_spin = QSpinBox()
        self._rec_fps_spin.setRange(5, 60)
        self._rec_fps_spin.setValue(qs.value("recording/fps", 18, type=int))
        out_form.addRow(t("rec_set_fps"), self._rec_fps_spin)

        self._rec_segment_spin = QSpinBox()
        self._rec_segment_spin.setRange(2, 60)
        self._rec_segment_spin.setValue(
            qs.value("recording/segment_seconds", 8, type=int)
        )
        out_form.addRow(t("rec_set_segment"), self._rec_segment_spin)

        self._rec_ffmpeg_edit = QLineEdit(
            qs.value("recording/ffmpeg_path", "", type=str) or ""
        )
        out_form.addRow(t("rec_set_ffmpeg_path"), self._rec_ffmpeg_edit)

        layout.addWidget(out_group)

        # ── Capture toggles ───────────────────────────────────────────────
        cap_group = QGroupBox(t("settings_tab_recording"))
        cap_layout = QVBoxLayout(cap_group)
        self._rec_cursor_check = QCheckBox(t("rec_set_cursor"))
        self._rec_cursor_check.setChecked(
            qs.value("recording/capture_cursor", True, type=bool)
        )
        self._rec_mic_check = QCheckBox(t("rec_set_microphone"))
        self._rec_mic_check.setChecked(
            qs.value("recording/capture_microphone", True, type=bool)
        )
        self._rec_sysaudio_check = QCheckBox(t("rec_set_system_audio"))
        self._rec_sysaudio_check.setChecked(
            qs.value("recording/capture_system_audio", True, type=bool)
        )
        self._rec_concat_check = QCheckBox(t("rec_set_concat"))
        self._rec_concat_check.setChecked(
            qs.value("recording/concat_on_stop", True, type=bool)
        )
        for cb in (
            self._rec_cursor_check,
            self._rec_mic_check,
            self._rec_sysaudio_check,
            self._rec_concat_check,
        ):
            cap_layout.addWidget(cb)
        layout.addWidget(cap_group)

        # ── Audio devices / mix ───────────────────────────────────────────
        layout.addWidget(self._build_recording_audio_group(qs))

        # ── Screens to record ─────────────────────────────────────────────
        layout.addWidget(self._build_recording_display_group(qs))

        layout.addStretch()
        scroll.setWidget(widget)
        return scroll

    def _build_recording_audio_group(self, qs) -> QWidget:
        """Audio device selection, per-source volume sliders and mute toggles."""
        group = QGroupBox(t("rec_audio_group"))
        form = QFormLayout(group)

        # Device enumeration spawns an ffmpeg subprocess, so it runs in the
        # background; the combos start with "Automatic" + the stored device
        # and the discovered names are merged in when the probe finishes.
        def _device_combo(stored_key: str) -> QComboBox:
            combo = QComboBox()
            combo.addItem(t("rec_audio_auto"), userData="")
            stored = qs.value(stored_key, "", type=str) or ""
            if stored:
                combo.addItem(stored, userData=stored)
                combo.setCurrentIndex(combo.findData(stored))
            return combo

        self._rec_audio_input_combo = _device_combo("recording/audio_input_device")
        form.addRow(t("rec_audio_input_device"), self._rec_audio_input_combo)
        self._rec_system_audio_combo = _device_combo("recording/system_audio_device")
        form.addRow(t("rec_system_audio_device"), self._rec_system_audio_combo)

        self._audio_thread = _track_thread(
            _AudioDevicesThread(qs.value("recording/ffmpeg_path", "", type=str) or "")
        )
        self._audio_thread.result_ready.connect(self._on_audio_devices_ready)
        self._audio_thread.start()

        def _volume_slider(stored_key: str) -> QSlider:
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 200)  # percent of unity gain
            slider.setSingleStep(5)
            stored = qs.value(stored_key, 1.0, type=float)
            slider.setValue(int(round(stored * 100)))
            return slider

        self._rec_mic_volume_slider = _volume_slider("recording/mic_volume")
        form.addRow(t("rec_mic_volume"), self._rec_mic_volume_slider)
        self._rec_system_volume_slider = _volume_slider("recording/system_volume")
        form.addRow(t("rec_system_volume"), self._rec_system_volume_slider)

        self._rec_mute_mic_check = QCheckBox(t("rec_mute_microphone"))
        self._rec_mute_mic_check.setChecked(
            qs.value("recording/mute_microphone", False, type=bool)
        )
        form.addRow(self._rec_mute_mic_check)
        self._rec_mute_system_check = QCheckBox(t("rec_mute_system_audio"))
        self._rec_mute_system_check.setChecked(
            qs.value("recording/mute_system_audio", False, type=bool)
        )
        form.addRow(self._rec_mute_system_check)
        return group

    def _build_recording_display_group(self, qs) -> QWidget:
        """Display-mode radios + dynamic monitor checklist."""
        from app.services.screen_recorder_service import (
            RecordingDisplayMode,
            format_display_label,
        )
        from app.ui.display_utils import enumerate_displays

        group = QGroupBox(t("rec_set_display_group"))
        vbox = QVBoxLayout(group)

        cur_mode = qs.value("recording/display_mode", "all", type=str)
        self._rec_mode_group = QButtonGroup(group)
        self._rec_mode_active = QRadioButton(t("rec_set_mode_active"))
        self._rec_mode_all = QRadioButton(t("rec_set_mode_all"))
        self._rec_mode_selected = QRadioButton(t("rec_set_mode_selected"))
        for btn, value in (
            (self._rec_mode_active, RecordingDisplayMode.ACTIVE_WINDOW.value),
            (self._rec_mode_all, RecordingDisplayMode.ALL_DISPLAYS.value),
            (self._rec_mode_selected, RecordingDisplayMode.SELECTED_DISPLAYS.value),
        ):
            btn.setProperty("rec_mode", value)
            self._rec_mode_group.addButton(btn)
            vbox.addWidget(btn)
            if value == cur_mode:
                btn.setChecked(True)
        if self._rec_mode_group.checkedButton() is None:
            self._rec_mode_all.setChecked(True)

        # Dynamic monitor checklist.
        saved_ids = qs.value("recording/selected_display_ids", None)
        if isinstance(saved_ids, str):
            saved_ids = [saved_ids] if saved_ids else []
        saved_set = {str(x) for x in (saved_ids or [])}

        self._rec_display_checks: list[QCheckBox] = []
        displays = enumerate_displays()
        primary_marker = t("rec_set_primary_marker")
        monitor_word = t("rec_set_monitor_word")
        if displays:
            for ordinal, info in enumerate(displays, start=1):
                cb = QCheckBox(
                    format_display_label(
                        info, ordinal, monitor_word, primary_marker
                    )
                )
                cb.setProperty("display_id", info.id)
                cb.setChecked(info.id in saved_set)
                self._rec_display_checks.append(cb)
                vbox.addWidget(cb)
        else:
            vbox.addWidget(QLabel(t("rec_set_no_displays")))

        self._rec_auto_fps_check = QCheckBox(t("rec_set_auto_fps"))
        self._rec_auto_fps_check.setChecked(
            qs.value("recording/auto_reduce_fps", True, type=bool)
        )
        vbox.addWidget(self._rec_auto_fps_check)

        # Only enable the monitor checklist in "selected" mode.
        def _sync_checklist_enabled() -> None:
            enabled = self._rec_mode_selected.isChecked()
            for cb in self._rec_display_checks:
                cb.setEnabled(enabled)

        self._rec_mode_group.buttonToggled.connect(
            lambda *_: _sync_checklist_enabled()
        )
        _sync_checklist_enabled()
        return group

    def _on_audio_devices_ready(self, devices: list) -> None:
        """Merge discovered device names into the combos, keeping selection."""
        for combo in (self._rec_audio_input_combo, self._rec_system_audio_combo):
            current = combo.currentData()
            for name in devices:
                if combo.findData(name) < 0:
                    combo.addItem(name, userData=name)
            idx = combo.findData(current)
            combo.setCurrentIndex(max(0, idx))

    def _on_browse_recording_dir(self) -> None:
        start = self._rec_dir_edit.text() or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(self, t("rec_choose_dir"), start)
        if chosen:
            self._rec_dir_edit.setText(chosen)

    def _build_tab_shortcuts(self) -> QWidget:
        """Build the keyboard shortcuts configuration tab."""
        from app.ui.dialogs.shortcuts_settings_tab import ShortcutsSettingsTab
        return ShortcutsSettingsTab(self)

    def _build_tab_gdrive(self) -> QWidget:
        """Build the Google Drive tab — implementation lives in its own module."""
        from app.ui.dialogs.gdrive_settings_tab import GDriveSettingsTab
        self._gdrive_tab = GDriveSettingsTab(self)
        return self._gdrive_tab

    def _build_tab_debug(self) -> QWidget:
        """Build the Debug tab with AI visualization and decision log toggles."""
        from PySide6.QtWidgets import QCheckBox, QGroupBox, QPushButton, QScrollArea

        qs = _qsettings()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(14)
        layout.setContentsMargins(2, 8, 2, 2)

        # ── AI Visualization ─────────────────────────────────────────────
        viz_group = QGroupBox(t("debug_ai_viz_group"))
        viz_layout = QVBoxLayout(viz_group)

        viz_desc = QLabel(t("debug_ai_viz_desc"))
        viz_desc.setWordWrap(True)
        viz_desc.setStyleSheet("color: #aaa; font-size: 11px;")
        viz_layout.addWidget(viz_desc)

        self._debug_viz_check = QCheckBox(t("debug_ai_viz_check"))
        self._debug_viz_check.setChecked(
            qs.value("debug/ai_visualization", False, type=bool)
        )
        viz_layout.addWidget(self._debug_viz_check)
        layout.addWidget(viz_group)

        # ── Decision Log ─────────────────────────────────────────────────
        log_group = QGroupBox(t("debug_log_group"))
        log_layout = QVBoxLayout(log_group)

        log_desc = QLabel(t("debug_log_desc"))
        log_desc.setWordWrap(True)
        log_desc.setStyleSheet("color: #aaa; font-size: 11px;")
        log_layout.addWidget(log_desc)

        self._debug_log_check = QCheckBox(t("debug_log_check"))
        self._debug_log_check.setChecked(
            qs.value("debug/ai_log_enabled", False, type=bool)
        )
        log_layout.addWidget(self._debug_log_check)

        btn_row = QHBoxLayout()
        open_log_btn = QPushButton(t("debug_log_open_btn"))
        open_log_btn.clicked.connect(self._on_open_debug_log)
        btn_row.addWidget(open_log_btn)

        clear_log_btn = QPushButton(t("debug_log_clear_btn"))
        clear_log_btn.clicked.connect(self._on_clear_debug_log)
        btn_row.addWidget(clear_log_btn)
        btn_row.addStretch()
        log_layout.addLayout(btn_row)
        layout.addWidget(log_group)

        # ── Detection Debug Log ───────────────────────────────────────────
        det_log_group = QGroupBox(t("debug_detection_log_group"))
        det_log_layout = QVBoxLayout(det_log_group)

        det_log_desc = QLabel(t("debug_detection_log_desc"))
        det_log_desc.setWordWrap(True)
        det_log_desc.setStyleSheet("color: #aaa; font-size: 11px;")
        det_log_layout.addWidget(det_log_desc)

        self._debug_detection_log_check = QCheckBox(t("debug_detection_log_check"))
        self._debug_detection_log_check.setChecked(
            qs.value("debug/detection_log_enabled", False, type=bool)
        )
        det_log_layout.addWidget(self._debug_detection_log_check)

        det_log_btn_row = QHBoxLayout()
        open_det_log_btn = QPushButton(t("debug_detection_log_open_btn"))
        open_det_log_btn.clicked.connect(self._on_open_detection_log_folder)
        det_log_btn_row.addWidget(open_det_log_btn)
        det_log_btn_row.addStretch()
        det_log_layout.addLayout(det_log_btn_row)
        layout.addWidget(det_log_group)

        # ── Neural Network Graph ─────────────────────────────────────────────
        nn_group = QGroupBox(t("debug_nn_group"))
        nn_layout = QVBoxLayout(nn_group)

        nn_desc = QLabel(t("debug_nn_group_desc"))
        nn_desc.setWordWrap(True)
        nn_desc.setStyleSheet("color: #aaa; font-size: 11px;")
        nn_layout.addWidget(nn_desc)

        open_nn_btn = QPushButton(t("debug_nn_open_btn"))
        open_nn_btn.clicked.connect(self._on_open_nn_graph)
        nn_layout.addWidget(open_nn_btn)
        layout.addWidget(nn_group)

        layout.addStretch()
        scroll.setWidget(inner)
        return scroll

    def _on_open_detection_log_folder(self) -> None:
        import subprocess
        import sys
        from app.paths import detection_logs_dir
        folder = detection_logs_dir()
        folder.mkdir(parents=True, exist_ok=True)
        # Check if there are any log files yet.
        files = list(folder.glob("detection_*.log"))
        if not files:
            QMessageBox.information(
                self,
                t("debug_detection_log_group"),
                t("debug_detection_log_folder_empty"),
            )
            return
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            elif sys.platform == "win32":
                subprocess.Popen(["explorer", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except Exception:  # noqa: BLE001
            QMessageBox.information(
                self, t("debug_detection_log_group"), str(folder.resolve())
            )

    def _on_open_nn_graph(self) -> None:
        # Settings runs exec() → application-modal on macOS, any new window is blocked.
        # Solution: mark the main window to open the graph AFTER this dialog closes,
        # then accept (save) settings now.
        from PySide6.QtWidgets import QApplication
        for w in QApplication.topLevelWidgets():
            if hasattr(w, "_open_nn_graph_window"):
                w._pending_open_nn_graph = True
                break
        self.accept()

    def _on_open_debug_log(self) -> None:
        import subprocess
        from pathlib import Path
        log_path = Path("data/deep_debug.jsonl")
        if not log_path.exists():
            QMessageBox.information(self, t("debug_log_group"), t("debug_log_not_found"))
            return
        try:
            subprocess.Popen(["open", str(log_path)])
        except Exception:  # noqa: BLE001
            try:
                subprocess.Popen(["xdg-open", str(log_path)])
            except Exception:  # noqa: BLE001
                QMessageBox.information(
                    self, t("debug_log_group"), str(log_path.resolve())
                )

    def _on_clear_debug_log(self) -> None:
        from pathlib import Path
        log_path = Path("data/deep_debug.jsonl")
        if log_path.exists():
            log_path.unlink()
        QMessageBox.information(self, t("debug_log_group"), t("debug_log_cleared"))

    def _build_tab_tasks(self) -> QWidget:
        """Build the Task Manager tab (finished-task auto-cleanup)."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(14)
        layout.setContentsMargins(2, 2, 2, 2)

        group = QGroupBox(t("tasks_settings_group"))
        group_layout = QVBoxLayout(group)

        self._tasks_autocleanup_check = QCheckBox(t("tasks_settings_autocleanup"))
        self._tasks_autocleanup_check.setChecked(
            _qsettings().value("tasks/auto_cleanup_enabled", True, type=bool)
        )
        group_layout.addWidget(self._tasks_autocleanup_check)

        note = QLabel(t("tasks_settings_note"))
        note.setWordWrap(True)
        note.setStyleSheet("color: #888; font-size: 11px;")
        group_layout.addWidget(note)

        layout.addWidget(group)
        layout.addStretch(1)
        scroll.setWidget(inner)
        return scroll

    def _build_tab_face_quality(self) -> QWidget:
        """Build the Face Quality tab."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(14)
        layout.setContentsMargins(2, 2, 2, 2)

        # ── Main enable toggle ────────────────────────────────────────────
        fq_group = QGroupBox(t("fq_group"))
        fq_layout = QVBoxLayout(fq_group)

        self._fq_exclude_check = QCheckBox(t("fq_exclude_toggle"))
        self._fq_exclude_check.setChecked(
            _qsettings().value("face_quality/exclude_low_quality", True, type=bool)
        )
        self._fq_exclude_check.setToolTip(t("fq_exclude_tip"))
        fq_layout.addWidget(self._fq_exclude_check)

        layout.addWidget(fq_group)

        # ── Re-evaluate button ────────────────────────────────────────────
        reeval_group = QGroupBox(t("fq_thresholds_group"))
        reeval_layout = QVBoxLayout(reeval_group)

        note = QLabel(
            t("fq_exclude_tip")
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #aaa; font-size: 11px;")
        reeval_layout.addWidget(note)

        reeval_btn_row = QHBoxLayout()
        self._fq_reanalyze_btn = QPushButton(t("fq_reanalyze_btn"))
        self._fq_reanalyze_btn.clicked.connect(self._on_reanalyze_quality)
        reeval_btn_row.addWidget(self._fq_reanalyze_btn)
        reeval_btn_row.addStretch()
        reeval_layout.addLayout(reeval_btn_row)

        layout.addWidget(reeval_group)

        # ── Re-recognition (image browser) ────────────────────────────────
        rerec_group = QGroupBox(t("rerec_settings_group"))
        rerec_layout = QVBoxLayout(rerec_group)

        self._rerec_enabled_check = QCheckBox(t("rerec_settings_enabled"))
        self._rerec_enabled_check.setChecked(
            _qsettings().value("rerecognition/enabled", True, type=bool)
        )
        rerec_layout.addWidget(self._rerec_enabled_check)

        rerec_form = QFormLayout()
        auto_default = int(round(
            _qsettings().value("rerecognition/auto_threshold", 0.72, type=float) * 100
        ))
        suggest_default = int(round(
            _qsettings().value("rerecognition/suggest_threshold", 0.55, type=float) * 100
        ))
        self._rerec_auto_spin = QSpinBox()
        self._rerec_auto_spin.setRange(1, 100)
        self._rerec_auto_spin.setSuffix(" %")
        self._rerec_auto_spin.setValue(auto_default)
        self._rerec_suggest_spin = QSpinBox()
        self._rerec_suggest_spin.setRange(1, 100)
        self._rerec_suggest_spin.setSuffix(" %")
        self._rerec_suggest_spin.setValue(suggest_default)
        rerec_form.addRow(t("rerec_settings_auto"), self._rerec_auto_spin)
        rerec_form.addRow(t("rerec_settings_suggest"), self._rerec_suggest_spin)
        rerec_layout.addLayout(rerec_form)

        rerec_note = QLabel(t("rerec_settings_note"))
        rerec_note.setWordWrap(True)
        rerec_note.setStyleSheet("color: #aaa; font-size: 11px;")
        rerec_layout.addWidget(rerec_note)

        layout.addWidget(rerec_group)
        layout.addStretch()
        scroll.setWidget(inner)
        return scroll

    # ------------------------------------------------------------------
    # AI model tab
    # ------------------------------------------------------------------

    _AI_MAX_LAYERS = 6
    _AI_DEFAULT_LAYERS = (256, 192, 128, 64)

    def _build_tab_ai_model(self) -> QWidget:
        """Build the AI Model tab — neural-network layer configuration."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(14)
        layout.setContentsMargins(2, 2, 2, 2)

        group = QGroupBox(t("ai_model_layers_group"))
        group_layout = QVBoxLayout(group)

        desc = QLabel(t("ai_model_layers_desc"))
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #aaa; font-size: 11px;")
        group_layout.addWidget(desc)

        current = list(self._app_config.deep_recognition.hidden_layers)
        if not current:
            current = list(self._AI_DEFAULT_LAYERS)

        form = QFormLayout()
        self._ai_layer_count_spin = QSpinBox()
        self._ai_layer_count_spin.setRange(1, self._AI_MAX_LAYERS)
        self._ai_layer_count_spin.setValue(
            min(len(current), self._AI_MAX_LAYERS)
        )
        self._ai_layer_count_spin.valueChanged.connect(
            self._on_ai_layer_count_changed
        )
        form.addRow(t("ai_model_layer_count"), self._ai_layer_count_spin)

        # Sizes for layers beyond the currently configured ones, shown when
        # the user raises the layer count.
        fallback = [256, 192, 128, 64, 48, 32]
        self._ai_layer_rows: list[tuple[QLabel, QSpinBox]] = []
        for i in range(self._AI_MAX_LAYERS):
            spin = QSpinBox()
            spin.setRange(8, 2048)
            spin.setSingleStep(8)
            spin.setValue(current[i] if i < len(current) else fallback[i])
            label = QLabel(t("ai_model_layer_n").format(n=i + 1))
            form.addRow(label, spin)
            self._ai_layer_rows.append((label, spin))
        group_layout.addLayout(form)

        reset_row = QHBoxLayout()
        reset_btn = QPushButton(t("ai_model_reset_btn"))
        reset_btn.clicked.connect(self._on_ai_layers_reset)
        reset_row.addWidget(reset_btn)
        reset_row.addStretch()
        group_layout.addLayout(reset_row)

        note = QLabel(t("ai_model_rebuild_note"))
        note.setWordWrap(True)
        note.setStyleSheet("color: #aaa; font-size: 11px;")
        group_layout.addWidget(note)

        layout.addWidget(group)
        layout.addStretch()
        scroll.setWidget(inner)

        self._on_ai_layer_count_changed(self._ai_layer_count_spin.value())
        return scroll

    # ------------------------------------------------------------------
    # AI Packages tab
    # ------------------------------------------------------------------

    def _build_tab_packages(self) -> QWidget:
        """Build the AI Packages tab — install / status for optional backends."""
        self._pkg_install_threads: dict[str, _PackageInstallThread] = {}
        self._pkg_row_widgets: dict[str, dict] = {}  # import_name → {btn, status_lbl}

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(16)
        layout.setContentsMargins(4, 8, 4, 4)

        # ── Intro text ────────────────────────────────────────────────────
        intro = QLabel(t("pkg_tab_intro"))
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(intro)

        # ── Embedding backend group ───────────────────────────────────────
        emb_group = QGroupBox(t("pkg_group_embedding"))
        emb_layout = QVBoxLayout(emb_group)

        # Embedding model selector — MobileFaceNet (default) vs ArcFace.
        model_label = QLabel(t("pkg_emb_model_label"))
        model_label.setStyleSheet("font-size: 11px;")
        emb_layout.addWidget(model_label)

        self._emb_backend_group = QButtonGroup(emb_group)
        self._emb_radio_mobilefacenet = QRadioButton(t("pkg_emb_mobilefacenet"))
        self._emb_radio_mobilefacenet.setProperty("emb_backend", "mobilefacenet")
        self._emb_radio_arcface = QRadioButton(t("pkg_emb_arcface"))
        self._emb_radio_arcface.setProperty("emb_backend", "arcface")
        self._emb_backend_group.addButton(self._emb_radio_mobilefacenet)
        self._emb_backend_group.addButton(self._emb_radio_arcface)
        emb_layout.addWidget(self._emb_radio_mobilefacenet)
        emb_layout.addWidget(self._emb_radio_arcface)

        if self._app_config.embedding.backend == "arcface":
            self._emb_radio_arcface.setChecked(True)
        else:
            self._emb_radio_mobilefacenet.setChecked(True)
        # Remember the backend at open time so we can detect a real change on save.
        self._emb_backend_initial = self._app_config.embedding.backend

        arcface_note = QLabel(t("pkg_emb_arcface_needs"))
        arcface_note.setWordWrap(True)
        arcface_note.setStyleSheet("color: #888; font-size: 10px;")
        emb_layout.addWidget(arcface_note)

        # Apple Silicon only: opt into running embeddings on the Neural Engine
        # via Core ML.  The control is hidden entirely on Windows / Linux / Intel
        # Macs, where it would have no effect.
        from app import accel
        self._emb_coreml_check = None
        self._emb_coreml_initial = bool(self._app_config.embedding.prefer_coreml)
        if accel.is_apple_silicon():
            self._emb_coreml_check = QCheckBox(t("pkg_emb_apple_ane"))
            self._emb_coreml_check.setChecked(self._emb_coreml_initial)
            emb_layout.addWidget(self._emb_coreml_check)

            ane_note = QLabel(t("pkg_emb_apple_ane_note", accel=accel.summary()))
            ane_note.setWordWrap(True)
            ane_note.setStyleSheet("color: #888; font-size: 10px;")
            emb_layout.addWidget(ane_note)

        sep_top = QFrame()
        sep_top.setFrameShape(QFrame.Shape.HLine)
        sep_top.setStyleSheet("color: #444;")
        emb_layout.addWidget(sep_top)

        self._active_backend_label = QLabel()
        self._active_backend_label.setStyleSheet("color: #888; font-size: 11px;")
        emb_layout.addWidget(self._active_backend_label)

        # Model file row — label + optional download button + progress bar
        model_row = QHBoxLayout()
        self._model_file_label = QLabel()
        self._model_file_label.setStyleSheet("color: #888; font-size: 11px;")
        model_row.addWidget(self._model_file_label, stretch=1)

        self._model_download_btn = QPushButton(t("pkg_model_download_btn"))
        self._model_download_btn.setFixedWidth(130)
        self._model_download_btn.setVisible(False)
        self._model_download_btn.clicked.connect(self._on_download_model)
        model_row.addWidget(self._model_download_btn)
        emb_layout.addLayout(model_row)

        from PySide6.QtWidgets import QProgressBar
        self._model_progress_bar = QProgressBar()
        self._model_progress_bar.setRange(0, 100)
        self._model_progress_bar.setVisible(False)
        self._model_progress_bar.setMaximumHeight(8)
        self._model_progress_bar.setTextVisible(False)
        emb_layout.addWidget(self._model_progress_bar)

        self._model_download_thread: Optional[_ModelDownloadThread] = None

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #444;")
        emb_layout.addWidget(sep)

        from app.services.dependency_installer import CANDIDATE_PACKAGES
        for import_name, pip_spec, display_name, size_hint in CANDIDATE_PACKAGES:
            if import_name == "ai_edge_litert":
                emb_layout.addWidget(
                    self._build_package_row(import_name, pip_spec, display_name, size_hint)
                )

        layout.addWidget(emb_group)

        # ── Detection / Verification group ────────────────────────────────
        det_group = QGroupBox(t("pkg_group_detection"))
        det_layout = QVBoxLayout(det_group)

        det_note = QLabel(t("pkg_detection_note"))
        det_note.setWordWrap(True)
        det_note.setStyleSheet("color: #888; font-size: 11px;")
        det_layout.addWidget(det_note)

        for import_name, pip_spec, display_name, size_hint in CANDIDATE_PACKAGES:
            if import_name in ("onnxruntime", "insightface"):
                det_layout.addWidget(
                    self._build_package_row(import_name, pip_spec, display_name, size_hint)
                )

        layout.addWidget(det_group)

        # ── Bottom action row ─────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._install_all_btn = QPushButton(t("pkg_install_all_btn"))
        self._install_all_btn.clicked.connect(self._on_install_all_packages)
        btn_row.addWidget(self._install_all_btn)

        refresh_btn = QPushButton(t("pkg_refresh_btn"))
        refresh_btn.clicked.connect(self._refresh_packages_tab)
        btn_row.addWidget(refresh_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        layout.addStretch()
        scroll.setWidget(inner)

        self._refresh_packages_tab()
        return scroll

    def _build_package_row(
        self,
        import_name: str,
        pip_spec: str,
        display_name: str,
        size_hint: str,
    ) -> QWidget:
        """Build one package row: icon + name + size + status label + install button."""
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 2, 0, 2)

        self._pkg_row_widgets[import_name] = {}

        status_icon = QLabel("…")
        status_icon.setFixedWidth(20)
        self._pkg_row_widgets[import_name]["icon"] = status_icon
        h.addWidget(status_icon)

        name_lbl = QLabel(f"<b>{display_name}</b>")
        name_lbl.setFixedWidth(130)
        h.addWidget(name_lbl)

        size_lbl = QLabel(size_hint)
        size_lbl.setStyleSheet("color: #666; font-size: 11px;")
        size_lbl.setFixedWidth(140)
        h.addWidget(size_lbl)

        status_lbl = QLabel()
        status_lbl.setStyleSheet("font-size: 11px;")
        self._pkg_row_widgets[import_name]["status"] = status_lbl
        h.addWidget(status_lbl, stretch=1)

        btn = QPushButton(t("pkg_install_btn"))
        btn.setFixedWidth(100)
        btn.clicked.connect(lambda checked=False, n=import_name, s=pip_spec, d=display_name:
                            self._on_install_package(n, s, d))
        self._pkg_row_widgets[import_name]["btn"] = btn
        h.addWidget(btn)

        return row

    def _refresh_packages_tab(self) -> None:
        """Re-check all package states and update the UI."""
        from app.services.dependency_installer import CANDIDATE_PACKAGES, _is_importable
        from app.diagnostics import check_tflite_backend

        any_missing = False
        for import_name, pip_spec, display_name, _ in CANDIDATE_PACKAGES:
            available = _is_importable(import_name)
            widgets = self._pkg_row_widgets.get(import_name)
            if not widgets:
                continue
            if available:
                widgets["icon"].setText("✓")
                widgets["icon"].setStyleSheet("color: #4caf50; font-weight: bold;")
                widgets["status"].setText(t("pkg_status_installed"))
                widgets["status"].setStyleSheet("color: #4caf50; font-size: 11px;")
                widgets["btn"].setVisible(False)
            else:
                widgets["icon"].setText("✗")
                widgets["icon"].setStyleSheet("color: #f57c00; font-weight: bold;")
                widgets["status"].setText(t("pkg_status_missing"))
                widgets["status"].setStyleSheet("color: #f57c00; font-size: 11px;")
                widgets["btn"].setVisible(True)
                widgets["btn"].setEnabled(True)
                widgets["btn"].setText(t("pkg_install_btn"))
                any_missing = True

        self._install_all_btn.setVisible(any_missing)

        # TFLite active backend + model file status
        try:
            diag = check_tflite_backend()
            backend = diag.active_backend or t("pkg_backend_none")
            backend_color = "#4caf50" if diag.active_backend else "#f57c00"
            self._active_backend_label.setText(
                t("pkg_active_backend", backend=backend)
            )
            self._active_backend_label.setStyleSheet(
                f"color: {backend_color}; font-size: 11px;"
            )
            if diag.model_exists:
                self._model_file_label.setText(
                    t("pkg_model_file_ok", path=diag.model_path)
                )
                self._model_file_label.setStyleSheet("color: #4caf50; font-size: 11px;")
                self._model_download_btn.setVisible(False)
            else:
                self._model_file_label.setText(
                    t("pkg_model_file_missing", path=diag.model_path)
                )
                self._model_file_label.setStyleSheet("color: #f57c00; font-size: 11px;")
                downloading = (
                    self._model_download_thread is not None
                    and self._model_download_thread.isRunning()
                )
                self._model_download_btn.setVisible(not downloading)
        except Exception:  # noqa: BLE001
            pass

    def _on_install_package(
        self, import_name: str, pip_spec: str, display_name: str
    ) -> None:
        """Start background install for one package."""
        widgets = self._pkg_row_widgets.get(import_name, {})
        btn = widgets.get("btn")
        status_lbl = widgets.get("status")
        icon_lbl = widgets.get("icon")

        if btn:
            btn.setEnabled(False)
            btn.setText(t("pkg_installing_btn"))
        if icon_lbl:
            icon_lbl.setText("⏳")
            icon_lbl.setStyleSheet("color: #888;")
        if status_lbl:
            status_lbl.setText(t("pkg_status_installing"))
            status_lbl.setStyleSheet("color: #888; font-size: 11px;")

        thread = _track_thread(_PackageInstallThread(pip_spec, display_name))
        thread.done.connect(lambda ok, name, iname=import_name:
                            self._on_package_install_done(iname, name, ok))
        self._pkg_install_threads[import_name] = thread
        thread.start()

    def _on_install_all_packages(self) -> None:
        """Install all missing packages one by one."""
        from app.services.dependency_installer import CANDIDATE_PACKAGES, _is_importable
        for import_name, pip_spec, display_name, _ in CANDIDATE_PACKAGES:
            if not _is_importable(import_name):
                self._on_install_package(import_name, pip_spec, display_name)

    def _on_download_model(self) -> None:
        from app.paths import resource_path
        dest = str(resource_path("models/mobilefacenet.tflite"))

        self._model_download_btn.setVisible(False)
        self._model_progress_bar.setValue(0)
        self._model_progress_bar.setVisible(True)
        self._model_file_label.setText(t("pkg_model_downloading"))
        self._model_file_label.setStyleSheet("color: #888; font-size: 11px;")

        self._model_download_thread = _track_thread(_ModelDownloadThread(dest))
        self._model_download_thread.progress.connect(self._model_progress_bar.setValue)
        self._model_download_thread.done.connect(self._on_model_download_done)
        self._model_download_thread.start()

    def _on_model_download_done(self, success: bool, dest_path: str) -> None:
        self._model_progress_bar.setVisible(False)
        if success:
            self._model_file_label.setText(t("pkg_model_file_ok", path=dest_path))
            self._model_file_label.setStyleSheet("color: #4caf50; font-size: 11px;")
            self._model_download_btn.setVisible(False)
            # Refresh the active-backend label — the model is now loadable
            self._refresh_packages_tab()
        else:
            self._model_file_label.setText(t("pkg_model_download_failed"))
            self._model_file_label.setStyleSheet("color: #e53935; font-size: 11px;")
            self._model_download_btn.setVisible(True)
            self._model_download_btn.setText(t("pkg_model_retry_btn"))

    def _on_package_install_done(
        self, import_name: str, display_name: str, success: bool
    ) -> None:
        self._pkg_install_threads.pop(import_name, None)
        widgets = self._pkg_row_widgets.get(import_name, {})
        btn = widgets.get("btn")
        status_lbl = widgets.get("status")
        icon_lbl = widgets.get("icon")

        if success:
            if icon_lbl:
                icon_lbl.setText("✓")
                icon_lbl.setStyleSheet("color: #4caf50; font-weight: bold;")
            if status_lbl:
                status_lbl.setText(t("pkg_status_installed"))
                status_lbl.setStyleSheet("color: #4caf50; font-size: 11px;")
            if btn:
                btn.setVisible(False)
            # Warm up onnxruntime on the main thread after insightface/onnxruntime install
            if import_name in ("onnxruntime", "insightface"):
                try:
                    from app.detectors.factory import warm_up_onnxruntime
                    warm_up_onnxruntime(self._app_config.detection)
                except Exception:  # noqa: BLE001
                    pass
        else:
            if icon_lbl:
                icon_lbl.setText("✗")
                icon_lbl.setStyleSheet("color: #e53935; font-weight: bold;")
            if status_lbl:
                status_lbl.setText(t("pkg_status_failed"))
                status_lbl.setStyleSheet("color: #e53935; font-size: 11px;")
            if btn:
                btn.setEnabled(True)
                btn.setText(t("pkg_install_btn"))

        # Refresh backend/model info line
        self._refresh_packages_tab()
        # Hide "Install All" if nothing missing anymore
        from app.services.dependency_installer import CANDIDATE_PACKAGES, _is_importable
        still_missing = any(
            not _is_importable(n) for n, *_ in CANDIDATE_PACKAGES
        )
        self._install_all_btn.setVisible(still_missing)

    def _on_ai_layer_count_changed(self, count: int) -> None:
        for i, (label, spin) in enumerate(self._ai_layer_rows):
            visible = i < count
            label.setVisible(visible)
            spin.setVisible(visible)

    def _on_ai_layers_reset(self) -> None:
        self._ai_layer_count_spin.setValue(len(self._AI_DEFAULT_LAYERS))
        for (label, spin), size in zip(
            self._ai_layer_rows, self._AI_DEFAULT_LAYERS
        ):
            spin.setValue(size)

    def _selected_hidden_layers(self) -> tuple:
        count = self._ai_layer_count_spin.value()
        return tuple(spin.value() for _, spin in self._ai_layer_rows[:count])

    # ------------------------------------------------------------------
    # TPU
    # ------------------------------------------------------------------

    def _start_tpu_probe(self) -> None:
        """Launch TPU probe in background — dialog opens immediately."""
        self._tpu_summary_label.setText(f"⏳ {t('tpu_checking')}")
        self._tpu_summary_label.setStyleSheet("color: #888;")
        self._probe_thread = _track_thread(_TpuProbeThread())
        self._probe_thread.result_ready.connect(self._on_tpu_probe_done)
        self._probe_thread.start()

    def _on_tpu_probe_done(self, info: dict) -> None:
        ok = info["delegate_ok"]
        if ok:
            self._tpu_summary_label.setText(f"✓ {t('tpu_ok_label')}")
            self._tpu_summary_label.setStyleSheet("color: #4caf50;")
            self._tpu_fix_btn.setVisible(False)
        else:
            parts = []
            if not info["ai_edge_litert"]:
                parts.append(t("ai_edge_missing"))
            if not info["libedgetpu"]:
                parts.append(t("libedgetpu_missing_short"))
            if info["error"]:
                parts.append(info["error"])
            detail = "; ".join(parts) if parts else t("tpu_none")
            self._tpu_summary_label.setText(f"✗ {t('tpu_warn_label')} — {detail}")
            self._tpu_summary_label.setStyleSheet("color: #f57c00;")
            self._tpu_fix_btn.setVisible(True)

    def _on_tpu_check(self) -> None:
        from app.ui.dialogs.tpu_status_dialog import TpuStatusDialog
        dlg = TpuStatusDialog(parent=self)
        dlg.exec()
        self._start_tpu_probe()

    def _on_tpu_fix(self) -> None:
        from app.ui.dialogs.tpu_status_dialog import TpuStatusDialog
        dlg = TpuStatusDialog(parent=self)
        dlg.exec()
        self._start_tpu_probe()

    # ------------------------------------------------------------------
    # Image Library
    # ------------------------------------------------------------------

    def _refresh_library_status(self) -> None:
        """Update the library root label and stats."""
        svc = get_image_library_optional()
        if svc is None:
            self._lib_root_label.setText(t("img_lib_not_configured"))
            self._lib_root_label.setStyleSheet("color: #888;")
            self._lib_stats_label.clear()
            return

        root = svc.library_root
        if root is None:
            self._lib_root_label.setText(t("img_lib_not_configured"))
            self._lib_root_label.setStyleSheet("color: #888;")
        elif svc.is_available():
            self._lib_root_label.setText(t("img_lib_status_ok", path=str(root)))
            self._lib_root_label.setStyleSheet("color: #4caf50;")
        else:
            self._lib_root_label.setText(t("img_lib_status_missing", path=str(root)))
            self._lib_root_label.setStyleSheet("color: #f57c00;")

        self._update_library_stats()

    def _update_library_stats(self) -> None:
        """Refresh the relative/absolute path counts (asynchronously).

        During initial construction the gdrive labels don't exist yet; the
        first stats pass is started by ``_build_ui`` instead.
        """
        if hasattr(self, "_gdrive_cache_size_label"):
            self._start_general_stats()

    def _on_change_library_root(self) -> None:
        svc = get_image_library_optional()
        start = str(svc.library_root) if (svc and svc.library_root) else str(Path.home())
        folder = QFileDialog.getExistingDirectory(
            self, t("img_lib_select_title"), start
        )
        if not folder:
            return
        try:
            if svc is not None:
                svc.set_library_root(folder)
            else:
                from app.services.image_library_service import get_image_library
                get_image_library().set_library_root(folder)
            self._refresh_library_status()
            QMessageBox.information(self, t("img_lib_group"), t("img_lib_root_changed"))
        except (NotADirectoryError, RuntimeError) as exc:
            QMessageBox.warning(self, t("error"), str(exc))

    def _on_migrate_paths(self) -> None:
        svc = get_image_library_optional()
        if svc is None or not svc.is_configured():
            QMessageBox.warning(
                self, t("img_lib_migrate_title"), t("img_lib_no_root_for_migrate")
            )
            return
        if not svc.is_available():
            QMessageBox.warning(
                self,
                t("img_lib_migrate_title"),
                t("img_lib_status_missing", path=str(svc.library_root)),
            )
            return

        # Count images without relative_path
        try:
            from app.db.database import session_scope
            from app.db.models import Image
            with session_scope() as session:
                n = session.query(Image).filter(
                    Image.relative_path.is_(None)
                ).count()
                session.query(Image).count()  # noqa: B018 (side-effect: validates DB)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, t("error"), str(exc))
            return

        if n == 0:
            QMessageBox.information(
                self,
                t("img_lib_migrate_title"),
                t("img_lib_migrate_done", migrated=0, skipped=0),
            )
            return

        reply = QMessageBox.question(
            self,
            t("img_lib_migrate_title"),
            t("img_lib_migrate_confirm", n=n, root=str(svc.library_root)),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        from app.ui.dialogs.image_library_dialog import MigrateLibraryDialog
        dlg = MigrateLibraryDialog(
            db_path=self._current_db_path,
            library_root=str(svc.library_root),
            total_images=n,
            parent=self,
        )
        dlg.exec()
        self._refresh_library_status()

    # ------------------------------------------------------------------
    # DB
    # ------------------------------------------------------------------

    def _on_new_db(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, t("db_new_title"), str(Path.home() / "faces.db"),
            "SQLite (*.db *.sqlite)",
        )
        if path:
            self._new_db_path = path
            self._db_label.setText(path)

    def _on_open_db(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, t("db_open_title"), str(Path.home()),
            "SQLite (*.db *.sqlite);;All files (*)",
        )
        if path:
            self._new_db_path = path
            self._db_label.setText(path)

    # ------------------------------------------------------------------
    # Accept
    # ------------------------------------------------------------------

    def _on_check_update(self) -> None:
        import sys

        # On macOS, when SparkleHelper is bundled, delegate to Sparkle.
        # Sparkle shows its own native update dialog and handles the download.
        if sys.platform == "darwin":
            from app.updater import check_for_updates
            if check_for_updates():
                self._update_status_label.setText(f"🔄 {t('checking')}")
                self._update_status_label.setStyleSheet("")
                return

        from app.services.update_service import fetch_latest_release, is_newer
        from app.ui.dialogs.update_dialog import UpdateDialog

        self._check_upd_btn.setEnabled(False)
        self._update_status_label.setText(f"⏳ {t('checking')}")
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()

        release = fetch_latest_release()
        self._check_upd_btn.setEnabled(True)

        if release is None:
            self._update_status_label.setText(
                f"⚠ {t('could_not_reach_github')}"
            )
            return

        if not is_newer(release.version, __version__):
            self._update_status_label.setText(
                f"✓ {t('up_to_date_version', version=__version__)}"
            )
            self._update_status_label.setStyleSheet("color: #4caf50;")
            return

        self._update_status_label.setText(
            f"🆕 {t('new_version_status', new=release.version, current=__version__)}"
        )
        self._update_status_label.setStyleSheet("color: #ffcc00;")
        dlg = UpdateDialog(release, parent=self)
        dlg.exec()

    def _on_reanalyze_quality(self) -> None:
        """Re-evaluate quality for all faces in the DB."""
        try:
            from app.db.database import session_scope
            from app.db.models import Face as _Face
            with session_scope() as session:
                n = session.query(_Face).count()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, t("error"), str(exc))
            return

        reply = QMessageBox.question(
            self,
            t("fq_group"),
            t("fq_reanalyze_confirm", n=n),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._fq_reanalyze_btn.setEnabled(False)

        def work(ctx):  # noqa: ANN001 — worker thread
            from app.db.database import session_scope
            from app.services.face_quality_service import FaceQualityBatchService

            def on_progress(current: int, total: int) -> None:
                ctx.checkpoint()
                ctx.report(int(current / total * 100) if total else 0, "")

            with session_scope() as session:
                return FaceQualityBatchService(session).evaluate_all(
                    progress_cb=on_progress
                )

        def on_done(processed: object) -> None:
            self._fq_reanalyze_btn.setEnabled(True)
            QMessageBox.information(
                self, t("fq_group"), t("fq_reanalyze_done", n=processed)
            )

        def on_error(message: str) -> None:
            self._fq_reanalyze_btn.setEnabled(True)
            QMessageBox.critical(self, t("error"), message)

        from app.tasks import get_task_manager
        get_task_manager().submit(
            t("task_quality_reanalyze"),
            work,
            supports_pause=True,
            on_done=on_done,
            on_error=on_error,
            on_cancelled=lambda: self._fq_reanalyze_btn.setEnabled(True),
        )

    def _on_accept(self) -> None:
        qs = _qsettings()
        qs.setValue("updates/notify", self._notify_check.isChecked())
        qs.setValue("deoldified/auto_pair", self._deoldified_check.isChecked())
        qs.setValue("deoldified/auto_sync", self._deoldified_sync_check.isChecked())
        qs.setValue("geocoding/enabled", self._geocoding_check.isChecked())
        qs.setValue(
            "face_quality/exclude_low_quality", self._fq_exclude_check.isChecked()
        )
        qs.setValue(
            "tasks/auto_cleanup_enabled",
            self._tasks_autocleanup_check.isChecked(),
        )
        qs.setValue("rerecognition/enabled", self._rerec_enabled_check.isChecked())
        # Keep suggest threshold strictly below the auto-merge threshold.
        auto_pct = self._rerec_auto_spin.value()
        suggest_pct = min(self._rerec_suggest_spin.value(), auto_pct - 1)
        qs.setValue("rerecognition/auto_threshold", auto_pct / 100.0)
        qs.setValue("rerecognition/suggest_threshold", max(1, suggest_pct) / 100.0)
        # The Recording tab is built lazily — only persist its values when the
        # user actually visited it (its widgets exist).
        if hasattr(self, "_rec_dir_edit"):
            qs.setValue("recording/output_dir", self._rec_dir_edit.text().strip())
            qs.setValue("recording/quality", self._rec_quality_combo.currentData())
            qs.setValue("recording/fps", self._rec_fps_spin.value())
            qs.setValue("recording/segment_seconds", self._rec_segment_spin.value())
            qs.setValue("recording/ffmpeg_path", self._rec_ffmpeg_edit.text().strip())
            qs.setValue("recording/capture_cursor", self._rec_cursor_check.isChecked())
            qs.setValue(
                "recording/capture_microphone", self._rec_mic_check.isChecked()
            )
            qs.setValue(
                "recording/capture_system_audio", self._rec_sysaudio_check.isChecked()
            )
            qs.setValue("recording/concat_on_stop", self._rec_concat_check.isChecked())
            qs.setValue(
                "recording/audio_input_device",
                self._rec_audio_input_combo.currentData() or "",
            )
            qs.setValue(
                "recording/system_audio_device",
                self._rec_system_audio_combo.currentData() or "",
            )
            qs.setValue(
                "recording/mic_volume", self._rec_mic_volume_slider.value() / 100.0
            )
            qs.setValue(
                "recording/system_volume",
                self._rec_system_volume_slider.value() / 100.0,
            )
            qs.setValue(
                "recording/mute_microphone", self._rec_mute_mic_check.isChecked()
            )
            qs.setValue(
                "recording/mute_system_audio", self._rec_mute_system_check.isChecked()
            )
            mode_btn = self._rec_mode_group.checkedButton()
            if mode_btn is not None:
                qs.setValue("recording/display_mode", mode_btn.property("rec_mode"))
            qs.setValue(
                "recording/selected_display_ids",
                [
                    cb.property("display_id")
                    for cb in self._rec_display_checks
                    if cb.isChecked()
                ],
            )
            qs.setValue(
                "recording/auto_reduce_fps", self._rec_auto_fps_check.isChecked()
            )
        qs.setValue("debug/ai_visualization", self._debug_viz_check.isChecked())
        qs.setValue("debug/ai_log_enabled", self._debug_log_check.isChecked())
        qs.setValue(
            "debug/detection_log_enabled",
            self._debug_detection_log_check.isChecked(),
        )
        # Flush explicitly — the transient QSettings objects above would
        # otherwise only persist on destruction, which is unreliable on
        # Windows and can silently drop the writes.
        qs.sync()
        # Persist a changed neural-network layout into the YAML config and the
        # live config object; the model itself retrains on the next AI run.
        new_layers = self._selected_hidden_layers()
        if new_layers != tuple(self._app_config.deep_recognition.hidden_layers):
            from app.config import save_deep_recognition_values
            self._app_config.deep_recognition.hidden_layers = new_layers
            save_deep_recognition_values({"hidden_layers": list(new_layers)})
        # Persist a changed embedding backend (AI Packages tab is built lazily,
        # so only act when its widgets were actually created).
        if hasattr(self, "_emb_backend_group"):
            checked = self._emb_backend_group.checkedButton()
            new_backend = (
                checked.property("emb_backend") if checked else "mobilefacenet"
            )
            if new_backend != self._emb_backend_initial:
                from app.config import save_embedding_values
                self._app_config.embedding.backend = new_backend
                save_embedding_values({"backend": new_backend})
                _BACKEND_LABELS = {
                    "mobilefacenet": "MobileFaceNet (192-dim)",
                    "arcface": "ArcFace ResNet-50 (512-dim)",
                }
                QMessageBox.information(
                    self,
                    t("pkg_emb_switch_title"),
                    t(
                        "pkg_emb_switch_msg",
                        old=_BACKEND_LABELS.get(self._emb_backend_initial, self._emb_backend_initial),
                        new=_BACKEND_LABELS.get(new_backend, new_backend),
                    ),
                )
        # Persist the Apple Neural Engine (Core ML) preference (Apple Silicon
        # only; the checkbox is None elsewhere).  Same model + dimensionality,
        # so toggling it needs no re-embed — it only changes the runtime device.
        if getattr(self, "_emb_coreml_check", None) is not None:
            new_pref = self._emb_coreml_check.isChecked()
            if new_pref != self._emb_coreml_initial:
                from app.config import save_embedding_values
                self._app_config.embedding.prefer_coreml = new_pref
                save_embedding_values({"prefer_coreml": new_pref})
        selected_lang = self._lang_combo.currentData()
        if selected_lang != current_language():
            set_language(selected_lang)
            self._language_changed = True
        self.accept()

    def selected_db_path(self) -> Optional[str]:
        return self._new_db_path

    def language_changed(self) -> bool:
        return self._language_changed
