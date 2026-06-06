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


class _TpuProbeThread(QThread):
    """Runs probe_tpu() in background so the dialog doesn't freeze."""

    result_ready = Signal(dict)

    def run(self) -> None:
        from app.ui.dialogs.tpu_status_dialog import probe_tpu
        self.result_ready.emit(probe_tpu())


class SettingsDialog(QDialog):
    """Settings dialog: language, database management, and TPU status."""

    def __init__(self, current_db_path: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("settings_title"))
        self.setMinimumWidth(540)
        self.setMinimumHeight(300)
        self.resize(560, 640)
        self._current_db_path = current_db_path
        self._new_db_path: Optional[str] = None
        self._language_changed = False
        self._probe_thread: Optional[_TpuProbeThread] = None
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        tabs = QTabWidget()
        tabs.addTab(self._build_tab_general(), t("settings_tab_general"))
        tabs.addTab(self._build_tab_pairing(), t("settings_tab_pairing"))
        tabs.addTab(self._build_tab_face_quality(), t("settings_tab_quality"))
        tabs.addTab(self._build_tab_shortcuts(), t("settings_tab_shortcuts"))
        tabs.addTab(self._build_tab_recording(), t("settings_tab_recording"))
        tabs.addTab(self._build_tab_gdrive(), t("settings_tab_gdrive"))
        outer.addWidget(tabs, stretch=1)

        # ── Buttons — always visible outside the tabs ─────────────────────
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        outer.addWidget(btns)

        self._start_tpu_probe()

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
        from app.gdrive.cache import get_cache_manager
        _mb = get_cache_manager().current_size_bytes() / (1024 * 1024)
        self._gdrive_cache_size_label = QLabel(t("gdrive_cache_size_value", mb=_mb))
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
        from app.services.screen_recorder_service import (
            list_audio_devices,
            resolve_ffmpeg,
        )

        group = QGroupBox(t("rec_audio_group"))
        form = QFormLayout(group)

        # Best-effort device enumeration; an empty list just leaves "Automatic".
        ffmpeg = resolve_ffmpeg(
            qs.value("recording/ffmpeg_path", "", type=str) or None
        )
        try:
            devices = list_audio_devices(ffmpeg)
        except Exception:  # noqa: BLE001 — never let probing break Settings
            devices = []

        def _device_combo(stored_key: str) -> QComboBox:
            combo = QComboBox()
            combo.addItem(t("rec_audio_auto"), userData="")
            for name in devices:
                combo.addItem(name, userData=name)
            stored = qs.value(stored_key, "", type=str) or ""
            idx = combo.findData(stored)
            # Keep a stored device even if it is not currently present.
            if idx < 0 and stored:
                combo.addItem(stored, userData=stored)
                idx = combo.findData(stored)
            combo.setCurrentIndex(max(0, idx))
            return combo

        self._rec_audio_input_combo = _device_combo("recording/audio_input_device")
        form.addRow(t("rec_audio_input_device"), self._rec_audio_input_combo)
        self._rec_system_audio_combo = _device_combo("recording/system_audio_device")
        form.addRow(t("rec_system_audio_device"), self._rec_system_audio_combo)

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
        layout.addStretch()
        scroll.setWidget(inner)
        return scroll

    # ------------------------------------------------------------------
    # TPU
    # ------------------------------------------------------------------

    def _start_tpu_probe(self) -> None:
        """Launch TPU probe in background — dialog opens immediately."""
        self._tpu_summary_label.setText(f"⏳ {t('tpu_checking')}")
        self._tpu_summary_label.setStyleSheet("color: #888;")
        self._probe_thread = _TpuProbeThread(self)
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
        """Show counts of relative vs. absolute paths in a subtle label."""
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
            parts = []
            if n_rel:
                parts.append(t("img_lib_n_relative", n=n_rel))
            if n_abs:
                parts.append(t("img_lib_n_absolute", n=n_abs))
            self._lib_stats_label.setText("  ".join(parts))
        except Exception:  # noqa: BLE001
            self._lib_stats_label.clear()

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
        try:
            from app.db.database import session_scope
            from app.services.face_quality_service import FaceQualityBatchService
            with session_scope() as session:
                svc = FaceQualityBatchService(session)
                processed = svc.evaluate_all()
            QMessageBox.information(
                self,
                t("fq_group"),
                t("fq_reanalyze_done", n=processed),
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, t("error"), str(exc))
        finally:
            self._fq_reanalyze_btn.setEnabled(True)

    def _on_accept(self) -> None:
        qs = _qsettings()
        qs.setValue("updates/notify", self._notify_check.isChecked())
        qs.setValue("deoldified/auto_pair", self._deoldified_check.isChecked())
        qs.setValue("deoldified/auto_sync", self._deoldified_sync_check.isChecked())
        qs.setValue(
            "face_quality/exclude_low_quality", self._fq_exclude_check.isChecked()
        )
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
        # Flush explicitly — the transient QSettings objects above would
        # otherwise only persist on destruction, which is unreliable on
        # Windows and can silently drop the writes.
        qs.sync()
        selected_lang = self._lang_combo.currentData()
        if selected_lang != current_language():
            set_language(selected_lang)
            self._language_changed = True
        self.accept()

    def selected_db_path(self) -> Optional[str]:
        return self._new_db_path

    def language_changed(self) -> bool:
        return self._language_changed
