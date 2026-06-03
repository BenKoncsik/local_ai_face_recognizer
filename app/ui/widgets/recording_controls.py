"""Toolbar widget that drives the screen recorder.

Emits intent signals (``start_requested`` / ``pause_toggle_requested`` /
``stop_requested``); the :class:`MainWindow` wires these to a
:class:`~app.services.screen_recorder_service.ScreenRecorderService` and feeds
the recorder state back via :meth:`set_state`.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from app.services.screen_recorder_service import RecorderState
from app.ui.i18n import t
from app.ui.widgets.audio_level_meter import AudioLevelMeter


class RecordingControls(QWidget):
    """Record / Pause-Resume / Stop buttons with a state indicator."""

    start_requested = Signal()
    pause_toggle_requested = Signal()
    stop_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._state = RecorderState.IDLE
        self._build_ui()
        self.set_state(RecorderState.IDLE)

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._record_btn = QPushButton("●")  # ●
        self._record_btn.setToolTip(t("rec_start_tip"))
        self._record_btn.clicked.connect(self.start_requested)
        layout.addWidget(self._record_btn)

        self._pause_btn = QPushButton("⏸")  # ⏸
        self._pause_btn.setToolTip(t("rec_pause_tip"))
        self._pause_btn.clicked.connect(self.pause_toggle_requested)
        layout.addWidget(self._pause_btn)

        self._stop_btn = QPushButton("■")  # ■
        self._stop_btn.setToolTip(t("rec_stop_tip"))
        self._stop_btn.clicked.connect(self.stop_requested)
        layout.addWidget(self._stop_btn)

        self._status_lbl = QLabel()
        layout.addWidget(self._status_lbl)

        # Live input-level VU meter (mixed mic + system bus).  Hidden until a
        # recording starts.
        self._meter = AudioLevelMeter(t("rec_audio_meter_label"))
        self._meter.setVisible(False)
        layout.addWidget(self._meter)

    # ------------------------------------------------------------------

    def set_audio_level(self, db: float) -> None:
        """Feed a peak dBFS reading from the recorder into the VU meter."""
        self._meter.set_level_db(db)

    def set_state(self, state: RecorderState) -> None:
        """Reflect the recorder *state* in button enablement and the label."""
        self._state = state
        recording = state is RecorderState.RECORDING
        paused = state is RecorderState.PAUSED
        active = recording or paused
        finalizing = state is RecorderState.FINALIZING

        self._record_btn.setEnabled(state in (RecorderState.IDLE, RecorderState.ERROR))
        self._pause_btn.setEnabled(active)
        self._stop_btn.setEnabled(active)

        # Pause button toggles to a resume glyph while paused.
        self._pause_btn.setText("▶" if paused else "⏸")  # ▶ / ⏸
        self._pause_btn.setToolTip(t("rec_resume_tip") if paused else t("rec_pause_tip"))

        text, color = {
            RecorderState.IDLE: (t("rec_state_idle"), "#A6ADC8"),
            RecorderState.RECORDING: (t("rec_state_recording"), "#F38BA8"),
            RecorderState.PAUSED: (t("rec_state_paused"), "#F9E2AF"),
            RecorderState.FINALIZING: (t("rec_state_finalizing"), "#89DCEB"),
            RecorderState.ERROR: (t("rec_state_error"), "#F38BA8"),
        }[state]
        self._status_lbl.setText(text)
        self._status_lbl.setStyleSheet(f"color: {color}; font-size: 11px;")

        # The VU meter is only meaningful while frames are flowing.
        if recording:
            if not self._meter.isVisible():
                self._meter.setVisible(True)
            self._meter.start()
        else:
            self._meter.stop()
            self._meter.setVisible(active)  # keep visible (idle) while paused
        # Avoid an unused-var lint while keeping intent explicit.
        _ = finalizing

    def set_elapsed(self, seconds: int) -> None:
        """Append the elapsed time to the status label while recording."""
        if self._state in (RecorderState.RECORDING, RecorderState.PAUSED):
            h, rem = divmod(int(seconds), 3600)
            m, s = divmod(rem, 60)
            base = (
                t("rec_state_recording")
                if self._state is RecorderState.RECORDING
                else t("rec_state_paused")
            )
            self._status_lbl.setText(f"{base}  {h:02d}:{m:02d}:{s:02d}")

    def retranslate(self) -> None:
        self._record_btn.setToolTip(t("rec_start_tip"))
        self._stop_btn.setToolTip(t("rec_stop_tip"))
        self.set_state(self._state)
