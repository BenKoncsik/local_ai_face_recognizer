"""A compact horizontal audio VU meter driven by peak dBFS readings.

The recorder emits the mixed-bus peak level (``ScreenRecorderService.audio_level``)
roughly ten times a second; :meth:`AudioLevelMeter.set_level_db` maps that onto a
coloured fill bar with a short decay so the bar falls smoothly between peaks.

When no signal is seen for a while the meter shows a muted "no audio" state so a
silent recording is visible *while it happens*, not only after validation.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QWidget

# Display range: -60 dBFS (floor) .. 0 dBFS (full scale).
_DB_FLOOR = -60.0
# Below this peak we treat the input as effectively silent.
_SILENCE_DB = -55.0
# Consecutive silent updates before the "no audio" warning latches.
_SILENCE_TICKS = 12


def level_to_fraction(db: Optional[float]) -> float:
    """Map a peak dBFS value onto a 0..1 bar fraction (clamped)."""
    if db is None:
        return 0.0
    if db <= _DB_FLOOR:
        return 0.0
    if db >= 0.0:
        return 1.0
    return (db - _DB_FLOOR) / (0.0 - _DB_FLOOR)


class AudioLevelMeter(QWidget):
    """Thin horizontal peak meter with green/amber/red zones."""

    def __init__(self, label: str = "", parent=None) -> None:
        super().__init__(parent)
        self._label = label
        self._fraction = 0.0
        self._silent_ticks = 0
        self._active = False  # True while a recording is running
        self.setMinimumSize(90, 16)
        self.setToolTip(label)
        # Decay the bar smoothly so it falls between the ~10 Hz peak updates.
        self._decay = QTimer(self)
        self._decay.setInterval(60)
        self._decay.timeout.connect(self._on_decay)

    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin metering: reset state and run the decay animation."""
        self._active = True
        self._fraction = 0.0
        self._silent_ticks = 0
        self._decay.start()
        self.update()

    def stop(self) -> None:
        """Stop metering and clear the bar."""
        self._active = False
        self._fraction = 0.0
        self._silent_ticks = 0
        self._decay.stop()
        self.update()

    def set_level_db(self, db: float) -> None:
        """Feed a fresh peak reading (dBFS) into the meter."""
        if not self._active:
            return
        frac = level_to_fraction(db)
        # Instant attack, the timer handles the decay.
        if frac > self._fraction:
            self._fraction = frac
        if db <= _SILENCE_DB:
            self._silent_ticks = min(_SILENCE_TICKS, self._silent_ticks + 1)
        else:
            self._silent_ticks = 0
        self.update()

    @property
    def is_silent(self) -> bool:
        """True once no meaningful signal has arrived for a short while."""
        return self._active and self._silent_ticks >= _SILENCE_TICKS

    # ------------------------------------------------------------------

    def _on_decay(self) -> None:
        if self._fraction > 0.0:
            self._fraction = max(0.0, self._fraction - 0.06)
            self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt signature)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = self.rect().adjusted(0, 0, -1, -1)
        # Track.
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#313244"))
        p.drawRoundedRect(QRectF(rect), 3, 3)

        if self._active and self._fraction > 0.0:
            w = rect.width() * self._fraction
            fill = QRectF(rect.x(), rect.y(), w, rect.height())
            # Colour by how hot the peak is.
            if self._fraction > 0.9:
                colour = QColor("#F38BA8")   # red — near clipping
            elif self._fraction > 0.7:
                colour = QColor("#F9E2AF")   # amber
            else:
                colour = QColor("#A6E3A1")   # green
            p.setBrush(colour)
            p.drawRoundedRect(fill, 3, 3)

        # Label / silence warning overlaid on the bar.
        if self._active and self.is_silent:
            p.setPen(QColor("#F38BA8"))
            text = f"{self._label}: —"
        else:
            p.setPen(QColor("#CDD6F4"))
            text = self._label
        if text:
            font = p.font()
            font.setPixelSize(10)
            p.setFont(font)
            p.drawText(rect.adjusted(4, 0, -2, 0), Qt.AlignVCenter | Qt.AlignLeft, text)
        p.end()
