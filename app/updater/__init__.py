"""Sparkle-based auto-updater for macOS.

On all other platforms this module is a no-op: every public function returns False
and can be called unconditionally.
"""

from .sparkle_bridge import check_for_updates, is_sparkle_available, start_background_update_check

__all__ = ["check_for_updates", "is_sparkle_available", "start_background_update_check"]
