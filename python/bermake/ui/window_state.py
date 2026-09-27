"""Window geometry and toolbar-layout persistence (M7.2).

Bermake's first non-document state. Everything else the application remembers
lives inside the .berm file; this is per-user, per-machine preference.

Every function takes the QSettings object as a parameter rather than
constructing one. On Windows a default-constructed QSettings writes to the
real registry, so tests inject a temporary IniFormat store and the suite
touches no real user state.

Restore never raises. A corrupt or stale blob returns False and the caller
keeps Qt's default layout -- a bad settings store must not be able to stop
the application from starting.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QMainWindow

logger = logging.getLogger(__name__)

ORGANIZATION_NAME = "Parrow Horrizon Studio"
APPLICATION_NAME = "Bermake"

GEOMETRY_KEY = "window/geometry"
STATE_KEY = "window/state"

# Bump when the set of toolbars or docks changes. Qt's restoreState() compares
# this against the saved value and refuses a mismatch, so an old layout is
# discarded cleanly instead of being half-applied.
#
# 1 -> 2 (M7.3): the Materials, Tags and Scenes docks became tabs inside one
# Properties dock. A v1 blob names three docks that no longer exist, and
# restoring it would leave the new panel unplaced.
WINDOW_STATE_VERSION = 2


def save_window_state(window: QMainWindow, settings: QSettings) -> None:
    """Persist size/position and the dock + toolbar arrangement."""
    settings.setValue(GEOMETRY_KEY, window.saveGeometry())
    settings.setValue(STATE_KEY, window.saveState(WINDOW_STATE_VERSION))


def restore_window_state(window: QMainWindow, settings: QSettings) -> bool:
    """Reapply a saved layout. Returns True only if both parts restored.

    Atomic: either both geometry and state end up applied, or neither does.
    If the state restore fails after geometry already changed, the geometry
    is rolled back to what it was before this call.

    False means "use the defaults" and is the normal result on first run.
    """
    geometry = settings.value(GEOMETRY_KEY)
    state = settings.value(STATE_KEY)
    if geometry is None or state is None:
        return False

    previous_geometry = window.saveGeometry()

    try:
        if not window.restoreGeometry(geometry):
            return False
        if window.restoreState(state, WINDOW_STATE_VERSION):
            return True
        window.restoreGeometry(previous_geometry)
        return False
    except (TypeError, ValueError):
        # A blob of the wrong type entirely (hand-edited INI, foreign writer).
        logger.warning("discarding unreadable saved window state", exc_info=True)
        window.restoreGeometry(previous_geometry)
        return False


def reset_window_state(settings: QSettings) -> None:
    """Forget the saved layout. Backs View > Reset Toolbars."""
    settings.remove(GEOMETRY_KEY)
    settings.remove(STATE_KEY)
