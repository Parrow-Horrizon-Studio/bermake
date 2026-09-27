"""Application colour scheme (M7.7).

Light and Dark only, with Light the default. No System option: SketchUp is
light on every OS regardless of the system setting, so light-unless-asked is
the faithful behaviour, and a third state would triple what the palette-refresh
path has to be tested against for nothing a user can see. It stays trivially
addable later, because Qt.ColorScheme.Unknown already means follow-the-system.

Qt is 6.11, so QStyleHints.setColorScheme is available directly and no custom
stylesheet is needed: the icons already recolour themselves from the palette
(see bermake.ui.icons), they simply never refreshed when it changed (#101).
"""

from __future__ import annotations

from enum import Enum

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication


class ThemeChoice(Enum):
    LIGHT = "light"
    DARK = "dark"


_SCHEMES = {
    ThemeChoice.LIGHT: Qt.ColorScheme.Light,
    ThemeChoice.DARK: Qt.ColorScheme.Dark,
}


def theme_for_name(name: str | None) -> ThemeChoice:
    """The ThemeChoice for a stored name, defaulting to Light.

    Falls back rather than raising: ThemeChoice(name) would raise ValueError on
    a corrupt preference, and this is read during startup before the main window
    exists, so it must not be able to stop the application from opening.
    """
    try:
        return ThemeChoice(name)
    except ValueError:
        return ThemeChoice.LIGHT


def apply_theme(app: QApplication, choice: ThemeChoice) -> None:
    """Set the application-wide colour scheme.

    Qt emits a palette change from this, which MainWindow.changeEvent picks up
    to rebuild the icons.
    """
    app.styleHints().setColorScheme(_SCHEMES[choice])
