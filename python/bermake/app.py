"""Bermake application entry point."""

import sys

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from bermake import __version__
from bermake.ui import preferences
from bermake.ui.main_window import MainWindow
from bermake.ui.theme import apply_theme, theme_for_name
from bermake.ui.window_state import APPLICATION_NAME, ORGANIZATION_NAME


def main() -> int:
    """Application entry point. Returns process exit code."""
    print(f"Bermake {__version__}")
    app = QApplication(sys.argv)
    # QSettings keys off these; without them it has no per-application store.
    app.setOrganizationName(ORGANIZATION_NAME)
    app.setApplicationName(APPLICATION_NAME)

    # Before MainWindow() so the first icon build already uses the right
    # palette (M7.7, #101).
    apply_theme(app, theme_for_name(preferences.read_theme(QSettings())))

    window = MainWindow()
    window.show()

    # After show(), so the dialog is modal over a real window rather than over
    # nothing, and so there is no second "no document yet" code path: the
    # template applies to the live document the window already built.
    if preferences.read_show_welcome(QSettings()):
        window.show_welcome_dialog()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
