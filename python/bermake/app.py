"""Bermake application entry point.

Startup is an ordered sequence, and the order is load-bearing (M7.9):

1. Qt's organisation and application names are set first, because the log
   folder and every QSettings read are derived from them.
2. Logging and crash capture are configured before anything else can fail.
3. Compatibility rendering, if requested, switches PyOpenGL and Qt to the
   bundled Mesa. It must precede QApplication and any import of OpenGL.GL,
   which is why MainWindow is imported inside _build_main_window.
4. QApplication is created, then MainWindow is built inside start_window, so
   a failure while building it still reaches the tester.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QSettings

from bermake import __version__
from bermake.diagnostics import logs
from bermake.diagnostics.compat_rendering import default_mesa_dir, enable_compatibility_rendering
from bermake.diagnostics.error_dialog import deferred_error_dialog, show_error_dialog
from bermake.diagnostics.launch import parse_launch_args
from bermake.ui import preferences
from bermake.ui.theme import apply_theme, theme_for_name
from bermake.ui.window_state import APPLICATION_NAME, ORGANIZATION_NAME

logger = logging.getLogger("bermake")


def start_window(
    window_factory: Callable[[], object],
    reporter: logs.ErrorReporter,
    show_now: Callable[[Path], None],
) -> object | None:
    """Build the main window, or report why not.

    No event loop runs yet, so the dialog must be shown synchronously; a
    deferred one would never appear and the application would just exit.
    """
    try:
        return window_factory()
    except Exception:
        reporter.report_startup_failure(*sys.exc_info(), show_now)
        return None


def _build_main_window():
    from bermake.ui.main_window import MainWindow

    window = MainWindow()
    window.show()
    # After show(), so the dialog is modal over a real window rather than over
    # nothing, and so there is no second "no document yet" code path: the
    # template applies to the live document the window already built.
    if preferences.read_show_welcome(QSettings()):
        window.show_welcome_dialog()
    return window


def main(argv: list[str] | None = None) -> int:
    """Application entry point. Returns process exit code."""
    args = parse_launch_args(sys.argv if argv is None else argv)
    QCoreApplication.setOrganizationName(ORGANIZATION_NAME)
    QCoreApplication.setApplicationName(APPLICATION_NAME)

    log_dir = logs.default_log_directory()
    logs.configure_file_logging(log_dir)
    native_crash_stream = logs.enable_native_crash_log(log_dir)
    logs.install_qt_message_handler()
    reporter = logs.ErrorReporter(deferred_error_dialog, log_dir)
    logs.install_exception_hooks(reporter)
    logger.info(logs.session_header(__version__))

    # Before QApplication and before anything imports OpenGL.GL (spec 2.4.1).
    if args.compatibility_rendering or preferences.read_compatibility_rendering(QSettings()):
        mesa_dir = default_mesa_dir()
        if mesa_dir is None:
            logger.warning(
                "compatibility rendering was requested but Mesa was not found; "
                "using the system OpenGL driver"
            )
        else:
            enable_compatibility_rendering(mesa_dir)
            logger.info("compatibility rendering enabled from %s", mesa_dir)

    from PySide6.QtWidgets import QApplication

    app = QApplication(args.qt_argv)

    # Before MainWindow() so the first icon build already uses the right
    # palette (M7.7, #101).
    apply_theme(app, theme_for_name(preferences.read_theme(QSettings())))

    window = start_window(_build_main_window, reporter, show_error_dialog)
    if window is None:
        return 1
    exit_code = app.exec()
    native_crash_stream.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
