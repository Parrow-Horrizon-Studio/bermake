"""Bermake application entry point.

Startup is an ordered sequence, and the order is load-bearing (M7.9):

1. Qt's organisation and application names are set first, because the log
   folder and every QSettings read are derived from them.
2. Logging and crash capture are configured before anything else can fail.
3. Compatibility rendering, if requested, switches PyOpenGL and Qt to the
   bundled Mesa. It must precede QApplication and any import of OpenGL.GL,
   which is why MainWindow is imported inside _build_main_window. A failure
   to switch is logged and startup continues on the system driver, so a
   stored preference can never lock the tester out.
4. QApplication is created and themed, then the OpenGL preflight asks Qt for
   a context before any window exists (gl_preflight), because on an OpenGL
   1.x driver the viewport never gets one and its own check never runs.
5. MainWindow is built inside start_window, so a failure while building it
   still reaches the tester.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QSettings

from bermake import __version__
from bermake.diagnostics import compat_rendering, logs
from bermake.diagnostics.compat_rendering import default_mesa_dir, enable_compatibility_rendering
from bermake.diagnostics.error_dialog import (
    deferred_error_dialog,
    show_error_dialog,
    show_gl_fallback_dialog,
)
from bermake.diagnostics.gl_preflight import PreflightOutcome, run_gl_preflight
from bermake.diagnostics.launch import LaunchArgs, parse_launch_args
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


def _build_main_window(*, gl_reported: bool = False, compat_failed: bool = False):
    """Build and show the main window.

    `gl_reported`: the startup check already showed an OpenGL problem, so the
    viewport must not show its version twice. `compat_failed`: switching to
    compatibility rendering was requested and failed, so the window must not
    offer the restart into it. Both are set before show(), because the
    viewport's initializeGL runs on show.
    """
    from bermake.ui.main_window import MainWindow

    window = MainWindow()
    if gl_reported:
        window.note_gl_reported_at_startup()
    if compat_failed:
        window.disable_compatibility_offer()
    window.show()
    # After show(), so the dialog is modal over a real window rather than over
    # nothing, and so there is no second "no document yet" code path: the
    # template applies to the live document the window already built.
    if preferences.read_show_welcome(QSettings()):
        window.show_welcome_dialog()
    return window


def wants_compatibility_rendering(args: LaunchArgs, stored: bool) -> bool:
    """--no-compatibility-rendering wins over the stored preference and over
    --compatibility-rendering: it is the way back when compatibility rendering
    itself stops Bermake starting (final review I3)."""
    if args.no_compatibility_rendering:
        return False
    return args.compatibility_rendering or stored


def apply_compatibility_rendering(
    requested: bool,
    *,
    find_mesa: Callable[[], Path | None] = default_mesa_dir,
    enable: Callable[[Path], None] = enable_compatibility_rendering,
) -> bool:
    """Switch to the bundled Mesa if requested. True if it is now on.

    A failure to switch (antivirus blocking the DLL, for one) is logged and
    Bermake continues on the system driver. Letting it propagate would stop
    every later launch the same way, because the preference is stored, with
    the Help menu that turns it off out of reach (final review I3).
    """
    if not requested:
        return False
    mesa_dir = find_mesa()
    if mesa_dir is None:
        logger.warning(
            "compatibility rendering was requested but Mesa was not found; "
            "using the system OpenGL driver"
        )
        return False
    try:
        enable(mesa_dir)
    except Exception:
        logger.exception(
            "could not enable compatibility rendering from %s; using the system OpenGL driver",
            mesa_dir,
        )
        return False
    logger.info("compatibility rendering enabled from %s", mesa_dir)
    return True


def mesa_restart_available(
    *, compat_failed: bool, find_mesa: Callable[[], Path | None] = default_mesa_dir
) -> bool:
    """Whether "restart using compatibility rendering" can help. Not when
    switching to it already failed this launch: the restart would land in the
    same failure and offer itself again, every time."""
    return not compat_failed and find_mesa() is not None


def _store_compatibility_preference() -> None:
    settings = QSettings()
    preferences.write_compatibility_rendering(settings, True)
    settings.sync()


def main(argv: list[str] | None = None) -> int:
    """Application entry point. Returns process exit code."""
    args = parse_launch_args(sys.argv if argv is None else argv)
    QCoreApplication.setOrganizationName(ORGANIZATION_NAME)
    QCoreApplication.setApplicationName(APPLICATION_NAME)

    # Before logging and before any window: the smoke test reports through its
    # own JSON file and must not touch the tester's log or preferences.
    if args.smoke_report is not None:
        from bermake.diagnostics.smoke import run_smoke

        return run_smoke(args.smoke_report)

    log_dir = logs.default_log_directory()
    logs.configure_file_logging(log_dir)
    native_crash_stream = logs.enable_native_crash_log(log_dir, __version__)
    try:
        return _run(args, log_dir)
    finally:
        logs.close_native_crash_log(native_crash_stream)


def _run(args: LaunchArgs, log_dir: Path) -> int:
    logs.install_qt_message_handler()
    reporter = logs.ErrorReporter(deferred_error_dialog, log_dir)
    logs.install_exception_hooks(reporter)
    logger.info(logs.session_header(__version__))

    # Before QApplication and before anything imports OpenGL.GL (spec 2.4.1).
    stored = preferences.read_compatibility_rendering(QSettings())
    if args.no_compatibility_rendering:
        logger.info("compatibility rendering is off for this launch (--no-compatibility-rendering)")
    requested = wants_compatibility_rendering(args, stored)
    compat_failed = requested and not apply_compatibility_rendering(requested)

    from PySide6.QtWidgets import QApplication

    app = QApplication(args.qt_argv)

    # Before MainWindow() so the first icon build already uses the right
    # palette (M7.7, #101), and before the preflight so its dialog does too.
    apply_theme(app, theme_for_name(preferences.read_theme(QSettings())))

    outcome = run_gl_preflight(
        mesa_available=mesa_restart_available(compat_failed=compat_failed),
        compat_active=compat_rendering.compatibility_rendering_active(),
        prompt=show_gl_fallback_dialog,
        store_preference=_store_compatibility_preference,
        relaunch=compat_rendering.relaunch,
    )
    if outcome is PreflightOutcome.RELAUNCHED:
        return 0

    window = start_window(
        lambda: _build_main_window(
            gl_reported=outcome is PreflightOutcome.REPORTED, compat_failed=compat_failed
        ),
        reporter,
        show_error_dialog,
    )
    if window is None:
        return 1
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
