"""app.py's startup sequence (M7.9)."""

import logging
import subprocess
import sys
from pathlib import Path

import pytest
from bermake.app import apply_compatibility_rendering, start_window, wants_compatibility_rendering
from bermake.diagnostics.launch import parse_launch_args
from bermake.diagnostics.logs import ErrorReporter


def test_a_failing_window_factory_shows_the_error_now_and_returns_none(tmp_path):
    immediate, deferred = [], []
    reporter = ErrorReporter(deferred.append, tmp_path)

    def broken_factory():
        raise RuntimeError("MainWindow could not be built")

    assert start_window(broken_factory, reporter, immediate.append) is None
    assert immediate == [tmp_path]
    assert deferred == []


def test_a_working_window_factory_returns_the_window(tmp_path):
    immediate = []
    reporter = ErrorReporter(lambda _log_dir: None, tmp_path)
    window = object()

    assert start_window(lambda: window, reporter, immediate.append) is window
    assert immediate == []


def test_importing_the_app_module_does_not_import_opengl():
    """Compatibility rendering must be decided before OpenGL.GL is imported
    (spec 2.4.1), so the module that makes the decision must not import it."""
    code = "import sys, bermake.app; print('OpenGL.GL' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.stdout.strip() == "False", result.stderr


@pytest.mark.parametrize(
    ("flags", "stored", "wanted"),
    [
        ([], False, False),
        ([], True, True),
        (["--compatibility-rendering"], False, True),
        (["--no-compatibility-rendering"], True, False),
        (["--compatibility-rendering", "--no-compatibility-rendering"], False, False),
        (["--no-compatibility-rendering", "--compatibility-rendering"], True, False),
    ],
)
def test_the_no_compatibility_flag_wins_over_everything(flags, stored, wanted):
    args = parse_launch_args(["Bermake.exe", *flags])
    assert wants_compatibility_rendering(args, stored) is wanted


def test_compatibility_rendering_not_requested_touches_nothing():
    def unexpected(*_args):
        raise AssertionError("should not be called")

    assert apply_compatibility_rendering(False, find_mesa=unexpected, enable=unexpected) is False


def test_compatibility_rendering_is_enabled_from_the_found_mesa():
    enabled = []
    mesa = Path("C:/B/_internal/mesa")

    assert apply_compatibility_rendering(True, find_mesa=lambda: mesa, enable=enabled.append)
    assert enabled == [mesa]


def test_missing_mesa_continues_on_the_system_driver(caplog):
    with caplog.at_level(logging.WARNING, logger="bermake"):
        assert apply_compatibility_rendering(True, find_mesa=lambda: None, enable=None) is False
    assert "Mesa was not found" in caplog.text


def test_a_failure_to_enable_is_logged_and_startup_continues(caplog):
    """A stored preference plus a DLL the antivirus blocks must not stop
    every later launch before the Help menu is reachable (final review I3)."""

    def blocked(mesa_dir):
        raise OSError("[WinError 225] Operation did not complete successfully")

    with caplog.at_level(logging.ERROR, logger="bermake"):
        result = apply_compatibility_rendering(
            True, find_mesa=lambda: Path("C:/B/_internal/mesa"), enable=blocked
        )

    assert result is False
    [record] = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert "could not enable compatibility rendering" in record.getMessage()
    assert record.exc_info is not None
    assert "WinError 225" in str(record.exc_info[1])
