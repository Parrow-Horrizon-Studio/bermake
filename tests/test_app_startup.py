"""app.py's startup sequence (M7.9)."""

import subprocess
import sys

from bermake.app import start_window
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
