"""app.py's startup sequence (M7.9)."""

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
