"""app.py's startup sequence (M7.9)."""

import logging
import subprocess
import sys
from pathlib import Path

import bermake.app as app_module
import pytest
from bermake.app import (
    apply_compatibility_rendering,
    mesa_restart_available,
    start_window,
    wants_compatibility_rendering,
)
from bermake.diagnostics.gl_check import GlInfo
from bermake.diagnostics.gl_preflight import PreflightOutcome, preflight_problem
from bermake.diagnostics.launch import parse_launch_args
from bermake.diagnostics.logs import ErrorReporter
from PySide6.QtWidgets import QApplication


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


# --- M7.9 pre-tag fix: no repeated dialog, no offer loop ---------------------


def test_a_failed_switch_to_compatibility_rendering_offers_no_restart():
    """The decision the startup check is given, end to end."""

    def found():
        return Path("mesa")

    assert mesa_restart_available(compat_failed=False, find_mesa=found) is True
    assert mesa_restart_available(compat_failed=True, find_mesa=found) is False
    assert mesa_restart_available(compat_failed=False, find_mesa=lambda: None) is False

    low = GlInfo((2, 1), "2.1 Mesa", "llvmpipe")
    offered = preflight_problem(
        low,
        mesa_available=mesa_restart_available(compat_failed=False, find_mesa=found),
        compat_active=False,
    )
    not_offered = preflight_problem(
        low,
        mesa_available=mesa_restart_available(compat_failed=True, find_mesa=found),
        compat_active=False,
    )
    assert offered.offer_restart is True
    assert not_offered.offer_restart is False
    assert "OpenGL 2.1" in not_offered.message


class _FakeWindow:
    def __init__(self):
        self.calls = []

    def note_gl_reported_at_startup(self):
        self.calls.append("reported")

    def disable_compatibility_offer(self):
        self.calls.append("no_offer")

    def show(self):
        self.calls.append("show")

    def show_welcome_dialog(self):
        self.calls.append("welcome")


@pytest.fixture
def fake_window(monkeypatch):
    import bermake.ui.main_window as main_window_module

    window = _FakeWindow()
    monkeypatch.setattr(main_window_module, "MainWindow", lambda: window)
    monkeypatch.setattr(app_module.preferences, "read_show_welcome", lambda settings: False)
    return window


def test_the_window_is_told_before_it_is_shown(fake_window):
    """initializeGL runs on show, so the flags must already be set."""
    app_module._build_main_window(gl_reported=True, compat_failed=True)
    assert fake_window.calls == ["reported", "no_offer", "show"]


def test_a_plain_start_tells_the_window_nothing(fake_window):
    app_module._build_main_window()
    assert fake_window.calls == ["show"]


def _built_window(qtbot, monkeypatch, *, mesa, **flags):
    """The real MainWindow through the real _build_main_window."""
    import bermake.ui.main_window as main_window_module

    monkeypatch.setattr(app_module.preferences, "read_show_welcome", lambda settings: False)
    monkeypatch.setattr(main_window_module, "compatibility_rendering_active", lambda: False)
    monkeypatch.setattr(
        main_window_module, "default_mesa_dir", lambda: Path("mesa") if mesa else None
    )
    # Nothing here may open a modal if the offscreen platform reports no GL.
    monkeypatch.setattr(
        main_window_module.MainWindow, "_prompt_gl_fallback", lambda self, message, offer: False
    )
    window = app_module._build_main_window(**flags)
    qtbot.addWidget(window)
    return window


def _note_of(window):
    from PySide6.QtWidgets import QLabel

    return window._viewport.findChild(QLabel, "GlUnavailableNote")


def test_a_reported_start_shows_the_restart_note_over_the_view(qtbot, monkeypatch):
    window = _built_window(qtbot, monkeypatch, mesa=True, gl_reported=True)
    assert not _note_of(window).isHidden()
    assert "Help > Use Compatibility Rendering and restart Bermake" in _note_of(window).text()


def test_a_reported_start_without_a_restart_shows_the_details_note(qtbot, monkeypatch):
    window = _built_window(qtbot, monkeypatch, mesa=True, gl_reported=True, compat_failed=True)
    assert "Help > About Bermake has details" in _note_of(window).text()

    window = _built_window(qtbot, monkeypatch, mesa=False, gl_reported=True)
    assert "Help > About Bermake has details" in _note_of(window).text()


def test_a_clean_start_shows_no_note(qtbot, monkeypatch):
    window = _built_window(qtbot, monkeypatch, mesa=True)
    assert _note_of(window).isHidden()


class _FakeQApplication:
    """Stands in for QApplication while _run is driven. The real application
    already exists in the suite, and pytest-qt still calls instance() on
    whatever the module attribute is at teardown."""

    def __init__(self, argv):
        pass

    instance = staticmethod(QApplication.instance)

    def exec(self):
        return 0


@pytest.fixture
def run_harness(monkeypatch):
    """_run with everything but the wiring stubbed out."""
    import PySide6.QtWidgets

    built = []
    monkeypatch.setattr(PySide6.QtWidgets, "QApplication", _FakeQApplication)
    monkeypatch.setattr(app_module.logs, "install_qt_message_handler", lambda: None)
    monkeypatch.setattr(app_module.logs, "install_exception_hooks", lambda reporter: None)
    monkeypatch.setattr(app_module, "apply_theme", lambda app, theme: None)
    monkeypatch.setattr(app_module.preferences, "read_compatibility_rendering", lambda s: False)
    monkeypatch.setattr(app_module.preferences, "read_theme", lambda s: "dark")
    monkeypatch.setattr(app_module, "default_mesa_dir", lambda: Path("mesa"))
    monkeypatch.setattr(
        app_module, "_build_main_window", lambda **flags: built.append(flags) or object()
    )
    return built


def _run_with(monkeypatch, tmp_path, *, requested, enabled, outcome):
    seen = {}
    monkeypatch.setattr(
        app_module, "apply_compatibility_rendering", lambda wanted: enabled if wanted else False
    )

    def preflight(**kwargs):
        seen.update(kwargs)
        return outcome

    monkeypatch.setattr(app_module, "run_gl_preflight", preflight)
    argv = ["bermake", "--compatibility-rendering"] if requested else ["bermake"]
    code = app_module._run(parse_launch_args(argv), tmp_path)
    return code, seen


def test_run_passes_a_reported_problem_on_to_the_window(run_harness, monkeypatch, tmp_path):
    code, _ = _run_with(
        monkeypatch, tmp_path, requested=False, enabled=False, outcome=PreflightOutcome.REPORTED
    )
    assert code == 0
    assert run_harness == [{"gl_reported": True, "compat_failed": False}]


def test_run_tells_the_window_nothing_when_the_check_was_clean(run_harness, monkeypatch, tmp_path):
    _run_with(monkeypatch, tmp_path, requested=False, enabled=False, outcome=PreflightOutcome.OK)
    assert run_harness == [{"gl_reported": False, "compat_failed": False}]


def test_run_stops_without_a_window_after_a_relaunch(run_harness, monkeypatch, tmp_path):
    code, _ = _run_with(
        monkeypatch, tmp_path, requested=False, enabled=False, outcome=PreflightOutcome.RELAUNCHED
    )
    assert code == 0
    assert run_harness == []


def test_run_withdraws_the_offer_when_the_requested_switch_failed(
    run_harness, monkeypatch, tmp_path
):
    _, seen = _run_with(
        monkeypatch, tmp_path, requested=True, enabled=False, outcome=PreflightOutcome.REPORTED
    )
    assert seen["mesa_available"] is False
    assert run_harness == [{"gl_reported": True, "compat_failed": True}]


def test_run_keeps_the_offer_when_nothing_was_requested_or_the_switch_worked(
    run_harness, monkeypatch, tmp_path
):
    _, seen = _run_with(
        monkeypatch, tmp_path, requested=False, enabled=False, outcome=PreflightOutcome.OK
    )
    assert seen["mesa_available"] is True
    _, seen = _run_with(
        monkeypatch, tmp_path, requested=True, enabled=True, outcome=PreflightOutcome.OK
    )
    assert seen["mesa_available"] is True
    assert run_harness[-1] == {"gl_reported": False, "compat_failed": False}


def test_run_offers_nothing_when_mesa_is_not_there(run_harness, monkeypatch, tmp_path):
    """Machine-independent: the lookup is patched, not read from build/mesa."""
    monkeypatch.setattr(app_module, "default_mesa_dir", lambda: None)
    _, seen = _run_with(
        monkeypatch, tmp_path, requested=False, enabled=False, outcome=PreflightOutcome.OK
    )
    assert seen["mesa_available"] is False


def test_the_mesa_lookup_is_resolved_when_called(monkeypatch):
    monkeypatch.setattr(app_module, "default_mesa_dir", lambda: Path("patched"))
    assert mesa_restart_available(compat_failed=False) is True
    monkeypatch.setattr(app_module, "default_mesa_dir", lambda: None)
    assert mesa_restart_available(compat_failed=False) is False


def test_apply_compatibility_rendering_resolves_the_mesa_lookup_at_call_time(monkeypatch):
    """A default argument would freeze default_mesa_dir at definition time and
    make this patch do nothing."""
    enabled = []
    monkeypatch.setattr(app_module, "default_mesa_dir", lambda: Path("patched"))
    assert apply_compatibility_rendering(True, enable=enabled.append) is True
    assert enabled == [Path("patched")]

    monkeypatch.setattr(app_module, "default_mesa_dir", lambda: None)
    assert apply_compatibility_rendering(True, enable=enabled.append) is False
    assert enabled == [Path("patched")]


def test_the_smoke_flag_runs_the_smoke_path_and_touches_neither_logs_nor_preferences(
    monkeypatch, tmp_path
):
    """main() hands --smoke-test to run_smoke before logging is configured.

    run_smoke is replaced by a recorder (the real run builds a QApplication and
    compiles shaders; tests/test_smoke.py covers it). The log folder points at
    a tmp path that must stay missing, and any QSettings construction or
    logging setup fails the test."""
    import bermake.diagnostics.smoke as smoke

    log_dir = tmp_path / "logs"
    report = tmp_path / "report.json"
    smoke_calls = []

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the smoke path must not reach this")

    monkeypatch.setattr(smoke, "run_smoke", lambda path: smoke_calls.append(path) or 7)
    monkeypatch.setattr(app_module.logs, "default_log_directory", lambda: log_dir)
    monkeypatch.setattr(app_module.logs, "configure_file_logging", forbidden)
    monkeypatch.setattr(app_module.logs, "enable_native_crash_log", forbidden)
    monkeypatch.setattr(app_module.logs, "install_qt_message_handler", forbidden)
    monkeypatch.setattr(app_module.logs, "install_exception_hooks", forbidden)
    monkeypatch.setattr(app_module, "QSettings", forbidden)

    assert app_module.main(["Bermake.exe", "--smoke-test", str(report)]) == 7

    assert smoke_calls == [report]
    assert not log_dir.exists()
