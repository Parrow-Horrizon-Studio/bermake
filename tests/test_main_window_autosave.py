"""Autosave in the window (M7.12, #77; spec 4.2 to 4.4).

The window's scheduler is swapped for one on a fake clock, and the last input
is pushed far into the past, so each test decides exactly when an autosave is
due and whether the user is idle. The recovery folder is the per-test one the
conftest autouse fixture installs.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import bermake
import bermake.ui.main_window as main_window_module
import numpy as np
import pytest
from bermake.commands.scene_commands import AddVertexCommand
from bermake.document import DocumentSettings
from bermake.io import BermakeFormatError, load_document, save_document
from bermake.io.document_codec import CameraState
from bermake.model import Model
from bermake.recovery.liveness import current_process_started_at
from bermake.recovery.scheduler import AutosaveScheduler
from bermake.recovery.store import SessionMeta
from bermake.ui import preferences
from bermake.ui.main_window import MainWindow
from bermake.viewport.camera import Camera
from bermake.viewport.render_style import RenderStyle
from PySide6.QtCore import QEvent, QPointF, QSettings, QStandardPaths, Qt
from PySide6.QtGui import QCloseEvent, QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

INTERVAL = 5
DUE = INTERVAL * 60 + 1

AUTOSAVE_ENTRIES = [
    ("file_autosave_off", 0),
    ("file_autosave_1", 1),
    ("file_autosave_5", 5),
    ("file_autosave_10", 10),
    ("file_autosave_30", 30),
]


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def win(main_window, clock):
    """A window whose autosave the test drives tick by tick."""
    main_window._autosave_timer.stop()
    main_window._autosave = AutosaveScheduler(INTERVAL, clock=clock)
    main_window._last_input = time.monotonic() - 3600.0
    return main_window


def _change(window, x: float = 1.0) -> None:
    window._command_stack.execute(
        AddVertexCommand(np.array([x, 2.0, 3.0])), window._model.active_scene
    )


def _make_due(window, clock, x: float = 1.0) -> None:
    _change(window, x)
    clock.advance(DUE)


def _session_files(window) -> tuple[Path, Path]:
    folder = window._recovery_store.folder
    return folder / f"{window._session_id}.berm", folder / f"{window._session_id}.json"


def _exist(paths) -> list[bool]:
    return [p.exists() for p in paths]


def _meta(window) -> dict:
    return json.loads(_session_files(window)[1].read_text(encoding="utf-8"))


def _autosave_once(window, clock) -> tuple[Path, Path]:
    _make_due(window, clock)
    window._autosave_tick()
    files = _session_files(window)
    assert _exist(files) == [True, True]
    return files


def _vertex_count(model) -> int:
    return len(list(model.root.mesh.vertices_iter()))


# --- The conftest guard -------------------------------------------------


def test_window_writes_only_under_the_patched_recovery_folder(win, clock, tmp_path):
    """The controller ruling: MainWindow resolves the folder through
    bermake.recovery.paths, so the conftest patch of that one attribute
    covers every MainWindow built anywhere in the suite."""
    patched = tmp_path / "recovery"
    assert win._recovery_store.folder == patched

    berm, meta = _autosave_once(win, clock)

    assert berm.parent == patched and meta.parent == patched
    assert sorted(p.name for p in patched.iterdir()) == sorted([berm.name, meta.name])
    real = (
        Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation))
        / "recovery"
    )
    assert real != patched
    assert not (real / berm.name).exists()
    assert not (real / meta.name).exists()


def test_a_directly_built_window_uses_the_patched_folder(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    assert window._recovery_store.folder == tmp_path / "recovery"


# --- Session state --------------------------------------------------------


def test_a_new_window_has_a_fresh_session(qtbot, main_window):
    other = MainWindow()
    qtbot.addWidget(other)
    assert len(main_window._session_id) == 32
    int(main_window._session_id, 16)  # a uuid4 hex string
    assert main_window._session_id != other._session_id
    assert main_window._suggested_save_path is None


def test_the_autosave_timer_ticks_every_second(main_window):
    timer = main_window._autosave_timer
    assert timer.interval() == 1000
    assert timer.isActive()
    assert timer.parent() is main_window


def test_the_running_timer_drives_a_due_autosave(qtbot, main_window, clock):
    """End to end through the real QTimer, not a direct _autosave_tick call."""
    main_window._autosave = AutosaveScheduler(INTERVAL, clock=clock)
    main_window._last_input = time.monotonic() - 3600.0
    _make_due(main_window, clock)
    berm, meta = _session_files(main_window)
    qtbot.waitUntil(lambda: berm.exists() and meta.exists(), timeout=5000)


@pytest.mark.parametrize(
    "make_event",
    [
        lambda: QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier),
        lambda: QMouseEvent(
            QEvent.Type.MouseMove,
            QPointF(5, 5),
            QPointF(5, 5),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        ),
    ],
    ids=["key", "mouse-move"],
)
def test_input_events_anywhere_reset_the_idle_clock(main_window, make_event):
    main_window._last_input = 0.0
    QApplication.sendEvent(main_window, make_event())
    assert main_window._last_input > 0.0


def test_other_events_do_not_reset_the_idle_clock(main_window):
    main_window._last_input = 0.0
    QApplication.sendEvent(main_window, QEvent(QEvent.Type.User))
    assert main_window._last_input == 0.0


# --- No-op cases ----------------------------------------------------------


def test_an_unchanged_document_is_never_written(win, clock):
    clock.advance(DUE * 3)
    win._autosave_tick()
    assert _exist(_session_files(win)) == [False, False]
    assert not win._recovery_store.folder.exists()


def test_interval_off_writes_nothing(win, clock):
    win._set_autosave_interval(0)
    _make_due(win, clock)
    clock.advance(3600)
    win._autosave_tick()
    assert _exist(_session_files(win)) == [False, False]


def test_a_change_before_the_interval_is_not_written_yet(win, clock):
    _change(win)
    clock.advance(INTERVAL * 60 - 1)
    win._autosave_tick()
    assert _exist(_session_files(win)) == [False, False]


# --- A due, idle tick writes the session --------------------------------


def test_a_due_idle_tick_writes_the_session(win, clock):
    _make_due(win, clock)
    win._autosave_tick()

    berm, meta_path = _session_files(win)
    assert _exist((berm, meta_path)) == [True, True]

    meta = _meta(win)
    assert meta["format"] == "bermake-recovery"
    assert meta["session_id"] == win._session_id
    assert meta["display_name"] == "Untitled"
    assert meta["original_path"] is None
    assert meta["app_version"] == bermake.__version__
    assert meta["pid"] == os.getpid()
    assert meta["process_started_at"] == current_process_started_at()
    assert datetime.fromisoformat(meta["saved_at"]).tzinfo is not None

    loaded = load_document(berm)
    assert _vertex_count(loaded.model) == 1
    assert "Autosaved" in win._status_bar.prompt_text()
    assert win._autosave.changed_since_autosave is False


def test_a_saved_document_records_its_path_and_name(win, clock, tmp_path):
    assert win._save_to(tmp_path / "Plan.berm")
    _autosave_once(win, clock)
    meta = _meta(win)
    assert Path(meta["original_path"]) == tmp_path / "Plan.berm"
    assert meta["display_name"] == "Plan.berm"


def test_a_written_session_is_not_rewritten_without_a_new_change(win, clock):
    berm, _ = _autosave_once(win, clock)
    berm.unlink()  # so a rewrite would show
    clock.advance(DUE * 2)
    win._autosave_tick()
    assert not berm.exists()


def test_autosave_now_returns_true_on_success(win):
    _change(win)
    assert win._autosave_now() is True
    assert _exist(_session_files(win)) == [True, True]


def test_the_autosaved_message_clears_after_a_few_seconds(win, clock):
    _autosave_once(win, clock)
    timer = win._autosave_message_timer
    assert timer.isSingleShot()
    assert timer.isActive()
    assert 2000 <= timer.interval() <= 5000
    win._clear_autosaved_message()
    assert "Autosaved" not in win._status_bar.prompt_text()


def test_clearing_the_autosaved_message_keeps_a_newer_one(win, clock):
    _autosave_once(win, clock)
    win._status_bar.set_message("Saved House.berm")
    win._clear_autosaved_message()
    assert "Saved House.berm" in win._status_bar.prompt_text()


# --- It waits for a safe moment ------------------------------------------


def test_a_modal_dialog_blocks_the_autosave(win, clock, monkeypatch):
    _make_due(win, clock)
    modal = QWidget()
    monkeypatch.setattr(QApplication, "activeModalWidget", lambda: modal)
    clock.advance(600)  # long past forced: a modal still blocks it
    win._autosave_tick()
    assert _exist(_session_files(win)) == [False, False]

    monkeypatch.setattr(QApplication, "activeModalWidget", lambda: None)
    win._autosave_tick()
    assert _exist(_session_files(win)) == [True, True]


def test_a_held_mouse_button_blocks_the_autosave(win, clock, monkeypatch):
    _make_due(win, clock)
    monkeypatch.setattr(QApplication, "mouseButtons", lambda: Qt.MouseButton.LeftButton)
    clock.advance(600)
    win._autosave_tick()
    assert _exist(_session_files(win)) == [False, False]

    monkeypatch.setattr(QApplication, "mouseButtons", lambda: Qt.MouseButton.NoButton)
    win._autosave_tick()
    assert _exist(_session_files(win)) == [True, True]


def test_a_tool_holding_uncommitted_changes_blocks_the_autosave(win, clock, monkeypatch):
    win._activate("eraser")
    tool_class = type(win._tool_manager.active)
    monkeypatch.setattr(tool_class, "holds_uncommitted_changes", property(lambda self: True))
    _make_due(win, clock)
    clock.advance(600)
    win._autosave_tick()
    assert _exist(_session_files(win)) == [False, False]

    monkeypatch.setattr(tool_class, "holds_uncommitted_changes", property(lambda self: False))
    win._autosave_tick()
    assert _exist(_session_files(win)) == [True, True]


def test_recent_input_delays_the_autosave_until_forced(win, clock):
    _make_due(win, clock)
    win._last_input = time.monotonic()  # the user is typing right now
    win._autosave_tick()
    assert _exist(_session_files(win)) == [False, False]

    clock.advance(60)  # due for 60 s without a pause: forced
    win._last_input = time.monotonic()
    win._autosave_tick()
    assert _exist(_session_files(win)) == [True, True]


def test_a_swapped_qapplication_module_attribute_does_not_break_the_tick(win, clock, monkeypatch):
    """tests/test_app_startup.py swaps PySide6.QtWidgets.QApplication for a fake
    while windows from earlier tests are still ticking."""
    import PySide6.QtWidgets

    class FakeQApplication:  # pytest-qt still calls instance() at teardown
        instance = staticmethod(QApplication.instance)

    monkeypatch.setattr(PySide6.QtWidgets, "QApplication", FakeQApplication)
    _make_due(win, clock)
    win._autosave_tick()
    assert _exist(_session_files(win)) == [True, True]


def test_a_failing_check_never_escapes_the_tick(win, clock, monkeypatch, caplog):
    def broken():
        raise RuntimeError("no screen")

    monkeypatch.setattr(QApplication, "mouseButtons", broken)
    _make_due(win, clock)
    with caplog.at_level(logging.ERROR, logger=main_window_module.__name__):
        win._autosave_tick()  # nothing propagates
    assert any(r.exc_info is not None for r in caplog.records)
    assert _exist(_session_files(win)) == [False, False]


def test_a_stuck_tick_logs_its_error_once_until_a_tick_succeeds(win, clock, monkeypatch, caplog):
    """Fix round 1, M2: a fault that repeats every second must not flood the
    log with a traceback a second; it is logged again only after a tick has
    worked in between."""

    def broken():
        raise RuntimeError("no screen")

    def errors():
        return [r for r in caplog.records if r.levelno >= logging.ERROR and r.exc_info]

    with caplog.at_level(logging.DEBUG, logger=main_window_module.__name__):
        monkeypatch.setattr(QApplication, "mouseButtons", broken)
        for _ in range(5):
            win._autosave_tick()
        assert len(errors()) == 1

        monkeypatch.setattr(QApplication, "mouseButtons", lambda: Qt.MouseButton.NoButton)
        win._autosave_tick()  # a tick that works clears the latch
        assert len(errors()) == 1

        monkeypatch.setattr(QApplication, "mouseButtons", broken)
        win._autosave_tick()
        win._autosave_tick()
        assert len(errors()) == 2


# --- Failure --------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [OSError(28, "No space left on device"), BermakeFormatError("cannot encode")],
    ids=["OSError", "BermakeIOError"],
)
def test_a_failed_autosave_warns_and_retries(win, clock, monkeypatch, caplog, error):
    _make_due(win, clock)

    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(main_window_module, "save_document", fail)
    with caplog.at_level(logging.WARNING, logger=main_window_module.__name__):
        win._autosave_tick()  # nothing propagates

    assert "Autosave failed; will retry" in win._status_bar.prompt_text()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings and warnings[0].exc_info is not None
    assert win._autosave.changed_since_autosave is True
    assert _exist(_session_files(win)) == [False, False]

    # Retried one interval later, not on the next tick.
    monkeypatch.setattr(main_window_module, "save_document", save_document)
    win._autosave_tick()
    assert _exist(_session_files(win)) == [False, False]
    clock.advance(DUE)
    win._autosave_tick()
    assert _exist(_session_files(win)) == [True, True]


def test_autosave_now_returns_false_on_failure(win, monkeypatch):
    _change(win)

    def fail(*_args, **_kwargs):
        raise OSError("locked")

    monkeypatch.setattr(main_window_module, "save_document", fail)
    assert win._autosave_now() is False


def test_an_unexpected_error_never_escapes_an_autosave(win, monkeypatch, caplog):
    """`_autosave_now` never raises: a bug in the writer must not reach the
    event loop once a second."""
    _change(win)

    def fail(*_args, **_kwargs):
        raise ValueError("a bug")

    monkeypatch.setattr(main_window_module, "save_document", fail)
    with caplog.at_level(logging.WARNING, logger=main_window_module.__name__):
        assert win._autosave_now() is False
    assert "Autosave failed; will retry" in win._status_bar.prompt_text()
    assert any(r.exc_info is not None for r in caplog.records)
    assert win._autosave.changed_since_autosave is True


def test_an_unwritable_recovery_folder_is_a_failed_autosave(win, clock, tmp_path):
    """Spec 5: a folder that cannot be created behaves as a failed write."""
    blocker = tmp_path / "not-a-folder"
    blocker.write_text("a file where the folder should be", encoding="utf-8")
    win._recovery_store.folder = blocker / "recovery"
    _make_due(win, clock)
    win._autosave_tick()
    assert "Autosave failed; will retry" in win._status_bar.prompt_text()
    assert win._autosave.changed_since_autosave is True


# --- Each cleanup trigger deletes the session's files ---------------------


def test_a_successful_save_deletes_the_session_files(win, clock, tmp_path):
    files = _autosave_once(win, clock)
    _change(win, 2.0)
    assert win._save_to(tmp_path / "House.berm") is True
    assert _exist(files) == [False, False]
    assert win._autosave.changed_since_autosave is False


def test_an_accepted_close_deletes_the_session_files(win, clock, monkeypatch):
    files = _autosave_once(win, clock)
    monkeypatch.setattr(win, "_confirm_discard_if_dirty", lambda: True)
    event = QCloseEvent()
    win.closeEvent(event)
    assert event.isAccepted()
    assert _exist(files) == [False, False]


def test_the_suite_stops_every_live_windows_autosave_timer(qtbot):
    """Fix round 1, M1: a window a test never closes (a bare MainWindow(),
    kept alive by its own signal cycles until the garbage collector runs) must
    not tick, and possibly autosave, during a later test. The conftest
    teardown stops the timer of every live MainWindow; this drives that
    helper directly and checks the autouse fixture that runs it is active."""
    from tests import conftest

    leftover = MainWindow()  # deliberately not given to qtbot, like test_main_window_fileio.py
    try:
        assert leftover._autosave_timer.isActive()
        conftest.stop_autosave_timers()
        assert not leftover._autosave_timer.isActive()
    finally:
        leftover.deleteLater()


def test_the_timer_stopping_fixture_is_autouse_and_stops_leftovers_at_teardown(qtbot, request):
    """Drives the fixture's own generator (through pytest's fixture manager, a
    private API: an upgrade that changes it fails here loudly, not silently)."""
    name = "_stop_leftover_autosave_timers"
    assert name in request.fixturenames
    (fixturedef,) = request._fixturemanager.getfixturedefs(name, request.node)
    leftover = MainWindow()  # not given to qtbot
    try:
        steps = fixturedef.func()
        next(steps)  # a test body would run here
        assert leftover._autosave_timer.isActive()
        with pytest.raises(StopIteration):
            next(steps)  # teardown
        assert not leftover._autosave_timer.isActive()
    finally:
        leftover.deleteLater()


def test_an_accepted_close_stops_the_timer(main_window, monkeypatch):
    monkeypatch.setattr(main_window, "_confirm_discard_if_dirty", lambda: True)
    main_window.closeEvent(QCloseEvent())
    assert not main_window._autosave_timer.isActive()


def test_file_new_deletes_the_session_files(win, clock):
    files = _autosave_once(win, clock)
    old_id = win._session_id
    win._prompt_discard = lambda: "discard"
    win._on_file_new()
    assert _exist(files) == [False, False]
    assert win._session_id != old_id


def test_file_open_deletes_the_session_files(win, clock, monkeypatch, tmp_path):
    other = tmp_path / "Other.berm"
    save_document(other, Model(), Camera(), DocumentSettings(), RenderStyle())
    loaded = load_document(other)
    files = _autosave_once(win, clock)
    old_id = win._session_id

    win._prompt_discard = lambda: "discard"
    monkeypatch.setattr(win, "_prompt_open_path", lambda: str(other))
    monkeypatch.setattr(main_window_module, "load_document", lambda _path: loaded)
    win._on_file_open()

    assert win._doc_controller.current_path == other
    assert _exist(files) == [False, False]
    assert win._session_id != old_id


# --- Keeping files ----------------------------------------------------------


def test_a_failed_save_keeps_the_session_files(win, clock, monkeypatch, tmp_path):
    files = _autosave_once(win, clock)

    def fail(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(main_window_module, "save_document", fail)
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: None)
    assert win._save_to(tmp_path / "House.berm") is False
    assert _exist(files) == [True, True]


def test_a_cancelled_close_keeps_the_session_files(win, clock, monkeypatch):
    files = _autosave_once(win, clock)
    monkeypatch.setattr(win, "_confirm_discard_if_dirty", lambda: False)
    event = QCloseEvent()
    win.closeEvent(event)
    assert not event.isAccepted()
    assert _exist(files) == [True, True]


def test_a_cancelled_new_keeps_the_session_files(win, clock):
    files = _autosave_once(win, clock)
    _change(win, 2.0)
    win._prompt_discard = lambda: "cancel"
    win._on_file_new()
    assert _exist(files) == [True, True]


def test_a_locked_recovery_file_does_not_fail_the_save(win, clock, monkeypatch, tmp_path, caplog):
    _autosave_once(win, clock)

    def locked(_session_id):
        raise PermissionError("in use by another process")

    monkeypatch.setattr(win._recovery_store, "delete", locked)
    with caplog.at_level(logging.WARNING, logger=main_window_module.__name__):
        assert win._save_to(tmp_path / "House.berm") is True
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_a_locked_recovery_file_does_not_block_the_close(win, clock, monkeypatch):
    _autosave_once(win, clock)

    def locked(_session_id):
        raise PermissionError("in use by another process")

    monkeypatch.setattr(win._recovery_store, "delete", locked)
    monkeypatch.setattr(win, "_confirm_discard_if_dirty", lambda: True)
    event = QCloseEvent()
    win.closeEvent(event)
    assert event.isAccepted()


# --- Interval menu ----------------------------------------------------------


@pytest.mark.parametrize(("action_id", "minutes"), AUTOSAVE_ENTRIES)
def test_an_autosave_menu_entry_sets_the_preference_and_interval(win, action_id, minutes):
    win._actions[action_id].trigger()
    assert preferences.read_autosave_interval(win._settings) == minutes
    assert win._autosave.interval_minutes == minutes
    assert win._actions[action_id].isChecked()


def test_changing_the_interval_restarts_the_countdown(win, clock):
    _change(win)
    clock.advance(DUE - 10)
    win._actions["file_autosave_5"].trigger()
    clock.advance(20)  # past the first due point, but not one interval since the change
    win._autosave_tick()
    assert _exist(_session_files(win)) == [False, False]


def _window_with_stored_interval(qtbot, monkeypatch, tmp_path, stored):
    ini = str(tmp_path / "stored-interval.ini")
    store = QSettings(ini, QSettings.Format.IniFormat)
    store.setValue(preferences.AUTOSAVE_INTERVAL_KEY, stored)
    store.sync()
    monkeypatch.setattr(
        main_window_module,
        "QSettings",
        lambda *_a, **_k: QSettings(ini, QSettings.Format.IniFormat),
    )
    window = MainWindow()
    qtbot.addWidget(window)
    return window, ini


def _checked_autosave_entries(window) -> list[str]:
    return [i for i, _ in AUTOSAVE_ENTRIES if window._actions[i].isChecked()]


def test_the_startup_checkmark_matches_a_stored_ten(qtbot, monkeypatch, tmp_path):
    window, _ = _window_with_stored_interval(qtbot, monkeypatch, tmp_path, 10)
    assert _checked_autosave_entries(window) == ["file_autosave_10"]
    assert window._autosave.interval_minutes == 10


def test_an_invalid_stored_interval_shows_five(qtbot, monkeypatch, tmp_path):
    window, ini = _window_with_stored_interval(qtbot, monkeypatch, tmp_path, "7")
    assert _checked_autosave_entries(window) == ["file_autosave_5"]
    assert window._autosave.interval_minutes == 5
    # Restoring the checkmark wrote nothing back.
    stored = QSettings(ini, QSettings.Format.IniFormat)
    assert stored.value(preferences.AUTOSAVE_INTERVAL_KEY) == "7"


def test_with_nothing_stored_the_default_five_is_checked(main_window):
    assert _checked_autosave_entries(main_window) == ["file_autosave_5"]
    assert main_window._autosave.interval_minutes == 5


# --- Review Focus 5 -----------------------------------------------------------


def test_an_untitled_drawing_saved_under_a_name_autosaves_under_that_name(win, clock, tmp_path):
    berm, meta_path = _autosave_once(win, clock)
    assert _meta(win)["display_name"] == "Untitled"

    assert win._save_to(tmp_path / "House.berm") is True
    assert _exist((berm, meta_path)) == [False, False]

    _make_due(win, clock, 4.0)
    win._autosave_tick()

    meta = _meta(win)
    assert meta["original_path"].endswith("House.berm")
    assert Path(meta["original_path"]) == tmp_path / "House.berm"
    assert meta["display_name"] == "House.berm"
    assert _vertex_count(load_document(_session_files(win)[0]).model) == 2


# --- Sessions -----------------------------------------------------------------


def _reset(window, **kwargs) -> None:
    window._reset_document(
        Model(),
        CameraState.from_camera(Camera()),
        window._doc.units,
        RenderStyle(),
        None,
        environment=window._doc.environment,
        **kwargs,
    )


def test_reset_document_starts_a_new_session(win, clock):
    files = _autosave_once(win, clock)
    old_id = win._session_id
    _change(win, 2.0)
    _reset(win)
    assert win._session_id != old_id
    assert len(win._session_id) == 32
    assert _exist(files) == [False, False]
    # The new document has nothing to protect yet.
    assert win._autosave.changed_since_autosave is False


def _write_recovered_session(window, session_id: str) -> tuple[Path, Path]:
    """A session left by another (crashed) Bermake, as Task 6's Recover finds it."""
    meta = SessionMeta(
        session_id=session_id,
        original_path="C:/drawings/House.berm",
        display_name="House.berm",
        saved_at="2026-10-04T14:32:05+08:00",
        app_version=bermake.__version__,
        pid=999999,
        process_started_at="2026-10-04T13:58:41+08:00",
    )
    store = window._recovery_store
    store.write(
        meta,
        lambda target: save_document(target, Model(), Camera(), DocumentSettings(), RenderStyle()),
    )
    folder = store.folder
    return folder / f"{session_id}.berm", folder / f"{session_id}.json"


def test_reset_document_handing_over_deletes_only_the_windows_own_files(win, clock):
    """Controller ruling (fix round 1, I1): Recover replaces the window's
    document like New or Open, so the window's own outgoing session goes; only
    the recovered session's files, named by hand_over_from, are left alone for
    the caller to hand over."""
    own = _autosave_once(win, clock)
    own_id = win._session_id
    recovered_id = "f" * 32
    recovered = _write_recovered_session(win, recovered_id)

    _reset(win, hand_over_from=recovered_id)

    assert _exist(own) == [False, False]
    assert _exist(recovered) == [True, True]
    assert win._session_id not in (own_id, recovered_id)
    assert len(win._session_id) == 32


def test_reset_document_forgets_a_suggested_save_path(win):
    win._suggested_save_path = "C:/somewhere/House.berm"
    _reset(win)
    assert win._suggested_save_path is None


def test_a_suggested_save_path_names_an_untitled_autosave(win, clock):
    """A recovered document has no path but remembers where it came from
    (Task 6). Its next autosave must keep that name, or a second crash would
    recover it as Untitled."""
    win._suggested_save_path = "C:/drawings/House.berm"
    _autosave_once(win, clock)
    meta = _meta(win)
    assert meta["original_path"] == "C:/drawings/House.berm"
    assert meta["display_name"] == "House.berm"
