"""The compatibility rendering switch in MainWindow (M7.9, spec 2.4).

conftest.py redirects MainWindow's QSettings() to tmp_path/window_state.ini,
so _settings() below opens the same store the window reads and writes.
"""

import logging

import bermake.ui.main_window as main_window_module
import pytest
from bermake.commands.scene_commands import ClearSceneCommand
from bermake.ui import preferences
from bermake.ui.main_window import MainWindow
from PySide6.QtCore import QSettings

ACTION = "help_compatibility_rendering"


def _settings(tmp_path):
    return QSettings(str(tmp_path / "window_state.ini"), QSettings.Format.IniFormat)


@pytest.fixture
def relaunches(main_window):
    calls = []
    main_window._relaunch = lambda: calls.append(True) or True
    return calls


def test_restoring_the_checkmark_at_startup_does_not_offer_a_restart(tmp_path, qtbot, monkeypatch):
    store = _settings(tmp_path)
    preferences.write_compatibility_rendering(store, True)
    store.sync()
    prompts = []
    monkeypatch.setattr(
        MainWindow,
        "_prompt_restart_for_rendering",
        lambda self, enabled: prompts.append(enabled) or False,
    )

    window = MainWindow()
    qtbot.addWidget(window)

    assert window._actions[ACTION].isChecked()
    assert prompts == []


def test_toggling_stores_the_preference_and_offers_a_restart(main_window, tmp_path, relaunches):
    prompts = []
    main_window._prompt_restart_for_rendering = lambda enabled: prompts.append(enabled) or False

    main_window._actions[ACTION].setChecked(True)

    assert prompts == [True]
    assert relaunches == []
    assert preferences.read_compatibility_rendering(_settings(tmp_path)) is True


def test_accepting_the_restart_closes_then_relaunches(main_window, relaunches):
    main_window._prompt_restart_for_rendering = lambda enabled: True
    main_window.show()

    main_window._actions[ACTION].setChecked(True)

    assert relaunches == [True]
    assert not main_window.isVisible()


def test_cancelling_the_unsaved_prompt_cancels_the_relaunch(main_window, relaunches):
    main_window._command_stack.execute(ClearSceneCommand(), main_window._model.active_scene)
    main_window._prompt_discard = lambda: "cancel"
    main_window._prompt_restart_for_rendering = lambda enabled: True
    main_window.show()

    main_window._actions[ACTION].setChecked(True)

    assert relaunches == []
    assert main_window.isVisible()


@pytest.mark.parametrize(
    ("mesa_available", "already_active", "offered"),
    [(True, False, True), (False, False, False), (True, True, False)],
)
def test_the_gl_failure_offers_the_switch_only_when_it_can_help(
    main_window, monkeypatch, mesa_available, already_active, offered
):
    main_window._mesa_available = mesa_available
    monkeypatch.setattr(
        main_window_module, "compatibility_rendering_active", lambda: already_active
    )
    seen = []
    main_window._prompt_gl_fallback = lambda message, offer: seen.append((message, offer)) or False

    main_window._on_gl_unavailable("no usable OpenGL")

    assert seen == [("no usable OpenGL", offered)]


def test_noting_the_startup_report_marks_the_viewport(main_window):
    assert main_window._viewport.version_failure_reported is False

    main_window.note_gl_reported_at_startup()

    assert main_window._viewport.version_failure_reported is True


def test_a_failed_switch_at_startup_removes_the_restart_offer(main_window, monkeypatch):
    """Otherwise the restart relaunches into the same failed switch and the
    same offer, every time (M7.9 pre-tag fix)."""
    main_window._mesa_available = True
    monkeypatch.setattr(main_window_module, "compatibility_rendering_active", lambda: False)
    seen = []
    main_window._prompt_gl_fallback = lambda message, offer: seen.append((message, offer)) or False

    main_window.disable_compatibility_offer()
    main_window._on_gl_unavailable("no usable OpenGL")

    assert seen == [("no usable OpenGL", False)]


def test_accepting_the_gl_fallback_stores_it_and_relaunches(
    main_window, monkeypatch, tmp_path, relaunches
):
    main_window._mesa_available = True
    monkeypatch.setattr(main_window_module, "compatibility_rendering_active", lambda: False)
    main_window._prompt_gl_fallback = lambda message, offer: True
    prompts = []
    main_window._prompt_restart_for_rendering = lambda enabled: prompts.append(enabled) or False
    main_window.show()

    main_window._on_gl_unavailable("no usable OpenGL")

    assert relaunches == [True]
    assert prompts == []
    assert main_window._actions[ACTION].isChecked()
    assert preferences.read_compatibility_rendering(_settings(tmp_path)) is True


def test_a_modal_dialog_defers_the_gl_fallback_prompt(main_window, monkeypatch, qtbot):
    """The Welcome dialog's startup nested event loop must not see this.

    While a modal is active, _on_gl_unavailable re-posts itself instead of
    prompting; once the modal is gone it proceeds normally.
    """
    main_window._mesa_available = True
    monkeypatch.setattr(main_window_module, "compatibility_rendering_active", lambda: False)
    seen = []
    main_window._prompt_gl_fallback = lambda message, offer: seen.append((message, offer)) or False
    main_window._active_modal = lambda: object()

    main_window._on_gl_unavailable("no usable OpenGL")

    assert seen == []

    main_window._active_modal = lambda: None

    def prompted():
        assert seen == [("no usable OpenGL", True)]

    # The retry fires 200 ms after the modal goes, but on the CI Windows
    # runner the window's own queued startup work has delayed it by several
    # seconds; waitUntil returns as soon as the prompt arrives.
    qtbot.waitUntil(prompted, timeout=10000)


def test_a_failed_relaunch_is_logged(main_window, caplog):
    main_window._relaunch = lambda: False

    with caplog.at_level(logging.ERROR, logger="bermake.ui.main_window"):
        main_window._restart()

    assert any(record.levelno == logging.ERROR for record in caplog.records)


def test_mesa_unavailable_disables_the_switch_at_startup(qtbot, monkeypatch):
    monkeypatch.setattr(main_window_module, "default_mesa_dir", lambda: None)

    window = MainWindow()
    qtbot.addWidget(window)

    action = window._actions[ACTION]
    assert not action.isEnabled()
    assert "tools/fetch_mesa.py" in action.toolTip()


def test_the_viewport_signal_reaches_the_window(main_window):
    seen = []
    main_window._prompt_gl_fallback = lambda message, offer: seen.append(message) or False

    main_window._viewport.gl_unavailable.emit("from the viewport")

    assert seen == ["from the viewport"]


def test_the_switch_is_in_the_help_menu(main_window):
    labels = [a.text() for a in main_window._menus["Help"].actions()]
    assert "Use Compatibility Rendering" in labels


def test_the_window_uses_the_dialog_the_startup_preflight_uses(main_window, monkeypatch):
    """One dialog for both checks (final review I1), parented to the window."""
    import bermake.diagnostics.error_dialog as error_dialog

    calls = []
    monkeypatch.setattr(
        error_dialog,
        "show_gl_fallback_dialog",
        lambda message, offer, parent=None: calls.append((message, offer, parent)) or True,
    )

    assert main_window._prompt_gl_fallback("no usable OpenGL", True) is True
    assert calls == [("no usable OpenGL", True, main_window)]
