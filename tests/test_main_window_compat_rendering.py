"""The compatibility rendering switch in MainWindow (M7.9, spec 2.4).

conftest.py redirects MainWindow's QSettings() to tmp_path/window_state.ini,
so _settings() below opens the same store the window reads and writes.
"""

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


def test_restoring_the_checkmark_at_startup_does_not_offer_a_restart(
    tmp_path, qtbot, monkeypatch
):
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


def test_the_viewport_signal_reaches_the_window(main_window):
    seen = []
    main_window._prompt_gl_fallback = lambda message, offer: seen.append(message) or False

    main_window._viewport.gl_unavailable.emit("from the viewport")

    assert seen == ["from the viewport"]


def test_the_switch_is_in_the_help_menu(main_window):
    labels = [a.text() for a in main_window._menus["Help"].actions()]
    assert "Use Compatibility Rendering" in labels
