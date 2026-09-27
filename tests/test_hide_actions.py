"""Hide / Unhide / Unhide All (M7.3 Task 14)."""

from __future__ import annotations

import numpy as np
import pytest
from bermake.ui import actions
from bermake.ui.context_menu import is_enabled


def _square(window):
    scene = window._model.active_context.mesh
    v = [
        scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32)),
    ]
    scene.add_face_from_loop(v)


def _group(window):
    window._on_select_all()
    window._on_make_group()
    window._selection.clear()
    return window._model.root.children[-1]


@pytest.mark.parametrize(
    "action_id,shortcut",
    [("edit_hide", "H"), ("edit_unhide", "Shift+H"), ("edit_unhide_all", None)],
)
def test_the_actions_are_declared_with_their_shortcuts(action_id, shortcut):
    spec = next((s for s in actions.ACTIONS if s.id == action_id), None)
    assert spec is not None, f"{action_id} is not declared"
    assert spec.shortcut == shortcut


def test_h_does_not_collide_with_a_tool_shortcut():
    tool_keys = {
        s.shortcut for s in actions.ACTIONS if s.group == actions.TOOL_GROUP
    }
    assert "H" not in tool_keys


def test_the_three_entries_are_in_the_edit_menu():
    edit = next(m for m in actions.MENUS if m.title == "Edit")
    for action_id in ("edit_hide", "edit_unhide", "edit_unhide_all"):
        assert action_id in edit.action_ids


def test_hide_is_offered_on_an_instance_right_click():
    assert "edit_hide" in actions.CONTEXT_MENUS[actions.ContextTarget.INSTANCE]


def test_unhide_all_is_offered_on_empty_space():
    # A hidden object cannot be right-clicked, so the empty-space menu is the
    # only place the escape hatch can live.
    assert "edit_unhide_all" in actions.CONTEXT_MENUS[actions.ContextTarget.EMPTY]


def test_hide_needs_a_selection():
    kwargs = dict(
        target=actions.ContextTarget.INSTANCE,
        in_group=False,
        is_component=False,
    )
    assert is_enabled("edit_hide", has_selection=True, **kwargs) is True
    assert is_enabled("edit_hide", has_selection=False, **kwargs) is False


def test_unhide_needs_a_hidden_instance_selected():
    # Per the spec (docs/2026-08-03-M7.3-outliner-properties-design.md
    # sec.1.3), edit_unhide is enabled when the *selection* contains a
    # hidden instance -- not merely when something, anything, is selected.
    # has_selection=True here (a normal, visible selection) must NOT be
    # enough on its own: this is what the bare has_selection gate got wrong.
    kwargs = dict(
        target=actions.ContextTarget.INSTANCE,
        in_group=False,
        is_component=False,
        has_selection=True,
    )
    assert is_enabled("edit_unhide", has_hidden_selected=True, **kwargs) is True
    assert is_enabled("edit_unhide", has_hidden_selected=False, **kwargs) is False


def test_unhide_all_needs_something_hidden():
    kwargs = dict(
        target=actions.ContextTarget.EMPTY,
        in_group=False,
        is_component=False,
        has_selection=False,
    )
    assert is_enabled("edit_unhide_all", has_hidden=True, **kwargs) is True
    assert is_enabled("edit_unhide_all", has_hidden=False, **kwargs) is False


def test_hiding_the_selection_is_undoable(main_window):
    _square(main_window)
    instance = _group(main_window)
    main_window._selection.replace(instances=[instance.id])

    main_window._on_hide()

    assert instance.hidden is True
    main_window._command_stack.undo()
    assert instance.hidden is False


def test_unhiding_a_selected_hidden_instance_works(main_window):
    # End-to-end: the edit_unhide gating change must not make Unhide itself
    # unreachable. A hidden instance can be selected via the Outliner (not
    # exercised here directly, but selection.replace stands in for it, since
    # _on_unhide only cares about what is selected) and Unhide must still
    # reveal it and remain undoable.
    _square(main_window)
    instance = _group(main_window)
    instance.hidden = True
    main_window._selection.replace(instances=[instance.id])

    main_window._on_unhide()

    assert instance.hidden is False
    main_window._command_stack.undo()
    assert instance.hidden is True


def test_hide_with_no_selection_does_nothing(main_window):
    _square(main_window)
    _group(main_window)

    main_window._on_hide()

    assert main_window._command_stack.can_undo is True  # only the make-group
    assert not any(i.hidden for i in main_window._model.root.children)


def test_unhide_all_reveals_everything_in_the_context(main_window):
    _square(main_window)
    first = _group(main_window)
    _square(main_window)
    second = _group(main_window)
    first.hidden = True
    second.hidden = True

    main_window._on_unhide_all()

    assert first.hidden is False
    assert second.hidden is False
    main_window._command_stack.undo()
    assert first.hidden is True


def test_unhide_all_with_nothing_hidden_pushes_no_command(main_window):
    _square(main_window)
    _group(main_window)
    depth = len(main_window._command_stack._undo)

    main_window._on_unhide_all()

    assert len(main_window._command_stack._undo) == depth


def test_the_h_shortcut_hides_the_selection(qtbot, main_window):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    _square(main_window)
    instance = _group(main_window)
    main_window._selection.replace(instances=[instance.id])
    main_window.show()
    # WindowShortcut-context actions only fire while their window is the
    # active one. show()/waitExposed() alone leaves activeWindow() as None
    # in this headless/offscreen environment, so the shortcut would silently
    # not fire without forcing activation explicitly.
    main_window.activateWindow()
    main_window.raise_()
    with qtbot.waitActive(main_window, timeout=2000):
        pass

    QTest.keyClick(main_window, Qt.Key.Key_H)

    assert instance.hidden is True
