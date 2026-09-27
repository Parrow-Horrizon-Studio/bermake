"""Right-click resolution and menu construction (M7.2 Task 14)."""

from __future__ import annotations

import numpy as np

from bermake.ui.actions import ContextTarget
from bermake.ui.context_menu import build_context_menu


def _labels(menu):
    return [a.text() for a in menu.actions() if not a.isSeparator()]


def test_empty_space_menu_is_built_from_the_registry(qtbot, main_window):
    menu = build_context_menu(main_window, ContextTarget.EMPTY, None)
    assert "Select All" in _labels(menu)
    assert "Zoom Extents" in _labels(menu)


def test_close_group_is_disabled_at_the_root(qtbot, main_window):
    menu = build_context_menu(main_window, ContextTarget.EMPTY, None)
    action = next(a for a in menu.actions() if a.text() == "Close Group")
    assert not action.isEnabled()


def test_instance_menu_carries_the_assign_tag_submenu(qtbot, main_window_with_group):
    window = main_window_with_group
    instance = window._model.active_context.children[-1]
    window._selection.clear()
    window._selection.toggle_instance(instance.id)

    menu = build_context_menu(window, ContextTarget.INSTANCE, instance.id)

    assign = next(a for a in menu.actions() if a.text() == "Assign Tag")
    assert assign.menu() is not None
    # One entry per tag in the library, including the Untagged sentinel.
    assert len(assign.menu().actions()) == len(window._model.tags.tags())


def test_face_menu_has_no_assign_tag_submenu(qtbot, main_window_with_square):
    # M5c is instances-only tagging (#69).
    menu = build_context_menu(main_window_with_square, ContextTarget.FACE, None)
    assert "Assign Tag" not in _labels(menu)


def test_menu_actions_are_the_same_objects_as_the_menu_bar_uses(qtbot, main_window):
    menu = build_context_menu(main_window, ContextTarget.EMPTY, None)
    action = next(a for a in menu.actions() if a.text() == "Select All")
    assert action is main_window._actions["edit_select_all"]


def test_right_click_on_empty_space_leaves_the_selection_alone(qtbot, main_window_with_square):
    window = main_window_with_square
    window._on_select_all()
    before = set(window._selection.faces)

    window._on_context_menu_requested(-100, -100)  # off-model

    assert set(window._selection.faces) == before


def test_viewport_exposes_a_context_menu_signal(qtbot, main_window):
    assert hasattr(main_window._viewport, "context_menu_requested")


def test_reenabling_after_the_menu_closes_restores_the_menu_bar(qtbot, main_window, monkeypatch):
    """The point of this task: build_context_menu disables entries on the
    SAME QAction objects the menu bar holds. If MainWindow only re-enabled
    the ones it happened to disable -- or skipped re-enabling on some early
    return -- a disabled context entry would leave the matching menu-bar
    item permanently greyed out. Exercise the real exec_menu=True path (the
    one the live signal connection uses) with QMenu.exec monkeypatched to a
    no-op so the test never enters a modal loop, then assert the menu-bar
    action came back enabled.
    """
    window = main_window
    close_group = window._actions["edit_close_group"]

    # Replace the pop-the-menu step on the instance, NOT QMenu.exec itself:
    # PySide6 dispatches QMenu.exec straight to C++, so patching the class is
    # accepted silently but has no effect and the real modal loop runs --
    # which under the offscreen platform CI uses never returns.
    #
    # The stub runs while the menu is notionally open, which is the only
    # moment the disabled state exists. Sampling it here proves the restore
    # afterwards undid real work rather than passing vacuously.
    observed: dict[str, bool] = {}

    def _capture(menu, pos):
        observed["during"] = close_group.isEnabled()

    monkeypatch.setattr(window, "_exec_context_menu", _capture)

    window._on_context_menu_requested(0, 0, exec_menu=True)  # empty space at root

    assert observed, "the exec_menu=True path did not reach _exec_context_menu"
    assert observed["during"] is False, "the EMPTY menu should disable Close Group at root"
    assert close_group.isEnabled()


def test_right_click_on_an_already_selected_face_keeps_the_multi_selection(
    qtbot, main_window, monkeypatch
):
    """Right-clicking one of several selected faces must not collapse the
    selection to just that face -- SketchUp's behaviour, and the guard
    _selection_contains/_select_only exist to preserve."""
    import bermake.ui.context_menu as context_menu_module

    window = main_window
    scene = window._model.active_context.mesh
    v1 = [
        scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32)),
    ]
    face_a = scene.add_face_from_loop(v1)
    v2 = [
        scene.add_vertex(np.array([2.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([3.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([3.0, 1.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([2.0, 1.0, 0.0], dtype=np.float32)),
    ]
    face_b = scene.add_face_from_loop(v2)
    window._selection.replace(faces=[face_a, face_b])

    monkeypatch.setattr(
        context_menu_module,
        "resolve_context_target",
        lambda *args, **kwargs: (ContextTarget.FACE, face_a),
    )

    window._on_context_menu_requested(0, 0)

    assert window._selection.faces == {face_a, face_b}


def test_reenabling_is_registry_wide_not_scoped_to_the_closed_menu(
    qtbot, main_window, monkeypatch
):
    """The discriminating case the same-target test cannot see.

    edit_close_group appears only on the EMPTY table. Disable it by
    building an EMPTY menu, then open and close a FACE menu -- which never
    lists it. A sweep scoped to the closed menu's own entries would leave
    it stuck disabled; only a whole-registry sweep brings it back.
    """
    from bermake.ui import context_menu as context_menu_module

    window = main_window
    close_group = window._actions["edit_close_group"]

    assert "edit_close_group" not in context_menu_module.context_menu_ids(
        ContextTarget.FACE, in_group=False, has_selection=True
    ), "precondition: the FACE menu must not list Close Group"

    # Disable it the way the EMPTY menu does, then run a FACE cycle. A
    # restore scoped to the closed menu's own entries would never touch
    # Close Group, since FACE does not list it, and it would stay disabled.
    close_group.setEnabled(False)
    monkeypatch.setattr(window, "_exec_context_menu", lambda menu, pos: None)
    monkeypatch.setattr(
        context_menu_module,
        "resolve_context_target",
        lambda *args, **kwargs: (ContextTarget.FACE, None),
    )
    window._on_context_menu_requested(0, 0, exec_menu=True)

    # The snapshot covers every registered id, so Close Group comes back to
    # exactly the state the handler found it in -- disabled, here.
    assert not close_group.isEnabled()


def test_reenabling_survives_a_raise_while_the_menu_is_being_built(
    qtbot, main_window, monkeypatch
):
    """build_context_menu disables entries as it goes, so it must run inside
    the try -- otherwise a raise partway through strands whatever it had
    already disabled."""
    from bermake.ui import context_menu as context_menu_module

    window = main_window
    close_group = window._actions["edit_close_group"]

    def _boom(win, target, entity_id):
        close_group.setEnabled(False)  # partial work, as the real builder does
        raise RuntimeError("menu construction blew up")

    monkeypatch.setattr(context_menu_module, "build_context_menu", _boom)
    monkeypatch.setattr(
        context_menu_module,
        "resolve_context_target",
        lambda *args, **kwargs: (ContextTarget.EMPTY, None),
    )

    try:
        window._on_context_menu_requested(0, 0, exec_menu=True)
    except RuntimeError:
        pass

    assert close_group.isEnabled(), "a raise during build left an action stuck disabled"


def test_right_click_is_suppressed_while_a_tool_is_mid_gesture(
    qtbot, main_window, monkeypatch
):
    """Right-click during a live gesture means cancel, which Esc owns."""
    window = main_window
    # MainWindow's live connection runs the handler with exec_menu=True, so
    # without this the emitted signal reaches the real modal QMenu.exec and
    # hangs headlessly -- the same trap _exec_context_menu exists to defuse.
    monkeypatch.setattr(window, "_exec_context_menu", lambda menu, pos: None)
    window._activate("line")  # Line tool
    line = window._viewport.tool_manager.active
    seen: list[tuple[int, int]] = []
    window._viewport.context_menu_requested.connect(lambda x, y: seen.append((x, y)))

    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QContextMenuEvent

    def _fire():
        window._viewport.contextMenuEvent(
            QContextMenuEvent(QContextMenuEvent.Reason.Mouse, QPoint(5, 5))
        )

    _fire()
    assert seen, "an idle Line tool must not suppress the context menu"

    seen.clear()
    from bermake.tools.line_tool import _State

    line._state = _State.DRAWING  # mid multi-click draw
    assert line.has_active_gesture
    _fire()
    assert not seen, "a mid-gesture tool must suppress the context menu"


def test_select_tool_suppresses_only_during_a_live_box_drag(
    qtbot, main_window_with_square, monkeypatch
):
    """SelectTool.has_active_gesture is also True for a merely non-empty
    selection -- right-clicking a selection is the ordinary case and must
    not be swallowed. A live box drag is the one Select case that is a
    gesture in the click sense."""
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QContextMenuEvent

    window = main_window_with_square
    monkeypatch.setattr(window, "_exec_context_menu", lambda menu, pos: None)
    window._activate("select")  # Select tool
    select = window._viewport.tool_manager.active
    seen: list[tuple[int, int]] = []
    window._viewport.context_menu_requested.connect(lambda x, y: seen.append((x, y)))

    def _fire():
        window._viewport.contextMenuEvent(
            QContextMenuEvent(QContextMenuEvent.Reason.Mouse, QPoint(5, 5))
        )

    faces = list(window._model.active_context.mesh.faces_iter())
    window._selection.replace(faces=[faces[0].id])
    assert select.has_active_gesture, "precondition: a selection reports a gesture"
    assert not select.is_box_selecting
    _fire()
    assert seen, "right-click on an existing selection must still open the menu"

    seen.clear()
    select._is_box = True  # live box-select drag
    assert select.is_box_selecting
    _fire()
    assert not seen, "a live box drag must suppress the context menu"


def test_context_menu_does_not_clobber_a_disable_it_did_not_make(
    qtbot, main_window, monkeypatch
):
    """The sweep restores prior state rather than forcing everything on.

    Nothing else in the app disables a registry action today, but the
    moment something reasonable does -- Undo greyed on an empty stack,
    Save on a clean document -- a blanket re-enable would silently undo it
    on the next right-click.
    """
    from bermake.ui import context_menu as context_menu_module

    window = main_window
    monkeypatch.setattr(window, "_exec_context_menu", lambda menu, pos: None)
    monkeypatch.setattr(
        context_menu_module,
        "resolve_context_target",
        lambda *args, **kwargs: (ContextTarget.EMPTY, None),
    )

    undo = window._actions["edit_undo"]
    undo.setEnabled(False)  # as an empty-command-stack guard would

    window._on_context_menu_requested(0, 0, exec_menu=True)

    assert not undo.isEnabled(), "the sweep re-enabled an action it did not disable"


def test_edit_group_is_disabled_for_a_multi_instance_selection(qtbot, main_window_with_group):
    """_on_edit_group returns early unless exactly one instance is selected,
    so offering it enabled for two would be a control that does nothing."""
    window = main_window_with_group
    instance = window._model.active_context.children[-1]

    window._selection.clear()
    window._selection.toggle_instance(instance.id)
    menu = build_context_menu(window, ContextTarget.INSTANCE, instance.id)
    single = next(a for a in menu.actions() if a.text() == "Edit Group")
    assert single.isEnabled()

    window._selection.toggle_instance(instance.id + 999)  # a second instance
    menu = build_context_menu(window, ContextTarget.INSTANCE, instance.id)
    multi = next(a for a in menu.actions() if a.text() == "Edit Group")
    assert not multi.isEnabled()


def test_edit_text_is_disabled_for_a_dimension(qtbot, main_window):
    """_on_edit_label_text returns early for any annotation that is not a
    Label, so a Dimension must not offer it enabled."""
    from bermake.model.annotation import Dimension, Label

    window = main_window
    context = window._model.active_context
    origin = np.zeros(3, dtype=np.float32)
    unit = np.ones(3, dtype=np.float32)

    label = Label(id=901, anchor=origin, text_pos=unit, text="hello")
    context.annotations.append(label)
    window._selection.clear()
    window._selection.toggle_annotation(label.id)
    menu = build_context_menu(window, ContextTarget.ANNOTATION, label.id)
    on_label = next(a for a in menu.actions() if a.text().startswith("Edit Text"))
    assert on_label.isEnabled()

    dim = Dimension(id=902, p1=origin, p2=unit, offset=origin)
    context.annotations.append(dim)
    window._selection.clear()
    window._selection.toggle_annotation(dim.id)
    menu = build_context_menu(window, ContextTarget.ANNOTATION, dim.id)
    on_dim = next(a for a in menu.actions() if a.text().startswith("Edit Text"))
    assert not on_dim.isEnabled()
