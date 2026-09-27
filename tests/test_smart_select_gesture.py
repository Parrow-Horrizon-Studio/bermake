"""Smart-select driven as the real gesture, through a real ViewportWidget.

Final review C1. Every test in tests/test_smart_select.py calls SelectTool's
hooks directly and stops there, and every test in
tests/test_viewport_triple_click.py drives presses without ever sending a
release. Between them those two files cover the whole of smart-select except
the one thing the user actually does, which is let go of the mouse button.

Qt's real double-click sequence is Press, Release, DblClick, Release, and
`ViewportWidget.mouseReleaseEvent` dispatches `on_mouse_release` on every
left-button release. So the trailing release ran the ordinary single-click
pick and replaced the selection smart-select had just built. The headline
feature of the milestone was invisible to the user and eleven task-scoped
reviews missed it, because no test ever sent that release.

These tests therefore drive whole gestures -- press, release and all -- into
a MainWindow's own viewport, with its own SelectTool armed through its own
ToolManager. Nothing here reaches past the widget boundary except to read the
resulting Selection.
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QMouseEvent

_W = 800
_H = 600


def _quad_pair(scene):
    """Two coplanar unit quads sharing one edge, at z=0.

    d---c---f
    |   |   |
    a---b---e
    """
    a = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    c = scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    d = scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    e = scene.add_vertex(np.array([2.0, 0.0, 0.0], dtype=np.float32))
    f = scene.add_vertex(np.array([2.0, 1.0, 0.0], dtype=np.float32))
    left = scene.add_face_from_loop((a, b, c, d))
    right = scene.add_face_from_loop((b, e, f, c))
    return {"a": a, "b": b, "left": left, "right": right}


@pytest.fixture
def armed(main_window):
    """A MainWindow with two quads, the Select tool armed, a fixed viewport
    size and an injected clock for the multi-click run counter."""
    win = main_window
    ids = _quad_pair(win._model.active_context.mesh)
    win._viewport.resize(_W, _H)
    win._viewport.camera.aspect = float(_W) / float(_H)
    clock = [0.0]
    win._viewport.set_now_ms_provider(lambda: clock[0])
    assert win._tool_manager.activate_by_shortcut("Space")
    return win, ids, clock


def _screen(win, world_point):
    s = win._viewport.camera.world_to_screen(np.asarray(world_point, dtype=np.float32), _W, _H)
    assert s is not None, f"{world_point} does not project on screen"
    return float(s[0]), float(s[1])


def _event(kind, x, y, modifiers=Qt.KeyboardModifier.NoModifier):
    return QMouseEvent(
        kind,
        QPointF(x, y),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        modifiers,
    )


def _press(win, x, y, modifiers=Qt.KeyboardModifier.NoModifier):
    win._viewport.mousePressEvent(_event(QMouseEvent.Type.MouseButtonPress, x, y, modifiers))


def _release(win, x, y, modifiers=Qt.KeyboardModifier.NoModifier):
    win._viewport.mouseReleaseEvent(_event(QMouseEvent.Type.MouseButtonRelease, x, y, modifiers))


def _dblclick(win, x, y, modifiers=Qt.KeyboardModifier.NoModifier):
    win._viewport.mouseDoubleClickEvent(
        _event(QMouseEvent.Type.MouseButtonDblClick, x, y, modifiers)
    )


def test_the_trailing_release_does_not_erase_a_double_click_selection(armed):
    """Press, Release, DblClick, Release -- the whole sequence Qt delivers for
    one physical double-click. Before the fix the selection read
    `faces [0] edges [0, 1, 2, 3]` after the DblClick and
    `faces [0] edges []` after the trailing Release."""
    win, ids, clock = armed
    x, y = _screen(win, (0.5, 0.5, 0.0))

    clock[0] = 0.0
    _press(win, x, y)
    _release(win, x, y)
    clock[0] = 120.0
    _dblclick(win, x, y)
    assert win._selection.faces == {ids["left"]}
    assert len(win._selection.edges) == 4

    clock[0] = 130.0
    _release(win, x, y)
    assert win._selection.faces == {ids["left"]}
    assert len(win._selection.edges) == 4


def test_the_trailing_release_does_not_erase_a_triple_click_flood(armed):
    """Press, Release, DblClick, Release, Press, Release. The flood takes both
    quads and all seven edges, which is what distinguishes it from the
    double-click's one face and four edges."""
    win, ids, clock = armed
    x, y = _screen(win, (0.5, 0.5, 0.0))

    clock[0] = 0.0
    _press(win, x, y)
    _release(win, x, y)
    clock[0] = 120.0
    _dblclick(win, x, y)
    _release(win, x, y)
    clock[0] = 240.0
    _press(win, x, y)
    assert win._selection.faces == {ids["left"], ids["right"]}
    assert len(win._selection.edges) == 7

    clock[0] = 250.0
    _release(win, x, y)
    assert win._selection.faces == {ids["left"], ids["right"]}
    assert len(win._selection.edges) == 7


def test_an_ordinary_single_click_still_picks_one_face(armed):
    """The suppression must be one-shot. A plain click after a double-click
    gesture has to behave like a plain click, or eating the trailing release
    would have traded one broken gesture for another."""
    win, ids, clock = armed
    x, y = _screen(win, (0.5, 0.5, 0.0))

    clock[0] = 0.0
    _press(win, x, y)
    _release(win, x, y)
    clock[0] = 120.0
    _dblclick(win, x, y)
    _release(win, x, y)

    # A separate, unhurried click well outside the multi-click run interval.
    clock[0] = 9000.0
    _press(win, x, y)
    _release(win, x, y)
    assert win._selection.faces == {ids["left"]}
    assert win._selection.edges == set()


def test_a_double_click_that_missed_still_lets_the_release_through(armed):
    """Suppression is set only where a smart selection was actually applied,
    so a double-click on empty space leaves the trailing release to do what it
    has always done. Inside no group, that is clearing the selection."""
    win, ids, clock = armed
    empty_x, empty_y = _screen(win, (50.0, 50.0, 0.0))
    win._selection.replace(faces={ids["left"]})

    clock[0] = 0.0
    _dblclick(win, empty_x, empty_y)
    assert win._selection.faces == {ids["left"]}  # the miss itself changes nothing
    clock[0] = 10.0
    _release(win, empty_x, empty_y)
    assert win._selection.faces == set()


def test_shift_double_click_adds_and_survives_the_trailing_release(armed):
    win, ids, clock = armed
    x, y = _screen(win, (0.5, 0.5, 0.0))
    win._selection.replace(faces={ids["right"]})
    shift = Qt.KeyboardModifier.ShiftModifier

    clock[0] = 0.0
    _press(win, x, y, shift)
    _release(win, x, y, shift)
    clock[0] = 120.0
    _dblclick(win, x, y, shift)
    clock[0] = 130.0
    _release(win, x, y, shift)
    assert win._selection.faces == {ids["left"], ids["right"]}


# --- Final review I3: a multi-click run does not survive a context change ---


def test_a_click_after_entering_a_group_does_not_flood_the_new_context(main_window):
    """Measured before the fix: Press(t=0), DblClick(t=120), Press(t=380) at
    one pixel gave `tool.triples == 1`. `ClickRuns` measures each press
    against the PREVIOUS press only, so a click a third of a second after a
    double-click continues the run to three -- and in bermake a double-click
    ENTERS a group, so that follow-on click flood-selected the context the
    user had only just stepped into.

    Masked by C1 until C1 was fixed: the flood used to be erased by its own
    trailing release, which is why the two had to be fixed together.
    """
    win = main_window
    ids = _quad_pair(win._model.active_context.mesh)
    win._on_select_all()
    win._on_make_group()
    win._selection.clear()

    win._viewport.resize(_W, _H)
    win._viewport.camera.aspect = float(_W) / float(_H)
    clock = [0.0]
    win._viewport.set_now_ms_provider(lambda: clock[0])
    assert win._tool_manager.activate_by_shortcut("Space")

    x, y = _screen(win, (0.5, 0.5, 0.0))
    clock[0] = 0.0
    _press(win, x, y)
    _release(win, x, y)
    clock[0] = 120.0
    _dblclick(win, x, y)
    _release(win, x, y)
    assert win._model.active_path, "the double-click should have entered the group"

    # A single click inside the freshly entered group, 260ms later: well
    # inside any plausible double-click interval, and therefore a third
    # press in the run as far as the old counter was concerned.
    clock[0] = 380.0
    _press(win, x, y)
    _release(win, x, y)
    assert win._selection.faces == {ids["left"]}
    assert win._selection.edges == set(), "the click flooded the newly entered context"
