"""Gesture tests for the 2-Point Arc tool."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent


def _snap(world):  # noqa: ANN001
    from bermake.viewport.snap_engine import SnapKind, SnapResult

    return SnapResult(
        kind=SnapKind.GRID,
        world_position=np.array(world, dtype=np.float32),
        axis=None,
        vertex_id=None,
        label="t",
    )


def _make_tool(scene, stack=None):  # noqa: ANN001
    from bermake.tools import ToolContext
    from bermake.tools.arc_tool import ArcTool

    tool = ArcTool()
    tool.activate(ToolContext(scene=scene, command_stack=stack))
    return tool


def test_arc_three_clicks_make_open_curve_no_face():
    from bermake.scene import Scene

    scene = Scene()
    tool = _make_tool(scene)
    tool.on_mouse_press(None, _snap((-1.0, 0.0, 0.0)))
    tool.on_mouse_press(None, _snap((1.0, 0.0, 0.0)))
    tool.on_mouse_press(None, _snap((0.0, 1.0, 0.0)))
    assert len(list(scene.vertices_iter())) == 13
    assert len(list(scene.edges_iter())) == 12
    assert len(list(scene.faces_iter())) == 0


def test_arc_points_lie_on_expected_circle():
    from bermake.scene import Scene

    scene = Scene()
    tool = _make_tool(scene)
    tool.on_mouse_press(None, _snap((-1.0, 0.0, 0.0)))
    tool.on_mouse_press(None, _snap((1.0, 0.0, 0.0)))
    tool.on_mouse_press(None, _snap((0.0, 1.0, 0.0)))
    for v in scene.vertices_iter():
        assert abs(float(np.hypot(v.position[0], v.position[1])) - 1.0) < 1e-3
        assert abs(float(v.position[2])) < 1e-6


def test_arc_commit_is_atomically_undoable():
    from bermake.commands import CommandStack
    from bermake.scene import Scene

    scene = Scene()
    stack = CommandStack()
    tool = _make_tool(scene, stack)
    tool.on_mouse_press(None, _snap((-1.0, 0.0, 0.0)))
    tool.on_mouse_press(None, _snap((1.0, 0.0, 0.0)))
    tool.on_mouse_press(None, _snap((0.0, 1.0, 0.0)))
    assert stack.can_undo
    stack.undo()
    assert len(list(scene.vertices_iter())) == 0
    stack.redo()
    assert len(list(scene.edges_iter())) == 12


def test_arc_esc_after_two_clicks_cancels_cleanly():
    from bermake.scene import Scene

    scene = Scene()
    tool = _make_tool(scene)
    tool.on_mouse_press(None, _snap((-1.0, 0.0, 0.0)))
    tool.on_mouse_press(None, _snap((1.0, 0.0, 0.0)))
    ev = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
    tool.on_key_press(ev)
    assert tool.has_active_gesture is False
    assert len(list(scene.vertices_iter())) == 0


def test_arc_degenerate_end_ignored():
    from bermake.scene import Scene

    scene = Scene()
    tool = _make_tool(scene)
    tool.on_mouse_press(None, _snap((0.0, 0.0, 0.0)))
    tool.on_mouse_press(None, _snap((0.0, 0.0, 0.0)))  # end == start → ignored
    assert len(list(scene.vertices_iter())) == 0
    assert tool.has_active_gesture is True


def test_arc_overlay_shows_chord_in_placing_end():
    from bermake.scene import Scene

    scene = Scene()
    tool = _make_tool(scene)
    tool.on_mouse_press(None, _snap((-1.0, 0.0, 0.0)))     # start → PLACING_END
    tool.on_mouse_move(None, _snap((1.0, 0.0, 0.0)))       # hover the end
    seg = tool.overlay().rubber_band_segments
    assert seg.shape == (2, 3)  # one chord segment


def test_arc_esc_after_one_click_cancels():
    from bermake.scene import Scene

    scene = Scene()
    tool = _make_tool(scene)
    tool.on_mouse_press(None, _snap((-1.0, 0.0, 0.0)))
    assert tool.has_active_gesture is True
    ev = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
    tool.on_key_press(ev)
    assert tool.has_active_gesture is False
    assert len(list(scene.vertices_iter())) == 0


def test_arc_flat_bulge_commits_straight_two_vertex_segment():
    from bermake.scene import Scene

    scene = Scene()
    tool = _make_tool(scene)
    tool.on_mouse_press(None, _snap((-1.0, 0.0, 0.0)))   # start
    tool.on_mouse_press(None, _snap((1.0, 0.0, 0.0)))    # end
    tool.on_mouse_press(None, _snap((0.0, 0.0, 0.0)))    # bulge on the chord → straight
    assert len(list(scene.vertices_iter())) == 2
    assert len(list(scene.edges_iter())) == 1
    assert len(list(scene.faces_iter())) == 0
