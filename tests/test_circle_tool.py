"""Gesture tests for the Circle tool."""

from __future__ import annotations

import numpy as np


def _snap(world, *, kind=None, face_id=None):  # noqa: ANN001
    from bermake.viewport.snap_engine import SnapKind, SnapResult

    return SnapResult(
        kind=kind if kind is not None else SnapKind.GRID,
        world_position=np.array(world, dtype=np.float32),
        axis=None,
        vertex_id=None,
        label="t",
        face_id=face_id,
    )


def _make_tool(scene, stack=None):  # noqa: ANN001
    from bermake.tools import ToolContext
    from bermake.tools.circle_tool import CircleTool

    tool = CircleTool()
    tool.activate(ToolContext(scene=scene, command_stack=stack))
    return tool


def test_circle_idle_overlay_empty():
    from bermake.scene import Scene

    tool = _make_tool(Scene())
    assert tool.overlay().rubber_band_segments.shape == (0, 3)


def test_circle_two_clicks_make_24_segments_and_a_face():
    from bermake.scene import Scene

    scene = Scene()
    tool = _make_tool(scene)
    tool.on_mouse_press(None, _snap((0.0, 0.0, 0.0)))
    tool.on_mouse_press(None, _snap((2.0, 0.0, 0.0)))
    assert len(list(scene.vertices_iter())) == 24
    assert len(list(scene.edges_iter())) == 24
    assert len(list(scene.faces_iter())) == 1
    for v in scene.vertices_iter():
        assert abs(float(np.hypot(v.position[0], v.position[1])) - 2.0) < 1e-3
        assert abs(float(v.position[2])) < 1e-6


def test_circle_face_normal_points_up_on_ground():
    from bermake.scene import Scene

    scene = Scene()
    tool = _make_tool(scene)
    tool.on_mouse_press(None, _snap((0.0, 0.0, 0.0)))
    tool.on_mouse_press(None, _snap((3.0, 0.0, 0.0)))
    face = next(iter(scene.faces_iter()))
    assert float(scene.face_normal(face.id)[2]) > 0.99


def test_circle_zero_radius_does_not_commit():
    from bermake.scene import Scene

    scene = Scene()
    tool = _make_tool(scene)
    tool.on_mouse_press(None, _snap((1.0, 1.0, 0.0)))
    tool.on_mouse_press(None, _snap((1.0, 1.0, 0.0)))
    assert len(list(scene.vertices_iter())) == 0


def test_circle_commit_is_atomically_undoable():
    from bermake.commands import CommandStack
    from bermake.scene import Scene

    scene = Scene()
    stack = CommandStack()
    tool = _make_tool(scene, stack)
    tool.on_mouse_press(None, _snap((0.0, 0.0, 0.0)))
    tool.on_mouse_press(None, _snap((2.0, 0.0, 0.0)))
    assert stack.can_undo
    stack.undo()
    assert len(list(scene.vertices_iter())) == 0
    stack.redo()
    assert len(list(scene.faces_iter())) == 1


def test_circle_draws_on_a_vertical_face():
    from bermake.scene import Scene
    from bermake.viewport.snap_engine import SnapKind

    scene = Scene()
    a = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([0.0, 4.0, 0.0], dtype=np.float32))
    c = scene.add_vertex(np.array([0.0, 4.0, 4.0], dtype=np.float32))
    d = scene.add_vertex(np.array([0.0, 0.0, 4.0], dtype=np.float32))
    fid = scene.add_face_from_loop((a, b, c, d))
    base_verts = len(list(scene.vertices_iter()))

    tool = _make_tool(scene)
    tool.on_mouse_press(None, _snap((0.0, 2.0, 2.0), kind=SnapKind.ON_FACE, face_id=fid))
    tool.on_mouse_press(None, _snap((0.0, 3.0, 2.0), kind=SnapKind.ON_FACE, face_id=fid))
    new = [v for v in scene.vertices_iter() if v.id >= base_verts]
    assert len(new) == 24
    for v in new:
        assert abs(float(v.position[0])) < 1e-4


def test_circle_esc_mid_gesture_resets():
    from bermake.scene import Scene
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent

    scene = Scene()
    tool = _make_tool(scene)
    tool.on_mouse_press(None, _snap((0.0, 0.0, 0.0)))
    assert tool.has_active_gesture
    ev = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
    tool.on_key_press(ev)
    assert not tool.has_active_gesture
    assert tool.overlay().rubber_band_segments.shape == (0, 3)
    assert len(list(scene.vertices_iter())) == 0
