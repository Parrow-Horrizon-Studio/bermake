"""Gesture tests for the Polygon tool."""

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


def _key(qt_key):  # noqa: ANN001
    return QKeyEvent(QKeyEvent.Type.KeyPress, qt_key, Qt.KeyboardModifier.NoModifier)


def _make_tool(scene):  # noqa: ANN001
    from bermake.tools import ToolContext
    from bermake.tools.polygon_tool import PolygonTool

    tool = PolygonTool()
    tool.activate(ToolContext(scene=scene))
    return tool


def test_polygon_default_six_sides():
    from bermake.scene import Scene

    scene = Scene()
    tool = _make_tool(scene)
    tool.on_mouse_press(None, _snap((0.0, 0.0, 0.0)))
    tool.on_mouse_press(None, _snap((2.0, 0.0, 0.0)))
    assert len(list(scene.vertices_iter())) == 6
    assert len(list(scene.edges_iter())) == 6
    assert len(list(scene.faces_iter())) == 1


def test_polygon_up_down_adjusts_side_count():
    from bermake.scene import Scene

    scene = Scene()
    tool = _make_tool(scene)
    tool.on_mouse_press(None, _snap((0.0, 0.0, 0.0)))
    tool.on_key_press(_key(Qt.Key.Key_Up))
    tool.on_key_press(_key(Qt.Key.Key_Up))
    tool.on_key_press(_key(Qt.Key.Key_Down))
    tool.on_mouse_press(None, _snap((2.0, 0.0, 0.0)))
    assert len(list(scene.vertices_iter())) == 7


def test_polygon_sides_clamped_to_min_three():
    from bermake.scene import Scene

    scene = Scene()
    tool = _make_tool(scene)
    tool.on_mouse_press(None, _snap((0.0, 0.0, 0.0)))
    for _ in range(10):
        tool.on_key_press(_key(Qt.Key.Key_Down))
    tool.on_mouse_press(None, _snap((2.0, 0.0, 0.0)))
    assert len(list(scene.vertices_iter())) == 3


def test_polygon_side_count_remembered_across_gestures():
    from bermake.scene import Scene

    scene = Scene()
    tool = _make_tool(scene)
    tool.on_mouse_press(None, _snap((0.0, 0.0, 0.0)))
    tool.on_key_press(_key(Qt.Key.Key_Up))
    tool.on_mouse_press(None, _snap((2.0, 0.0, 0.0)))
    tool.on_mouse_press(None, _snap((10.0, 0.0, 0.0)))
    tool.on_mouse_press(None, _snap((12.0, 0.0, 0.0)))
    counts = [len(scene.face_loop(f.id)) for f in scene.faces_iter()]
    assert sorted(counts) == [7, 7]


def test_polygon_vertices_inscribed_at_radius():
    from bermake.scene import Scene

    scene = Scene()
    tool = _make_tool(scene)
    tool.on_mouse_press(None, _snap((0.0, 0.0, 0.0)))
    tool.on_mouse_press(None, _snap((3.0, 0.0, 0.0)))
    for v in scene.vertices_iter():
        assert abs(float(np.hypot(v.position[0], v.position[1])) - 3.0) < 1e-3


def test_polygon_esc_mid_gesture_resets():
    from bermake.scene import Scene

    scene = Scene()
    tool = _make_tool(scene)
    tool.on_mouse_press(None, _snap((0.0, 0.0, 0.0)))
    assert tool.has_active_gesture
    tool.on_key_press(_key(Qt.Key.Key_Escape))
    assert not tool.has_active_gesture
    assert len(list(scene.vertices_iter())) == 0
