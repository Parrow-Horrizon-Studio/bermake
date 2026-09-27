"""apply_typed_value on Move/Rotate/Scale resolves the gesture at the exact value."""

from __future__ import annotations

import types

import numpy as np
from bermake.commands.command_stack import CommandStack
from bermake.scene.scene import Scene
from bermake.selection import Selection
from bermake.tools.move_tool import MoveTool
from bermake.tools.rotate_tool import RotateTool
from bermake.tools.scale_tool import ScaleTool
from bermake.tools.tool import ToolContext
from bermake.tools.transform_support import GripSpec
from bermake.units import Units
from bermake.viewport.snap_engine import SnapKind
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

U = Units()  # metric meters


def _press():
    return QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(0, 0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )


def _snap(p):
    return types.SimpleNamespace(
        kind=SnapKind.ENDPOINT,
        world_position=np.asarray(p, np.float32),
        axis=None,
        vertex_id=None,
        edge_id=None,
        edge_t=None,
    )


def _square(s):
    a = s.add_vertex(np.array([0, 0, 0], np.float32))
    b = s.add_vertex(np.array([2, 0, 0], np.float32))
    c = s.add_vertex(np.array([2, 2, 0], np.float32))
    d = s.add_vertex(np.array([0, 2, 0], np.float32))
    f = s.add_face_from_loop([a, b, c, d])
    return a, b, c, d, f


def _ctx(s, stack, sel):
    return ToolContext(
        scene=s,
        command_stack=stack,
        camera=None,
        widget_size_provider=lambda: (800, 600),
        selection=sel,
        units_provider=lambda: U,
    )


def test_move_typed_distance(qtbot):
    s = Scene()
    a, _b, _c, _d, f = _square(s)
    sel = Selection()
    sel.replace(faces=[f])
    stack = CommandStack()
    t = MoveTool()
    t.activate(_ctx(s, stack, sel))
    t.on_mouse_press(_press(), _snap([0, 0, 0]))   # grab
    t.on_mouse_move(_press(), _snap([0, 0, 1]))    # direction +Z (any magnitude)
    assert t.apply_typed_value("5", U) is True     # 5 m along +Z
    assert np.allclose(s.vertex(a).position, [0, 0, 5])
    assert stack.can_undo


def test_rotate_typed_angle(qtbot, monkeypatch):
    s = Scene()
    a = s.add_vertex(np.array([1, 0, 0], np.float32))
    b = s.add_vertex(np.array([2, 0, 0], np.float32))
    e = s.add_edge(a, b)
    sel = Selection()
    sel.replace(edges=[e])
    stack = CommandStack()
    t = RotateTool()
    t.activate(_ctx(s, stack, sel))
    monkeypatch.setattr(t, "_pick_plane_normal", lambda ev: np.array([0, 0, 1], np.float32))
    t.on_mouse_press(_press(), _snap([0, 0, 0]))   # center
    t.on_mouse_press(_press(), _snap([1, 0, 0]))   # start dir +X
    t.on_mouse_move(_press(), _snap([1, 1, 0]))    # sweeping CCW (+)
    assert t.apply_typed_value("90", U) is True
    assert np.allclose(s.vertex(a).position, [0, 1, 0], atol=1e-4)
    _ = b  # b is added to scene; keep var to satisfy scene refcount


def test_scale_typed_factor(qtbot, monkeypatch):
    s = Scene()
    _a, _b, c, _d, f = _square(s)
    sel = Selection()
    sel.replace(faces=[f])
    stack = CommandStack()
    t = ScaleTool()
    t.activate(_ctx(s, stack, sel))
    grip = GripSpec(
        position=np.array([2, 2, 0], np.float32),
        opposite=np.array([0, 0, 0], np.float32),
        axes=(0, 1),
    )
    monkeypatch.setattr(t, "_pick_grip", lambda ev: grip)
    monkeypatch.setattr(t, "_cursor_world", lambda ev: np.array([3, 3, 0], np.float32))
    t.on_mouse_press(_press(), None)               # arm grip (anchor origin)
    assert t.apply_typed_value("2", U) is True     # 2x about origin
    assert np.allclose(s.vertex(c).position, [4, 4, 0])
