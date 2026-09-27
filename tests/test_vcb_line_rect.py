from __future__ import annotations

import types

import numpy as np
from bermake.commands.command_stack import CommandStack
from bermake.scene.scene import Scene
from bermake.tools.line_tool import LineTool
from bermake.tools.rectangle_tool import RectangleTool
from bermake.tools.tool import ToolContext
from bermake.units import Units
from bermake.viewport.snap_engine import SnapKind
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

U = Units()


def _press():
    return QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(0, 0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )


def _snap(p, kind=SnapKind.ENDPOINT):
    return types.SimpleNamespace(
        kind=kind,
        world_position=np.asarray(p, np.float32),
        axis=None,
        vertex_id=None,
        edge_id=None,
        edge_t=None,
    )


def _ctx(s, stack):
    return ToolContext(
        scene=s,
        command_stack=stack,
        camera=None,
        widget_size_provider=lambda: (800, 600),
        units_provider=lambda: U,
    )


def test_line_typed_length(qtbot):
    s = Scene()
    stack = CommandStack()
    t = LineTool()
    t.activate(_ctx(s, stack))
    t.on_mouse_press(_press(), _snap([0, 0, 0]))     # start
    t.on_mouse_move(_press(), _snap([1, 0, 0]))      # +X direction
    assert t.apply_typed_value("3", U) is True       # 3 m along +X
    # the second vertex must be at (3,0,0)
    assert any(np.allclose(v.position, [3, 0, 0]) for v in s.vertices_iter())


def test_rectangle_typed_dims(qtbot):
    s = Scene()
    stack = CommandStack()
    t = RectangleTool()
    t.activate(_ctx(s, stack))
    t.on_mouse_press(_press(), _snap([0, 0, 0]))     # first corner
    t.on_mouse_move(_press(), _snap([1, 1, 0]))      # drag into +X/+Y quadrant
    assert t.apply_typed_value("4x2", U) is True     # 4 wide, 2 tall
    assert any(np.allclose(v.position, [4, 2, 0]) for v in s.vertices_iter())
    assert stack.can_undo
