from __future__ import annotations

import types

import numpy as np
import pytest
from bermake.commands.command_stack import CommandStack
from bermake.scene.scene import Scene
from bermake.tools.circle_tool import CircleTool
from bermake.tools.polygon_tool import PolygonTool
from bermake.tools.tool import ToolContext
from bermake.units import Units
from bermake.viewport.snap_engine import SnapKind
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

U = Units()


def _press():
    return QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(0, 0),
                       Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier)


def _snap(p):
    return types.SimpleNamespace(kind=SnapKind.ON_FACE, world_position=np.asarray(p, np.float32),
                                 axis=None, vertex_id=None, edge_id=None, edge_t=None,
                                 face_id=None)


def _ctx(s, stack):
    return ToolContext(scene=s, command_stack=stack, camera=None,
                       widget_size_provider=lambda: (800, 600), units_provider=lambda: U)


def _max_radius_from_origin(s):
    return max(float(np.linalg.norm(v.position[:2])) for v in s.vertices_iter())


def test_circle_typed_radius(qtbot):
    s = Scene()
    stack = CommandStack()
    t = CircleTool()
    t.activate(_ctx(s, stack))
    t.on_mouse_press(_press(), _snap([0, 0, 0]))     # center on ground
    assert t.apply_typed_value("2", U) is True       # radius 2 m
    assert _max_radius_from_origin(s) == pytest.approx(2.0, abs=1e-3)
    assert stack.can_undo


def test_polygon_sides_then_radius(qtbot):
    s = Scene()
    stack = CommandStack()
    t = PolygonTool()
    t.activate(_ctx(s, stack))
    t.on_mouse_press(_press(), _snap([0, 0, 0]))     # center
    assert t.apply_typed_value("8s", U) is True      # set 8 sides, keep drawing
    assert t._sides == 8
    assert t.has_active_gesture                       # still drawing
    assert t.apply_typed_value("2", U) is True        # radius 2 → commit
    assert sum(1 for _ in s.vertices_iter()) == 8
