from __future__ import annotations

import types

import numpy as np
from bermake.commands.command_stack import CommandStack
from bermake.scene.scene import Scene
from bermake.tools.arc_tool import ArcTool
from bermake.tools.push_pull_tool import PushPullTool
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


def _snap(p):
    return types.SimpleNamespace(
        kind=SnapKind.ON_FACE,
        world_position=np.asarray(p, np.float32),
        axis=None,
        vertex_id=None,
        edge_id=None,
        edge_t=None,
        face_id=None,
    )


def _ctx(s, stack):
    return ToolContext(
        scene=s,
        command_stack=stack,
        camera=None,
        widget_size_provider=lambda: (800, 600),
        units_provider=lambda: U,
    )


def test_arc_typed_chord_then_bulge(qtbot):
    s = Scene()
    stack = CommandStack()
    t = ArcTool()
    t.activate(_ctx(s, stack))
    t.on_mouse_press(_press(), _snap([0, 0, 0]))      # start
    t.on_mouse_move(_press(), _snap([1, 0, 0]))       # +X dir for the chord
    assert t.apply_typed_value("4", U) is True        # chord length 4 → places end, advances
    t.on_mouse_move(_press(), _snap([2, 1, 0]))       # bulge side preview
    assert t.apply_typed_value("1", U) is True        # sagitta 1 → commit
    assert stack.can_undo


def test_pushpull_typed_distance(qtbot, monkeypatch):
    s = Scene()
    # build a unit square face on the ground
    a = s.add_vertex(np.array([0, 0, 0], np.float32))
    b = s.add_vertex(np.array([1, 0, 0], np.float32))
    c = s.add_vertex(np.array([1, 1, 0], np.float32))
    d = s.add_vertex(np.array([0, 1, 0], np.float32))
    f = s.add_face_from_loop([a, b, c, d])
    stack = CommandStack()
    t = PushPullTool()
    t.activate(_ctx(s, stack))
    t._arm_face(f)                                     # enter DRAGGING directly
    assert t.apply_typed_value("3", U) is True         # extrude 3 m
    # a top vertex must exist at z = 3
    assert any(np.allclose(v.position, [0, 0, 3]) for v in s.vertices_iter())
    assert stack.can_undo
