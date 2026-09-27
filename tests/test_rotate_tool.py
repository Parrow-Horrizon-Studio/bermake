"""RotateTool: 3-click flow, 15 degree snap, plane/axis selection."""

from __future__ import annotations

import math
import types

import numpy as np
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

from bermake.commands.command_stack import CommandStack
from bermake.scene.scene import Scene
from bermake.selection import Selection
from bermake.tools.rotate_tool import RotateTool
from bermake.tools.tool import ToolContext
from bermake.viewport.snap_engine import SnapKind


def _press():
    return QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(0, 0),
                       Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier)


def _snap(pos, kind=SnapKind.ENDPOINT):
    return types.SimpleNamespace(
        kind=kind, world_position=np.asarray(pos, np.float32),
        axis=None, vertex_id=None, edge_id=None, edge_t=None,
    )


def _line(s: Scene):
    a = s.add_vertex(np.array([1, 0, 0], np.float32))
    b = s.add_vertex(np.array([2, 0, 0], np.float32))
    e = s.add_edge(a, b)
    return a, b, e


def _ctx(s, stack, sel):
    return ToolContext(scene=s, command_stack=stack, camera=None,
                       widget_size_provider=lambda: (800, 600), selection=sel)


def test_rotate_90_about_z(qtbot, monkeypatch):
    s = Scene()
    a, b, e = _line(s)
    sel = Selection(); sel.replace(edges=[e])
    stack = CommandStack()
    tool = RotateTool(); tool.activate(_ctx(s, stack, sel))
    monkeypatch.setattr(tool, "_pick_plane_normal", lambda ev: np.array([0, 0, 1], np.float32))
    tool.on_mouse_press(_press(), _snap([0, 0, 0]))      # center at origin
    tool.on_mouse_press(_press(), _snap([1, 0, 0]))      # start dir = +X
    tool.on_mouse_press(_press(), _snap([0, 1, 0]))      # end dir = +Y -> +90
    assert np.allclose(s.vertex(a).position, [0, 1, 0], atol=1e-4)
    assert np.allclose(s.vertex(b).position, [0, 2, 0], atol=1e-4)
    assert stack.can_undo


def test_rotate_snaps_to_15_degrees(qtbot, monkeypatch):
    s = Scene()
    a, b, e = _line(s)
    sel = Selection(); sel.replace(edges=[e])
    tool = RotateTool(); tool.activate(_ctx(s, CommandStack(), sel))
    monkeypatch.setattr(tool, "_pick_plane_normal", lambda ev: np.array([0, 0, 1], np.float32))
    tool.on_mouse_press(_press(), _snap([0, 0, 0]))
    tool.on_mouse_press(_press(), _snap([1, 0, 0]))
    ang = math.radians(20)
    tool.on_mouse_press(_press(), _snap([math.cos(ang), math.sin(ang), 0]))
    exp = np.array([math.cos(math.radians(15)), math.sin(math.radians(15)), 0], np.float32)
    assert np.allclose(s.vertex(a).position, exp, atol=1e-4)


def test_rotate_esc_resets(qtbot, monkeypatch):
    s = Scene()
    a, b, e = _line(s)
    sel = Selection(); sel.replace(edges=[e])
    stack = CommandStack()
    tool = RotateTool(); tool.activate(_ctx(s, stack, sel))
    monkeypatch.setattr(tool, "_pick_plane_normal", lambda ev: np.array([0, 0, 1], np.float32))
    tool.on_mouse_press(_press(), _snap([0, 0, 0]))
    tool.on_mouse_press(_press(), _snap([1, 0, 0]))
    from PySide6.QtGui import QKeyEvent
    tool.on_key_press(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier))
    assert not tool.has_active_gesture
    assert not stack.can_undo
    assert np.allclose(s.vertex(a).position, [1, 0, 0])
