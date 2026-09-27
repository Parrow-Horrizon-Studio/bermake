"""ScaleTool factor math + grip selection + commit."""

from __future__ import annotations

import numpy as np

from bermake.commands.command_stack import CommandStack
from bermake.scene.scene import Scene
from bermake.selection import Selection
from bermake.tools.scale_tool import ScaleTool
from bermake.tools.tool import ToolContext
from bermake.tools.transform_support import GripSpec


def _square(s: Scene):
    a = s.add_vertex(np.array([0, 0, 0], np.float32))
    b = s.add_vertex(np.array([2, 0, 0], np.float32))
    c = s.add_vertex(np.array([2, 2, 0], np.float32))
    d = s.add_vertex(np.array([0, 2, 0], np.float32))
    f = s.add_face_from_loop([a, b, c, d])
    return a, b, c, d, f


def _ctx(s, stack, sel):
    return ToolContext(scene=s, command_stack=stack, camera=None,
                       widget_size_provider=lambda: (800, 600), selection=sel)


def test_face_grip_single_axis_factor():
    tool = ScaleTool()
    grip = GripSpec(position=np.array([2, 1, 0], np.float32),
                    opposite=np.array([0, 1, 0], np.float32), axes=(0,))
    extent = np.array([2, 2, 0], np.float32)
    f = tool._factors(grip, anchor=grip.opposite, cursor=np.array([4, 1, 0], np.float32),
                      extent=extent, uniform=False)
    assert np.allclose(f, [2, 1, 1])


def test_corner_grip_uniform_factor():
    tool = ScaleTool()
    grip = GripSpec(position=np.array([2, 2, 0], np.float32),
                    opposite=np.array([0, 0, 0], np.float32), axes=(0, 1))
    extent = np.array([2, 2, 0], np.float32)
    f = tool._factors(grip, anchor=grip.opposite, cursor=np.array([4, 4, 0], np.float32),
                      extent=extent, uniform=False)
    assert np.allclose(f, [2, 2, 1])


def test_zero_extent_axis_stays_unit():
    tool = ScaleTool()
    grip = GripSpec(position=np.array([2, 2, 0], np.float32),
                    opposite=np.array([0, 0, 0], np.float32), axes=(0, 1, 2))
    extent = np.array([2, 2, 0], np.float32)  # z extent 0
    f = tool._factors(grip, anchor=grip.opposite, cursor=np.array([4, 4, 5], np.float32),
                      extent=extent, uniform=False)
    assert f[2] == 1.0


def test_factor_epsilon_clamp_no_mirror():
    tool = ScaleTool()
    grip = GripSpec(position=np.array([2, 1, 0], np.float32),
                    opposite=np.array([0, 1, 0], np.float32), axes=(0,))
    extent = np.array([2, 2, 0], np.float32)
    f = tool._factors(grip, anchor=grip.opposite, cursor=np.array([-1, 1, 0], np.float32),
                      extent=extent, uniform=False)
    assert f[0] > 0.0


def test_scale_commit_applies_to_selection(qtbot, monkeypatch):
    s = Scene()
    a, b, c, d, f = _square(s)
    sel = Selection(); sel.replace(faces=[f])
    stack = CommandStack()
    tool = ScaleTool(); tool.activate(_ctx(s, stack, sel))
    grip = GripSpec(position=np.array([2, 2, 0], np.float32),
                    opposite=np.array([0, 0, 0], np.float32), axes=(0, 1))
    monkeypatch.setattr(tool, "_pick_grip", lambda ev: grip)
    monkeypatch.setattr(tool, "_cursor_world", lambda ev: np.array([4, 4, 0], np.float32))
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    press = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(0, 0),
                        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                        Qt.KeyboardModifier.NoModifier)
    rel = QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(0, 0),
                      Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
                      Qt.KeyboardModifier.NoModifier)
    tool.on_mouse_press(press, None)
    tool.on_mouse_release(rel, None)
    assert np.allclose(s.vertex(c).position, [4, 4, 0])
    assert np.allclose(s.vertex(b).position, [4, 0, 0])
    assert stack.can_undo
