"""MoveTool gesture: press → drag → release commits a translate."""

from __future__ import annotations

import types

import numpy as np
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

from bermake.commands.command_stack import CommandStack
from bermake.scene.scene import Scene
from bermake.selection import Selection
from bermake.tools.move_tool import MoveTool
from bermake.tools.tool import ToolContext
from bermake.viewport.snap_engine import SnapKind


def _press(x=0.0, y=0.0):
    return QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(x, y),
                       Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier)


def _release(x=0.0, y=0.0):
    return QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(x, y),
                       Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
                       Qt.KeyboardModifier.NoModifier)


def _snap(pos, kind=SnapKind.ENDPOINT):
    return types.SimpleNamespace(
        kind=kind, world_position=np.asarray(pos, np.float32),
        axis=None, vertex_id=None, edge_id=None, edge_t=None,
    )


def _square(s: Scene):
    a = s.add_vertex(np.array([0, 0, 0], np.float32))
    b = s.add_vertex(np.array([2, 0, 0], np.float32))
    c = s.add_vertex(np.array([2, 2, 0], np.float32))
    d = s.add_vertex(np.array([0, 2, 0], np.float32))
    f = s.add_face_from_loop([a, b, c, d])
    return a, b, c, d, f


def _ctx(scene, stack, selection):
    return ToolContext(scene=scene, command_stack=stack, camera=None,
                       widget_size_provider=lambda: (800, 600), selection=selection)


def test_move_translates_selection(qtbot):
    s = Scene()
    a, b, c, d, f = _square(s)
    sel = Selection(); sel.replace(faces=[f])
    stack = CommandStack()
    tool = MoveTool(); tool.activate(_ctx(s, stack, sel))
    tool.on_mouse_press(_press(), _snap([0, 0, 0]))
    tool.on_mouse_move(_press(), _snap([0, 0, 3]))   # button-held move
    tool.on_mouse_release(_release(), _snap([0, 0, 3]))
    for vid, base in ((a, [0, 0, 0]), (b, [2, 0, 0]), (c, [2, 2, 0]), (d, [0, 2, 0])):
        assert np.allclose(s.vertex(vid).position, np.array(base) + [0, 0, 3])
    assert stack.can_undo
    stack.undo()
    assert np.allclose(s.vertex(a).position, [0, 0, 0])


def test_move_noop_on_empty_selection(qtbot):
    s = Scene(); _square(s)
    sel = Selection()  # empty
    stack = CommandStack()
    tool = MoveTool(); tool.activate(_ctx(s, stack, sel))
    tool.on_mouse_press(_press(), _snap([0, 0, 0]))
    tool.on_mouse_release(_release(), _snap([0, 0, 3]))
    assert not stack.can_undo


def test_move_esc_cancels_without_commit(qtbot):
    s = Scene()
    a, b, c, d, f = _square(s)
    sel = Selection(); sel.replace(faces=[f])
    stack = CommandStack()
    tool = MoveTool(); tool.activate(_ctx(s, stack, sel))
    tool.on_mouse_press(_press(), _snap([0, 0, 0]))
    tool.on_mouse_move(_press(), _snap([0, 0, 3]))
    tool.deactivate()  # mid-drag bail
    assert not stack.can_undo
    assert np.allclose(s.vertex(a).position, [0, 0, 0])  # mesh untouched
