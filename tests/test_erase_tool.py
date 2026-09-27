"""Gesture tests for the Eraser tool (drag-erase edges, cascade to faces)."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent


def _cam(w, h):
    from bermake.viewport.camera import Camera
    c = Camera()
    c.aspect = float(w) / float(h)
    return c


def _press(x, y):
    return QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(x, y),
                       Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier)


def _move(x, y):
    return QMouseEvent(QEvent.Type.MouseMove, QPointF(x, y),
                       Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier)


def _release(x, y):
    return QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(x, y),
                       Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
                       Qt.KeyboardModifier.NoModifier)


def _quad_scene():
    from bermake.scene import Scene
    s = Scene()
    a = s.add_vertex(np.array([-1, -1, 0], dtype=np.float32))
    b = s.add_vertex(np.array([1, -1, 0], dtype=np.float32))
    c = s.add_vertex(np.array([1, 1, 0], dtype=np.float32))
    d = s.add_vertex(np.array([-1, 1, 0], dtype=np.float32))
    fid = s.add_face_from_loop((a, b, c, d))
    return s, fid


def _make(scene, stack, w=800, h=600):
    from bermake.tools import ToolContext
    from bermake.tools.erase_tool import EraserTool
    cam = _cam(w, h)
    tool = EraserTool()
    tool.activate(ToolContext(scene=scene, command_stack=stack, camera=cam,
                              widget_size_provider=lambda: (w, h)))
    return tool, cam


def test_erase_edge_removes_edge_and_its_face(qtbot):
    from bermake.commands import CommandStack
    scene, fid = _quad_scene()
    stack = CommandStack()
    tool, cam = _make(scene, stack)
    faces0 = len(list(scene.faces_iter()))
    edges0 = len(list(scene.edges_iter()))
    sx, sy, _ = cam.world_to_screen(np.array([0.0, -1.0, 0.0], dtype=np.float32), 800, 600)
    tool.on_mouse_press(_press(sx, sy), None)
    tool.on_mouse_release(_release(sx, sy), None)
    assert len(list(scene.faces_iter())) == faces0 - 1
    assert len(list(scene.edges_iter())) == edges0 - 1


def test_erase_is_atomically_undoable(qtbot):
    from bermake.commands import CommandStack
    scene, fid = _quad_scene()
    stack = CommandStack()
    tool, cam = _make(scene, stack)
    v0, e0, f0 = (len(list(scene.vertices_iter())), len(list(scene.edges_iter())),
                  len(list(scene.faces_iter())))
    sx, sy, _ = cam.world_to_screen(np.array([0.0, -1.0, 0.0], dtype=np.float32), 800, 600)
    tool.on_mouse_press(_press(sx, sy), None)
    tool.on_mouse_release(_release(sx, sy), None)
    assert stack.can_undo
    stack.undo()
    assert (len(list(scene.vertices_iter())), len(list(scene.edges_iter())),
            len(list(scene.faces_iter()))) == (v0, e0, f0)
    stack.redo()
    assert len(list(scene.faces_iter())) == f0 - 1


def test_drag_erase_two_edges_is_one_undo(qtbot):
    from bermake.commands import CommandStack
    scene, fid = _quad_scene()
    stack = CommandStack()
    tool, cam = _make(scene, stack)
    e0 = len(list(scene.edges_iter()))
    p_ab = cam.world_to_screen(np.array([0.0, -1.0, 0.0], dtype=np.float32), 800, 600)
    p_bc = cam.world_to_screen(np.array([1.0, 0.0, 0.0], dtype=np.float32), 800, 600)
    tool.on_mouse_press(_press(p_ab[0], p_ab[1]), None)
    tool.on_mouse_move(_move(p_bc[0], p_bc[1]), None)
    tool.on_mouse_release(_release(p_bc[0], p_bc[1]), None)
    assert len(list(scene.edges_iter())) <= e0 - 2
    stack.undo()
    assert len(list(scene.edges_iter())) == e0


def test_miss_click_pushes_nothing(qtbot):
    from bermake.commands import CommandStack
    scene, fid = _quad_scene()
    stack = CommandStack()
    tool, cam = _make(scene, stack)
    f0 = len(list(scene.faces_iter()))
    # A far corner pixel that hits no edge.
    tool.on_mouse_press(_press(2.0, 2.0), None)
    tool.on_mouse_release(_release(2.0, 2.0), None)
    assert not stack.can_undo
    assert len(list(scene.faces_iter())) == f0


def test_erase_a_crease_edge_removes_both_faces(qtbot):
    from bermake.commands import CommandStack
    from bermake.scene import Scene

    # Two quads meeting at a right-angle crease along (1,0)-(1,1) -- NOT
    # coplanar. M7.6a (#31) makes a COPLANAR interior seam dissolve into one
    # face instead of cascading (see test_erase_dissolves.py); this fixture
    # is folded specifically so it keeps exercising the cascade path.
    s = Scene()
    v00 = s.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    v10 = s.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    v11 = s.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    v01 = s.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    v20 = s.add_vertex(np.array([1.0, 0.0, 1.0], dtype=np.float32))
    v21 = s.add_vertex(np.array([1.0, 1.0, 1.0], dtype=np.float32))
    s.add_face_from_loop((v00, v10, v11, v01))
    s.add_face_from_loop((v10, v20, v21, v11))
    assert len(list(s.faces_iter())) == 2

    stack = CommandStack()
    tool, cam = _make(s, stack)
    f0 = len(list(s.faces_iter()))
    # Erase the shared edge at its midpoint (1, 0.5, 0).
    sx, sy, _ = cam.world_to_screen(np.array([1.0, 0.5, 0.0], dtype=np.float32), 800, 600)
    tool.on_mouse_press(_press(sx, sy), None)
    tool.on_mouse_release(_release(sx, sy), None)
    assert len(list(s.faces_iter())) == f0 - 2   # both faces cascaded
    stack.undo()
    assert len(list(s.faces_iter())) == f0        # both restored
