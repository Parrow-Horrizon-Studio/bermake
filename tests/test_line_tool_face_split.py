"""LineTool: drawing a chord across a face divides it (M7.6a task 5).

Reuses the _FakeEvent / _snap / _make_tool idiom from test_line_tool_split.py.
"""

from __future__ import annotations

import numpy as np
from bermake.commands.command_stack import CommandStack
from bermake.scene.scene import Scene
from bermake.tools.line_tool import LineTool
from bermake.tools.tool import ToolContext
from bermake.viewport.snap_engine import SnapKind, SnapResult
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent


class _FakeEvent:
    """LineTool.on_mouse_press/on_key_press never touch the event; a stub suffices."""


def _snap(kind, pos, **kw):
    return SnapResult(
        kind=kind,
        world_position=np.array(pos, dtype=np.float32),
        axis=kw.get("axis"),
        vertex_id=kw.get("vertex_id"),
        label=kw.get("label", ""),
        edge_id=kw.get("edge_id"),
        face_id=kw.get("face_id"),
        edge_t=kw.get("edge_t"),
    )


def _make_tool(scene):
    stack = CommandStack()
    tool = LineTool()
    tool.activate(
        ToolContext(scene=scene, command_stack=stack, camera=None, widget_size_provider=None)
    )
    return tool, stack


def _enter():
    return QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier)


def _quad(scene, z=0.0, size=2.0):
    v = [
        scene.add_vertex(np.array(p, dtype=np.float32))
        for p in [(0, 0, z), (size, 0, z), (size, size, z), (0, size, z)]
    ]
    return scene.add_face_from_loop(v), v


def test_drawing_a_chord_then_enter_splits_the_face_into_two():
    scene = Scene()
    fid, v = _quad(scene)
    tool, stack = _make_tool(scene)

    tool.on_mouse_press(_FakeEvent(), _snap(SnapKind.ENDPOINT, [0, 0, 0], vertex_id=v[0]))
    tool.on_mouse_press(_FakeEvent(), _snap(SnapKind.ENDPOINT, [2, 2, 0], vertex_id=v[2]))
    tool.on_key_press(_enter())

    assert sum(1 for _ in scene.faces_iter()) == 2
    assert not scene._mesh.face_is_live(fid)
    assert stack.can_undo


def test_one_undo_after_a_splitting_line_returns_to_one_face():
    """The split must land in the SAME undo step as the line: one Ctrl+Z,
    not two, restores the original single face.

    Face count and fid liveness alone do not discriminate this: a BROKEN
    implementation that pushes the line composite and then separately
    push_executed()s the SplitFaceCommand as a second undo entry also shows
    exactly one face and a live fid after one undo (undoing the split alone
    already restores the parent face). What a broken version leaves behind
    that a correct one does not is the chord EDGE (never removed, because
    undoing only the split doesn't touch it) and a non-empty undo stack
    (the line composite is still sitting there, unpushed by this one undo).
    Both are asserted below.
    """
    scene = Scene()
    fid, v = _quad(scene)
    tool, stack = _make_tool(scene)

    tool.on_mouse_press(_FakeEvent(), _snap(SnapKind.ENDPOINT, [0, 0, 0], vertex_id=v[0]))
    tool.on_mouse_press(_FakeEvent(), _snap(SnapKind.ENDPOINT, [2, 2, 0], vertex_id=v[2]))
    tool.on_key_press(_enter())
    assert sum(1 for _ in scene.faces_iter()) == 2
    edges_after_split = sum(1 for _ in scene.edges_iter())

    stack.undo()

    assert sum(1 for _ in scene.faces_iter()) == 1
    assert scene._mesh.face_is_live(fid)
    # The chord edge must be gone too: a second, separately-pushed undo
    # entry for the split would leave it behind (one undo only reverses the
    # split, not the line that drew the chord).
    assert sum(1 for _ in scene.edges_iter()) == edges_after_split - 1
    # And the whole gesture -- line plus split -- must be gone from the
    # stack in that one undo, not just the split half of it.
    assert stack.can_undo is False


def test_a_line_that_does_not_cross_a_face_leaves_face_count_unchanged():
    scene = Scene()
    fid, v = _quad(scene)
    tool, _stack = _make_tool(scene)

    # Two adjacent corners: the segment runs along the boundary, not across
    # the interior -- chain_cuts_face rejects this (see its own test suite).
    tool.on_mouse_press(_FakeEvent(), _snap(SnapKind.ENDPOINT, [0, 0, 0], vertex_id=v[0]))
    tool.on_mouse_press(_FakeEvent(), _snap(SnapKind.ENDPOINT, [2, 0, 0], vertex_id=v[1]))
    tool.on_key_press(_enter())

    assert sum(1 for _ in scene.faces_iter()) == 1
    assert scene._mesh.face_is_live(fid)


def test_closing_a_loop_still_creates_a_face_and_does_not_split():
    """Loop closure is the face-creation path (branch 1 in on_mouse_press),
    and it never calls _finish_open_polyline -- NOT because chain_cuts_face
    would reject the closing gesture itself. It would not: branch 1 never
    re-appends the first vertex id to _gesture_vertex_ids, so by the time a
    3-click loop closes, the ids on hand are 3 distinct, non-repeating
    vertices and chain_cuts_face's own degenerate-chain guard never fires.

    So this test has to discriminate on a shape chain_cuts_face WOULD accept
    as a real split if it were ever asked: start and end on an existing
    face's boundary, with the third point strictly interior to it (not one
    of the face's own corners, so rule 2's loop-membership check does not
    reject it either). If loop closure ever gets wired to call
    _finish_open_polyline by mistake, this reads as one extra face (3
    instead of 2, and the original quad no longer live)."""
    scene = Scene()
    fid, v = _quad(scene, size=4.0)
    tool, stack = _make_tool(scene)

    tool.on_mouse_press(_FakeEvent(), _snap(SnapKind.ENDPOINT, [0, 0, 0], vertex_id=v[0]))
    tool.on_mouse_press(_FakeEvent(), _snap(SnapKind.GRID, [2, 2, 0]))  # strictly interior
    tool.on_mouse_press(_FakeEvent(), _snap(SnapKind.ENDPOINT, [4, 4, 0], vertex_id=v[2]))
    tool.on_mouse_press(_FakeEvent(), _snap(SnapKind.ENDPOINT, [0, 0, 0], vertex_id=v[0]))

    assert sum(1 for _ in scene.faces_iter()) == 2
    assert scene._mesh.face_is_live(fid)
    assert tool.has_active_gesture is False
    assert stack.can_undo


def test_double_click_finishes_the_polyline_and_can_split():
    scene = Scene()
    _fid, v = _quad(scene)
    tool, stack = _make_tool(scene)

    tool.on_mouse_press(_FakeEvent(), _snap(SnapKind.ENDPOINT, [0, 0, 0], vertex_id=v[0]))
    tool.on_mouse_press(_FakeEvent(), _snap(SnapKind.ENDPOINT, [2, 2, 0], vertex_id=v[2]))
    tool.on_mouse_double_click(_FakeEvent(), _snap(SnapKind.ENDPOINT, [2, 2, 0], vertex_id=v[2]))

    assert sum(1 for _ in scene.faces_iter()) == 2
    assert tool.has_active_gesture is False
    assert stack.can_undo
