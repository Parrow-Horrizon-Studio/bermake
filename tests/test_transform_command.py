"""TransformVerticesCommand do/undo/redo round-trip."""

from __future__ import annotations

import numpy as np
from bermake.commands.command_stack import CommandStack
from bermake.commands.scene_commands import TransformVerticesCommand
from bermake.scene.scene import Scene


def _two_verts(s: Scene):
    a = s.add_vertex(np.array([0, 0, 0], np.float32))
    b = s.add_vertex(np.array([1, 0, 0], np.float32))
    return a, b


def test_do_moves_and_undo_restores():
    s = Scene()
    a, b = _two_verts(s)
    moves = {
        a: (np.array([0, 0, 0], np.float32), np.array([0, 0, 5], np.float32)),
        b: (np.array([1, 0, 0], np.float32), np.array([1, 0, 5], np.float32)),
    }
    cmd = TransformVerticesCommand(moves)
    cmd.do(s)
    assert np.allclose(s.vertex(a).position, [0, 0, 5])
    assert np.allclose(s.vertex(b).position, [1, 0, 5])
    cmd.undo(s)
    assert np.allclose(s.vertex(a).position, [0, 0, 0])
    assert np.allclose(s.vertex(b).position, [1, 0, 0])


def test_redo_via_stack():
    s = Scene()
    a, _b = _two_verts(s)
    stack = CommandStack()
    cmd = TransformVerticesCommand(
        {a: (np.array([0, 0, 0], np.float32), np.array([9, 0, 0], np.float32))}
    )
    stack.execute(cmd, s)
    assert np.allclose(s.vertex(a).position, [9, 0, 0])
    stack.undo()
    assert np.allclose(s.vertex(a).position, [0, 0, 0])
    stack.redo()
    assert np.allclose(s.vertex(a).position, [9, 0, 0])


def test_noop_moves_are_dropped():
    a_old = np.array([1, 2, 3], np.float32)
    cmd = TransformVerticesCommand({7: (a_old, a_old.copy())})
    assert cmd.is_empty()
