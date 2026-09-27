from __future__ import annotations

import numpy as np
from bermake.commands.material_commands import PaintFaceCommand
from bermake.scene.scene import Scene


def _one_face_scene():
    s = Scene()
    v = [
        s.add_vertex(np.array([0.0, 0.0, 0.0])),
        s.add_vertex(np.array([1.0, 0.0, 0.0])),
        s.add_vertex(np.array([0.0, 1.0, 0.0])),
    ]
    return s, s.add_face_from_loop(v)


def test_do_paints_and_undo_restores_default():
    s, f = _one_face_scene()
    cmd = PaintFaceCommand(f, 4)
    cmd.do(s)
    assert s.face_material(f) == 4
    cmd.undo(s)
    assert s.face_material(f) == 0


def test_overpaint_undo_restores_previous_material():
    s, f = _one_face_scene()
    s.set_face_material(f, 2)
    cmd = PaintFaceCommand(f, 9)
    cmd.do(s)
    assert s.face_material(f) == 9
    cmd.undo(s)
    assert s.face_material(f) == 2


def test_paint_default_clears_and_undo_restores():
    s, f = _one_face_scene()
    s.set_face_material(f, 6)
    cmd = PaintFaceCommand(f, 0)        # paint Default -> clear
    cmd.do(s)
    assert s.face_material(f) == 0
    cmd.undo(s)
    assert s.face_material(f) == 6


def test_redo_after_undo_is_idempotent():
    s, f = _one_face_scene()
    cmd = PaintFaceCommand(f, 5)
    cmd.do(s)
    cmd.undo(s)
    cmd.do(s)
    assert s.face_material(f) == 5
