from __future__ import annotations

import numpy as np
from bermake.commands.group_commands import MakeComponentCommand, MakeGroupCommand
from bermake.model.model import Model


def _model_with_face():
    m = Model()
    s = m.active_scene
    v = [s.add_vertex(np.array([0.0, 0.0, 0.0])),
         s.add_vertex(np.array([1.0, 0.0, 0.0])),
         s.add_vertex(np.array([0.0, 1.0, 0.0]))]
    f = s.add_face_from_loop(v)
    return m, v, f


def test_make_group_inherits_tag_id():
    m, v, f = _model_with_face()
    cmd = MakeGroupCommand(m.active_context, v, [], [f], tag_id=4)
    cmd.do(m)
    assert cmd.created_instance.tag_id == 4


def test_make_group_defaults_untagged():
    m, v, f = _model_with_face()
    cmd = MakeGroupCommand(m.active_context, v, [], [f])
    cmd.do(m)
    assert cmd.created_instance.tag_id == 0


def test_tag_survives_undo_redo():
    m, v, f = _model_with_face()
    cmd = MakeGroupCommand(m.active_context, v, [], [f], tag_id=4)
    cmd.do(m)
    cmd.undo(m)
    cmd.do(m)                                  # routes to _redo, reusing the instance object
    assert cmd.created_instance.tag_id == 4


def test_make_component_inherits_tag_id():
    m, v, f = _model_with_face()
    cmd = MakeComponentCommand(m.active_context, v, [], [f], name="C", tag_id=9)
    cmd.do(m)
    assert cmd.created_instance.tag_id == 9
