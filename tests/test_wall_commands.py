from __future__ import annotations

from bermake.commands.wall_commands import CreateWallCommand
from bermake.model.model import Model


def test_do_adds_wall_group_then_undo_removes_it():
    model = Model()
    target = model.active_context
    before = len(target.children)
    cmd = CreateWallCommand((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), 0.2, 2.4, target)
    cmd.do(model)
    assert len(target.children) == before + 1
    wall_def = target.children[-1].definition
    assert wall_def.is_group and wall_def.name == "Wall"
    assert len(list(wall_def.mesh.vertices_iter())) == 8
    assert len(list(wall_def.mesh.faces_iter())) == 6
    cmd.undo(model)
    assert len(target.children) == before


def test_redo_rebuilds_and_double_undo_is_noop():
    model = Model()
    target = model.active_context
    cmd = CreateWallCommand((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), 0.2, 2.4, target)
    cmd.do(model)
    cmd.undo(model)
    cmd.undo(model)                     # guarded
    assert len(target.children) == 0
    cmd.do(model)                       # redo re-runs do()
    assert len(target.children) == 1


def test_degenerate_segment_adds_nothing():
    model = Model()
    target = model.active_context
    cmd = CreateWallCommand((1.0, 1.0, 0.0), (1.0, 1.0, 0.0), 0.2, 2.4, target)
    cmd.do(model)
    assert len(target.children) == 0    # wall_box returned empty -> no group
