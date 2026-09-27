from __future__ import annotations

import numpy as np
from bermake.commands.roof_commands import CreateRoofCommand
from bermake.model.model import Model


def _cmd(model, kind="gable"):
    return CreateRoofCommand(kind, 4.0, 6.0, 30.0, np.eye(4), model.active_context)


def test_creates_one_roof_group():
    model = Model()
    target = model.active_context
    _cmd(model).do(model)
    assert len(target.children) == 1
    defn = target.children[0].definition
    assert defn.is_group is True
    assert defn.name == "Roof"
    assert len(list(defn.mesh.faces_iter())) == 5


def test_each_placement_is_its_own_definition():
    model = Model()
    target = model.active_context
    _cmd(model).do(model)
    _cmd(model).do(model)
    assert target.children[0].definition is not target.children[1].definition


def test_undo_detaches_and_redo_reuses_same_instance():
    model = Model()
    target = model.active_context
    cmd = _cmd(model)
    cmd.do(model)
    inst = target.children[0]
    defn = inst.definition
    assert len(defn.instances) == 1
    cmd.undo(model)
    assert len(target.children) == 0
    assert len(defn.instances) == 0          # detached from both
    cmd.do(model)                            # redo
    assert len(target.children) == 1
    assert target.children[0] is inst        # SAME object reused
    assert len(defn.instances) == 1


def test_transform_is_applied():
    model = Model()
    t = np.eye(4)
    t[:3, 3] = [5.0, 6.0, 2.4]
    CreateRoofCommand("gable", 4.0, 6.0, 30.0, t, model.active_context).do(model)
    inst = model.active_context.children[-1]
    assert np.allclose(inst.transform[:3, 3], [5.0, 6.0, 2.4])


def test_degenerate_adds_nothing():
    model = Model()
    CreateRoofCommand("gable", 0.0, 6.0, 30.0, np.eye(4), model.active_context).do(model)
    assert len(model.active_context.children) == 0
