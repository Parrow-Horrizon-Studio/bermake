import numpy as np
from bermake.model.model import Model
from bermake.commands.instance_commands import CreateInstanceCommand, TransformInstanceCommand


def _xlate(x):
    t = np.eye(4); t[0, 3] = x; return t


def test_transform_instance_sets_and_restores():
    m = Model()
    d = m.new_definition("G", is_group=True)
    inst = m.new_instance(d)
    m.root.children.append(inst)
    cmd = TransformInstanceCommand(inst, _xlate(5))
    cmd.do(m)
    assert np.allclose(inst.transform[:3, 3], [5, 0, 0])
    cmd.undo(m)
    assert np.allclose(inst.transform, np.eye(4))


def test_create_instance_adds_and_removes():
    m = Model()
    d = m.new_definition("Chair", is_group=False)
    base = m.new_instance(d)
    m.root.children.append(base)
    cmd = CreateInstanceCommand(m.root, d, _xlate(9))
    cmd.do(m)
    assert cmd.created_instance in m.root.children
    assert len(d.instances) == 2
    cmd.undo(m)
    assert cmd.created_instance not in m.root.children


def test_create_instance_redo_reuses_same_instance():
    """Redo must reuse the same Instance object (same id), not create a second one."""
    m = Model()
    d = m.new_definition("Table", is_group=False)
    cmd = CreateInstanceCommand(m.root, d, _xlate(3))

    # First do
    cmd.do(m)
    inst_id = cmd.created_instance.id
    assert len(d.instances) == 1
    assert cmd.created_instance in m.root.children

    # Undo
    cmd.undo(m)
    assert cmd.created_instance not in m.root.children
    assert len(d.instances) == 0

    # Redo (second do) — same instance id, count returns to 1 (not 2)
    cmd.do(m)
    assert cmd.created_instance.id == inst_id
    assert len(d.instances) == 1
    assert cmd.created_instance in m.root.children
