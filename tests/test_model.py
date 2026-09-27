import numpy as np
from bermake.model.model import Model


def test_fresh_model_is_at_root_identity():
    m = Model()
    assert m.root.name == "Model"
    assert m.active_path == []
    assert m.active_context is m.root
    assert m.active_scene is m.root.mesh
    assert np.allclose(m.active_world_transform, np.eye(4))


def test_enter_exit_changes_active_context():
    m = Model()
    d = m.new_definition("Group #1", is_group=True)
    t = np.eye(4); t[:3, 3] = [2, 0, 0]
    inst = m.new_instance(d, t)
    m.root.children.append(inst)

    m.enter(inst)
    assert m.active_context is d
    assert m.active_scene is d.mesh
    assert np.allclose(m.active_world_transform[:3, 3], [2, 0, 0])

    m.exit_one()
    assert m.active_context is m.root
    assert np.allclose(m.active_world_transform, np.eye(4))


def test_new_instance_registers_backref():
    m = Model()
    d = m.new_definition("Chair", is_group=False)
    a = m.new_instance(d)
    b = m.new_instance(d)
    assert d.instances == [a, b]
    assert a.id != b.id


def test_revalidate_pops_destroyed_context():
    m = Model()
    d = m.new_definition("Group #1", is_group=True)
    inst = m.new_instance(d)
    m.root.children.append(inst)
    m.enter(inst)
    # Simulate undo destroying the instance:
    m.root.children.remove(inst)
    m.revalidate_active_path()
    assert m.active_path == []
    assert m.active_context is m.root
