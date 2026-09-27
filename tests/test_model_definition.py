import numpy as np
from bermake.model.definition import Definition
from bermake.model.instance import Instance


def test_definition_defaults_to_empty_scene():
    d = Definition(0, "Model", is_group=False)
    assert d.id == 0
    assert d.name == "Model"
    assert d.is_group is False
    assert d.children == []
    assert d.instances == []
    assert d.local_aabb() is None  # empty mesh → no bbox


def test_instance_defaults_to_identity_transform():
    d = Definition(1, "Group #1", is_group=True)
    inst = Instance(7, d)
    assert inst.id == 7
    assert inst.definition is d
    assert np.allclose(inst.transform, np.eye(4))


def test_local_aabb_spans_mesh_vertices():
    d = Definition(2, "Box", is_group=True)
    d.mesh.add_vertex(np.array([-1, -2, 0], np.float32))
    d.mesh.add_vertex(np.array([3, 4, 5], np.float32))
    lo, hi = d.local_aabb()
    assert np.allclose(lo, [-1, -2, 0])
    assert np.allclose(hi, [3, 4, 5])
