import numpy as np
from bermake.model.model import Model


def test_traverse_root_only():
    m = Model()
    out = list(m.traverse())
    assert len(out) == 1
    d, t = out[0]
    assert d is m.root
    assert np.allclose(t, np.eye(4))


def test_traverse_accumulates_transforms():
    m = Model()
    g = m.new_definition("G", is_group=True)
    t = np.eye(4); t[:3, 3] = [5, 0, 0]
    inst = m.new_instance(g, t)
    m.root.children.append(inst)
    out = list(m.traverse())
    defs = [d for d, _ in out]
    assert defs == [m.root, g]
    assert np.allclose(out[1][1][:3, 3], [5, 0, 0])


def test_traverse_yields_shared_definition_twice():
    m = Model()
    chair = m.new_definition("Chair", is_group=False)
    a = m.new_instance(chair, _xlate(1, 0, 0))
    b = m.new_instance(chair, _xlate(9, 0, 0))
    m.root.children += [a, b]
    out = [(d.id, tuple(t[:3, 3])) for d, t in m.traverse()]
    # root + chair@1 + chair@9
    assert (chair.id, (1.0, 0.0, 0.0)) in out
    assert (chair.id, (9.0, 0.0, 0.0)) in out


def _xlate(x, y, z):
    t = np.eye(4); t[:3, 3] = [x, y, z]; return t
