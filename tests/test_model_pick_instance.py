import numpy as np
from bermake.model.model import Model


def _unit_quad(scene):
    a = scene.add_vertex(np.array([-1, -1, 0], np.float32))
    b = scene.add_vertex(np.array([1, -1, 0], np.float32))
    c = scene.add_vertex(np.array([1, 1, 0], np.float32))
    d = scene.add_vertex(np.array([-1, 1, 0], np.float32))
    scene.add_face_from_loop([a, b, c, d])


def test_pick_instance_hits_translated_child():
    m = Model()
    g = m.new_definition("G", is_group=True)
    _unit_quad(g.mesh)
    t = np.eye(4); t[:3, 3] = [10, 0, 0]
    inst = m.new_instance(g, t); m.root.children.append(inst)
    # Ray from above the translated quad pointing down (-z):
    hit = m.pick_instance(np.array([10, 0, 5], np.float32), np.array([0, 0, -1], np.float32))
    assert hit is inst
    # A ray over the origin (where nothing is) misses:
    miss = m.pick_instance(np.array([0, 0, 5], np.float32), np.array([0, 0, -1], np.float32))
    assert miss is None
