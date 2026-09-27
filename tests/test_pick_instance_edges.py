import numpy as np
from bermake.model.model import Model


def test_edge_only_group_is_pickable():
    model = Model()
    defn = model.new_definition("EdgeOnly", is_group=True)
    v0 = defn.mesh.add_vertex(np.array([0.0, 0.0, 0.0], np.float32))
    v1 = defn.mesh.add_vertex(np.array([1.0, 0.0, 0.0], np.float32))
    defn.mesh.add_edge(v0, v1)  # a single edge, no faces
    inst = model.new_instance(defn, np.eye(4, dtype=np.float64))
    model.root.children.append(inst)

    # Ray straight down onto the middle of the edge.
    origin = np.array([0.5, 0.0, 5.0], np.float64)
    direction = np.array([0.0, 0.0, -1.0], np.float64)
    assert model.pick_instance(origin, direction) is inst


def test_ray_far_from_the_edge_misses():
    model = Model()
    defn = model.new_definition("EdgeOnly", is_group=True)
    v0 = defn.mesh.add_vertex(np.array([0.0, 0.0, 0.0], np.float32))
    v1 = defn.mesh.add_vertex(np.array([1.0, 0.0, 0.0], np.float32))
    defn.mesh.add_edge(v0, v1)
    inst = model.new_instance(defn, np.eye(4, dtype=np.float64))
    model.root.children.append(inst)

    origin = np.array([0.5, 50.0, 5.0], np.float64)  # far off to the side
    direction = np.array([0.0, 0.0, -1.0], np.float64)
    assert model.pick_instance(origin, direction) is None
