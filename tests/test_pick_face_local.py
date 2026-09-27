from __future__ import annotations

import numpy as np
from bermake.geometry.wall import wall_box
from bermake.model.model import Model


def _add_wall(model):
    # a wall along +X, thickness 0.2 (faces at y = +/-0.1), height 2.4
    verts, faces = wall_box((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), 0.2, 2.4)
    defn = model.new_definition("Wall", is_group=True)
    ids = [defn.mesh.add_vertex(np.array(v, dtype=np.float32)) for v in verts]
    for loop in faces:
        defn.mesh.add_face_from_loop([ids[i] for i in loop])
    inst = model.new_instance(defn)
    model.active_context.children.append(inst)
    return inst


def test_picks_wall_face_point_and_viewer_facing_normal():
    model = Model()
    _add_wall(model)
    # ray from +Y toward -Y at (1, 5, 1.2): hits the y=+0.1 face
    hit = model.pick_face_local(origin=(1.0, 5.0, 1.2), direction=(0.0, -1.0, 0.0))
    assert hit is not None
    point, normal = hit
    assert np.allclose(point, [1.0, 0.1, 1.2], atol=1e-5)
    # normal faces the viewer (+Y), i.e. opposite the ray direction
    assert np.allclose(normal / np.linalg.norm(normal), [0.0, 1.0, 0.0], atol=1e-5)


def test_miss_returns_none():
    model = Model()
    _add_wall(model)
    hit = model.pick_face_local(origin=(50.0, 50.0, 50.0), direction=(0.0, 0.0, 1.0))
    assert hit is None
