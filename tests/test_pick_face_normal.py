"""#92: face normals under a non-uniform instance scale.

Brief drift found while implementing (see task-13-brief.md): the brief's
example test builds the group via `main_window` + `group_factory` + a UI
`_on_make_group()` call. `pick_face_local` is a pure `Model` method with no
UI dependency (see the existing `tests/test_pick_face_local.py`), so this
test builds the instance directly on a bare `Model` -- no MainWindow/Qt
needed, and it sidesteps the group_factory nesting trap entirely since only
one instance level is involved.
"""

from __future__ import annotations

import numpy as np
from bermake.model.model import Model


def test_normal_is_correct_under_non_uniform_scale():
    """A 45-degree face inside an instance scaled 4x in x only.

    The linear block maps the untransformed face normal to a vector
    proportional to (-4, 0, -1) (normalized), which is NOT perpendicular to
    the transformed face. The inverse-transpose maps it to a vector
    proportional to (-1, 0, -4) (normalized), which is. The dot product
    between these two candidates is 8/17 (~0.4706) -- far enough from 1.0
    that a wrong implementation cannot pass by tolerance.
    """
    model = Model()
    defn = model.new_definition("G", is_group=True)
    # A face on the plane x + z = 0, so its (local) normal is proportional
    # to (1, 0, 1).
    v = [
        defn.mesh.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        defn.mesh.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32)),
        defn.mesh.add_vertex(np.array([1.0, 1.0, -1.0], dtype=np.float32)),
        defn.mesh.add_vertex(np.array([1.0, 0.0, -1.0], dtype=np.float32)),
    ]
    defn.mesh.add_face_from_loop(v)
    inst = model.new_instance(defn)
    inst.transform[0, 0] = 4.0  # non-uniform: x only
    model.active_context.children.append(inst)

    # Expected normal, computed independently from the transformed geometry
    # (not from pick_face_local's own output).
    local = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, -1.0]])
    world = (inst.transform[:3, :3] @ local.T).T
    expected = np.cross(world[1] - world[0], world[2] - world[0])
    expected = expected / np.linalg.norm(expected)

    # Sanity: the two candidate answers actually differ by far more than
    # our comparison tolerance (see docstring above for the numbers).
    wrong = inst.transform[:3, :3] @ np.array([-1.0, 0.0, -1.0])
    wrong = wrong / np.linalg.norm(wrong)
    assert abs(float(np.dot(wrong, expected))) < 0.5

    origin = np.array([2.0, 0.5, 5.0], dtype=np.float64)
    direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    hit = model.pick_face_local(origin, direction)
    assert hit is not None  # confirm the ray actually hits the face
    _point, normal = hit
    normal = normal / np.linalg.norm(normal)
    assert abs(abs(float(np.dot(normal, expected))) - 1.0) < 1e-6
