"""v0.7.1: earcut triangles must wind the same way as the face loop they came from.

`Scene._project_loop_to_2d_for_earcut` flattens a loop onto an axis-aligned
plane before triangulating. It used to choose the axis pair from `abs(normal)`,
which is handedness-blind: three of the six dominant-normal cases came out
mirrored, so earcut returned triangles wound backwards relative to the face's
own loop. Half of every closed primitive's triangles were inside-out (a box:
3 of its 6 faces), invisible until M7.5a gave front and back distinct looks and
a freshly created box started rendering three faces in the back material.

These tests pin the invariant directly, in the two forms that matter:
  - the projection preserves the loop's orientation, for all six normals;
  - the triangles of a closed primitive all point away from its centroid.
"""

from __future__ import annotations

import numpy as np
import pytest
from bermake._core import make_box, make_cone, make_cylinder, make_sphere
from bermake.scene.mesh_builder import build_mesh_into_scene
from bermake.scene.scene import Scene, _project_loop_to_2d_for_earcut

_AXES = {
    "+Z": (0.0, 0.0, 1.0),
    "-Z": (0.0, 0.0, -1.0),
    "+X": (1.0, 0.0, 0.0),
    "-X": (-1.0, 0.0, 0.0),
    "+Y": (0.0, 1.0, 0.0),
    "-Y": (0.0, -1.0, 0.0),
}


def _unit_square_with_normal(normal) -> np.ndarray:
    """A CCW unit square in the plane whose outward normal is `normal`."""
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    seed = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(seed, n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    corners = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    return np.array([a * u + b * v for a, b in corners], dtype=np.float32)


def _shoelace(ring_2d: np.ndarray) -> float:
    """Twice the signed area: positive iff the 2D ring is counter-clockwise."""
    total = 0.0
    n = len(ring_2d)
    for i in range(n):
        x0, y0 = float(ring_2d[i][0]), float(ring_2d[i][1])
        x1, y1 = float(ring_2d[(i + 1) % n][0]), float(ring_2d[(i + 1) % n][1])
        total += x0 * y1 - x1 * y0
    return total


def _loop_normal(points: np.ndarray) -> np.ndarray:
    """Newell normal of a closed loop of 3D points."""
    p = np.asarray(points, dtype=np.float64)
    q = np.roll(p, -1, axis=0)
    n = np.array(
        [
            np.sum((p[:, 1] - q[:, 1]) * (p[:, 2] + q[:, 2])),
            np.sum((p[:, 2] - q[:, 2]) * (p[:, 0] + q[:, 0])),
            np.sum((p[:, 0] - q[:, 0]) * (p[:, 1] + q[:, 1])),
        ]
    )
    return n / np.linalg.norm(n)


def _face_into_scene(points: np.ndarray):
    scene = Scene()
    ids = [scene.add_vertex(np.asarray(p, dtype=np.float32)) for p in points]
    return scene, scene.add_face_from_loop(ids)


def _triangle_windings(scene, face_id, reference) -> tuple[int, int]:
    """(agreeing, reversed) triangle counts against a reference direction."""
    positions = {v.id: np.asarray(v.position, dtype=np.float64) for v in scene.vertices_iter()}
    agree = reversed_ = 0
    for tri in scene.face(face_id).triangles:
        a, b, c = (positions[int(i)] for i in tri)
        if float(np.dot(np.cross(b - a, c - a), reference)) > 0.0:
            agree += 1
        else:
            reversed_ += 1
    return agree, reversed_


# --- the projection itself ------------------------------------------------


@pytest.mark.parametrize("axis", sorted(_AXES))
def test_projection_preserves_loop_orientation_on_every_axis(axis):
    """A CCW loop must stay CCW after projection, for all six dominant normals.

    Under the abs()-based projection the -Z, -X and +Y rows came out negative:
    a mirrored ring, hence backwards triangles out of earcut.
    """
    ring = _project_loop_to_2d_for_earcut(_unit_square_with_normal(_AXES[axis]))
    assert _shoelace(ring) > 0.0, f"projection for a {axis} normal is mirrored"


@pytest.mark.parametrize("axis", sorted(_AXES))
def test_face_triangles_agree_with_the_loop_normal_on_every_axis(axis):
    points = _unit_square_with_normal(_AXES[axis])
    scene, face_id = _face_into_scene(points)
    agree, reversed_ = _triangle_windings(scene, face_id, _loop_normal(points))
    assert (agree, reversed_) == (2, 0)


# --- faces the axis-aligned projection must still cope with ---------------


def test_a_tilted_non_axis_aligned_face_still_triangulates():
    """The bug the projection originally fixed: vertical/tilted faces used to
    collapse to collinear points in XY and earcut returned zero triangles."""
    square = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    axis = np.array([0.37, 0.81, 0.45])
    axis /= np.linalg.norm(axis)
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=np.float64
    )
    angle = 0.9
    rotation = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)
    points = ((square @ rotation.T) + np.array([0.3, -0.7, 1.1])).astype(np.float32)

    scene, face_id = _face_into_scene(points)
    triangles = scene.face(face_id).triangles
    assert len(triangles) == 2
    agree, reversed_ = _triangle_windings(scene, face_id, _loop_normal(points))
    assert (agree, reversed_) == (2, 0)


@pytest.mark.parametrize(
    "normal",
    [(1, 1, 0), (0, 1, 1), (1, 0, 1), (-1, -1, 0), (1, 1, 1), (-1, -1, -1)],
    ids=["45-xy", "45-yz", "45-xz", "45-xy-neg", "body-diagonal", "body-diagonal-neg"],
)
def test_a_face_on_a_45_degree_diagonal_normal_is_handled(normal):
    """Exactly-tied dominant components: the tie-break must still be handed."""
    points = _unit_square_with_normal(normal)
    scene, face_id = _face_into_scene(points)
    assert len(scene.face(face_id).triangles) == 2
    agree, reversed_ = _triangle_windings(scene, face_id, _loop_normal(points))
    assert (agree, reversed_) == (2, 0)


def test_a_concave_loop_winds_outward_even_with_a_reflex_corner_first():
    """The plane is chosen from the whole loop (Newell), not the first three
    vertices — whose cross product points the WRONG WAY when the loop starts at
    a reflex corner, mirroring the projection and reversing every triangle."""
    el = np.array(
        [[0, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0], [1, 2, 0], [0, 2, 0]], dtype=np.float32
    )
    points = np.roll(el, -2, axis=0)  # reflex corner lands at index 1
    scene, face_id = _face_into_scene(points)
    triangles = scene.face(face_id).triangles
    assert len(triangles) == 4
    agree, reversed_ = _triangle_windings(scene, face_id, _loop_normal(points))
    assert (agree, reversed_) == (4, 0)


# --- closed solids --------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(lambda: make_box(2.0, 3.0, 4.0), id="box"),
        pytest.param(lambda: make_cylinder(1.0, 2.0, 16), id="cylinder"),
        pytest.param(lambda: make_cone(1.0, 2.0, 16), id="cone"),
        pytest.param(lambda: make_sphere(1.0, 8, 16), id="sphere"),
    ],
)
def test_every_triangle_of_a_closed_primitive_winds_outward(factory):
    """Each rendered triangle's geometric normal must point away from the solid.

    These are convex solids, so "away from the centroid" is the whole test.
    Before the fix: box 6 outward / 6 inward, cylinder 30/30, cone 8/22,
    sphere 112/112.
    """
    scene = Scene()
    build_mesh_into_scene(factory(), scene)
    positions = {v.id: np.asarray(v.position, dtype=np.float64) for v in scene.vertices_iter()}
    centroid = np.mean(np.array(list(positions.values())), axis=0)

    inward = []
    outward = 0
    for face in scene.faces_iter():
        for tri in face.triangles:
            a, b, c = (positions[int(i)] for i in tri)
            normal = np.cross(b - a, c - a)
            assert np.linalg.norm(normal) > 1e-12, f"degenerate triangle on face {face.id}"
            if float(np.dot(normal, (a + b + c) / 3.0 - centroid)) > 0.0:
                outward += 1
            else:
                inward.append(face.id)

    assert not inward, f"{len(inward)} inward triangles on faces {sorted(set(inward))}"
    assert outward > 0
