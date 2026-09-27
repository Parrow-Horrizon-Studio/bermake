"""Python binding smoke tests for ray_intersect_mesh."""

import numpy as np
import pytest


def _make_ground_rect():
    from bermake._core import HalfEdgeMesh

    mesh = HalfEdgeMesh()
    v0 = mesh.add_vertex(0.0, 0.0, 0.0)
    v1 = mesh.add_vertex(1.0, 0.0, 0.0)
    v2 = mesh.add_vertex(1.0, 1.0, 0.0)
    v3 = mesh.add_vertex(0.0, 1.0, 0.0)
    mesh.add_halfedge_pair(v0, v1)
    mesh.add_halfedge_pair(v1, v2)
    mesh.add_halfedge_pair(v2, v3)
    mesh.add_halfedge_pair(v3, v0)
    f = mesh.add_face_from_loop([v0, v1, v2, v3], [v0, v1, v2, v0, v2, v3])
    return mesh, f


def test_ray_intersect_mesh_hit_returns_face_id_t_and_point():
    from bermake._core import ray_intersect_mesh

    mesh, f = _make_ground_rect()
    hit = ray_intersect_mesh(mesh, [0.5, 0.5, 5.0], [0.0, 0.0, -1.0])
    assert hit is not None
    assert hit.face_id == f
    assert hit.t == pytest.approx(5.0, abs=1e-5)
    assert tuple(hit.point) == pytest.approx((0.5, 0.5, 0.0), abs=1e-5)


def test_ray_intersect_mesh_miss_returns_none():
    from bermake._core import ray_intersect_mesh

    mesh, _ = _make_ground_rect()
    hit = ray_intersect_mesh(mesh, [5.0, 5.0, 5.0], [0.0, 0.0, -1.0])
    assert hit is None


def test_ray_intersect_mesh_empty_mesh_returns_none():
    from bermake._core import HalfEdgeMesh, ray_intersect_mesh

    mesh = HalfEdgeMesh()
    hit = ray_intersect_mesh(mesh, [0.0, 0.0, 5.0], [0.0, 0.0, -1.0])
    assert hit is None


def test_ray_intersect_mesh_accepts_numpy_arrays():
    """Common caller shape: pass numpy float32 (3,) arrays directly.

    nanobind's stl/array converter accepts numpy ndarrays without explicit
    list conversion — this exercises the code path PushPullTool will use."""
    from bermake._core import ray_intersect_mesh

    mesh, f = _make_ground_rect()
    origin = np.array([0.5, 0.5, 5.0], dtype=np.float32)
    direction = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    hit = ray_intersect_mesh(mesh, origin, direction)
    assert hit is not None
    assert hit.face_id == f
