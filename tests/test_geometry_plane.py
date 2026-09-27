"""Unit tests for DrawingPlane (pure geometry, no Qt)."""

from __future__ import annotations

import numpy as np

from bermake.geometry import DrawingPlane


def _orthonormal(plane: DrawingPlane) -> None:
    for a in (plane.u, plane.v, plane.normal):
        assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-9
    assert abs(float(plane.u @ plane.v)) < 1e-9
    assert abs(float(plane.u @ plane.normal)) < 1e-9
    assert abs(float(plane.v @ plane.normal)) < 1e-9
    assert np.allclose(np.cross(plane.u, plane.v), plane.normal, atol=1e-9)


def test_horizontal_plane_is_world_aligned():
    p = DrawingPlane.horizontal(np.array([2.0, 3.0, 5.0]))
    _orthonormal(p)
    assert np.allclose(p.u, [1.0, 0.0, 0.0])
    assert np.allclose(p.v, [0.0, 1.0, 0.0])
    assert np.allclose(p.normal, [0.0, 0.0, 1.0])


def test_to_world_project_round_trip():
    p = DrawingPlane.horizontal(np.array([1.0, 1.0, 4.0]))
    uv = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0], [-1.5, 2.5]])
    world = p.to_world(uv)
    assert np.allclose(world[0], [1.0, 1.0, 4.0])
    assert np.allclose(world[:, 2], 4.0)
    back = p.project(world)
    assert np.allclose(back, uv, atol=1e-9)
    assert np.allclose(p.project(p.to_world(np.array([2.0, 3.0]))), [2.0, 3.0])


def test_from_normal_builds_orthonormal_basis():
    p = DrawingPlane.from_normal(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 5.0]))
    _orthonormal(p)
    assert np.allclose(p.normal, [0.0, 0.0, 1.0])
    q = DrawingPlane.from_normal(np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    _orthonormal(q)
    assert np.allclose(q.normal, [0.0, 1.0, 0.0])
    r = DrawingPlane.from_normal(np.zeros(3), np.array([1.0, 0.0, 0.0]))
    _orthonormal(r)
    assert np.allclose(r.normal, [1.0, 0.0, 0.0])


def test_from_normal_rejects_degenerate():
    import pytest

    with pytest.raises(ValueError):
        DrawingPlane.from_normal(np.zeros(3), np.zeros(3))


def test_from_face_uses_scene_face_normal():
    from bermake.scene import Scene

    scene = Scene()
    a = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    c = scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    d = scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    fid = scene.add_face_from_loop((a, b, c, d))

    p = DrawingPlane.from_face(scene, fid, np.array([0.5, 0.5, 0.0]))
    _orthonormal(p)
    assert np.allclose(p.normal, scene.face_normal(fid), atol=1e-6)
