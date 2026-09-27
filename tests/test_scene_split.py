"""Scene-level split_edge wrapper + edge geometry helpers."""

from __future__ import annotations

import numpy as np
from bermake.scene.scene import Scene


def _quad(scene: Scene):
    v0 = scene.add_vertex(np.array([0, 0, 0], dtype=np.float32))
    v1 = scene.add_vertex(np.array([2, 0, 0], dtype=np.float32))
    v2 = scene.add_vertex(np.array([2, 2, 0], dtype=np.float32))
    v3 = scene.add_vertex(np.array([0, 2, 0], dtype=np.float32))
    for a, b in [(v0, v1), (v1, v2), (v2, v3), (v3, v0)]:
        scene.add_edge(a, b)
    f = scene.add_face_from_loop([v0, v1, v2, v3])
    e01 = scene.add_edge(v0, v1)  # idempotent → existing edge id
    return f, e01, (v0, v1, v2, v3)


def test_point_on_edge_midpoint():
    scene = Scene()
    _, e01, _ = _quad(scene)
    p = scene.point_on_edge(e01, 0.5)
    np.testing.assert_allclose(p, [1.0, 0.0, 0.0], atol=1e-6)


def test_closest_point_on_edge_clamps_to_segment():
    scene = Scene()
    _, e01, _ = _quad(scene)
    p, t = scene.closest_point_on_edge(e01, np.array([5.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(p, [2.0, 0.0, 0.0], atol=1e-6)
    assert t == 1.0


def test_split_edge_returns_result_and_inserts_vertex():
    scene = Scene()
    f, e01, (v0, v1, v2, v3) = _quad(scene)
    res = scene.split_edge(e01, 0.5)
    assert res is not None
    np.testing.assert_allclose(scene.vertex(res.vertex).position, [1.0, 0.0, 0.0], atol=1e-6)
    # Boundary edge → one rebuilt face (face_a), other side None.
    assert (res.face_a is None) != (res.face_b is None)
    live_face = res.face_a if res.face_a is not None else res.face_b
    assert len(scene.face(live_face).loop_vertex_ids) == 5


def test_split_edge_invalid_returns_none():
    scene = Scene()
    _, e01, _ = _quad(scene)
    assert scene.split_edge(e01, 0.0) is None
    assert scene.split_edge(e01, 1.0) is None
