from __future__ import annotations

from collections import Counter

import numpy as np
from bermake.geometry.wall import wall_box


def _bbox(vertices):
    a = np.array(vertices, dtype=np.float64)
    return a.min(axis=0), a.max(axis=0)


def test_axis_aligned_wall_dimensions_and_centering():
    verts, faces = wall_box((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), thickness=0.2, height=2.4)
    assert len(verts) == 8
    assert len(faces) == 6
    lo, hi = _bbox(verts)
    assert np.allclose(lo, [0.0, -0.1, 0.0])
    assert np.allclose(hi, [4.0, 0.1, 2.4])   # length 4, thickness 0.2 centered, height 2.4


def test_closed_solid_every_edge_shared_by_two_faces():
    _, faces = wall_box((0.0, 0.0, 0.0), (3.0, 0.0, 0.0), 0.2, 2.4)
    edges = Counter()
    for f in faces:
        n = len(f)
        for i in range(n):
            a, b = f[i], f[(i + 1) % n]
            edges[frozenset((a, b))] += 1
    assert len(edges) == 12
    assert all(c == 2 for c in edges.values())   # closed manifold box


def test_diagonal_segment_length_and_height():
    verts, _ = wall_box((0.0, 0.0, 0.0), (3.0, 4.0, 0.0), 0.2, 2.4)
    a = np.array(verts)
    assert np.isclose(a[:, 2].min(), 0.0) and np.isclose(a[:, 2].max(), 2.4)
    # the two base "start" corners are ±perp around (0,0); base centroid near origin end
    base = a[a[:, 2] < 1e-6]
    assert len(base) == 4


def test_bottom_face_normal_points_down():
    verts, faces = wall_box((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), 0.2, 2.4)
    v = np.array(verts)
    loop = faces[0]                     # bottom
    p0, p1, p2 = v[loop[0]], v[loop[1]], v[loop[2]]
    n = np.cross(p1 - p0, p2 - p0)
    assert n[2] < 0                     # outward (down) for the bottom face


def test_degenerate_returns_empty():
    assert wall_box((1, 1, 0), (1, 1, 0), 0.2, 2.4) == ([], [])
    assert wall_box((0, 0, 0), (1, 0, 0), 0.0, 2.4) == ([], [])
    assert wall_box((0, 0, 0), (1, 0, 0), 0.2, 0.0) == ([], [])
