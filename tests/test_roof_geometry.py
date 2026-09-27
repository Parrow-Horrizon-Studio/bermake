from __future__ import annotations

import math
from collections import Counter

import numpy as np
from bermake.geometry.roof import roof_solid


def _edge_counts(faces):
    edges = Counter()
    for f in faces:
        n = len(f)
        for i in range(n):
            edges[frozenset((f[i], f[(i + 1) % n]))] += 1
    return edges


def _closed(faces):
    return all(c == 2 for c in _edge_counts(faces).values())


def _bbox(verts):
    a = np.array(verts, dtype=np.float64)
    return a.min(axis=0), a.max(axis=0)


def test_shed_counts_and_closed():
    verts, faces = roof_solid("shed", width=4.0, depth=6.0, angle=30.0)
    assert len(verts) == 6
    assert len(faces) == 5
    assert _closed(faces)


def test_gable_counts_and_closed():
    verts, faces = roof_solid("gable", width=4.0, depth=6.0, angle=30.0)
    assert len(verts) == 6
    assert len(faces) == 5
    assert _closed(faces)


def test_gable_ridge_height_and_extent():
    w, d, ang = 4.0, 6.0, 30.0
    verts, _ = roof_solid("gable", w, d, ang)
    lo, hi = _bbox(verts)
    # footprint centred; base at z=0; ridge height = (w/2)*tan(angle)
    assert np.allclose(lo, [-w / 2, -d / 2, 0.0])
    assert np.isclose(hi[2], (w / 2) * math.tan(math.radians(ang)))
    assert np.allclose([hi[0], hi[1]], [w / 2, d / 2])


def test_shed_high_edge_height():
    w, d, ang = 4.0, 6.0, 30.0
    verts, _ = roof_solid("shed", w, d, ang)
    _, hi = _bbox(verts)
    # shed rises across the full width: H = w*tan(angle)
    assert np.isclose(hi[2], w * math.tan(math.radians(ang)))


def test_identical_params_identical_geometry():
    a = roof_solid("gable", 4.0, 6.0, 30.0)
    b = roof_solid("gable", 4.0, 6.0, 30.0)
    assert a[0] == b[0] and a[1] == b[1]


def test_degenerate_returns_empty():
    assert roof_solid("gable", 0.0, 6.0, 30.0) == ([], [])
    assert roof_solid("gable", 4.0, 0.0, 30.0) == ([], [])
    assert roof_solid("gable", 4.0, 6.0, 0.0) == ([], [])
    assert roof_solid("gable", 4.0, 6.0, 90.0) == ([], [])
    assert roof_solid("shed", 4.0, 6.0, -5.0) == ([], [])


def test_hip_ridge_case_counts_and_closed():
    # depth > width -> a ridge set back from both ends
    verts, faces = roof_solid("hip", width=4.0, depth=6.0, angle=30.0)
    assert len(verts) == 6
    assert len(faces) == 5
    assert _closed(faces)


def test_hip_pyramid_case_counts_and_closed():
    # depth <= width -> pyramidal apex (single point)
    verts, faces = roof_solid("hip", width=6.0, depth=4.0, angle=30.0)
    assert len(verts) == 5
    assert len(faces) == 5
    assert _closed(faces)


def test_hip_apex_height_equal_pitch():
    # apex height = min(w, d)/2 * tan(angle)
    for w, d in [(4.0, 6.0), (6.0, 4.0), (5.0, 5.0)]:
        verts, _ = roof_solid("hip", w, d, 35.0)
        _, hi = _bbox(verts)
        assert np.isclose(hi[2], min(w, d) / 2.0 * math.tan(math.radians(35.0)))


def test_hip_ridge_setback_length():
    # for d > w, ridge length along Y == d - w (hip run = w/2 each end)
    w, d = 4.0, 10.0
    verts, _ = roof_solid("hip", w, d, 30.0)
    a = np.array(verts)
    apex_z = (w / 2.0) * math.tan(math.radians(30.0))
    ridge = a[np.isclose(a[:, 2], apex_z)]
    assert len(ridge) == 2
    assert np.isclose(abs(ridge[:, 1].max() - ridge[:, 1].min()), d - w)
