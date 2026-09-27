from __future__ import annotations

from collections import Counter

import numpy as np
from bermake.geometry.opening import opening_frame


def _closed(faces):
    edges = Counter()
    for f in faces:
        n = len(f)
        for i in range(n):
            edges[frozenset((f[i], f[(i + 1) % n]))] += 1
    return edges


def _bbox(verts):
    a = np.array(verts, dtype=np.float64)
    return a.min(axis=0), a.max(axis=0)


def test_door_has_three_frame_members_plus_panel():
    verts, faces = opening_frame("door", width=0.9, height=2.1, depth=0.1)
    # 4 boxes (L jamb, R jamb, head, panel) -> 32 verts, 24 quad faces
    assert len(verts) == 32
    assert len(faces) == 24
    # every box is closed: each edge shared by exactly two faces of its own box
    assert all(c == 2 for c in _closed(faces).values())


def test_window_has_four_frame_members_plus_glazing():
    verts, faces = opening_frame("window", width=1.2, height=1.2, depth=0.1)
    # 5 boxes (L, R, head, sill, glazing) -> 40 verts, 30 quad faces
    assert len(verts) == 40
    assert len(faces) == 30
    assert all(c == 2 for c in _closed(faces).values())


def test_canonical_extents_and_origin():
    verts, _ = opening_frame("window", width=1.2, height=1.5, depth=0.1)
    lo, hi = _bbox(verts)
    # X centered, Y from 0, Z from 0 (bottom-center origin)
    assert np.allclose(lo, [-0.6, 0.0, 0.0])
    assert np.allclose(hi, [0.6, 0.1, 1.5])


def test_door_panel_does_not_protrude_when_depth_below_panel_thickness():
    """depth < panel thickness (0.04m) used to make iy0 negative and iy1 >
    depth, so the panel box protruded past the flush outer wall faces on
    both sides. opening_frame now clamps depth up to the infill thickness."""
    verts, _ = opening_frame("door", width=0.9, height=2.1, depth=0.01)
    a = np.array(verts, dtype=np.float64)
    y = a[:, 1]
    assert y.min() >= -1e-9, "panel must not protrude before the front wall face (y=0)"
    assert y.max() <= 0.04 + 1e-9, "panel must not protrude past the clamped depth"


def test_window_glazing_does_not_protrude_when_depth_below_glazing_thickness():
    """Same clamp, window side: depth < glazing thickness (0.006m)."""
    verts, _ = opening_frame("window", width=1.2, height=1.2, depth=0.001)
    a = np.array(verts, dtype=np.float64)
    y = a[:, 1]
    assert y.min() >= -1e-9, "glazing must not protrude before the front wall face (y=0)"
    assert y.max() <= 0.006 + 1e-9, "glazing must not protrude past the clamped depth"


def test_door_panel_vs_window_sill_at_the_floor():
    # At the floor (z==0), a door's solid panel is thin in depth, adding two
    # interior depth values (iy0, iy1) on top of the jambs' (0, d) -> 4 distinct
    # depths. A window's sill spans the full depth like the jambs -> only (0, d),
    # 2 distinct depths (its glazing starts above the sill, at z=profile).
    dv, _ = opening_frame("door", 0.9, 2.1, 0.1)
    wv, _ = opening_frame("window", 1.2, 1.2, 0.1)
    d = np.array(dv)
    w = np.array(wv)
    d_floor_y = np.unique(np.round(d[d[:, 2] < 1e-9][:, 1], 6))
    w_floor_y = np.unique(np.round(w[w[:, 2] < 1e-9][:, 1], 6))
    assert len(d_floor_y) == 4     # door: jambs (0, d) + panel (iy0, iy1)
    assert len(w_floor_y) == 2     # window: jambs + sill both span the full depth


def test_identical_params_identical_geometry():
    a = opening_frame("door", 0.9, 2.1, 0.1)
    b = opening_frame("door", 0.9, 2.1, 0.1)
    assert a[0] == b[0] and a[1] == b[1]


def test_degenerate_returns_empty():
    assert opening_frame("door", 0.0, 2.1, 0.1) == ([], [])
    assert opening_frame("door", 0.9, 2.1, 0.0) == ([], [])
    assert opening_frame("door", 0.1, 2.1, 0.1) == ([], [])   # width <= 2*profile
    assert opening_frame("window", 1.2, 0.1, 0.1) == ([], []) # height <= 2*profile
