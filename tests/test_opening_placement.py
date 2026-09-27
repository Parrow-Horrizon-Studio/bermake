from __future__ import annotations

import numpy as np
from bermake.geometry.opening import opening_placement_transform


def _apply(m, pt):
    return (m @ np.array([pt[0], pt[1], pt[2], 1.0]))[:3]


def test_wall_facing_plus_x():
    # A wall face at x=3 whose outward (viewer-facing) normal is +X.
    m = opening_placement_transform(point=(3.0, 1.0, 1.2), normal=(1.0, 0.0, 0.0), sill=0.9)
    assert m is not None
    # canonical origin (bottom-center, outer face) -> cursor x,y at height=sill
    assert np.allclose(_apply(m, (0.0, 0.0, 0.0)), [3.0, 1.0, 0.9])
    # canonical +Z (up) stays up
    assert np.allclose((m[:3, :3] @ np.array([0.0, 0.0, 1.0])), [0.0, 0.0, 1.0])
    # canonical +Y (depth) points INTO the wall (-X here)
    assert np.allclose((m[:3, :3] @ np.array([0.0, 1.0, 0.0])), [-1.0, 0.0, 0.0])
    # proper rotation (no mirroring)
    assert np.isclose(np.linalg.det(m[:3, :3]), 1.0)


def test_horizontalizes_a_tilted_normal():
    # a normal with a small +Z tilt still yields an upright opening
    m = opening_placement_transform((0.0, 0.0, 0.0), (1.0, 0.0, 0.2), sill=0.0)
    assert m is not None
    assert np.allclose((m[:3, :3] @ np.array([0.0, 0.0, 1.0])), [0.0, 0.0, 1.0])
    out_into_wall = m[:3, :3] @ np.array([0.0, 1.0, 0.0])
    assert np.isclose(out_into_wall[2], 0.0)        # depth axis horizontal


def test_horizontal_face_returns_none():
    assert opening_placement_transform((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.0) is None
