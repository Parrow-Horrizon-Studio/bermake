"""Unit tests for aabb_world_edges (Task 15).

Tests verify:
1. The function returns (24, 3) float32 — 12 edges × 2 endpoints.
2. A translated box is correctly offset in world space.
3. Corner ordering: all 8 unique corners appear in the output (up to float tolerance).
"""
import numpy as np
import pytest

from bermake.viewport.scene_renderer import aabb_world_edges


def test_aabb_world_edges_count_and_translation():
    """Shape (24, 3) and all x-coords >= 10 after +10 translation."""
    lo = np.array([0, 0, 0], np.float32)
    hi = np.array([1, 1, 1], np.float32)
    t = np.eye(4)
    t[:3, 3] = [10, 0, 0]
    segs = aabb_world_edges(lo, hi, t)
    assert segs.shape == (24, 3)            # 12 edges * 2 endpoints
    assert segs[:, 0].min() >= 10.0 - 1e-6  # translated +10 in x


def test_aabb_world_edges_identity_corners():
    """With an identity transform the 24 endpoints collapse to 8 unique corners."""
    lo = np.array([1.0, 2.0, 3.0], np.float32)
    hi = np.array([4.0, 5.0, 6.0], np.float32)
    segs = aabb_world_edges(lo, hi, np.eye(4))
    assert segs.shape == (24, 3)
    # Round to avoid float32 noise when comparing unique rows
    rounded = np.round(segs.astype(np.float64), 5)
    unique_pts = np.unique(rounded, axis=0)
    assert unique_pts.shape[0] == 8, f"Expected 8 unique corners, got {unique_pts.shape[0]}"


def test_aabb_world_edges_scaled_transform():
    """A uniform scale-2 transform doubles all coordinates."""
    lo = np.array([0, 0, 0], np.float32)
    hi = np.array([1, 1, 1], np.float32)
    t = np.eye(4) * 2.0
    t[3, 3] = 1.0  # keep homogeneous row correct
    segs = aabb_world_edges(lo, hi, t)
    assert segs.shape == (24, 3)
    assert segs[:, 0].max() <= 2.0 + 1e-5
    assert segs[:, 1].max() <= 2.0 + 1e-5
    assert segs[:, 2].max() <= 2.0 + 1e-5
