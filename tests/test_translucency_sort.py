"""M7.5a Task 5: back-to-front ordering primitives."""

from __future__ import annotations

import numpy as np
import pytest
from bermake.viewport.translucency import (
    order_back_to_front,
    transform_points,
    triangle_centroids,
    triangle_order_to_vertex_order,
)


def test_triangle_centroids_averages_each_triple():
    positions = np.array(
        [
            [0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0],   # centroid (1, 1, 0)
            [0.0, 0.0, 6.0], [6.0, 0.0, 6.0], [0.0, 6.0, 6.0],   # centroid (2, 2, 6)
        ],
        dtype=np.float64,
    )
    got = triangle_centroids(positions)
    assert got.shape == (2, 3)
    assert got[0] == pytest.approx([1.0, 1.0, 0.0])
    assert got[1] == pytest.approx([2.0, 2.0, 6.0])


def test_transform_points_is_column_vector():
    # 90 degrees about +Z, then translate by (10, 0, 0).
    m = np.array(
        [
            [0.0, -1.0, 0.0, 10.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    got = transform_points(np.array([[1.0, 0.0, 0.0]]), m)
    # column-vector: (1,0,0) rotates to (0,1,0), then translates to (10,1,0).
    # A row-vector reading would give (0,-1,0)+(10,0,0) = (10,-1,0), so the
    # sign of y is the discriminating bit.
    assert got[0] == pytest.approx([10.0, 1.0, 0.0])


def test_order_back_to_front_puts_the_farthest_first():
    pts = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    order = order_back_to_front(pts, np.array([0.0, 0.0, 0.0]))
    # distances 0, 10, 5 -> farthest first is index 1, then 2, then 0
    assert order.tolist() == [1, 2, 0]


def test_order_back_to_front_follows_the_camera():
    pts = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    from_origin = order_back_to_front(pts, np.array([0.0, 0.0, 0.0]))
    from_far_side = order_back_to_front(pts, np.array([20.0, 0.0, 0.0]))
    # Moving the camera to the other side must reverse the order. A sort that
    # ignored the camera, or used a fixed axis, would return the same list.
    assert from_origin.tolist() == list(reversed(from_far_side.tolist()))


def test_order_back_to_front_is_a_permutation():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(17, 3))
    order = order_back_to_front(pts, np.array([1.0, 2.0, 3.0]))
    assert sorted(order.tolist()) == list(range(17))


def test_order_back_to_front_handles_an_empty_input():
    order = order_back_to_front(np.zeros((0, 3)), np.array([0.0, 0.0, 0.0]))
    assert order.shape == (0,)


def test_order_back_to_front_breaks_ties_by_input_order():
    camera = np.array([0.0, 0.0, 0.0])
    pts = np.array(
        [
            [3.0, 4.0, 0.0],    # idx 0: dist2 25 (tied)
            [5.0, 0.0, 0.0],    # idx 1: dist2 25 (tied)
            [0.0, 5.0, 0.0],    # idx 2: dist2 25 (tied)
            [4.0, 3.0, 0.0],    # idx 3: dist2 25 (tied)
            [0.0, -5.0, 0.0],   # idx 4: dist2 25 (tied)
            [10.0, 0.0, 0.0],   # idx 5: dist2 100, strictly farther
            [1.0, 0.0, 0.0],    # idx 6: dist2 1, strictly closer
        ]
    )
    order = order_back_to_front(pts, camera)
    # order_back_to_front uses argsort(..., kind="stable") specifically so
    # that same-distance triangles keep their input order across frames.
    # Five points tied at distance 25 make a coincidental quicksort match
    # implausible; idx 5 and idx 6 confirm real sorting still happens around
    # the tied group.
    assert order.tolist() == [5, 0, 1, 2, 3, 4, 6]


def test_triangle_order_expands_to_vertex_order():
    # swap two triangles -> their three vertices each move together, in order
    got = triangle_order_to_vertex_order(np.array([1, 0], dtype=np.int64))
    assert got.tolist() == [3, 4, 5, 0, 1, 2]


def test_triangle_order_expansion_is_a_permutation():
    got = triangle_order_to_vertex_order(np.array([2, 0, 1], dtype=np.int64))
    # pin exact values (triangle 2 -> vertices 6,7,8, then triangle 0 ->
    # 0,1,2, then triangle 1 -> 3,4,5) so a wrong-order expansion (e.g.
    # expanding reversed(tri_order)) fails, not just a shape/range check.
    assert got.tolist() == [6, 7, 8, 0, 1, 2, 3, 4, 5]
    assert sorted(got.tolist()) == list(range(9))


def test_triangle_order_to_vertex_order_handles_an_empty_input():
    got = triangle_order_to_vertex_order(np.zeros(0, dtype=np.int64))
    assert got.shape == (0,)
