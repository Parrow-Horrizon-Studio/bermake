"""Unit tests for curve generators (pure 2D, no Qt)."""

from __future__ import annotations

import numpy as np

from bermake.geometry import circle, polygon


def test_circle_has_segment_count_points_on_radius():
    pts = circle(radius=2.0, segments=24)
    assert pts.shape == (24, 2)
    dists = np.linalg.norm(pts, axis=1)
    assert np.allclose(dists, 2.0, atol=1e-9)


def test_circle_first_vertex_follows_start_angle():
    pts = circle(radius=1.0, segments=24, start_angle=np.pi / 2)
    assert np.allclose(pts[0], [0.0, 1.0], atol=1e-9)


def test_polygon_is_inscribed_and_regular():
    pts = polygon(radius=3.0, sides=6)
    assert pts.shape == (6, 2)
    assert np.allclose(np.linalg.norm(pts, axis=1), 3.0, atol=1e-9)
    edges = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    assert np.allclose(edges, edges[0], atol=1e-9)


def test_ring_winding_is_ccw():
    pts = polygon(radius=1.0, sides=5)
    x, y = pts[:, 0], pts[:, 1]
    area2 = float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    assert area2 > 0.0


def test_arc_semicircle_lies_on_circle():
    from bermake.geometry import arc_2pt

    pts = arc_2pt(np.array([-1.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0]), segments=12)
    assert pts.shape == (13, 2)
    assert np.allclose(pts[0], [-1.0, 0.0], atol=1e-9)
    assert np.allclose(pts[-1], [1.0, 0.0], atol=1e-9)
    assert np.allclose(np.linalg.norm(pts, axis=1), 1.0, atol=1e-9)
    assert np.any(np.all(np.isclose(pts, [0.0, 1.0], atol=1e-9), axis=1))


def test_arc_general_samples_on_common_circle():
    from bermake.geometry import arc_2pt

    start, end, bulge = np.array([0.0, 0.0]), np.array([2.0, 0.0]), np.array([1.0, 0.5])
    pts = arc_2pt(start, end, bulge, segments=16)
    center = np.array([1.0, -0.75])
    assert np.allclose(np.linalg.norm(pts - center, axis=1), 1.25, atol=1e-9)
    assert np.allclose(pts[0], start, atol=1e-9)
    assert np.allclose(pts[-1], end, atol=1e-9)


def test_arc_flat_bulge_returns_straight_chord():
    from bermake.geometry import arc_2pt

    pts = arc_2pt(np.array([0.0, 0.0]), np.array([2.0, 0.0]), np.array([1.0, 0.0]))
    assert pts.shape == (2, 2)
    assert np.allclose(pts, [[0.0, 0.0], [2.0, 0.0]])


def test_arc_degenerate_chord_returns_single_point():
    from bermake.geometry import arc_2pt

    pts = arc_2pt(np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([2.0, 2.0]))
    assert pts.shape == (1, 2)


def test_semicircle_snap_pulls_near_semicircle_exact():
    from bermake.geometry import semicircle_snap

    start, end = np.array([-1.0, 0.0]), np.array([1.0, 0.0])
    snapped = semicircle_snap(start, end, np.array([0.0, 0.97]))
    assert np.allclose(snapped, [0.0, 1.0], atol=1e-9)
    far = semicircle_snap(start, end, np.array([0.0, 0.4]))
    assert np.allclose(far, [0.0, 0.4])


def test_arc_semicircle_negative_sagitta_bows_below_chord():
    from bermake.geometry import arc_2pt

    pts = arc_2pt(np.array([-1.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, -1.0]), segments=12)
    assert pts.shape == (13, 2)
    assert np.allclose(np.linalg.norm(pts, axis=1), 1.0, atol=1e-9)
    assert np.any(np.all(np.isclose(pts, [0.0, -1.0], atol=1e-9), axis=1))
    assert np.all(pts[:, 1] <= 1e-9)  # arc bows below the chord


def test_semicircle_snap_negative_sagitta_snaps_below():
    from bermake.geometry import semicircle_snap

    start, end = np.array([-1.0, 0.0]), np.array([1.0, 0.0])
    snapped = semicircle_snap(start, end, np.array([0.0, -0.97]))
    assert np.allclose(snapped, [0.0, -1.0], atol=1e-9)
