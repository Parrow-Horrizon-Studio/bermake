"""Offset polygon maths and its collapse clamp (M7.4 Task 4)."""

from __future__ import annotations

import numpy as np
import pytest
from bermake.tools.sweep_support import offset_polygon

Z = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _rect(w, h):
    return np.array(
        [[0.0, 0.0, 0.0], [w, 0.0, 0.0], [w, h, 0.0], [0.0, h, 0.0]], dtype=np.float64
    )


def test_outward_offset_keeps_edges_parallel():
    pts, clamped = offset_polygon(_rect(4.0, 2.0), Z, -0.5)
    # Outward by 0.5 on every side: 5 x 3 centred on the original.
    assert clamped == pytest.approx(-0.5)
    assert pts[:, 0].min() == pytest.approx(-0.5)
    assert pts[:, 0].max() == pytest.approx(4.5)
    assert pts[:, 1].min() == pytest.approx(-0.5)
    assert pts[:, 1].max() == pytest.approx(2.5)


def test_inward_offset_shrinks_by_the_distance_on_every_side():
    pts, clamped = offset_polygon(_rect(4.0, 2.0), Z, 0.25)
    assert clamped == pytest.approx(0.25)
    assert pts[:, 0].min() == pytest.approx(0.25)
    assert pts[:, 0].max() == pytest.approx(3.75)


def test_the_clamp_is_half_the_short_side_on_a_rectangle():
    # A 4x2 rectangle collapses to a line at 1.0, so asking for more returns
    # the clamped value, not the requested one -- and (fix round, Finding 1)
    # the clamp must land STRICTLY short of 1.0, not exactly at it: a
    # distance of exactly 1.0 collapses the two short edges to zero length,
    # which is the degenerate case this fix exists to avoid. The back-off is
    # a small fraction of the limit, so it stays close to 1.0 without ever
    # reaching it.
    _pts, clamped = offset_polygon(_rect(4.0, 2.0), Z, 5.0)
    assert clamped < 1.0
    assert clamped == pytest.approx(1.0, rel=1e-2)


def test_clamping_is_reported_so_the_caller_need_not_recompute_it():
    _pts, clamped = offset_polygon(_rect(4.0, 2.0), Z, 5.0)
    assert clamped < 5.0


def test_a_concave_loop_still_yields_a_simple_polygon():
    # An L shape. Its analytic per-edge limit is optimistic: the reflex
    # corner self-intersects before any single edge collapses, so the
    # binary-search stage has to catch it.
    l_shape = np.array(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 4.0, 0.0],
            [0.0, 4.0, 0.0],
        ],
        dtype=np.float64,
    )
    pts, clamped = offset_polygon(l_shape, Z, 3.0)
    assert clamped <= 3.0
    assert _is_simple(pts)


def _is_simple(pts) -> bool:
    """No two non-adjacent edges of the closed polygon intersect."""
    n = len(pts)
    for i in range(n):
        a1, a2 = pts[i][:2], pts[(i + 1) % n][:2]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or j == (i + 1) % n:
                continue
            b1, b2 = pts[j][:2], pts[(j + 1) % n][:2]
            if _segments_cross(a1, a2, b1, b2):
                return False
    return True


def _segments_cross(p1, p2, p3, p4, tol: float = 1e-9) -> bool:
    """Parametric segment-segment intersection.

    Deliberately a different algorithm from the implementation's own
    simplicity check (`sweep_support._is_simple_offset`, a four-orientation
    sign comparison): this solves ``p1 + t*(p2-p1) == p3 + u*(p4-p3)`` for
    the two segment parameters ``t`` and ``u`` directly and reports a
    crossing when both lie in ``[0, 1]`` (within `tol`). Using a genuinely
    different method -- rather than the same sign-comparison idea reapplied
    -- means this test can catch a bug in that shared idea, not only a bug
    in how the implementation applies it.

    Near-parallel segments (the 2x2 system's determinant is close to zero)
    are handled explicitly rather than falling through to a division: a
    collinear, overlapping pair counts as crossing; anything else parallel
    does not.
    """
    r = p2 - p1
    s = p4 - p3
    denom = r[0] * s[1] - r[1] * s[0]
    diff = p3 - p1

    if abs(denom) < tol:
        # Parallel or nearly parallel: only a collinear overlap counts.
        cross_diff_r = diff[0] * r[1] - diff[1] * r[0]
        if abs(cross_diff_r) > tol:
            return False  # parallel but offset apart -- can never meet
        len_sq = r[0] * r[0] + r[1] * r[1]
        if len_sq < tol:
            return False  # p1 == p2, degenerate segment
        t3 = (diff[0] * r[0] + diff[1] * r[1]) / len_sq
        t4 = ((p4 - p1)[0] * r[0] + (p4 - p1)[1] * r[1]) / len_sq
        lo, hi = min(t3, t4), max(t3, t4)
        return hi > tol and lo < 1.0 - tol

    t = (diff[0] * s[1] - diff[1] * s[0]) / denom
    u = (diff[0] * r[1] - diff[1] * r[0]) / denom
    return -tol <= t <= 1.0 + tol and -tol <= u <= 1.0 + tol
