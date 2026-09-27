from __future__ import annotations

import numpy as np
import pytest
from bermake.annotations.draw_plan import (
    _EXT_GAP_PX,
    _EXT_OVERSHOOT_PX,
    _TEXT_GAP_PX,
    _TICK_PX,
    CHAR_W_PX,
    FONT_PX,
    plan_annotation,
)
from bermake.model.annotation import Dimension
from bermake.units import Units


class _FlatCamera:
    """Orthographic-ish stand-in: x,y map straight to pixels, z is depth."""

    def world_to_screen(self, world_xyz, width, height):
        x, y, z = float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2])
        if z < 0.0:
            return None
        return (100.0 + x * 10.0, 200.0 - y * 10.0, 1.0 + z)


def _plan(dim, camera=None):
    return plan_annotation(dim, np.eye(4), camera or _FlatCamera(), 640, 480, Units())


def test_dimension_plan_has_line_ticks_extensions_and_text():
    d = Dimension(1, (0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, -2.0, 0.0))
    plan = _plan(d)
    assert plan is not None
    # 1 dimension line + 2 extension lines + 2 ticks = 5 segments
    assert len(plan.segments_px) == 5
    assert len(plan.texts) == 1
    assert len(plan.hit_boxes) >= 1


def test_dimension_text_is_derived_from_world_distance():
    d = Dimension(1, (0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, -2.0, 0.0))
    plan = _plan(d)
    # 4 m at default (metric) units
    assert "4" in plan.texts[0].text


def test_dimension_text_follows_the_world_transform_scale():
    # a 2x uniform scale makes the same local 4 m measure 8 m in the world
    d = Dimension(1, (0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, -2.0, 0.0))
    m = np.eye(4)
    m[0, 0] = m[1, 1] = m[2, 2] = 2.0
    plan = plan_annotation(d, m, _FlatCamera(), 640, 480, Units())
    assert "8" in plan.texts[0].text


def test_dimension_line_sits_at_the_offset_not_on_the_geometry():
    d = Dimension(1, (0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, -2.0, 0.0))
    plan = _plan(d)
    # geometry is at world y=0 -> screen y=200; offset -2 -> screen y=220
    ys = [seg[1] for seg in plan.segments_px] + [seg[3] for seg in plan.segments_px]
    assert max(ys) > 215.0


def test_point_behind_camera_yields_no_plan():
    class _Behind:
        def world_to_screen(self, world_xyz, width, height):
            return None

    d = Dimension(1, (0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, -2.0, 0.0))
    assert _plan(d, _Behind()) is None


def test_degenerate_zero_length_dimension_yields_no_plan():
    d = Dimension(1, (1.0, 1.0, 0.0), (1.0, 1.0, 0.0), (0.0, -2.0, 0.0))
    assert _plan(d) is None


def test_style_constants_match_the_documented_dimension_anatomy():
    """Pin the constants themselves, independent of any geometry derived from
    them, so a stray edit to any one is caught directly."""
    assert _EXT_GAP_PX == pytest.approx(4.0)
    assert _EXT_OVERSHOOT_PX == pytest.approx(6.0)
    assert _TICK_PX == pytest.approx(6.0)
    assert _TEXT_GAP_PX == pytest.approx(5.0)
    assert FONT_PX == pytest.approx(12.0)
    assert CHAR_W_PX == pytest.approx(6.6)


def test_dimension_plan_pins_exact_dimension_line_and_extension_geometry():
    """Axis-aligned fixture: pin the dimension line and both extension lines.

    Geometry (world y=0) projects to screen y=200; the offset (-2 in world y)
    projects to screen y=220. A wrong _EXT_GAP_PX or _EXT_OVERSHOOT_PX value
    changes these endpoints directly, and a wrong tick angle or length changes
    the tick endpoints directly.
    """
    d = Dimension(1, (0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, -2.0, 0.0))
    plan = _plan(d)
    assert plan is not None
    assert len(plan.segments_px) == 5

    dim_line, ext1, ext2, tick1, tick2 = plan.segments_px

    # dimension line runs along the offset (screen y=220) from x=100 to x=140
    assert dim_line == pytest.approx((100.0, 220.0, 140.0, 220.0))

    # extension line at p1: starts _EXT_GAP_PX off the geometry point (100, 200)
    # and ends _EXT_OVERSHOOT_PX past the dimension line (100, 220)
    assert ext1 == pytest.approx((100.0, 200.0 + _EXT_GAP_PX, 100.0, 220.0 + _EXT_OVERSHOOT_PX))
    # extension line at p2: same gap/overshoot magnitudes, at x=140
    assert ext2 == pytest.approx((140.0, 200.0 + _EXT_GAP_PX, 140.0, 220.0 + _EXT_OVERSHOOT_PX))

    # ticks: 45 degrees off the (axis-aligned) dimension line, half-length
    # _TICK_PX, centred exactly on each dimension-line endpoint
    half = _TICK_PX / np.sqrt(2.0)
    assert tick1 == pytest.approx((100.0 - half, 220.0 - half, 100.0 + half, 220.0 + half))
    assert tick2 == pytest.approx((140.0 - half, 220.0 - half, 140.0 + half, 220.0 + half))


def test_dimension_plan_pins_text_position_and_hit_boxes():
    d = Dimension(1, (0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, -2.0, 0.0))
    plan = _plan(d)
    assert plan is not None

    text = plan.texts[0]
    # centred over the dimension line (x mid-point 120) and _TEXT_GAP_PX further
    # away from the geometry: geometry is above at y=200, dim line at y=220, so
    # "away" continues in the same direction, down to y=220 + _TEXT_GAP_PX
    assert (text.x, text.y) == pytest.approx((120.0, 220.0 + _TEXT_GAP_PX))
    assert text.align == "center"

    assert len(plan.hit_boxes) == 2
    text_box, dim_box = plan.hit_boxes

    w = len(text.text) * CHAR_W_PX
    x0 = text.x - w / 2.0
    assert text_box == pytest.approx((x0, text.y - FONT_PX, x0 + w, text.y))

    pad = 3.0
    x0d, y0d, x1d, y1d = plan.segments_px[0]
    assert dim_box == pytest.approx(
        (min(x0d, x1d) - pad, min(y0d, y1d) - pad, max(x0d, x1d) + pad, max(y0d, y1d) + pad)
    )


def test_dimension_plan_ticks_stay_45_degrees_for_a_diagonal_dimension():
    """A hardcoded 'always diagonally down-right' tick would fail this fixture.

    World direction (3, 4, 0) maps, through the flat camera's 10x scale and
    y-flip, to a screen-space dimension line running (30, -40) pixels -- not
    axis-aligned -- so the tick must genuinely bisect the dimension line and
    its perpendicular, not just reuse the direction that happens to look right
    for an axis-aligned line.
    """
    d = Dimension(1, (0.0, 0.0, 0.0), (3.0, 4.0, 0.0), (0.0, -2.0, 0.0))
    plan = _plan(d)
    assert plan is not None
    assert len(plan.segments_px) == 5

    dim_line = plan.segments_px[0]
    # p1=(0,0,0) -> (100,200); p1+offset -> d1=(100,220)
    # p2=(3,4,0) -> (130,160); p2+offset -> d2=(130,180)
    assert dim_line == pytest.approx((100.0, 220.0, 130.0, 180.0))

    x1, y1, x2, y2 = dim_line
    along = np.array([x2 - x1, y2 - y1])
    along = along / np.linalg.norm(along)
    perp = np.array([-along[1], along[0]])
    tick_dir = along + perp
    tick_dir = tick_dir / np.linalg.norm(tick_dir)

    ends = ((x1, y1), (x2, y2))
    for end, tick_seg in zip(ends, plan.segments_px[3:5], strict=True):
        ax, ay, bx, by = tick_seg
        # tick is centred exactly on its dimension-line endpoint
        assert ((ax + bx) / 2.0, (ay + by) / 2.0) == pytest.approx(end)

        tick_vec = np.array([bx - ax, by - ay])
        tick_len = float(np.linalg.norm(tick_vec))
        assert tick_len == pytest.approx(2.0 * _TICK_PX)

        # generic 45-degree check: holds for any dimension-line orientation,
        # not only an axis-aligned one
        cos_to_along = abs(float(np.dot(tick_vec / tick_len, along)))
        assert cos_to_along == pytest.approx(float(np.sqrt(2.0) / 2.0))

        # exact position: genuinely bisects along/perp, not a fixed
        # screen-space diagonal
        expected_a = np.array(end) - tick_dir * _TICK_PX
        expected_b = np.array(end) + tick_dir * _TICK_PX
        assert (ax, ay) == pytest.approx(tuple(expected_a))
        assert (bx, by) == pytest.approx(tuple(expected_b))


def test_dimension_plan_text_is_centred_and_away_from_geometry_when_diagonal():
    d = Dimension(1, (0.0, 0.0, 0.0), (3.0, 4.0, 0.0), (0.0, -2.0, 0.0))
    plan = _plan(d)
    assert plan is not None

    dim_line = plan.segments_px[0]
    x1, y1, x2, y2 = dim_line
    mid_dim = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
    # geometry: p1=(0,0,0) -> (100,200); p2=(3,4,0) -> (130,160)
    mid_geom = np.array([(100.0 + 130.0) / 2.0, (200.0 + 160.0) / 2.0])

    text = plan.texts[0]
    text_at = np.array([text.x, text.y])
    offset = text_at - mid_dim

    along = np.array([x2 - x1, y2 - y1])
    along = along / np.linalg.norm(along)

    # centred along the dimension line: no component in the "along" direction
    assert float(np.dot(offset, along)) == pytest.approx(0.0, abs=1e-9)
    # sits exactly _TEXT_GAP_PX away from the dimension-line midpoint
    assert float(np.linalg.norm(offset)) == pytest.approx(_TEXT_GAP_PX)
    # on the side away from the geometry: continuing past the dimension line,
    # not back toward it
    assert float(np.dot(offset, mid_dim - mid_geom)) > 0.0


def test_dimension_plan_skips_a_degenerate_extension_line():
    """An offset with no lateral screen component makes a geometry point and
    its offset point project to the same pixel; that endpoint's extension
    line must be silently skipped rather than emit a zero-length segment.
    """
    # offset (0, 0, 5): the flat camera ignores z for x/y, so both p1/p2 and
    # their offset points project to the SAME screen pixel per endpoint ->
    # both extension lines are degenerate and skipped, leaving only the
    # dimension line + 2 ticks (3 segments, below the usual 5).
    d = Dimension(1, (0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 0.0, 5.0))
    plan = _plan(d)
    assert plan is not None
    assert len(plan.segments_px) == 3
    assert plan.segments_px[0] == pytest.approx((100.0, 200.0, 140.0, 200.0))
    assert len(plan.texts) == 1
    assert len(plan.hit_boxes) == 2
