"""Screen-space layout for infinite guide lines and guide points.

A guide has no endpoints, so its plan is the visible portion: clipped against
the camera near plane in 3D, then against the viewport rectangle in 2D.
"""

from __future__ import annotations

import math

import numpy as np
from bermake.units import Units

W, H = 1280, 800


def _camera_at_default():
    from bermake.viewport.camera import Camera

    cam = Camera()
    cam.aspect = W / H
    return cam


def _camera(azimuth_deg, elevation_deg, radius, fov_y_deg, near, target):
    """An orbit camera at the given spherical offset from `target`. Mirrors
    the fix-round-1 review's own probe camera so the regression test below
    reproduces its exact reported coordinates."""
    from bermake.viewport.camera import Camera

    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    offset = np.array(
        [
            radius * math.cos(el) * math.cos(az),
            radius * math.cos(el) * math.sin(az),
            radius * math.sin(el),
        ],
        dtype=np.float64,
    )
    cam = Camera()
    cam.target = np.asarray(target, dtype=np.float64)
    cam.position = cam.target + offset
    cam.aspect = W / H
    cam.fov_y_deg = fov_y_deg
    cam.near = near
    return cam


def _plan(annotation, cam):
    from bermake.annotations.draw_plan import plan_annotation

    return plan_annotation(annotation, None, cam, W, H, Units())


def _inside(x, y, pad=2.0):
    return -pad <= x <= W + pad and -pad <= y <= H + pad


def test_a_guide_through_the_view_lays_out_one_segment_inside_the_viewport():
    from bermake.model.annotation import Guide

    plan = _plan(Guide(1, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), _camera_at_default())
    assert plan is not None
    assert len(plan.segments_px) == 1
    x1, y1, x2, y2 = plan.segments_px[0]
    assert _inside(x1, y1) and _inside(x2, y2)
    assert np.hypot(x2 - x1, y2 - y1) > 10.0, "a visible guide must not collapse to a point"


def test_a_guide_entirely_behind_the_camera_lays_out_nothing():
    from bermake.model.annotation import Guide

    cam = _camera_at_default()
    origin, direction = cam.ray_from_screen(W / 2, H / 2, W, H)
    behind = np.asarray(origin, dtype=np.float64) - np.asarray(direction, dtype=np.float64) * 50.0
    # A guide through a point well behind the eye, running perpendicular to the view.
    perp = np.cross(np.asarray(direction, dtype=np.float64), np.array([0.0, 0.0, 1.0]))

    plan = _plan(Guide(1, tuple(behind), tuple(perp)), cam)
    assert plan is None or plan.segments_px == []


def test_a_guide_nearly_edge_on_to_the_view_does_not_produce_runaway_coordinates():
    """The failure mode of projecting two far-apart samples instead of clipping."""
    from bermake.model.annotation import Guide

    cam = _camera_at_default()
    _origin, direction = cam.ray_from_screen(W / 2, H / 2, W, H)
    d = np.asarray(direction, dtype=np.float64)
    d = d / float(np.linalg.norm(d))
    # Tilt a hair off the view direction so the guide runs almost straight away.
    almost = d + np.array([0.0, 0.0, 1e-3])

    plan = _plan(Guide(1, (0.0, 0.0, 0.0), tuple(almost)), cam)
    if plan is None or not plan.segments_px:
        return  # fully clipped is a valid answer
    for x1, y1, x2, y2 in plan.segments_px:
        for v in (x1, y1, x2, y2):
            assert abs(v) < 1e5, f"runaway screen coordinate {v}"


def test_a_guide_point_lays_out_a_small_cross():
    from bermake.model.annotation import GuidePoint

    plan = _plan(GuidePoint(1, (0.0, 0.0, 0.0)), _camera_at_default())
    assert plan is not None
    assert len(plan.segments_px) == 2, "a guide point draws as a two-stroke cross"


def test_a_guide_point_far_outside_the_viewport_lays_out_nothing():
    """Regression: projecting a point does not by itself exclude points that
    are in front of the camera but well off to the side. Unlike a guide's
    infinite line (clipped to the viewport by _clip_segment_to_viewport), an
    unguarded guide point would otherwise emit a cross at wildly
    out-of-range pixel coordinates instead of nothing."""
    from bermake.model.annotation import GuidePoint

    cam = _camera_at_default()
    origin, direction = cam.ray_from_screen(50 * W, H / 2, W, H)
    far_off_screen = (
        np.asarray(origin, dtype=np.float64) + np.asarray(direction, dtype=np.float64) * 10.0
    )

    plan = _plan(GuidePoint(1, tuple(far_off_screen)), cam)
    assert plan is None or plan.segments_px == []


def test_a_distant_guide_does_not_truncate_in_mid_viewport():
    """Fix round 1, Important finding 2: a fixed _VISIBLE_SPAN truncates the
    drawn segment well short of where the true infinite line stops being
    visible, once the guide is far enough from the eye. Reviewer's concrete
    case: azimuth 133, elevation -20, radius 5, fov 25, guide origin
    (300, 300, -50), direction (0.692, -0.692, 0.208). Before the fix the
    drawn end was (500.1, 662.5); the true line continues visibly to roughly
    (577.8, 654.1) -- the span must reach far enough that the drawn endpoint
    lands within a couple of pixels of that, not stop 78px short."""
    from bermake.model.annotation import Guide

    cam = _camera(133, -20, 5, 25, 0.01, (0.0, 0.0, 0.5))
    guide = Guide(1, (300.0, 300.0, -50.0), (0.692, -0.692, 0.208))
    plan = _plan(guide, cam)
    assert plan is not None and len(plan.segments_px) == 1
    x1, y1, x2, y2 = plan.segments_px[0]
    far_end = (x1, y1) if math.hypot(x1 - 0.0, y1 - 716.0) > 1.0 else (x2, y2)
    # Whichever endpoint is not the viewport-edge clip should sit close to the
    # true line's asymptotic direction, not stop far short of it.
    truncated_end = (500.1, 662.5)
    true_ish_end = (577.8, 654.1)
    dist_to_truncated = math.hypot(far_end[0] - truncated_end[0], far_end[1] - truncated_end[1])
    dist_to_true = math.hypot(far_end[0] - true_ish_end[0], far_end[1] - true_ish_end[1])
    assert dist_to_true < dist_to_truncated
    assert dist_to_true < 5.0, f"endpoint {far_end} is not close to the true line's extent"


def test_a_guide_between_the_eye_and_the_near_plane_lays_out_nothing():
    """Fix round 1, Important finding 3: a guide can be technically in front
    of the eye (depth > 0, so world_to_screen's own behind-eye check does not
    reject it) yet still sit inside the near plane (depth < camera.near).
    The parallel branch of _clip_line_to_near_plane (b ~= 0) has its own
    `a <= 0` rejection for exactly this case; deleting it would let such a
    guide draw as a full-width line where no geometry could ever appear."""
    from bermake.model.annotation import Guide

    cam = _camera_at_default()
    cam.near = 2.0
    eye = np.asarray(cam.position, dtype=np.float64)
    fwd = np.asarray(cam.target, dtype=np.float64) - eye
    fwd = fwd / float(np.linalg.norm(fwd))
    up = np.asarray(cam.up, dtype=np.float64)
    perp = np.cross(fwd, up)
    perp = perp / float(np.linalg.norm(perp))

    origin = eye + fwd * 1.0  # depth 1.0, inside the near plane (near=2.0)
    plan = _plan(Guide(1, tuple(origin), tuple(perp)), cam)
    assert plan is None or plan.segments_px == []
