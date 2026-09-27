"""Tests for the Scale tool's cursor projection (#57, #47).

`_cursor_world` used to intersect the cursor ray with a plane whose normal
was hardcoded to world-Z. That's wrong in any non-top-down view (#57) and
also wrong for single-axis grips, which should be dragged along their own
axis line rather than snapped to a plane at all (#47).

Fixture note: the plane tests below set `_anchor` but not `_active`, and
`_grips_are_local` defaults to True on a fresh ScaleTool. Since these tools
are never `activate()`d, we set `_grips_are_local = False` so the anchor is
treated as already-world and `_cursor_world` doesn't have to consult
`self._world_transform()` (which depends on `self._model`). This keeps the
plane tests focused purely on the plane-normal bug.
"""

import numpy as np
from bermake.tools.scale_tool import ScaleTool
from bermake.tools.transform_support import grip_specs
from bermake.viewport.camera import Camera
from PySide6.QtCore import QPointF


class _Ev:
    """Minimal stand-in for QMouseEvent: only position() is used by _cursor_world."""

    def __init__(self, x, y):
        self._p = QPointF(x, y)

    def position(self):
        return self._p


def _tool_with_camera(cam):
    tool = ScaleTool()
    tool._camera = cam
    tool._size_provider = lambda: (800, 600)
    tool._grips_are_local = False
    return tool


def test_cursor_plane_is_camera_facing_not_world_z():
    # Camera looking horizontally along -X: its view direction is world-X, so a
    # correct camera-facing plane is X-normal. A world-Z plane would be edge-on
    # to this ray and give a wildly different (or unstable) answer.
    cam = Camera()
    cam.position[:] = (10.0, 0.0, 0.0)
    cam.target[:] = (0.0, 0.0, 0.0)
    cam.aspect = 800.0 / 600.0
    tool = _tool_with_camera(cam)
    tool._anchor = np.array([0.0, 0.0, 0.0], np.float32)

    pt = tool._cursor_world(_Ev(400.0, 300.0))
    assert pt is not None
    # The returned point must lie on the plane through the origin with the
    # camera-facing normal (+X): its X component is ~0.
    assert abs(float(pt[0])) < 1e-4


def test_cursor_plane_normal_matches_analytic_camera_facing_intersection():
    # The two tests above put the anchor at camera.target and read the cursor
    # from screen-centre. ray_from_screen's screen-centre direction is, by
    # construction, normalize(target - position) -- i.e. the ray always
    # passes exactly *through* the target. So when anchor == target, the
    # ray/plane intersection lands on the anchor itself no matter which
    # (non-perpendicular) normal is used: they cannot distinguish the correct
    # camera-facing normal from any other plausible-but-wrong one (camera.up,
    # world-Z, a sign flip, ...).
    #
    # To make the normal actually matter we need the ray to MISS the anchor,
    # which requires both an oblique camera (view direction not axis-aligned)
    # and an anchor offset from camera.target. We then compare the tool's
    # answer against the analytically-computed intersection for the one
    # correct normal, n = normalize(camera.position - camera.target).
    cam = Camera()
    cam.position[:] = (10.0, 10.0, 10.0)
    cam.target[:] = (0.0, 0.0, 0.0)
    cam.aspect = 800.0 / 600.0
    tool = _tool_with_camera(cam)
    p0 = np.array([2.0, 0.0, 0.0], np.float64)
    tool._anchor = p0.astype(np.float32)

    pt = tool._cursor_world(_Ev(400.0, 300.0))
    assert pt is not None

    # Reference intersection, computed independently from first principles
    # (screen-centre ray: origin = camera.position, direction = normalize
    # (target - position); plane through p0 with normal n).
    origin = np.array([10.0, 10.0, 10.0], np.float64)
    direction = np.array([0.0, 0.0, 0.0], np.float64) - origin
    direction /= np.linalg.norm(direction)
    n_correct = np.array([10.0, 10.0, 10.0], np.float64)
    n_correct /= np.linalg.norm(n_correct)
    t_correct = np.dot(p0 - origin, n_correct) / np.dot(direction, n_correct)
    expected = origin + t_correct * direction

    assert np.allclose(pt, expected, atol=1e-4), f"expected {expected}, got {pt}"

    # Sanity check that this scenario is actually discriminating: a plausible
    # wrong normal (world-Z, which would also be "camera.up" for this camera)
    # gives a materially different point, so a wrong implementation could not
    # accidentally satisfy the assertion above.
    n_wrong = np.array([0.0, 0.0, 1.0])
    t_wrong = np.dot(p0 - origin, n_wrong) / np.dot(direction, n_wrong)
    wrong_point = origin + t_wrong * direction
    assert not np.allclose(expected, wrong_point, atol=1e-2), (
        "test scenario is degenerate: correct and wrong normals agree"
    )


def test_top_down_view_still_behaves_like_before():
    # Regression guard for the common case: with the camera overhead the
    # camera-facing normal IS world-Z, so behavior must be unchanged.
    cam = Camera()
    cam.position[:] = (0.0, 0.0, 10.0)
    cam.target[:] = (0.0, 0.0, 0.0)
    cam.aspect = 800.0 / 600.0
    tool = _tool_with_camera(cam)
    tool._anchor = np.array([0.0, 0.0, 0.0], np.float32)

    pt = tool._cursor_world(_Ev(400.0, 300.0))
    assert pt is not None
    assert abs(float(pt[2])) < 1e-4


def test_single_axis_grip_projects_onto_its_axis_line_not_a_plane():
    # Real single-axis grip from grip_specs: a face of [0,2]^3 driving only
    # axis 0 (the -X face, at world [0, 1, 1] -- first face-grip yielded by
    # grip_specs's iteration order).
    lo = np.array([0.0, 0.0, 0.0], np.float32)
    hi = np.array([2.0, 2.0, 2.0], np.float32)
    grip = next(g for g in grip_specs(lo, hi) if g.axes == (0,))
    assert np.allclose(grip.position, [0.0, 1.0, 1.0])

    # Top-down camera looking straight down -Z through screen-centre: the ray
    # is the vertical line x=0, y=0. A camera-facing (or world-Z) plane through
    # the grip would hit that ray at world [0, 0, 1] -- off the axis line's
    # Y=1 -- so a plane-based answer does NOT reproduce the grip's Y, Z.
    # The correct axis-projected answer lies on the line
    # (0, 1, 1) + t*(1, 0, 0), whose Y and Z are always 1, 1 regardless of t.
    cam = Camera()
    cam.position[:] = (0.0, 0.0, 10.0)
    cam.target[:] = (0.0, 0.0, 0.0)
    cam.aspect = 800.0 / 600.0
    tool = _tool_with_camera(cam)
    tool._active = grip

    pt = tool._cursor_world(_Ev(400.0, 300.0))
    assert pt is not None
    # _grips_are_local is False, so _cursor_world returns the world point
    # directly (no local conversion) -- assert on world coordinates.
    assert abs(float(pt[1]) - float(grip.position[1])) < 1e-4
    assert abs(float(pt[2]) - float(grip.position[2])) < 1e-4
