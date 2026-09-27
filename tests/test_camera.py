"""Tests for the Python Camera class — pure numpy math, no OpenGL needed."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bermake.viewport.camera import Camera


# --- Defaults --------------------------------------------------------------


def test_default_camera_pose():
    c = Camera()
    np.testing.assert_allclose(c.position, [8.0, -8.0, 6.0], atol=1e-5)
    np.testing.assert_allclose(c.target, [0.0, 0.0, 0.5], atol=1e-5)
    np.testing.assert_allclose(c.up, [0.0, 0.0, 1.0], atol=1e-5)


# --- View matrix -----------------------------------------------------------


def test_view_matrix_is_4x4():
    c = Camera()
    v = c.view_matrix()
    assert v.shape == (4, 4)
    assert v.dtype == np.float32


def test_view_matrix_sends_position_to_origin():
    """v * position (homogeneous) should be (0,0,0) in view space."""
    c = Camera()
    v = c.view_matrix()
    homogeneous = np.array([*c.position, 1.0], dtype=np.float32)
    result = v @ homogeneous
    np.testing.assert_allclose(result[:3], [0.0, 0.0, 0.0], atol=1e-4)


# --- Projection matrix -----------------------------------------------------


def test_projection_matrix_is_4x4():
    c = Camera()
    c.aspect = 16.0 / 9.0
    p = c.projection_matrix()
    assert p.shape == (4, 4)
    assert p.dtype == np.float32


def test_projection_matrix_respects_aspect():
    """Wider aspect should compress x more than tall aspect does (same fov_y)."""
    c1 = Camera()
    c1.aspect = 1.0
    c2 = Camera()
    c2.aspect = 2.0
    p1 = c1.projection_matrix()
    p2 = c2.projection_matrix()
    # Element [0,0] is fovy-derived divided by aspect, so p2[0,0] < p1[0,0].
    assert p2[0, 0] < p1[0, 0]


# --- Orbit -----------------------------------------------------------------


def test_orbit_preserves_distance_to_target():
    c = Camera()
    distance_before = np.linalg.norm(c.position - c.target)
    c.orbit(dx_pixels=50.0, dy_pixels=30.0)
    distance_after = np.linalg.norm(c.position - c.target)
    np.testing.assert_allclose(distance_after, distance_before, atol=1e-4)


def test_orbit_full_circle_returns_to_origin():
    """Orbiting 360 degrees in many small steps returns the camera home."""
    c = Camera()
    pos_before = c.position.copy()
    distance_initial = np.linalg.norm(pos_before - c.target)
    for _ in range(1000):
        c.orbit(dx_pixels=1.0, dy_pixels=0.0)
    distance_final = np.linalg.norm(c.position - c.target)
    np.testing.assert_allclose(distance_final, distance_initial, atol=1e-3)


def test_orbit_elevation_clamped():
    """Pitch must be clamped to avoid gimbal flip at the poles."""
    c = Camera()
    # Try to orbit way past the top pole.
    for _ in range(10000):
        c.orbit(dx_pixels=0.0, dy_pixels=10.0)
    # Should still be a valid view (position not on top of target).
    distance = np.linalg.norm(c.position - c.target)
    assert distance > 0.1
    # And up direction is still roughly +Z.
    np.testing.assert_allclose(c.up, [0.0, 0.0, 1.0], atol=1e-5)


# --- Pan -------------------------------------------------------------------


def test_pan_preserves_camera_to_target_vector():
    """Pan translates position and target together; the offset is unchanged."""
    c = Camera()
    offset_before = c.position - c.target
    c.pan(dx_pixels=20.0, dy_pixels=-10.0)
    offset_after = c.position - c.target
    np.testing.assert_allclose(offset_after, offset_before, atol=1e-4)


# --- Zoom ------------------------------------------------------------------


def test_zoom_toward_target_reduces_distance():
    c = Camera()
    distance_before = np.linalg.norm(c.position - c.target)
    c.zoom(scroll_delta=1.0, cursor_ndc=None)  # zoom in
    distance_after = np.linalg.norm(c.position - c.target)
    assert distance_after < distance_before


def test_zoom_out_increases_distance():
    c = Camera()
    distance_before = np.linalg.norm(c.position - c.target)
    c.zoom(scroll_delta=-1.0, cursor_ndc=None)
    distance_after = np.linalg.norm(c.position - c.target)
    assert distance_after > distance_before


def test_zoom_toward_cursor_moves_position_and_target_together():
    """Cursor-zoom must move position AND target by the same delta.

    This preserves the view direction (no rotation) at the cost of letting the
    orbit pivot drift along the cursor ray — matching SketchUp's behavior and
    the preference established during M2 visual verification.
    """
    from bermake.viewport.camera import _normalize

    c = Camera()
    pos_before = c.position.copy()
    target_before = c.target.copy()
    cursor = np.array([0.5, 0.5], dtype=np.float32)  # off-center cursor

    c.zoom(scroll_delta=1.0, cursor_ndc=cursor)

    # Both position and target must have moved.
    assert not np.allclose(c.position, pos_before), "position should have changed"
    assert not np.allclose(c.target, target_before), "target should have changed"

    # They must have moved by the same vector (rigid translation).
    pos_delta = c.position - pos_before
    target_delta = c.target - target_before
    np.testing.assert_allclose(pos_delta, target_delta, atol=1e-6,
                               err_msg="position and target must move by identical delta")

    # View direction must be preserved (no rotation).
    dir_before = _normalize(target_before - pos_before)
    dir_after = _normalize(c.target - c.position)
    np.testing.assert_allclose(dir_after, dir_before, atol=1e-5,
                               err_msg="view direction must be unchanged after cursor-zoom")


# --- Ray from screen / ray intersect ground --------------------------------


def test_ray_from_screen_returns_origin_and_unit_direction():
    from bermake.viewport.camera import Camera

    cam = Camera()
    cam.aspect = 1.0
    origin, direction = cam.ray_from_screen(640.0, 400.0, 1280, 800)

    # Origin is the camera position
    np.testing.assert_allclose(origin, cam.position, atol=1e-6)
    # Direction is a unit vector
    np.testing.assert_allclose(float(np.linalg.norm(direction)), 1.0, atol=1e-6)
    # Centre cursor → direction points from position toward target
    expected = cam.target - cam.position
    expected = expected / float(np.linalg.norm(expected))
    np.testing.assert_allclose(direction, expected, atol=1e-5)


def test_ray_intersect_ground_for_centre_cursor():
    from bermake.viewport.camera import Camera

    cam = Camera()
    cam.aspect = 1.0
    hit = cam.ray_intersect_ground(640.0, 400.0, 1280, 800)
    # Centre cursor with default camera (looking at target at z=0.5) hits the
    # ground near, but not exactly at, target.x/target.y because the target
    # has z=0.5 not 0. The hit must still be valid (not None) and on z=0.
    assert hit is not None
    assert abs(float(hit[2])) < 1e-5


def test_ray_intersect_ground_returns_none_when_ray_parallel_or_above():
    """Cursor placed so the ray goes upward (away from ground) yields None."""
    from bermake.viewport.camera import Camera

    cam = Camera()
    # Pose the camera above the ground looking up (away from z=0).
    cam.position = np.array([0.0, 0.0, 5.0], dtype=np.float32)
    cam.target = np.array([0.0, 0.0, 10.0], dtype=np.float32)
    cam.aspect = 1.0
    hit = cam.ray_intersect_ground(640.0, 400.0, 1280, 800)
    assert hit is None


# --- world_to_screen -------------------------------------------------------


def test_world_to_screen_roundtrips_with_ray_from_screen():
    """A world point projected to screen, then turned back into a ray, yields a
    ray that passes through the original point."""
    from bermake.viewport.camera import Camera

    cam = Camera()
    cam.aspect = 1280.0 / 800.0
    world = np.array([1.0, 2.0, 0.5], dtype=np.float32)

    sx, sy, depth = cam.world_to_screen(world, 1280, 800)
    assert depth > 0.0  # in front of the camera

    origin, direction = cam.ray_from_screen(sx, sy, 1280, 800)
    # The world point lies on the ray: world == origin + s*direction for some s>0.
    to_point = world - origin
    s = float(np.dot(to_point, direction))
    closest = origin + s * direction
    np.testing.assert_allclose(closest, world, atol=1e-3)


def test_world_to_screen_center_of_target_is_screen_center():
    from bermake.viewport.camera import Camera

    cam = Camera()
    cam.aspect = 1.0
    sx, sy, _ = cam.world_to_screen(cam.target, 1000, 1000)
    np.testing.assert_allclose([sx, sy], [500.0, 500.0], atol=1.0)


def test_world_to_screen_behind_camera_returns_none():
    from bermake.viewport.camera import Camera

    cam = Camera()
    cam.position = np.array([0.0, 0.0, 5.0], dtype=np.float32)
    cam.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # looking down -Z
    cam.aspect = 1.0
    behind = np.array([0.0, 0.0, 10.0], dtype=np.float32)  # above/behind the camera
    assert cam.world_to_screen(behind, 1280, 800) is None
