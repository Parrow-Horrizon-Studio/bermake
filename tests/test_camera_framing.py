"""Zoom Extents framing math (M7.2 Task 7). Pure numpy -- no Qt, no model."""

from __future__ import annotations

import math

import numpy as np
import pytest
from bermake.viewport.camera_framing import MIN_RADIUS, frame_bounds

_DIR = np.array([0.0, 1.0, -1.0], dtype=np.float64)


def _view_matrix(position: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Mirrors `Camera.view_matrix` (python/bermake/viewport/camera.py) in
    float64, so this test exercises the same right-handed look-at convention
    the real viewport uses instead of inventing its own."""
    forward = (target - position) / np.linalg.norm(target - position)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    cam_up = np.cross(right, forward)

    m = np.eye(4, dtype=np.float64)
    m[0, 0:3] = right
    m[1, 0:3] = cam_up
    m[2, 0:3] = -forward
    m[0, 3] = -float(np.dot(right, position))
    m[1, 3] = -float(np.dot(cam_up, position))
    m[2, 3] = float(np.dot(forward, position))
    return m


def _projection_matrix(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Mirrors `Camera.projection_matrix` (python/bermake/viewport/camera.py)
    in float64 -- standard OpenGL right-handed perspective, clip_w = -z_cam."""
    f = 1.0 / math.tan(fov_y / 2.0)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def test_target_is_the_centre_of_the_bounds():
    _, target = frame_bounds(
        np.array([0.0, 0.0, 0.0]), np.array([2.0, 4.0, 6.0]), _DIR, 1.5, np.radians(45)
    )
    assert np.allclose(target, [1.0, 2.0, 3.0])


def test_view_direction_is_preserved():
    position, target = frame_bounds(
        np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]), _DIR, 1.5, np.radians(45)
    )
    actual = target - position
    expected = _DIR / np.linalg.norm(_DIR)
    assert np.allclose(actual / np.linalg.norm(actual), expected)


def test_a_bigger_model_pushes_the_camera_further_back():
    args = (_DIR, 1.5, np.radians(45))
    near, target = frame_bounds(np.array([-1.0] * 3), np.array([1.0] * 3), *args)
    far, _ = frame_bounds(np.array([-10.0] * 3), np.array([10.0] * 3), *args)
    assert np.linalg.norm(far - target) > np.linalg.norm(near - target)


def test_the_whole_bounding_sphere_fits_in_the_frustum():
    bmin, bmax = np.array([-1.0, -2.0, -3.0]), np.array([4.0, 5.0, 6.0])
    fov_y = np.radians(45)
    position, target = frame_bounds(bmin, bmax, _DIR, 1.5, fov_y)

    radius = float(np.linalg.norm(bmax - bmin) / 2.0)
    distance = float(np.linalg.norm(target - position))
    # Half-angle of the vertical frustum must cover the sphere, with margin.
    assert distance * np.sin(fov_y / 2.0) >= radius


def test_a_narrow_viewport_pushes_further_back_than_a_wide_one():
    args = (np.array([-1.0] * 3), np.array([1.0] * 3), _DIR)
    fov_y = np.radians(45)
    narrow, target = frame_bounds(*args, 0.5, fov_y)
    wide, _ = frame_bounds(*args, 2.0, fov_y)
    assert np.linalg.norm(narrow - target) > np.linalg.norm(wide - target)


def test_a_zero_extent_model_does_not_divide_by_zero():
    point = np.array([3.0, 3.0, 3.0])
    position, target = frame_bounds(point, point, _DIR, 1.5, np.radians(45))
    assert np.all(np.isfinite(position))
    assert np.allclose(target, point)
    assert np.linalg.norm(target - position) >= MIN_RADIUS


def test_a_degenerate_view_direction_falls_back_to_a_default():
    position, target = frame_bounds(
        np.array([0.0] * 3), np.array([1.0] * 3), np.zeros(3), 1.5, np.radians(45)
    )
    assert np.all(np.isfinite(position))
    assert not np.allclose(position, target)


def test_extreme_scale_boxes_stay_finite_and_still_fit():
    """A near-point box (radius floored to MIN_RADIUS) and a huge box (radius
    ~1.7e12) must both yield finite poses whose frustum still covers the
    bounding sphere -- no overflow, no underflow-to-zero distance."""
    fov_y = np.radians(45)
    for scale in (1e-9, 1e12):
        bmin, bmax = np.array([-scale] * 3), np.array([scale] * 3)
        position, target = frame_bounds(bmin, bmax, _DIR, 1.5, fov_y)

        assert np.all(np.isfinite(position))
        assert np.all(np.isfinite(target))

        radius = max(float(np.linalg.norm(bmax - bmin) / 2.0), MIN_RADIUS)
        distance = float(np.linalg.norm(target - position))
        assert distance > 0.0
        assert distance * np.sin(fov_y / 2.0) >= radius


def test_all_box_corners_land_inside_the_ndc_volume():
    """Genuine frustum-containment check.

    `test_the_whole_bounding_sphere_fits_in_the_frustum` and
    `test_extreme_scale_boxes_stay_finite_and_still_fit` both fix
    aspect=1.5 (>1), where the limiting half-angle is always fov_y/2 by
    construction -- their assertion collapses to `radius * margin >= radius`,
    true for any margin > 1 no matter what the horizontal-fit logic does.

    This test instead builds the real view + projection matrices (mirroring
    Camera's conventions above), projects all eight box corners through the
    pose frame_bounds returns, and asserts every corner lands inside the
    normalised device volume. It is parametrised over aspect=0.5 (the
    regime the tests above cannot see -- horizontal is limiting) and
    aspect=2.0, with a non-cube box and oblique view directions, so it
    actually exercises both the vertical- and horizontal-fit branches.
    """
    up = np.array([0.0, 0.0, 1.0])
    fov_y = np.radians(45)
    near, far = 0.01, 1.0e6
    bmin = np.array([-1.0, -3.0, -0.2])
    bmax = np.array([2.0, 1.0, 5.3])  # extents [3, 4, 5.5] -- anisotropic, not a cube
    cases = [
        (0.5, np.array([0.3, -0.7, 0.5])),  # aspect < 1: horizontal limits
        (2.0, np.array([-0.6, 0.2, 0.9])),  # aspect > 1: vertical limits
    ]
    corners = np.array(
        [
            [x, y, z, 1.0]
            for x in (bmin[0], bmax[0])
            for y in (bmin[1], bmax[1])
            for z in (bmin[2], bmax[2])
        ]
    )

    for aspect, view_dir in cases:
        position, target = frame_bounds(bmin, bmax, view_dir, aspect, fov_y)
        view = _view_matrix(position, target, up)
        proj = _projection_matrix(fov_y, aspect, near, far)

        for corner in corners:
            clip = proj @ (view @ corner)
            assert clip[3] > 0.0, "corner must be in front of the camera"
            ndc = clip[:2] / clip[3]
            assert abs(ndc[0]) <= 1.0 + 1e-9, f"aspect={aspect}: ndc_x={ndc[0]!r} out of frustum"
            assert abs(ndc[1]) <= 1.0 + 1e-9, f"aspect={aspect}: ndc_y={ndc[1]!r} out of frustum"


def test_a_degrees_shaped_fov_y_is_rejected():
    """The reviewer's failure mode: passing 45.0 (degrees) where radians(45)
    was meant must raise loudly, not silently return a near-zero-angle pose."""
    with pytest.raises(ValueError):
        frame_bounds(np.array([0.0] * 3), np.array([1.0] * 3), _DIR, 1.5, 45.0)


def test_a_fov_y_near_the_top_of_the_valid_radian_range_still_works():
    position, target = frame_bounds(
        np.array([0.0] * 3), np.array([1.0] * 3), _DIR, 1.5, math.pi - 1e-3
    )
    assert np.all(np.isfinite(position))
    assert np.all(np.isfinite(target))
