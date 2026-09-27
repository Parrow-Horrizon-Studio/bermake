import math

import numpy as np

from bermake.io.document_codec import CameraState
from bermake.views.interpolate import interpolate_pose


def _cam(pos, target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0), fov=45.0):
    return CameraState(position=tuple(pos), target=tuple(target), up=tuple(up), fov_y_deg=fov)


def test_t0_reproduces_from_pose():
    a = _cam((5.0, -3.0, 4.0), target=(1.0, 1.0, 0.5), fov=40.0)
    b = _cam((-2.0, 6.0, 9.0), target=(0.0, 0.0, 0.0), fov=60.0)
    out = interpolate_pose(a, b, 0.0)
    assert np.allclose(out.position, a.position, atol=1e-6)
    assert np.allclose(out.target, a.target, atol=1e-6)
    assert out.fov_y_deg == 40.0


def test_t1_reproduces_to_pose():
    a = _cam((5.0, -3.0, 4.0), fov=40.0)
    b = _cam((-2.0, 6.0, 9.0), target=(1.0, 0.0, 2.0), fov=60.0)
    out = interpolate_pose(a, b, 1.0)
    assert np.allclose(out.position, b.position, atol=1e-6)
    assert np.allclose(out.target, b.target, atol=1e-6)
    assert out.fov_y_deg == 60.0


def test_midpoint_orbits_distance_is_pinned():
    # Same target, distances 2 and 8; the midpoint eye must sit at distance 5
    # from the (interpolated) target — proving orbit decomposition, not a
    # straight-line lerp of the eye (which would give a different distance).
    a = _cam((2.0, 0.0, 0.0))     # distance 2 along +X
    b = _cam((0.0, 8.0, 0.0))     # distance 8 along +Y
    out = interpolate_pose(a, b, 0.5)
    tgt = np.array(out.target)
    dist = np.linalg.norm(np.array(out.position) - tgt)
    assert abs(dist - 5.0) < 1e-6


def test_azimuth_takes_short_way_across_pi_seam():
    # 170° -> -170° must pass through 180°, not sweep back through 0°.
    a170 = math.radians(170.0)
    an170 = math.radians(-170.0)
    a = _cam((math.cos(a170), math.sin(a170), 0.0))
    b = _cam((math.cos(an170), math.sin(an170), 0.0))
    out = interpolate_pose(a, b, 0.5)
    p = np.array(out.position)
    assert p[0] < -0.99          # near (-1, 0, 0): azimuth 180°
    assert abs(p[1]) < 1e-6
    # NOT near (+1, 0, 0), which is what sweeping through 0° would give:
    assert p[0] < 0.0


def test_identical_poses_are_constant_with_no_nan():
    a = _cam((3.0, 3.0, 3.0), target=(1.0, 1.0, 1.0))
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        out = interpolate_pose(a, a, t)
        assert np.all(np.isfinite(out.position))
        assert np.allclose(out.position, a.position, atol=1e-6)


def test_degenerate_zero_distance_does_not_nan():
    # Eye == target (distance 0) must not produce NaN.
    a = _cam((1.0, 1.0, 1.0), target=(1.0, 1.0, 1.0))
    b = _cam((4.0, 0.0, 0.0), target=(0.0, 0.0, 0.0))
    out = interpolate_pose(a, b, 0.5)
    assert np.all(np.isfinite(out.position))
