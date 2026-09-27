"""Camera tween math (M7e): orbit-decomposition interpolation between two
CameraStates. PURE numpy — no Qt — so it is deterministically unit-testable.

Each endpoint is decomposed into (target, azimuth, elevation, distance, fov)
and interpolated component-wise, then recomposed. This makes the eye ORBIT the
model (constant-ish framing, monotonic distance) instead of straight-lining
through geometry. Azimuth interpolates the short way across the +/-pi seam.
World is Z-up: the horizontal plane is XY, elevation is the angle above it.
"""

from __future__ import annotations

import math

import numpy as np

from bermake.io.document_codec import CameraState

_EPS = 1e-9


def _decompose(cam: CameraState):
    """(target, azimuth, elevation, distance, fov) for a CameraState (Z-up)."""
    target = np.array(cam.target, dtype=np.float64)
    eye = np.array(cam.position, dtype=np.float64)
    d = eye - target
    distance = float(np.linalg.norm(d))
    if distance < _EPS:
        return target, 0.0, 0.0, 0.0, float(cam.fov_y_deg)
    dn = d / distance
    azimuth = math.atan2(float(dn[1]), float(dn[0]))
    elevation = math.asin(max(-1.0, min(1.0, float(dn[2]))))
    return target, azimuth, elevation, distance, float(cam.fov_y_deg)


def _wrap(a: float) -> float:
    """Map an angle delta into (-pi, pi] so interpolation takes the short way."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _nlerp(a, b, t: float):
    """Normalized lerp of two unit-ish vectors (for the up vector)."""
    va = np.array(a, dtype=np.float64)
    vb = np.array(b, dtype=np.float64)
    v = va + (vb - va) * t
    n = float(np.linalg.norm(v))
    if n < _EPS:
        return tuple(float(x) for x in va)
    return tuple(float(x) for x in (v / n))


def interpolate_pose(from_cam: CameraState, to_cam: CameraState, t: float) -> CameraState:
    """The eased-`t` pose between two CameraStates (t clamped to [0, 1])."""
    t = max(0.0, min(1.0, float(t)))
    tgt0, az0, el0, dist0, fov0 = _decompose(from_cam)
    tgt1, az1, el1, dist1, fov1 = _decompose(to_cam)

    target = tgt0 + (tgt1 - tgt0) * t
    azimuth = az0 + _wrap(az1 - az0) * t
    elevation = _lerp(el0, el1, t)
    distance = _lerp(dist0, dist1, t)
    fov = _lerp(fov0, fov1, t)

    ce = math.cos(elevation)
    direction = np.array(
        [math.cos(azimuth) * ce, math.sin(azimuth) * ce, math.sin(elevation)],
        dtype=np.float64,
    )
    position = target + direction * distance
    up = _nlerp(from_cam.up, to_cam.up, t)

    return CameraState(
        position=tuple(float(x) for x in position),
        target=tuple(float(x) for x in target),
        up=up,
        fov_y_deg=float(fov),
    )
