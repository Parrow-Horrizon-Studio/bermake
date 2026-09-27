"""Pure ray/segment closest-point helpers, shared by the snap engine and
model-layer picking (e.g. edge-proximity fallback in Model.pick_instance).

These are plain numpy math with no dependency on viewport/camera/scene
types, so they live in `geometry/` rather than `viewport/` — keeping the
model layer free to use them without importing the higher `viewport` layer.
"""

import numpy as np


def closest_points_two_lines(p1, d1, p2, d2):
    """Closest points between two infinite lines L1=p1+s*d1, L2=p2+t*d2.

    Returns (s, t, c1, c2). For parallel lines s=0 (and t follows). All inputs
    are float32 (3,) numpy arrays; d1/d2 need not be unit length.
    """
    r = p1 - p2
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))
    b = float(np.dot(d1, d2))
    c = float(np.dot(d1, r))
    denom = a * e - b * b
    s = 0.0 if abs(denom) < 1e-12 else (b * f - c * e) / denom
    t = (b * s + f) / e if e > 1e-12 else 0.0
    c1 = p1 + s * d1
    c2 = p2 + t * d2
    return s, t, c1.astype(np.float32), c2.astype(np.float32)


def closest_point_on_segment_to_ray(ray_origin, ray_dir, a, b):
    """Closest point ON segment [a, b] to the (infinite) ray line. Returns
    (point, t) with t clamped to [0, 1]."""
    d2 = b - a
    _, t, _, _ = closest_points_two_lines(ray_origin, ray_dir, a, d2)
    t = max(0.0, min(1.0, t))
    return (a + t * d2).astype(np.float32), float(t)
