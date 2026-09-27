"""Pure transform math for Move / Rotate / Scale.

All functions take and return (N, 3) float32 arrays and have no dependency on
Qt, OpenGL, or the Scene. Move/Rotate/Scale tools resolve which vertices to
transform, then call these to get the new positions.
"""

from __future__ import annotations

import numpy as np


def translate(points: np.ndarray, delta) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    d = np.asarray(delta, dtype=np.float32).reshape(3)
    return (pts + d).astype(np.float32)


def rotate(points: np.ndarray, center, axis, angle_rad: float) -> np.ndarray:
    """Rotate points about the line through `center` along `axis` (Rodrigues).

    `axis` need not be unit length. Raises ValueError on a near-zero axis.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3).astype(np.float64)
    c = np.asarray(center, dtype=np.float64).reshape(3)
    k = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(k))
    if norm < 1e-9:
        raise ValueError("rotate: degenerate (near-zero) axis")
    k = k / norm
    a = float(angle_rad)
    cos_a, sin_a = np.cos(a), np.sin(a)
    rel = pts - c
    cross = np.cross(np.broadcast_to(k, rel.shape), rel)
    dot = rel @ k
    rot = rel * cos_a + cross * sin_a + np.outer(dot, k) * (1.0 - cos_a)
    return (rot + c).astype(np.float32)


def scale(points: np.ndarray, anchor, factors) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    a = np.asarray(anchor, dtype=np.float32).reshape(3)
    f = np.asarray(factors, dtype=np.float32).reshape(3)
    return (a + (pts - a) * f).astype(np.float32)


# ---------------------------------------------------------------------------
# 4x4 matrix helpers (M4e: world↔local coordinate conversion)
# ---------------------------------------------------------------------------


def mat_translate(delta) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[:3, 3] = np.asarray(delta, dtype=np.float64).reshape(3)
    return m


def mat_scale(anchor, factors) -> np.ndarray:
    a = np.asarray(anchor, dtype=np.float64).reshape(3)
    f = np.asarray(factors, dtype=np.float64).reshape(3)
    m = np.eye(4, dtype=np.float64)
    m[0, 0], m[1, 1], m[2, 2] = f
    m[:3, 3] = a - f * a  # p' = a + (p-a)*f
    return m


def mat_rotate(center, axis, angle_rad: float) -> np.ndarray:
    c = np.asarray(center, dtype=np.float64).reshape(3)
    k = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(k))
    if norm < 1e-9:
        raise ValueError("mat_rotate: degenerate (near-zero) axis")
    k = k / norm
    x, y, z = k
    ca, sa = np.cos(angle_rad), np.sin(angle_rad)
    r = np.array(
        [
            [ca + x * x * (1 - ca), x * y * (1 - ca) - z * sa, x * z * (1 - ca) + y * sa],
            [y * x * (1 - ca) + z * sa, ca + y * y * (1 - ca), y * z * (1 - ca) - x * sa],
            [z * x * (1 - ca) - y * sa, z * y * (1 - ca) + x * sa, ca + z * z * (1 - ca)],
        ],
        dtype=np.float64,
    )
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = r
    m[:3, 3] = c - r @ c  # rotate about the line through center
    return m


def mat_compose(*mats) -> np.ndarray:
    """Compose transforms applied left-to-right: mat_compose(A, B) == B @ A."""
    out = np.eye(4, dtype=np.float64)
    for m in mats:
        out = np.asarray(m, dtype=np.float64) @ out
    return out


def is_identity_transform(m) -> bool:
    """True if m is None or numerically the 4x4 identity."""
    if m is None:
        return True
    return bool(np.allclose(np.asarray(m, dtype=np.float64), np.eye(4)))


def mat_invert(m) -> np.ndarray:
    return np.linalg.inv(np.asarray(m, dtype=np.float64))


def apply_mat(points, m) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    m = np.asarray(m, dtype=np.float64)
    h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    out = (h @ m.T)[:, :3]
    return out.astype(np.float32)
