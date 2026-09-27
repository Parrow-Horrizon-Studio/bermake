"""Curve point generators in 2D plane coordinates.

All functions return (N, 2) float64 arrays of plane coords centered on the
plane origin (0, 0). Tools lift them to world via DrawingPlane.to_world.
Rings are wound counter-clockwise so the resulting face normal aligns with the
plane normal.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-7


def _ring(radius: float, count: int, start_angle: float) -> np.ndarray:
    angles = start_angle + np.arange(count, dtype=np.float64) * (2.0 * np.pi / count)
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=-1)


def circle(radius: float, segments: int = 24, start_angle: float = 0.0) -> np.ndarray:
    """A `segments`-gon approximating a circle of `radius`, centered at (0, 0)."""
    return _ring(float(radius), int(segments), float(start_angle))


def polygon(radius: float, sides: int, start_angle: float = 0.0) -> np.ndarray:
    """A regular inscribed `sides`-gon of circumradius `radius`, centered at (0, 0)."""
    return _ring(float(radius), int(sides), float(start_angle))


def arc_2pt(
    start_uv: np.ndarray,
    end_uv: np.ndarray,
    bulge_uv: np.ndarray,
    segments: int = 12,
) -> np.ndarray:
    """Sample a circular arc through `start_uv` and `end_uv` whose bow is set by
    the perpendicular offset of `bulge_uv` from the chord.

    Returns (segments + 1, 2) inclusive of both endpoints. Degenerate cases:
    near-zero chord -> single point; near-zero sagitta -> the straight chord
    (2 points).
    """
    start = np.asarray(start_uv, dtype=np.float64).reshape(2)
    end = np.asarray(end_uv, dtype=np.float64).reshape(2)
    bulge = np.asarray(bulge_uv, dtype=np.float64).reshape(2)

    chord = end - start
    chord_len = float(np.linalg.norm(chord))
    if chord_len < _EPS:
        return start.reshape(1, 2).copy()

    chord_dir = chord / chord_len
    normal_dir = np.array([-chord_dir[1], chord_dir[0]])
    mid = 0.5 * (start + end)
    half = 0.5 * chord_len
    sagitta = float((bulge - mid) @ normal_dir)
    if abs(sagitta) < _EPS:
        return np.stack([start, end])

    yc = (sagitta * sagitta - half * half) / (2.0 * sagitta)
    center = mid + normal_dir * yc
    radius = float(np.linalg.norm(start - center))

    def _ang(p: np.ndarray) -> float:
        d = p - center
        return float(np.arctan2(d @ normal_dir, d @ chord_dir))

    apex = mid + normal_dir * sagitta
    ts = _ang(start)
    te_rel = (_ang(end) - ts) % (2.0 * np.pi)
    ta_rel = (_ang(apex) - ts) % (2.0 * np.pi)
    end_rel = te_rel if ta_rel <= te_rel else te_rel - 2.0 * np.pi

    thetas = ts + np.linspace(0.0, end_rel, int(segments) + 1)
    return center + radius * (
        np.outer(np.cos(thetas), chord_dir) + np.outer(np.sin(thetas), normal_dir)
    )


def semicircle_snap(
    start_uv: np.ndarray,
    end_uv: np.ndarray,
    bulge_uv: np.ndarray,
    rel_tol: float = 0.08,
) -> np.ndarray:
    """Snap `bulge_uv` to an exact semicircle (|sagitta| == half-chord) when it is
    within `rel_tol` of one; preserves the sign so the snapped point stays on the
    same side of the chord as the original bulge. Returns `bulge_uv` unchanged if
    outside tolerance."""
    start = np.asarray(start_uv, dtype=np.float64).reshape(2)
    end = np.asarray(end_uv, dtype=np.float64).reshape(2)
    bulge = np.asarray(bulge_uv, dtype=np.float64).reshape(2)

    chord = end - start
    chord_len = float(np.linalg.norm(chord))
    if chord_len < _EPS:
        return bulge.copy()
    normal_dir = np.array([-chord[1], chord[0]]) / chord_len
    mid = 0.5 * (start + end)
    half = 0.5 * chord_len
    sagitta = float((bulge - mid) @ normal_dir)
    if abs(abs(sagitta) - half) <= rel_tol * half:
        snapped = half if sagitta >= 0.0 else -half
        return mid + normal_dir * snapped
    return bulge.copy()
