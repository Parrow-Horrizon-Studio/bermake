"""Zoom Extents framing (M7.2).

Pure numpy in, pure numpy out -- this module imports neither Qt nor any model
type, so it is testable with plain arrays. The camera merely adopts the
position and target it returns.

The model is framed by its bounding *sphere* rather than its box: a sphere is
orientation-independent, so the result does not change as the user orbits, and
no corner of the box can poke outside the frustum at an oblique angle.
"""

from __future__ import annotations

import math

import numpy as np

MIN_RADIUS = 1e-3
DEFAULT_VIEW_DIR = np.array([1.0, 1.0, -1.0], dtype=np.float64)


def frame_bounds(
    bmin: np.ndarray,
    bmax: np.ndarray,
    view_dir: np.ndarray,
    aspect: float,
    fov_y: float,
    margin: float = 1.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Position a camera so the bounds fill the frame.

    bmin/bmax: world-space axis-aligned bounds.
    view_dir:  the camera's current forward vector; its direction is
               preserved so Zoom Extents reframes without reorienting.
    aspect:    viewport width / height.
    fov_y:     vertical field of view in radians.
    margin:    fraction of slack around the model (1.15 = 15% breathing room).

    Returns (position, target) as float64 arrays.

    Raises:
        ValueError: if `fov_y` is not in the open interval (0, pi) radians --
            this is almost always a caller passing degrees by mistake (e.g.
            `Camera.fov_y_deg` unconverted).
    """
    if not (0.0 < fov_y < math.pi):
        raise ValueError(
            f"fov_y must be in radians, in the open interval (0, pi); got {fov_y!r}. "
            "Did you pass degrees instead of radians (e.g. Camera.fov_y_deg)?"
        )

    lo = np.asarray(bmin, dtype=np.float64)
    hi = np.asarray(bmax, dtype=np.float64)
    target = (lo + hi) / 2.0

    radius = max(float(np.linalg.norm(hi - lo)) / 2.0, MIN_RADIUS)

    direction = np.asarray(view_dir, dtype=np.float64)
    length = float(np.linalg.norm(direction))
    if not np.isfinite(length) or length < 1e-12:
        direction = DEFAULT_VIEW_DIR
        length = float(np.linalg.norm(direction))
    forward = direction / length

    # Fit vertically, and horizontally when the viewport is narrower than tall.
    half_v = fov_y / 2.0
    half_h = np.arctan(np.tan(half_v) * max(aspect, 1e-6))
    limiting = min(half_v, half_h)
    distance = radius * margin / max(np.sin(limiting), 1e-6)

    position = target - forward * distance
    return position, target
