"""Pure geometry generator for the Wall tool (M7a).

wall_box builds a centered solid-box wall segment from two base-centerline
points + thickness + height. No Model/Scene/Qt/GL deps — the caller supplies
points in whatever frame it wants the box built in (the WallTool converts
world -> active-context-local first).
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-9


def wall_box(start, end, thickness, height):
    """Return (vertices, faces) for a centered wall box, or ([], []) if degenerate.

    vertices: 8 (x, y, z) tuples [A, B, C, D, A', B', C', D'] (base then top).
    faces: 6 quad loops (index tuples), each wound so its right-hand-rule normal
    points OUT of the solid.
    """
    s = np.asarray(start, dtype=np.float64)
    e = np.asarray(end, dtype=np.float64)
    d = e - s
    d[2] = 0.0  # centerline is in-plane
    length = float(np.linalg.norm(d))
    if length < _EPS or thickness <= 0.0 or height <= 0.0:
        return [], []
    d /= length
    perp = np.array([d[1], -d[0], 0.0])  # in-plane perpendicular, unit
    o = perp * (thickness / 2.0)
    up = np.array([0.0, 0.0, float(height)])
    base_z = float(s[2])
    s0 = np.array([s[0], s[1], base_z])
    e0 = np.array([e[0], e[1], base_z])
    corners = [s0 - o, s0 + o, e0 + o, e0 - o]  # A, B, C, D (base)
    corners += [c + up for c in corners]  # A', B', C', D' (top)
    vertices = [(float(c[0]), float(c[1]), float(c[2])) for c in corners]
    # 0=A 1=B 2=C 3=D 4=A' 5=B' 6=C' 7=D'
    faces = [
        (0, 3, 2, 1),  # bottom  (-Z)
        (4, 5, 6, 7),  # top     (+Z)
        (0, 1, 5, 4),  # start cap
        (1, 2, 6, 5),  # long side
        (2, 3, 7, 6),  # end cap
        (3, 0, 4, 7),  # long side
    ]
    return vertices, faces
