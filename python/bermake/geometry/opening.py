"""Pure geometry for the Door/Window tool (M7b).

opening_frame builds a framed door/window from closed solid sub-boxes in a
canonical local frame (origin at the opening's bottom-center; +X = width,
+Y = depth-into-wall, +Z = up). No Model/Scene/Qt/GL deps.

opening_placement_transform builds the 4x4 that places a canonical opening onto
a picked wall face (see Task 2).
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-9
_PROFILE = 0.06  # frame member width (m), fixed (not a user knob)
_PANEL_T = 0.04  # door panel thickness (m)
_GLAZING_T = 0.006  # window glazing thickness (m)

# Outward-wound quad faces for an axis-aligned box with corners
# 0:(x0,y0,z0) 1:(x1,y0,z0) 2:(x1,y1,z0) 3:(x0,y1,z0) and 4..7 = same at z1.
_BOX_FACES = (
    (0, 3, 2, 1),  # bottom -Z
    (4, 5, 6, 7),  # top    +Z
    (0, 1, 5, 4),  # front  -Y
    (1, 2, 6, 5),  # right  +X
    (2, 3, 7, 6),  # back   +Y
    (3, 0, 4, 7),  # left   -X
)


def _box(x0, x1, y0, y1, z0, z1):
    """Return (8 vertex tuples, 6 outward-wound quad loops) for a box."""
    verts = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    return verts, [tuple(loop) for loop in _BOX_FACES]


def opening_frame(kind, width, height, depth):
    """Return (vertices, faces) for a framed door/window, or ([], []) if degenerate.

    kind: "door" (open threshold + solid panel) or "window" (sill + glazing).
    """
    w = float(width)
    h = float(height)
    d = float(depth)
    p = _PROFILE
    if w <= 2.0 * p + _EPS or h <= 2.0 * p + _EPS or d <= _EPS:
        return [], []

    hx = w / 2.0
    is_window = kind == "window"
    infill_t = _GLAZING_T if is_window else _PANEL_T
    # Clamp depth to at least the infill thickness: below that, iy0 (=
    # (d-infill_t)/2) goes negative and iy1 (= (d+infill_t)/2) exceeds d, so
    # the panel/glazing box protrudes past the flush outer face on both
    # sides. Not reachable at the 100mm default depth; only the `d > _EPS`
    # guard existed before, which didn't cover this case.
    d = max(d, infill_t)
    iy0 = (d - infill_t) / 2.0
    iy1 = (d + infill_t) / 2.0

    boxes = [
        (-hx, -hx + p, 0.0, d, 0.0, h),  # left jamb
        (hx - p, hx, 0.0, d, 0.0, h),  # right jamb
        (-hx + p, hx - p, 0.0, d, h - p, h),  # head
    ]
    if is_window:
        boxes.append((-hx + p, hx - p, 0.0, d, 0.0, p))  # sill
        boxes.append((-hx + p, hx - p, iy0, iy1, p, h - p))  # glazing
    else:
        boxes.append((-hx + p, hx - p, iy0, iy1, 0.0, h - p))  # door panel (to floor)

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    for x0, x1, y0, y1, z0, z1 in boxes:
        bverts, bfaces = _box(x0, x1, y0, y1, z0, z1)
        off = len(vertices)
        vertices.extend((float(a), float(b), float(c)) for (a, b, c) in bverts)
        faces.extend(tuple(i + off for i in loop) for loop in bfaces)
    return vertices, faces


def opening_placement_transform(point, normal, sill):
    """Return the 4x4 placing a canonical opening onto a wall face, or None.

    point/normal are in the active-context-local frame; normal faces the viewer.
    The opening stands upright (up = local +Z); its outer face is flush with the
    wall face; its bottom-center sits at the cursor's horizontal position, at
    height `sill`. Returns None for a near-horizontal face (no valid upright).
    """
    up = np.array([0.0, 0.0, 1.0])
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    out = n - np.dot(n, up) * up  # horizontalize
    mag = float(np.linalg.norm(out))
    if mag < _EPS:
        return None  # near-horizontal face
    out /= mag
    along = np.cross(up, out)  # unit (up, out orthonormal)
    p = np.asarray(point, dtype=np.float64).reshape(3)
    m = np.eye(4, dtype=np.float64)
    m[:3, 0] = along  # canonical +X -> along-wall
    m[:3, 1] = -out  # canonical +Y -> into the wall
    m[:3, 2] = up  # canonical +Z -> up
    m[:3, 3] = np.array([p[0], p[1], float(sill)])
    return m
