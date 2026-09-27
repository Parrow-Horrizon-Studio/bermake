"""One face's texture coordinates, resolved in loop order.

Spec decision D9 requires the renderer and the exporter to agree about what a
face's UVs are. They cannot share one function: the renderer needs per triangle
corner, vectorised across a whole definition on every buffer upload, and calling
a per-face routine there made M7.5b's upload 15.7 times slower. So this module
is the per-face REFERENCE implementation, and tests/test_uv_resolve.py pins the
renderer's batched path to it corner for corner. Change one and you must change
the other, exactly as uv_projection.py's scalar and batched siblings already
work.
"""

from __future__ import annotations

import numpy as np

from bermake.scene.scene import Side
from bermake.viewport.uv_projection import apply_placement, project_corners


def resolve_face_uvs(scene, materials, face_id: int, side: Side = Side.FRONT) -> np.ndarray:
    """(L, 2) float32 UVs for one face and side, in its boundary loop's order.

    Base, then placement, per spec section 1.3: stored UVs when this face and
    side have them, otherwise the plane projection at the material's real-world
    texture_size, with the face's offset, scale and rotation applied on top of
    whichever base was used.

    `materials` may be None, in which case a unit texture size is assumed. That
    matches build_face_uvs, whose `model` argument is read with getattr.
    """
    loop = scene.face_loop(int(face_id))
    stored = scene.face_uvs(int(face_id), side)

    if stored is not None:
        base = np.asarray(stored, dtype=np.float64).reshape(-1, 2)
    else:
        positions = np.array(
            [scene.vertex(v).position for v in loop],
            dtype=np.float64,
        ).reshape(-1, 3)
        size = (1.0, 1.0)
        if materials is not None:
            size = materials.get(scene.face_material(int(face_id), side)).texture_size
        base = project_corners(
            positions,
            scene.face_normal(int(face_id)),
            scene.face_center(int(face_id)),
            size,
        )

    p = scene.face_placement(int(face_id), side)
    return apply_placement(base, p.offset_u, p.offset_v, p.scale, p.rotation).astype(np.float32)
