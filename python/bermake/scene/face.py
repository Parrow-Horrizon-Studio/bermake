"""A planar face — closed loop of vertices with eager triangulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True, eq=False)
class Face:
    """A planar face bounded by an ordered vertex loop.

    `loop_vertex_ids` is the ordered tuple of vertex IDs walking the boundary
    CCW from +Z (the ground-plane convention in M2). `plane_normal` is a unit
    vector — (0, 0, 1) for all M2 ground-plane faces. `triangles` is an
    (N, 3) int32 array of vertex IDs per triangle, produced by earcut at
    insertion time so the renderer never re-triangulates.

    Hashing and equality are by integer `id` (see Vertex for the rationale).
    """

    id: int
    loop_vertex_ids: tuple[int, ...]
    plane_normal: np.ndarray
    triangles: np.ndarray

    def __post_init__(self) -> None:
        normal = np.asarray(self.plane_normal, dtype=np.float32).reshape(3).copy()
        normal.flags.writeable = False
        object.__setattr__(self, "plane_normal", normal)

        tris = np.asarray(self.triangles, dtype=np.int32).copy()
        tris.flags.writeable = False
        object.__setattr__(self, "triangles", tris)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Face):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)
