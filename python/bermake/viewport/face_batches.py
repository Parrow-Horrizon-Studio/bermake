"""Group a definition's triangles by (front, back) material into draw batches.

M5b introduced single-material batching; M7.5a Task 4 generalizes it to a
(front, back) material pair per triangle (SketchUp-style two-sided faces) and
splits translucent triangles into a separate, contiguous suffix so the
renderer can draw them in a later depth-sorted pass (Task 5/6).

Pure Python + numpy — no GL, no `bermake.model` — so it is fully unit-testable
headlessly. The renderer reorders its interleaved face VBO by `vertex_order`
so each (front, back) pair's triangles are contiguous, then issues one
glDrawArrays per FaceBatch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FaceBatch:
    """A contiguous run of same-(front, back)-material vertices in the VBO."""

    front_material_id: int
    back_material_id: int
    first: int  # first vertex index
    count: int  # vertex count (a multiple of 3)


@dataclass(frozen=True, slots=True)
class BatchPlan:
    """The reordering and the two batch lists it produces."""

    vertex_order: np.ndarray
    opaque: list[FaceBatch]
    translucent: list[FaceBatch]
    translucent_first: int


_ID_BITS = 20  # material ids are monotonic and small; 1M per side is ample

# The inclusive upper bound on a material id. Public because the .berm loader
# validates against it: an id outside [0, MAX_MATERIAL_ID] would otherwise load
# clean and then raise from plan_face_batches inside render(), i.e. every frame.
# One constant so the loader's rule and the planner's cannot drift apart.
MAX_MATERIAL_ID = (1 << _ID_BITS) - 1


def plan_face_batches(
    front_ids: Sequence[int],
    back_ids: Sequence[int],
    translucent_mids: frozenset[int] = frozenset(),
) -> BatchPlan:
    """Stable-sort triangles by (is_translucent, front, back) into batches.

    Args:
        front_ids: front-side material id of each triangle, length T, in
            face-VBO order (e.g. Scene.face_triangle_materials(Side.FRONT)).
        back_ids: back-side material id of each triangle, same length and
            order (e.g. Scene.face_triangle_materials(Side.BACK)).
        translucent_mids: material ids that are translucent (Material.alpha
            < 1.0). A triangle is translucent when EITHER of its side
            materials is (spec D5).

    Because translucency leads the key, translucent triangles land as a
    contiguous suffix of `vertex_order`, starting at `translucent_first`.
    Task 5 re-permutes only that range and Task 6 re-uploads only that slice,
    so the suffix property is load-bearing rather than incidental.
    """
    front = np.asarray(front_ids, dtype=np.int64)
    back = np.asarray(back_ids, dtype=np.int64)
    t = int(front.shape[0])
    if int(back.shape[0]) != t:
        raise ValueError(f"front/back length mismatch: {t} vs {int(back.shape[0])}")
    if t == 0:
        return BatchPlan(np.zeros(0, dtype=np.int64), [], [], 0)

    id_limit = MAX_MATERIAL_ID + 1
    for name, ids in (("front_ids", front), ("back_ids", back)):
        invalid = ids[(ids < 0) | (ids >= id_limit)]
        if invalid.size:
            raise ValueError(
                f"{name} contains material id {int(invalid[0])}, "
                f"which is out of range: must be in [0, {id_limit})"
            )

    if translucent_mids:
        tl = np.fromiter(translucent_mids, dtype=np.int64, count=len(translucent_mids))
        is_tl = np.isin(front, tl) | np.isin(back, tl)
    else:
        is_tl = np.zeros(t, dtype=bool)

    key = (is_tl.astype(np.int64) << (2 * _ID_BITS)) | (front << _ID_BITS) | back
    tri_order = np.argsort(key, kind="stable")
    vertex_order = (tri_order[:, None] * 3 + np.arange(3)).reshape(-1).astype(np.int64)

    sorted_key = key[tri_order]
    sorted_front = front[tri_order]
    sorted_back = back[tri_order]
    sorted_tl = is_tl[tri_order]

    opaque: list[FaceBatch] = []
    translucent: list[FaceBatch] = []
    _, starts = np.unique(sorted_key, return_index=True)
    for k in range(len(starts)):
        tri_start = int(starts[k])
        tri_end = int(starts[k + 1]) if k + 1 < len(starts) else t
        batch = FaceBatch(
            front_material_id=int(sorted_front[tri_start]),
            back_material_id=int(sorted_back[tri_start]),
            first=tri_start * 3,
            count=(tri_end - tri_start) * 3,
        )
        (translucent if bool(sorted_tl[tri_start]) else opaque).append(batch)

    translucent_first = translucent[0].first if translucent else t * 3
    return BatchPlan(vertex_order, opaque, translucent, translucent_first)
