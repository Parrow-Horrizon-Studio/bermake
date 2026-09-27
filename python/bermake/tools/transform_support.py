"""Geometry helpers shared by the transform tools.

`selection_vertices` flattens an M4b Selection into the unique vertex ids it
covers. `selection_aabb` is their world axis-aligned bounding box. `grip_specs`
enumerates the Scale gizmo's handles (corner/edge/face) on that box, each
carrying the axes it drives and its opposite (anchor) position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def selection_vertices(scene, selection) -> list[int]:
    """Ordered-unique vertex ids covered by the selection.

    Union of each selected edge's two endpoints, each selected face's loop,
    and each directly-selected vertex. Ids whose entity is no longer live are
    skipped. Sorted for determinism.
    """
    seen: dict[int, None] = {}
    for e_id in sorted(selection.edges):
        try:
            e = scene.edge(e_id)
        except KeyError:
            continue
        seen.setdefault(int(e.v1_id), None)
        seen.setdefault(int(e.v2_id), None)
    for f_id in sorted(selection.faces):
        try:
            loop = scene.face_loop(f_id)
        except KeyError:
            continue
        for vid in loop:
            seen.setdefault(int(vid), None)
    # M7.6c: a directly-selected vertex is part of the transform set exactly
    # like an edge's endpoint. This is the single line that makes Move,
    # Rotate and Scale drag a bare vertex, since all three read this function
    # and nothing else.
    for v_id in sorted(selection.vertices):
        try:
            scene.vertex(v_id)
        except KeyError:
            continue
        seen.setdefault(int(v_id), None)
    return sorted(seen)


def selection_aabb(scene, vertex_ids):
    """(min_xyz, max_xyz) float32 over the given vertex ids, or None if empty."""
    if not vertex_ids:
        return None
    pts = np.array([scene.vertex(v).position for v in vertex_ids], dtype=np.float32)
    return pts.min(axis=0).astype(np.float32), pts.max(axis=0).astype(np.float32)


@dataclass(frozen=True)
class GripSpec:
    position: np.ndarray  # world handle position (3,) float32
    opposite: np.ndarray  # the anchor: opposite handle position (3,) float32
    axes: tuple[int, ...]  # which axes this grip drives (subset of {0,1,2})


def grip_specs(lo: np.ndarray, hi: np.ndarray) -> list[GripSpec]:
    """All non-degenerate Scale handles on the AABB [lo, hi].

    A handle sits at one of {lo, mid, hi} per axis. The axes it *drives* are
    those where it is at lo or hi (not mid). Corner = 3 driven axes, edge = 2,
    face = 1. Handles that coincide because an axis has zero extent are
    de-duplicated, so a flat selection yields the planar 8-grip set.
    """
    lo = np.asarray(lo, dtype=np.float32)
    hi = np.asarray(hi, dtype=np.float32)
    mid = (lo + hi) * 0.5
    coord = (lo, mid, hi)  # per-axis index: 0=lo, 1=mid, 2=hi
    degenerate = tuple(bool(hi[ax] == lo[ax]) for ax in (0, 1, 2))
    out: dict[tuple, GripSpec] = {}
    for ix in (0, 1, 2):
        for iy in (0, 1, 2):
            for iz in (0, 1, 2):
                if ix == 1 and iy == 1 and iz == 1:
                    continue  # centre is not a handle
                idx = (ix, iy, iz)
                pos = np.array([coord[ix][0], coord[iy][1], coord[iz][2]], dtype=np.float32)
                axes = tuple(ax for ax, i in zip((0, 1, 2), idx, strict=True) if i != 1)
                if not axes:
                    continue
                # Drop handles that drive only zero-extent axes (e.g. the centre
                # of a flat box's face): they coincide with the centre and add
                # no usable grip. A grip survives iff it drives a real axis.
                if all(degenerate[ax] for ax in axes):
                    continue
                opp_idx = tuple((2 - i) if i != 1 else 1 for i in idx)
                opp = np.array(
                    [coord[opp_idx[0]][0], coord[opp_idx[1]][1], coord[opp_idx[2]][2]],
                    dtype=np.float32,
                )
                key = tuple(np.round(pos, 5))
                if key not in out:
                    out[key] = GripSpec(position=pos, opposite=opp, axes=axes)
    return list(out.values())
