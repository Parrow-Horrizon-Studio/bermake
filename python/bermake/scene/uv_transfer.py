"""Carry a face's stored per-corner UVs across an edge split or a face split.

Pure arithmetic over loops, coordinates and triangulations, with no Scene and
no kernel, so these functions are testable in isolation and cheap to reason
about. `uv_at_point_in_face` uses numpy for the coordinate math; nothing here
touches the mesh.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def transfer_uvs_across_split(
    old_uvs: Sequence[tuple[float, float]],
    old_loop: Sequence[int],
    new_loop: Sequence[int],
    w: int,
    va: int,
    vb: int,
    t: float,
) -> list[tuple[float, float]] | None:
    """`old_uvs` re-ordered onto `new_loop`, with one lerped entry for `w`.

    `t` is measured from `va` toward `vb`, which is the kernel's convention and
    is NOT necessarily the direction this face's loop traverses the edge. When
    the loop runs vb-then-va the inserted corner sits at `1 - t` along the
    loop's own direction, and using `t` there would misplace the corner on
    roughly half of a model's faces.

    Returns None when the inputs are not a single clean insertion, which is the
    caller's signal to drop that face to the projection rather than guess.
    """
    if len(old_uvs) != len(old_loop):
        return None
    if len(new_loop) != len(old_loop) + 1:
        return None

    try:
        insert_at = list(new_loop).index(w)
    except ValueError:
        return None

    n_new = len(new_loop)
    prev_v = new_loop[insert_at - 1]
    next_v = new_loop[(insert_at + 1) % n_new]

    # Position of every surviving vertex in the OLD loop, so each keeps its own
    # UV regardless of where the insertion shifted it to.
    old_index = {int(v): i for i, v in enumerate(old_loop)}
    if len(old_index) != len(old_loop):
        return None  # a repeated vertex makes the mapping ambiguous
    if prev_v not in old_index or next_v not in old_index:
        return None
    if {int(prev_v), int(next_v)} != {int(va), int(vb)}:
        return None

    uv_prev = old_uvs[old_index[int(prev_v)]]
    uv_next = old_uvs[old_index[int(next_v)]]
    frac = float(t) if int(prev_v) == int(va) else 1.0 - float(t)
    inserted = (
        uv_prev[0] + (uv_next[0] - uv_prev[0]) * frac,
        uv_prev[1] + (uv_next[1] - uv_prev[1]) * frac,
    )

    out: list[tuple[float, float]] = []
    for i, v in enumerate(new_loop):
        if i == insert_at:
            out.append(inserted)
        else:
            uv = old_uvs[old_index[int(v)]]
            out.append((float(uv[0]), float(uv[1])))
    return out


def _barycentric(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, p: np.ndarray
) -> tuple[float, float, float] | None:
    """Barycentric weights of `p` against triangle `a, b, c`.

    Works directly in 3D: `p` is assumed coplanar with the triangle (both
    come from the same planar face), so no projection to 2D is needed. Returns
    None for a degenerate (zero-area) triangle rather than dividing by zero.
    """
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d20 = float(np.dot(v2, v0))
    d21 = float(np.dot(v2, v1))
    if d00 <= 0.0 or d11 <= 0.0:
        return None  # a or b or c coincide with a: zero-length edge vector
    denom = d00 * d11 - d01 * d01
    # `denom` is (twice the triangle's area)^2, so it scales as length^4. A
    # fixed absolute threshold here misclassifies a small-but-real triangle
    # as degenerate once its edges drop below roughly 1e-4 model units (a
    # face split near a tight edge loop is exactly where this bites).
    # denom / (d00 * d11) is sin^2 of the angle between edges v0 and v1,
    # which is dimensionless and scale-invariant, so comparing against a
    # small multiple of d00 * d11 reads as "degenerate below roughly this
    # angle" at any model scale instead of "degenerate below this absolute
    # area". The d00/d11 <= 0 check above covers the one case this ratio
    # test cannot: a genuinely zero-length edge, where d00 * d11 is itself
    # zero and the ratio is undefined rather than small.
    if abs(denom) < 1e-18 * d00 * d11:
        return None
    wb = (d11 * d20 - d01 * d21) / denom
    wc = (d00 * d21 - d01 * d20) / denom
    wa = 1.0 - wb - wc
    return wa, wb, wc


def uv_at_point_in_face(
    positions: np.ndarray,
    uvs: np.ndarray,
    triangles: Sequence[int],
    point: np.ndarray,
) -> tuple[float, float] | None:
    """The UV at `point`, interpolated inside whichever triangle contains it.

    `positions` and `uvs` are parallel to the face's boundary loop, `triangles`
    is a flat list of indices INTO THAT LOOP, three per triangle, and `point`
    lies in the face's plane.

    Barycentric rather than an affine fit of the plane to the UV layout. The
    affine answer is cheaper and is exact whenever the layout came from
    bermake's own planar projection, but it is wrong for an imported layout,
    which is what M7.5c added and what this would silently distort.

    Returns None when no triangle contains the point, which a caller should
    treat as "this face cannot supply a UV here" rather than as an error.
    """
    positions = np.asarray(positions, dtype=np.float64)
    uvs = np.asarray(uvs, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)
    tol = 1e-6

    n_triangles = len(triangles) // 3
    for k in range(n_triangles):
        ia, ib, ic = triangles[3 * k], triangles[3 * k + 1], triangles[3 * k + 2]
        bary = _barycentric(positions[ia], positions[ib], positions[ic], point)
        if bary is None:
            continue
        wa, wb, wc = bary
        if wa >= -tol and wb >= -tol and wc >= -tol:
            uv = wa * uvs[ia] + wb * uvs[ib] + wc * uvs[ic]
            return float(uv[0]), float(uv[1])
    return None
