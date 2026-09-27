"""Project face corners onto their own plane to get texture coordinates.

Faces store no UVs (spec D1). A material carries a real-world `texture_size`,
and each face's corners are projected onto a basis built on that face's plane,
divided by that size, then optionally adjusted by a per-face placement.

Pure numpy. No model, no scene, no Qt, no GL, so every function here is
testable with no QApplication and no GL context.
"""

from __future__ import annotations

import numpy as np

# A normal this close to a world axis makes that axis a poor choice of seed for
# the cross product, because the result approaches zero length.
_AXIS_ALIGNED = 0.9


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """An orthonormal (u, v) pair spanning the plane perpendicular to `normal`.

    Derived from the normal alone, deliberately. A basis taken from the face's
    first edge would change whenever an unrelated edit changed which vertex the
    loop starts at, spinning the texture on a face nobody touched.
    """
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(n))
    if length == 0.0:
        return (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        )
    nx, ny, nz = (float(c) / length for c in n)
    # cross(seed, n) and cross(n, u) written out as scalar arithmetic. The seed
    # is always a unit world axis, so the first cross collapses to two negations
    # and the second is three multiply-subtracts — the same floating-point
    # operations np.cross performs, without its moveaxis/broadcast dispatch.
    # That dispatch dominated the profile: building the UVs for a 9,600-face
    # definition spent 0.72 s of 1.14 s in np.cross on 3-vectors.
    if abs(nz) < _AXIS_ALIGNED:
        ux, uy, uz = -ny, nx, 0.0  # cross((0, 0, 1), n)
    else:
        ux, uy, uz = 0.0, -nz, ny  # cross((1, 0, 0), n)
    u_len = (ux * ux + uy * uy + uz * uz) ** 0.5
    ux, uy, uz = ux / u_len, uy / u_len, uz / u_len
    return (
        np.array([ux, uy, uz]),
        np.array([ny * uz - nz * uy, nz * ux - nx * uz, nx * uy - ny * ux]),
    )


def project_corners(
    positions: np.ndarray,
    normal: np.ndarray,
    origin: np.ndarray,
    texture_size: tuple[float, float],
) -> np.ndarray:
    """(N, 3) corners of one face to (N, 2) UVs, in texture tiles.

    `texture_size` is the image's extent in MODEL UNITS, so a 1x1 texture on a
    2x2 face spans two tiles rather than being stretched to fit.
    """
    pts = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    u_axis, v_axis = plane_basis(normal)
    rel = pts - np.asarray(origin, dtype=np.float64).reshape(3)
    su = float(texture_size[0]) or 1.0
    sv = float(texture_size[1]) or 1.0
    uvs = np.stack([rel @ u_axis / su, rel @ v_axis / sv], axis=1)
    return uvs.astype(np.float32)


def apply_placement(
    uvs: np.ndarray,
    offset_u: float,
    offset_v: float,
    scale: float,
    rotation: float,
) -> np.ndarray:
    """Scale, then rotate, then offset. That order is contractual.

    A larger `scale` makes the image cover more surface, so the UV is DIVIDED
    by it. Task 10's numeric fields and Task 12's drag both depend on this
    order matching, and a different one passes any test that varies a single
    parameter at a time.

    `rotation` is negated before it drives the UV rotation matrix. This
    function returns the coordinate a FIXED face point samples the texture
    at, and (as `paint_tool._pixel_delta_to_uv` already documents for offset)
    a change to that sample coordinate is the opposite of the corresponding
    change in what the viewer sees painted on the surface. Applying the UV
    frame's rotation unnegated is mathematically tidy but turns the image
    clockwise for a positive value; negating it here makes the stored
    `rotation` mean the image's own counter-clockwise turn, in a v-up frame,
    which is what the bare "Rotation" field in Properties promises.
    """
    out = np.asarray(uvs, dtype=np.float64).reshape(-1, 2)
    if out.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    s = float(scale) or 1.0
    out = out / s
    if rotation:
        r = -float(rotation)
        c, sn = np.cos(r), np.sin(r)
        rot = np.array([[c, -sn], [sn, c]])
        out = out @ rot.T
    out = out + np.array([float(offset_u), float(offset_v)])
    return out.astype(np.float32)


# --- Batched siblings -------------------------------------------------------
#
# The three functions above are the reference implementation and their tests are
# the specification; these do the same arithmetic one array at a time instead of
# one face at a time. The renderer bakes UVs for every corner of every face on
# every upload, where the per-face versions spent 94% of upload time in numpy
# call overhead rather than arithmetic — ~9,600 calls on (3,) arrays, where one
# call on an (N, 3) array costs almost nothing.
#
# tests/test_uv_projection.py pins the two paths against each other corner for
# corner. Change one of these and you must change its scalar twin.


def plane_bases(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """plane_basis for (N, 3) normals at once, returning (N, 3) u and v."""
    n = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    lengths = np.sqrt((n * n).sum(axis=1))
    degenerate = lengths == 0.0
    n = n / np.where(degenerate, 1.0, lengths)[:, None]

    nx, ny, nz = n[:, 0], n[:, 1], n[:, 2]
    # The same two seed cases, selected per row instead of per call.
    near_z = np.abs(nz) < _AXIS_ALIGNED
    u = np.empty_like(n)
    u[:, 0] = np.where(near_z, -ny, 0.0)
    u[:, 1] = np.where(near_z, nx, -nz)
    u[:, 2] = np.where(near_z, 0.0, ny)
    u_len = np.sqrt((u * u).sum(axis=1))
    u = u / np.where(u_len == 0.0, 1.0, u_len)[:, None]
    v = np.cross(n, u)

    # A zero-length normal has no plane; the scalar version answers with the
    # world XY basis, so this must too.
    if degenerate.any():
        u[degenerate] = (1.0, 0.0, 0.0)
        v[degenerate] = (0.0, 1.0, 0.0)
    return u, v


def project_onto_bases(
    positions: np.ndarray,
    u_axes: np.ndarray,
    v_axes: np.ndarray,
    origins: np.ndarray,
    texture_sizes: np.ndarray,
) -> np.ndarray:
    """project_corners' second half, per corner, with the bases already built.

    Every argument is per CORNER: (N, 3) positions, (N, 3) u and v axes, (N, 3)
    origins, (N, 2) texture sizes. The caller gathers a face's basis, centroid
    and texture size onto that face's corners, which is what lets a whole
    definition project in one call.
    """
    pts = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    rel = pts - np.asarray(origins, dtype=np.float64).reshape(-1, 3)
    sizes = np.asarray(texture_sizes, dtype=np.float64).reshape(-1, 2)
    sizes = np.where(sizes == 0.0, 1.0, sizes)  # matches the scalar `or 1.0`
    uvs = np.stack(
        [
            (rel * np.asarray(u_axes, dtype=np.float64)).sum(axis=1) / sizes[:, 0],
            (rel * np.asarray(v_axes, dtype=np.float64)).sum(axis=1) / sizes[:, 1],
        ],
        axis=1,
    )
    return uvs.astype(np.float32)


def apply_placements(
    uvs: np.ndarray,
    offsets: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
) -> np.ndarray:
    """apply_placement per corner: (N, 2) offsets, (N,) scales and rotations.

    Same contractual order — scale, then rotate, then offset. Rotation is
    applied unconditionally rather than under the scalar's `if rotation:`,
    which changes nothing: cos(0) and sin(0) are exactly 1.0 and 0.0, so a zero
    rotation is the identity to the last bit. `rotations` is negated exactly as
    the scalar `apply_placement` negates `rotation`, and for the same reason:
    the two are pinned together by an equivalence test and must agree bit for
    bit, not just in the identity case.
    """
    out = np.asarray(uvs, dtype=np.float64).reshape(-1, 2)
    if out.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    s = np.asarray(scales, dtype=np.float64).reshape(-1)
    out = out / np.where(s == 0.0, 1.0, s)[:, None]
    rot = -np.asarray(rotations, dtype=np.float64).reshape(-1)
    c, sn = np.cos(rot), np.sin(rot)
    out = np.stack([out[:, 0] * c - out[:, 1] * sn, out[:, 0] * sn + out[:, 1] * c], axis=1)
    out = out + np.asarray(offsets, dtype=np.float64).reshape(-1, 2)
    return out.astype(np.float32)
