"""The C++ kernel's Face::normal and Scene.face_normal must agree — sign included.

Issue #110, kernel half. Two independent implementations of the same quantity
now exist: `Scene.face_normal` in Python (Newell's method over the whole loop,
`_newell_normal` in python/bermake/scene/scene.py) and `Face::normal` in the
C++ kernel, read back through `face_triangle_buffer`. Cross-checking them
against each other is the strongest guard available, because it is the exact
check that would have caught the reopened bug: the kernel used to estimate the
normal from the loop's first three vertices and substitute a hardcoded
`(0, 0, 1)` when they were collinear, while Python computed the real one.

Why the kernel's copy is the one that shows: the renderer reads it out of
`face_triangle_buffer` for lighting, and since M7.5b `_face_uv_geometry` in
python/bermake/viewport/scene_renderer.py reads the very same block to build
each face's TEXTURE PROJECTION BASIS. A wall answering `(0, 0, 1)` is textured
as a floor.

Every case here is duplicated in a COLLINEAR-START form — the same face with
its first edge split, which is what an edge split leaves behind and which is
the shape that defeated the old estimate. The Python-side regression test for
this issue put its face in the XY plane, where the `(0, 0, 1)` fallback is
accidentally the right answer, and so missed the kernel half entirely; the
vertical and oblique cases below are where the two diverge.
"""

from __future__ import annotations

import numpy as np
import pytest

from bermake.scene import Scene

# Two unit vectors that should be identical, compared as a dot product: +1, not
# merely |1|. The sign is the whole point — lighting, picking, coplanarity,
# push/pull, offset, follow-me and the texture basis all steer by it.
_SAME_DIRECTION = 1.0 - 1e-5


def _scene_with_faces(loops: list[list[tuple[float, float, float]]]) -> tuple[Scene, list[int]]:
    """Build a Scene holding one face per loop, returning it and the face ids."""
    scene = Scene()
    face_ids = []
    for points in loops:
        vids = [scene.add_vertex(np.asarray(p, dtype=np.float32)) for p in points]
        for a, b in zip(vids, vids[1:] + vids[:1], strict=True):
            scene.add_edge(a, b)
        face_ids.append(scene.add_face_from_loop(vids))
    return scene, face_ids


def _kernel_normals(scene: Scene) -> dict[int, np.ndarray]:
    """Face id → the kernel's cached normal, read the way the renderer reads it.

    `face_triangle_buffer` is per CORNER, so this also asserts the block is
    constant within a face: the renderer relies on that to shade a face flat
    and `_face_uv_geometry` relies on it to give a face one texture basis.
    """
    _, normals = scene.face_triangle_buffer()
    face_ids = np.asarray(scene.face_triangle_face_ids())
    per_corner = np.repeat(face_ids, 3)
    assert per_corner.shape[0] == normals.shape[0], (
        "face_triangle_face_ids must align 1:1 with face_triangle_buffer"
    )

    out: dict[int, np.ndarray] = {}
    for f_id in np.unique(per_corner):
        block = normals[per_corner == f_id]
        np.testing.assert_allclose(block, np.broadcast_to(block[0], block.shape), atol=1e-6)
        out[int(f_id)] = np.asarray(block[0], dtype=np.float64)
    return out


def _split_first_edge(points):
    """Insert the midpoint of the loop's first edge at index 1.

    The first three vertices are then collinear — the #110 shape — while the
    face's area, plane and winding are untouched.
    """
    p0 = np.asarray(points[0], dtype=np.float64)
    p1 = np.asarray(points[1], dtype=np.float64)
    return [tuple(points[0]), tuple(0.5 * (p0 + p1))] + [tuple(p) for p in points[1:]]


def _rotated(points, axis, angle):
    """Rodrigues-rotate an iterable of 3-points about `axis`."""
    k = np.asarray(axis, dtype=np.float64)
    k = k / np.linalg.norm(k)
    c, s = float(np.cos(angle)), float(np.sin(angle))
    return [
        tuple(v * c + np.cross(k, v) * s + k * float(np.dot(k, v)) * (1.0 - c))
        for v in (np.asarray(p, dtype=np.float64) for p in points)
    ]


def _square_in_plane(orientation):
    """A unit square wound so its normal points along `orientation`.

    All six axis-aligned orientations, not one per axis: three of the six were
    the cases the v0.7.1 earcut winding bug mirrored, and `(0, 0, 1)` — the
    kernel's old fallback — is accidentally right on exactly one of them.
    """
    base = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    return {
        "+Z": [(x, y, 0.0) for x, y in base],
        "-Z": [(y, x, 0.0) for x, y in base],
        "+X": [(0.0, x, y) for x, y in base],
        "-X": [(0.0, y, x) for x, y in base],
        "+Y": [(y, 0.0, x) for x, y in base],
        "-Y": [(x, 0.0, y) for x, y in base],
    }[orientation]


_ORIENTATIONS = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")

_AXIS = {
    "+X": (1.0, 0.0, 0.0),
    "-X": (-1.0, 0.0, 0.0),
    "+Y": (0.0, 1.0, 0.0),
    "-Y": (0.0, -1.0, 0.0),
    "+Z": (0.0, 0.0, 1.0),
    "-Z": (0.0, 0.0, -1.0),
}

# Concave L wound CCW in XY, normal +Z. Its reflex corner (1,1) sits at index 3.
_L_SHAPE = [
    (0.0, 0.0, 0.0),
    (2.0, 0.0, 0.0),
    (2.0, 1.0, 0.0),
    (1.0, 1.0, 0.0),
    (1.0, 2.0, 0.0),
    (0.0, 2.0, 0.0),
]

_TRIANGLE = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]

# The issue's evidence, verbatim: a vertical wall in the XZ plane whose loop
# starts on a split edge. True normal (0, -1, 0); the old kernel said (0, 0, 1).
_ISSUE_WALL = [
    (0.0, 0.0, 0.0),
    (0.5, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 0.0, 1.0),
    (0.0, 0.0, 1.0),
]

_PLAIN_CASES = {f"square {o}": _square_in_plane(o) for o in _ORIENTATIONS} | {
    "triangle": _TRIANGLE,
    "triangle, reversed winding": list(reversed(_TRIANGLE)),
    "concave L": _L_SHAPE,
    "concave L, reversed winding": list(reversed(_L_SHAPE)),
    "oblique square": _rotated(_square_in_plane("+Z"), (1.0, 2.0, 3.0), 0.7),
    "oblique concave L": _rotated(_L_SHAPE, (2.0, -1.0, 0.5), 1.1),
}

# Every plain case again with its first edge split, plus the issue's own wall.
_COLLINEAR_START_CASES = {
    f"{name}, split first edge": _split_first_edge(pts) for name, pts in _PLAIN_CASES.items()
} | {"issue #110 vertical wall": _ISSUE_WALL}

_ALL_CASES = _PLAIN_CASES | _COLLINEAR_START_CASES


@pytest.mark.parametrize("case", sorted(_ALL_CASES))
def test_kernel_face_normal_matches_scene_face_normal(case):
    # The cross-check: two independent implementations of one quantity, in two
    # languages, asserted equal as vectors — not as axes.
    scene, (f_id,) = _scene_with_faces([_ALL_CASES[case]])
    kernel = _kernel_normals(scene)[f_id]
    python = np.asarray(scene.face_normal(f_id), dtype=np.float64)

    assert float(np.linalg.norm(kernel)) == pytest.approx(1.0, abs=1e-5)
    assert float(np.dot(kernel, python)) > _SAME_DIRECTION, (
        f"{case}: kernel normal {kernel} disagrees with Scene.face_normal {python}"
    )
    np.testing.assert_allclose(kernel, python, atol=1e-6)


@pytest.mark.parametrize("orientation", _ORIENTATIONS)
def test_a_collinear_start_wall_keeps_its_own_orientation(orientation):
    # The literal claim, stated against the six axes rather than against the
    # other implementation, so a shared mistake could not satisfy both.
    # `(0, 0, 1)` — the kernel's old answer for all six — passes only on "+Z".
    scene, (f_id,) = _scene_with_faces([_split_first_edge(_square_in_plane(orientation))])
    np.testing.assert_allclose(_kernel_normals(scene)[f_id], _AXIS[orientation], atol=1e-6)


def test_splitting_an_edge_does_not_change_the_kernel_normal():
    # Same wall, same winding, one extra collinear vertex. Both implementations
    # must be blind to it — this is the property an edge split has to preserve.
    plain = _square_in_plane("-Y")
    scene, (f_plain, f_split) = _scene_with_faces(
        [plain, [(x + 4.0, y, z) for x, y, z in _split_first_edge(plain)]]
    )
    kernel = _kernel_normals(scene)
    np.testing.assert_allclose(kernel[f_split], kernel[f_plain], atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(scene.face_normal(f_split), dtype=np.float64),
        np.asarray(scene.face_normal(f_plain), dtype=np.float64),
        atol=1e-6,
    )


def test_many_faces_in_one_scene_stay_matched_face_by_face():
    # The per-case tests above each hold one face, so they cannot catch a
    # misalignment between the buffer and its face ids. This holds all six
    # collinear-start orientations at once, spread apart so no vertex is
    # shared, and checks each face against its own Python normal.
    loops = [
        [(x + 4.0 * i, y, z) for x, y, z in _split_first_edge(_square_in_plane(o))]
        for i, o in enumerate(_ORIENTATIONS)
    ]
    scene, face_ids = _scene_with_faces(loops)
    kernel = _kernel_normals(scene)

    assert sorted(kernel) == sorted(face_ids)
    for f_id, orientation in zip(face_ids, _ORIENTATIONS, strict=True):
        np.testing.assert_allclose(kernel[f_id], _AXIS[orientation], atol=1e-6)
        np.testing.assert_allclose(
            kernel[f_id], np.asarray(scene.face_normal(f_id), dtype=np.float64), atol=1e-6
        )


def test_python_raises_on_a_collinear_loop_that_earcut_drops_entirely():
    # Two separate facts about one shape, both worth pinning and neither
    # implying the other.
    #
    # 1. Scene.face_normal RAISES on a face with no area. That is the
    #    deliberate difference between the two implementations: the kernel
    #    cannot raise into the render path without blanking the whole viewport,
    #    so it stores the {0, 0, 0} sentinel instead — not the old hardcoded
    #    (0, 0, 1), which is a specific plausible-looking direction and is how
    #    a wall came to be textured as a floor.
    #
    # 2. This particular loop emits nothing at all, because EARCUT finds no
    #    triangles in a fully collinear ring. That is a property of the
    #    TRIANGULATOR on this shape, NOT a kernel guarantee about degenerate
    #    faces — do not read it as one. A thin-but-real face whose area vector
    #    falls in the kernel's reject band (an XZ wall of side 2e-4 has an area
    #    vector of 8e-8) triangulates normally and emits corners carrying the
    #    sentinel. The kernel-side counterpart,
    #    HalfEdgeMeshTest.FaceNormalOfAZeroAreaFaceIsTheSentinelNotAnUpwardGuess,
    #    reads the sentinel straight out of face_triangle_buffer using a
    #    hand-built fan, and the two only look contradictory if this one is
    #    mistaken for a statement about the kernel.
    #
    # Consumers therefore have to cope with the sentinel rather than assume it
    # never arrives: plane_bases answers it with the world XY basis, and
    # phong.vert guards normalize() against it.
    scene, (f_id,) = _scene_with_faces(
        [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)]]
    )
    with pytest.raises(ValueError, match="degenerate"):
        scene.face_normal(f_id)

    assert scene.face_triangle_buffer()[1].shape[0] == 0
    assert f_id not in _kernel_normals(scene)


def test_a_thin_but_real_face_in_the_reject_band_still_emits_the_sentinel():
    # The measured counterexample to "a face that small covers no pixels",
    # kept as a test so the claim cannot quietly come back. A 2e-4 XZ square
    # has an area vector of 8e-8, inside the kernel's 1e-7 reject band, yet it
    # triangulates like any other quad and every corner it emits carries the
    # {0, 0, 0} sentinel. This is why each consumer needs its own guard.
    side = 2e-4
    scene, (f_id,) = _scene_with_faces(
        [[(0.0, 0.0, 0.0), (side, 0.0, 0.0), (side, 0.0, side), (0.0, 0.0, side)]]
    )
    _, normals = scene.face_triangle_buffer()
    assert normals.shape[0] > 0, "a thin face still triangulates; it is not dropped"
    np.testing.assert_allclose(normals, 0.0, atol=0.0)

    # Ten times wider is outside the band and gets a real normal, which pins
    # that the zeros above are the threshold talking and not a broken face.
    wide = 2e-3
    scene, (f_id,) = _scene_with_faces(
        [[(0.0, 0.0, 0.0), (wide, 0.0, 0.0), (wide, 0.0, wide), (0.0, 0.0, wide)]]
    )
    np.testing.assert_allclose(_kernel_normals(scene)[f_id], [0.0, -1.0, 0.0], atol=1e-6)
