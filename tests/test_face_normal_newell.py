"""Scene.face_normal uses Newell's method — and points the SAME WAY as before.

Issue #110. `face_normal` used to estimate the normal from `cross(p1-p0, p2-p0)`
and raise on a loop whose first three vertices are collinear, which is exactly
what an edge split leaves behind. Newell's method over the whole loop fixes it.

The risk in that change is not the fix, it is the SIGN. Six interactive tools
steer by this direction: push/pull extrudes along it, offset decides inward from
outward by it, paint resolves which side was picked with it. A flipped normal
would be a far worse bug than the one being fixed, and would be silent in any
test that only checks the axis. So the central test here is an equivalence
test — on every face the OLD estimate handled correctly, the new result must
agree in direction, not merely lie on the same line.

The old formula is recomputed inline below rather than kept as a preserved copy
of the implementation, so these tests state the property ("Newell agrees with a
first-three cross product wherever that is valid") instead of restating code.
"""

from __future__ import annotations

import numpy as np
import pytest

from bermake.scene import Scene

# Tolerance on a dot product of two unit vectors that should be identical.
_PARALLEL = 1.0 - 1e-5


def _face_from_points(points) -> tuple[Scene, int]:
    """Build a single-face Scene from an ordered (N, 3) loop of world points."""
    scene = Scene()
    vids = [scene.add_vertex(np.asarray(p, dtype=np.float32)) for p in points]
    for a, b in zip(vids, vids[1:] + vids[:1], strict=True):
        scene.add_edge(a, b)
    return scene, scene.add_face_from_loop(vids)


def _first_three_normal(scene: Scene, f_id: int) -> np.ndarray | None:
    """The pre-#110 estimate, recomputed here. None when it was degenerate.

    Deliberately written out rather than imported: the equivalence test is only
    meaningful if the thing being compared against is the property, not a copy
    of the implementation that could drift with it.
    """
    loop = scene.face_loop(f_id)
    p0, p1, p2 = (
        np.asarray(scene.vertex(loop[i]).position, dtype=np.float64) for i in (0, 1, 2)
    )
    n = np.cross(p1 - p0, p2 - p0)
    length = float(np.linalg.norm(n))
    if length < 1e-9:
        return None
    return n / length


def _rotated(points, axis, angle):
    """Rodrigues-rotate an iterable of 3-points about a unit axis."""
    k = np.asarray(axis, dtype=np.float64)
    k = k / np.linalg.norm(k)
    c, s = float(np.cos(angle)), float(np.sin(angle))
    out = []
    for p in points:
        v = np.asarray(p, dtype=np.float64)
        out.append(v * c + np.cross(k, v) * s + k * float(np.dot(k, v)) * (1.0 - c))
    return out


def _square_in_plane(orientation):
    """A unit square wound so its normal points along `orientation`.

    Six axis-aligned cases: three of six were the exact cases the v0.7.1 earcut
    winding bug mirrored, which is why all six are here rather than one per axis.
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


# A concave (L-shaped) polygon wound CCW in XY, with its reflex corner at
# (1,1), index 3. As written, index 1 holds a convex vertex, so the OLD
# estimate succeeds and points +Z — which is what makes it usable in the
# equivalence test. `cross(p1-p0, p2-p0)` is twice the signed area of triangle
# (p0, p1, p2), i.e. the ear test at p1, so it is a reflex vertex at INDEX 1
# that flips it, not one at the start. np.roll(_L_SHAPE, -2) puts it there;
# see test_the_reflex_at_index_1_case_is_where_newell_beats_the_old_estimate.
_L_SHAPE = [
    (0.0, 0.0, 0.0),
    (2.0, 0.0, 0.0),
    (2.0, 1.0, 0.0),
    (1.0, 1.0, 0.0),
    (1.0, 2.0, 0.0),
    (0.0, 2.0, 0.0),
]

_HEXAGON = [
    (float(np.cos(t)), float(np.sin(t)), 0.0)
    for t in np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
]

_TRIANGLE = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]

# Every face here is one the OLD implementation handled successfully. The
# equivalence test asserts the new one agrees with it in DIRECTION on all of
# them; anything that made the old estimate raise belongs in the regression
# test below instead, where there is nothing to compare against.
_EQUIVALENCE_CASES = {
    "square +Z": _square_in_plane("+Z"),
    "square -Z": _square_in_plane("-Z"),
    "square +X": _square_in_plane("+X"),
    "square -X": _square_in_plane("-X"),
    "square +Y": _square_in_plane("+Y"),
    "square -Y": _square_in_plane("-Y"),
    "triangle": _TRIANGLE,
    "triangle, reversed winding": list(reversed(_TRIANGLE)),
    "hexagon": _HEXAGON,
    "hexagon, reversed winding": list(reversed(_HEXAGON)),
    "concave L": _L_SHAPE,
    "concave L, reversed winding": list(reversed(_L_SHAPE)),
    # Oblique: no dominant axis handed to it, and no component near zero.
    "oblique square": _rotated(_square_in_plane("+Z"), (1.0, 2.0, 3.0), 0.7),
    "oblique concave L": _rotated(_L_SHAPE, (2.0, -1.0, 0.5), 1.1),
    "oblique triangle": _rotated(_TRIANGLE, (-1.0, 1.0, 4.0), 2.3),
}


@pytest.mark.parametrize("case", sorted(_EQUIVALENCE_CASES))
def test_newell_agrees_in_direction_with_the_first_three_vertex_estimate(case):
    # The whole point of issue #110's risk section: not "same axis", SAME WAY.
    # dot == +1, never -1. A sign flip here means push/pull pushes inward.
    scene, f_id = _face_from_points(_EQUIVALENCE_CASES[case])
    old = _first_three_normal(scene, f_id)
    assert old is not None, f"{case}: the old estimate must succeed to compare against"

    new = np.asarray(scene.face_normal(f_id), dtype=np.float64)
    assert float(np.dot(new, old)) > _PARALLEL, (
        f"{case}: Newell normal {new} disagrees with first-three estimate {old}"
    )


@pytest.mark.parametrize("case", sorted(_EQUIVALENCE_CASES))
def test_the_normal_is_finite_and_unit_length(case):
    scene, f_id = _face_from_points(_EQUIVALENCE_CASES[case])
    n = scene.face_normal(f_id)
    assert n.shape == (3,)
    assert n.dtype == np.float32
    assert np.all(np.isfinite(n))
    assert float(np.linalg.norm(n)) == pytest.approx(1.0, abs=1e-5)


def test_a_loop_starting_on_a_split_edge_no_longer_raises():
    # Issue #110 exactly: a square carrying a mid-edge vertex at the start of
    # its loop, which is the shape any edge split leaves behind. The first three
    # vertices are collinear, so the old estimate raised; the face is perfectly
    # ordinary geometry and renders fine.
    points = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    scene, f_id = _face_from_points([(x, y, 0.0) for x, y in points])

    assert _first_three_normal(scene, f_id) is None, (
        "this face must be one the old estimate could not handle, or it proves nothing"
    )

    n = scene.face_normal(f_id)
    assert np.all(np.isfinite(n))
    assert float(np.linalg.norm(n)) == pytest.approx(1.0, abs=1e-5)
    np.testing.assert_allclose(n, [0.0, 0.0, 1.0], atol=1e-6)


def test_the_normal_of_a_split_edge_face_matches_its_unsplit_square():
    # Same square, same winding, one extra collinear vertex. The normal must not
    # notice — including its sign.
    plain, f_plain = _face_from_points(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    )
    split, f_split = _face_from_points(
        [
            (0.0, 0.0, 0.0),
            (0.5, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ]
    )
    np.testing.assert_allclose(
        np.asarray(split.face_normal(f_split), dtype=np.float64),
        np.asarray(plain.face_normal(f_plain), dtype=np.float64),
        atol=1e-6,
    )


def test_the_reflex_at_index_1_case_is_where_newell_beats_the_old_estimate():
    # Every case above pins AGREEMENT. This one pins the IMPROVEMENT: a face
    # where the two intentionally differ, and the new answer is the right one.
    #
    # cross(p1-p0, p2-p0) is twice the signed area of triangle (p0, p1, p2) —
    # the ear test at p1 — so a REFLEX vertex at index 1 makes it point the
    # opposite way to the polygon's own normal. A loop merely starting at the
    # reflex corner does not do it; the vertex has to land at index 1.
    # np.roll(-2) puts the L's reflex corner (1,1) there, which is exactly the
    # construction tests/test_face_triangulation_winding.py uses for the same
    # class of bug on the earcut side.
    #
    # Without this test, an implementation that merely stopped raising on
    # collinear starts — while keeping the old sign — would pass the whole
    # file.
    points = np.roll(np.asarray(_L_SHAPE, dtype=np.float64), -2, axis=0)
    assert tuple(points[1]) == (1.0, 1.0, 0.0), "roll must land the reflex vertex at index 1"

    scene, f_id = _face_from_points(points)

    old = _first_three_normal(scene, f_id)
    np.testing.assert_allclose(old, [0.0, 0.0, -1.0], atol=1e-9)

    new = np.asarray(scene.face_normal(f_id), dtype=np.float64)
    np.testing.assert_allclose(new, [0.0, 0.0, 1.0], atol=1e-6)

    # Stated as the property, not just two literals: they must be opposed, and
    # the winding is unchanged from _L_SHAPE, whose normal is +Z.
    assert float(np.dot(new, old)) < -_PARALLEL


def test_a_genuinely_degenerate_zero_area_face_still_raises():
    # The bug was that an awkward loop START raised. A face with no area at all
    # has no normal to give, and must still say so.
    scene = Scene()
    vids = [
        scene.add_vertex(np.array([x, 0.0, 0.0], dtype=np.float32))
        for x in (0.0, 1.0, 2.0, 3.0)
    ]
    for a, b in zip(vids, vids[1:] + vids[:1], strict=True):
        scene.add_edge(a, b)
    f_id = scene.add_face_from_loop(vids)

    with pytest.raises(ValueError, match="degenerate"):
        scene.face_normal(f_id)


def test_a_face_normal_call_on_a_dead_face_still_raises_keyerror():
    scene = Scene()
    with pytest.raises(KeyError):
        scene.face_normal(999)
