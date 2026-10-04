"""Concave faces keep a correct fill after an edge split or an erase-to-merge.

The kernel's split_edge and dissolve_edge rebuild each affected face with a
triangle fan from loop[0]. A fan is only correct when loop[0] can see every
other corner, which a concave face does not guarantee: a U-shaped face of
area 7 came back from any edge split with triangles whose absolute areas sum
to 11, so the notch rendered filled and was clickable. Scene now re-earcuts
every face those two operations touch (M7.11 decision D5).

The check used throughout measures each triangle's signed area in the face's
own plane. A correct triangulation has no triangle wound against the face and
its areas add up to the polygon's area; a fan across a notch has at least one
triangle wound backwards (the one spanning the notch) and over-covers.
"""

from __future__ import annotations

import numpy as np
import pytest
from bermake.commands.command_stack import CommandStack
from bermake.commands.scene_commands import SplitEdgeCommand
from bermake.document import DocumentSettings
from bermake.io import load_document, save_document
from bermake.model.model import Model
from bermake.scene.scene import Scene, Side, TexturePlacement
from bermake.viewport.camera import Camera
from bermake.viewport.render_style import RenderStyle

# A 3 x 3 square with a 1 wide, 2 deep notch cut from the top: area 9 - 2 = 7.
# Wound CCW from +Z. The fan from (0, 0) crosses the notch.
U_POINTS = [(0, 0), (3, 0), (3, 3), (2, 3), (2, 1), (1, 1), (1, 3), (0, 3)]
U_AREA = 7.0

# An L of area 3 whose loop starts at (2, 1), not at the outer corner (0, 0).
# A fan from (2, 1) puts the triangle (2,1)-(1,1)-(1,2) outside the face.
L_POINTS = [(2, 1), (1, 1), (1, 2), (0, 2), (0, 0), (2, 0)]
L_AREA = 3.0

_AREA_TOL = 1e-5


# Where the 2D outline lands in 3D, with a ray that falls straight onto the
# notch (the empty part of the U) and one that falls onto its solid base. The
# earcut projection drops a different axis and winds the other way in each.
PLANES = {
    "xy": (
        lambda x, y: (x, y, 0.0),
        ((1.5, 2.0, 5.0), (0.0, 0.0, -1.0)),
        ((1.5, 0.5, 5.0), (0.0, 0.0, -1.0)),
    ),
    "xz": (
        lambda x, y: (x, 0.0, y),
        ((1.5, 5.0, 2.0), (0.0, -1.0, 0.0)),
        ((1.5, 5.0, 0.5), (0.0, -1.0, 0.0)),
    ),
    "yz": (
        lambda x, y: (0.0, x, y),
        ((5.0, 1.5, 2.0), (-1.0, 0.0, 0.0)),
        ((5.0, 1.5, 0.5), (-1.0, 0.0, 0.0)),
    ),
}


def _add_loop(
    scene: Scene, points, z: float = 0.0, dx: float = 0.0, to_3d=None
) -> tuple[int, list[int]]:
    if to_3d is None:

        def to_3d(x, y):
            return (x + dx, y, z)

    ids = [scene.add_vertex(np.array(to_3d(x, y), dtype=np.float32)) for x, y in points]
    return scene.add_face_from_loop(ids), ids


def _fill_problems(scene: Scene, fid: int) -> list[str]:
    """Everything wrong with face `fid`'s stored triangles, or [] if it is filled exactly."""
    face = scene.face(fid)
    loop = list(face.loop_vertex_ids)
    pos = {v: scene.vertex(v).position.astype(np.float64) for v in loop}
    ring = np.array([pos[v] for v in loop])
    nxt = np.roll(ring, -1, axis=0)
    newell = np.sum(np.cross(ring, nxt), axis=0)  # 2 * area vector
    polygon_area = 0.5 * float(np.linalg.norm(newell))
    unit = newell / np.linalg.norm(newell)

    problems: list[str] = []
    total = 0.0
    for tri in face.triangles:
        if any(int(v) not in pos for v in tri):
            problems.append(f"triangle {tuple(int(v) for v in tri)} names a vertex off the loop")
            continue
        a, b, c = (pos[int(v)] for v in tri)
        signed = 0.5 * float(np.dot(np.cross(b - a, c - a), unit))
        if signed <= _AREA_TOL:
            problems.append(f"triangle {tuple(int(v) for v in tri)} has signed area {signed:.4f}")
        total += signed
    if abs(total - polygon_area) > _AREA_TOL:
        problems.append(f"triangle areas sum to {total:.4f}, polygon area is {polygon_area:.4f}")
    return problems


def _assert_fills(scene: Scene, fid: int) -> None:
    problems = _fill_problems(scene, fid)
    assert problems == [], f"face {fid}: " + "; ".join(problems)


def _edge_ids_of(scene: Scene, ids: list[int]) -> list[int]:
    n = len(ids)
    return [scene.edge_between(ids[i], ids[(i + 1) % n]) for i in range(n)]


def _area(scene: Scene, fid: int) -> float:
    face = scene.face(fid)
    total = 0.0
    for tri in face.triangles:
        a, b, c = (scene.vertex(int(v)).position.astype(np.float64) for v in tri)
        total += 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))
    return total


# ---- the baseline: a freshly added face is earcut -----------------------------


def test_a_freshly_added_u_face_is_filled_exactly():
    s = Scene()
    f, _ids = _add_loop(s, U_POINTS)
    _assert_fills(s, f)
    assert _area(s, f) == pytest.approx(U_AREA)


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(lambda loop: [*loop[:3], loop[0]], id="not-a-multiple-of-3"),
        pytest.param(lambda loop: [loop[0], loop[1], 999], id="id-off-the-loop"),
    ],
)
def test_the_binding_rejects_bad_triangles_with_value_error_and_changes_nothing(bad):
    s = Scene()
    f, _ids = _add_loop(s, U_POINTS)
    before = s.face(f).triangles.copy()
    with pytest.raises(ValueError):
        s._mesh.set_face_triangles(f, bad(s.face_loop(f)))
    np.testing.assert_array_equal(s.face(f).triangles, before)


def test_the_binding_rejects_a_dead_face_with_value_error():
    s = Scene()
    f, _ids = _add_loop(s, U_POINTS)
    tris = list(s._mesh.face_triangles(f))
    s.remove_face(f)
    with pytest.raises(ValueError):
        s._mesh.set_face_triangles(f, tris)
    assert not s._mesh.face_is_live(f)


# ---- split_edge ---------------------------------------------------------------


@pytest.mark.parametrize("plane", PLANES)
@pytest.mark.parametrize("edge_index", range(len(U_POINTS)))
def test_splitting_any_edge_of_a_u_keeps_the_notch_empty(edge_index, plane):
    to_3d, (notch_origin, notch_dir), (base_origin, base_dir) = PLANES[plane]
    s = Scene()
    _f, ids = _add_loop(s, U_POINTS, to_3d=to_3d)
    e = _edge_ids_of(s, ids)[edge_index]

    res = s.split_edge(e, 0.5)

    assert res is not None
    affected = [fid for fid in (res.face_a, res.face_b) if fid is not None]
    assert len(affected) == 1
    for fid in affected:
        _assert_fills(s, fid)
        # The absolute areas sum to 11 with the fan: the notch (2) counted twice.
        assert _area(s, fid) == pytest.approx(U_AREA)
    # The user-visible symptom: the notch is empty, so a click there misses,
    # while a click on the solid base still picks the face.
    assert s.ray_pick_face(np.array(notch_origin), np.array(notch_dir)) is None
    assert s.ray_pick_face(np.array(base_origin), np.array(base_dir)) is not None


@pytest.mark.parametrize("edge_index", range(len(L_POINTS)))
def test_splitting_any_edge_of_an_l_that_starts_off_its_outer_corner(edge_index):
    s = Scene()
    _f, ids = _add_loop(s, L_POINTS)
    e = _edge_ids_of(s, ids)[edge_index]

    res = s.split_edge(e, 0.5)

    assert res is not None
    for fid in (res.face_a, res.face_b):
        if fid is not None:
            _assert_fills(s, fid)
            assert _area(s, fid) == pytest.approx(L_AREA)


def test_splitting_an_edge_shared_by_two_u_faces_fixes_both():
    s = Scene()
    _f1, ids1 = _add_loop(s, U_POINTS)
    # A second U to the right, sharing the left U's edge (3,0)-(3,3). Its loop
    # starts at (3,0), its own outer bottom-left corner, so its fan is wrong too.
    right_u = [(0, 0), (3, 0), (3, 3), (2, 3), (2, 1), (1, 1), (1, 3), (0, 3)]
    _f2, ids2 = _add_loop(s, right_u, dx=3.0)
    shared = s.edge_between(ids1[1], ids1[2])
    assert shared is not None and shared == s.edge_between(ids2[0], ids2[7])

    res = s.split_edge(shared, 0.5)

    assert res is not None and res.face_a is not None and res.face_b is not None
    _assert_fills(s, res.face_a)
    _assert_fills(s, res.face_b)


def test_split_edge_fixes_every_split_in_a_sequence():
    s = Scene()
    _f, ids = _add_loop(s, U_POINTS)
    e = s.edge_between(ids[3], ids[4])  # (2,3)-(2,1), the notch's right wall
    res = s.split_edge(e, 0.25)
    assert res is not None and res.face_a is not None
    res2 = s.split_edge(res.edge_b, 0.5)
    assert res2 is not None
    fid = res2.face_a if res2.face_a is not None else res2.face_b
    _assert_fills(s, fid)


# ---- dissolve_edge (erase-to-merge) -------------------------------------------


# Chords across the U, as indices into U_POINTS. Each one is a straight segment
# that stays inside the face, so split_face accepts it.
@pytest.mark.parametrize("chord", [(1, 4), (0, 4), (0, 5), (5, 7), (2, 4)])
def test_erasing_a_chord_of_a_u_refills_the_merged_face(chord):
    s = Scene()
    f, ids = _add_loop(s, U_POINTS)
    a, b = ids[chord[0]], ids[chord[1]]
    chord_edge = s.add_edge(a, b)
    halves = s.split_face(f, [a, b])
    assert halves is not None
    for half in halves:
        _assert_fills(s, half)

    merged = s.dissolve_edge(chord_edge)

    assert merged is not None
    assert len(s.face_loop(merged)) == len(U_POINTS)
    _assert_fills(s, merged)
    assert _area(s, merged) == pytest.approx(U_AREA)


# ---- Review Focus 1: undo / redo of a split through the command stack ---------


@pytest.mark.parametrize("edge_index", [0, 3, 4, 5])
def test_undo_and_redo_of_a_split_edge_command_keep_the_fill(edge_index):
    s = Scene()
    f, ids = _add_loop(s, U_POINTS)
    e = _edge_ids_of(s, ids)[edge_index]
    stack = CommandStack()

    cmd = SplitEdgeCommand(e, 0.5)
    stack.execute(cmd, s)
    after_do = [face.id for face in s.faces_iter()]
    assert len(after_do) == 1 and after_do[0] != f
    _assert_fills(s, after_do[0])

    assert stack.undo()
    assert [face.id for face in s.faces_iter()] == [f]
    _assert_fills(s, f)

    assert stack.redo()
    assert [face.id for face in s.faces_iter()] == after_do
    _assert_fills(s, after_do[0])


# ---- Review Focus 2: paint, stored UVs and placement survive ------------------


def _paint_everything(s: Scene, fid: int, loop: list[int]) -> dict:
    s.set_face_material(fid, 11, Side.FRONT)
    s.set_face_material(fid, 22, Side.BACK)
    front_uvs = [(float(i), float(i) * 0.5) for i in range(len(loop))]
    back_uvs = [(-float(i), float(i) + 0.25) for i in range(len(loop))]
    s.set_face_uvs(fid, front_uvs, Side.FRONT)
    s.set_face_uvs(fid, back_uvs, Side.BACK)
    front_placement = TexturePlacement(0.25, 0.5, 2.0, 0.75)
    back_placement = TexturePlacement(-0.5, 0.125, 0.5, 1.5)
    s.set_face_placement(fid, front_placement, Side.FRONT)
    s.set_face_placement(fid, back_placement, Side.BACK)
    return {
        Side.FRONT: (11, front_uvs, front_placement),
        Side.BACK: (22, back_uvs, back_placement),
    }


def _expected_after_split(loop, uvs, va, vb, w, t):
    """The loop with w inserted between va and vb, and its UVs with one value
    lerped at t from va's UV (va = the edge's lower vertex id) to vb's."""
    by_vertex = dict(zip(loop, uvs, strict=True))
    lerp = tuple((1.0 - t) * by_vertex[va][k] + t * by_vertex[vb][k] for k in range(2))
    n = len(loop)
    new_loop: list[int] = []
    new_uvs: list[tuple[float, float]] = []
    for i, v in enumerate(loop):
        new_loop.append(v)
        new_uvs.append(by_vertex[v])
        if {v, loop[(i + 1) % n]} == {va, vb}:
            new_loop.append(w)
            new_uvs.append(lerp)
    return new_loop, new_uvs


def _assert_sidecars_carried(s, fid, painted, old_loop, va, vb, w, t):
    for side, (material, uvs, placement) in painted.items():
        assert s.face_material(fid, side) == material
        assert s.face_placement(fid, side) == placement
        want_loop, want_uvs = _expected_after_split(old_loop, uvs, va, vb, w, t)
        assert s.face_loop(fid) == want_loop
        got = s.face_uvs(fid, side)
        assert got is not None
        np.testing.assert_allclose(got, np.asarray(want_uvs, dtype=np.float32), atol=1e-6)


@pytest.mark.parametrize("edge_index", range(len(U_POINTS)))
def test_a_painted_textured_u_keeps_every_sidecar_through_a_split(edge_index):
    s = Scene()
    f, ids = _add_loop(s, U_POINTS)
    old_loop = s.face_loop(f)
    painted = _paint_everything(s, f, old_loop)
    e = _edge_ids_of(s, ids)[edge_index]
    va, vb = s.edge(e).v1_id, s.edge(e).v2_id
    t = 0.25

    res = s.split_edge(e, t)

    assert res is not None
    fid = res.face_a if res.face_a is not None else res.face_b
    _assert_fills(s, fid)
    _assert_sidecars_carried(s, fid, painted, old_loop, va, vb, res.vertex, t)


def test_a_painted_textured_convex_quad_splits_exactly_as_before():
    # The convex case the fan always got right. Nothing about its sidecars, its
    # loop or its covered area may change now that Scene re-earcuts it.
    s = Scene()
    f, ids = _add_loop(s, [(0, 0), (1, 0), (1, 1), (0, 1)])
    old_loop = s.face_loop(f)
    painted = _paint_everything(s, f, old_loop)
    e = s.edge_between(ids[0], ids[1])
    va, vb = s.edge(e).v1_id, s.edge(e).v2_id

    res = s.split_edge(e, 0.25)

    assert res is not None and res.face_a is not None
    # Covered area, not _assert_fills: the old fan from loop[0] put a
    # zero-area sliver on the split edge, which covers nothing but is not a
    # positive triangle. What it covers is what has to match.
    assert _area(s, res.face_a) == pytest.approx(1.0)
    _assert_sidecars_carried(s, res.face_a, painted, old_loop, va, vb, res.vertex, 0.25)


# ---- Review Focus 3: loading always re-earcuts (regression pin) ---------------


def test_a_u_face_saved_with_fan_corrupted_triangles_loads_filled_exactly(tmp_path):
    model = Model()
    scene = model.root.mesh
    f, _ids = _add_loop(scene, U_POINTS)
    loop = scene.face_loop(f)
    # Reproduce what a pre-fix split left behind: the fan from loop[0].
    fan = [v for i in range(1, len(loop) - 1) for v in (loop[0], loop[i], loop[i + 1])]
    scene._mesh.set_face_triangles(f, fan)
    assert _fill_problems(scene, f) != [], "the fan should be visibly wrong before saving"

    path = tmp_path / "u.berm"
    save_document(path, model, Camera(), DocumentSettings(), RenderStyle())
    loaded = load_document(path).model.root.mesh

    faces = list(loaded.faces_iter())
    assert len(faces) == 1
    _assert_fills(loaded, faces[0].id)
    assert _area(loaded, faces[0].id) == pytest.approx(U_AREA)
