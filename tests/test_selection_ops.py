"""Tests for bermake.selection_ops: pure selection derivations.

Every test here builds a bare Scene and calls the function directly. No
widget, no QApplication, no Model unless the function under test needs one.
"""

from __future__ import annotations

import numpy as np
import pytest


def _quad_pair():
    """Two quads sharing one edge, lying in the z=0 plane.

        d---c---f
        |   |   |
        a---b---e

    Returns (scene, ids) where ids is a dict of the named vertices, the two
    face ids and the shared edge id.
    """
    from bermake.scene import Scene

    scene = Scene()
    a = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    c = scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    d = scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    e = scene.add_vertex(np.array([2.0, 0.0, 0.0], dtype=np.float32))
    f = scene.add_vertex(np.array([2.0, 1.0, 0.0], dtype=np.float32))
    left = scene.add_face_from_loop((a, b, c, d))
    right = scene.add_face_from_loop((b, e, f, c))
    shared = scene.edge_between(b, c)
    assert shared is not None
    return scene, {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "e": e,
        "f": f,
        "left": left,
        "right": right,
        "shared": shared,
    }


def test_bounding_edges_returns_all_four_edges_of_a_quad():
    from bermake.selection_ops import bounding_edges

    scene, ids = _quad_pair()
    got = bounding_edges(scene, {ids["left"]})
    assert len(got) == 4
    assert ids["shared"] in got


def test_bounding_edges_unions_two_faces_and_counts_the_shared_edge_once():
    from bermake.selection_ops import bounding_edges

    scene, ids = _quad_pair()
    got = bounding_edges(scene, {ids["left"], ids["right"]})
    # Four edges each, one shared: 4 + 4 - 1 == 7.
    assert len(got) == 7


def test_bounding_edges_skips_a_dead_face_id():
    from bermake.selection_ops import bounding_edges

    scene, ids = _quad_pair()
    got = bounding_edges(scene, {ids["left"], 9999})
    assert len(got) == 4


def test_adjacent_faces_of_the_shared_edge_is_both_quads():
    from bermake.selection_ops import adjacent_faces

    scene, ids = _quad_pair()
    got = adjacent_faces(scene, {ids["shared"]})
    assert got == {ids["left"], ids["right"]}


def test_adjacent_faces_drops_the_none_side_of_a_boundary_edge():
    from bermake.selection_ops import adjacent_faces

    scene, ids = _quad_pair()
    outer = scene.edge_between(ids["a"], ids["b"])
    got = adjacent_faces(scene, {outer})
    assert got == {ids["left"]}


def test_adjacent_faces_of_a_naked_edge_is_empty():
    from bermake.scene import Scene
    from bermake.selection_ops import adjacent_faces

    scene = Scene()
    p = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    q = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    naked = scene.add_edge(p, q)
    assert adjacent_faces(scene, {naked}) == set()


def test_adjacent_faces_skips_a_dead_edge_id():
    from bermake.selection_ops import adjacent_faces

    scene, ids = _quad_pair()
    assert adjacent_faces(scene, {ids["shared"], 9999}) == {ids["left"], ids["right"]}


def test_incident_edges_of_the_shared_corner_finds_every_touching_edge():
    from bermake.selection_ops import incident_edges

    scene, ids = _quad_pair()
    got = incident_edges(scene, {ids["b"]})
    # b touches a-b, b-c (shared) and b-e.
    assert len(got) == 3
    assert ids["shared"] in got


def test_incident_edges_of_no_vertices_is_empty():
    from bermake.selection_ops import incident_edges

    scene, _ids = _quad_pair()
    assert incident_edges(scene, set()) == set()


@pytest.mark.parametrize("fn_name", ["bounding_edges", "adjacent_faces", "incident_edges"])
def test_neighbour_queries_return_a_plain_set_of_ints(fn_name):
    """Callers union these into Selection sets, which hold ints. A numpy
    integer would compare equal but hash into a set that later fails an
    `id in selection.edges` check against a plain int in some code paths."""
    import bermake.selection_ops as ops

    scene, ids = _quad_pair()
    seed = {
        "bounding_edges": {ids["left"]},
        "adjacent_faces": {ids["shared"]},
        "incident_edges": {ids["b"]},
    }[fn_name]
    got = getattr(ops, fn_name)(scene, seed)
    assert isinstance(got, set)
    assert all(type(x) is int for x in got)


def _two_islands():
    """Two quads that share no vertex, plus one loose edge touching neither.

    Returns (scene, ids) with both face ids, a vertex of each, and the loose
    edge's id.
    """
    from bermake.scene import Scene

    scene = Scene()
    a = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    c = scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    d = scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    one = scene.add_face_from_loop((a, b, c, d))

    p = scene.add_vertex(np.array([5.0, 0.0, 0.0], dtype=np.float32))
    q = scene.add_vertex(np.array([6.0, 0.0, 0.0], dtype=np.float32))
    r = scene.add_vertex(np.array([6.0, 1.0, 0.0], dtype=np.float32))
    s = scene.add_vertex(np.array([5.0, 1.0, 0.0], dtype=np.float32))
    two = scene.add_face_from_loop((p, q, r, s))

    m = scene.add_vertex(np.array([9.0, 0.0, 0.0], dtype=np.float32))
    n = scene.add_vertex(np.array([9.0, 1.0, 0.0], dtype=np.float32))
    loose = scene.add_edge(m, n)
    return scene, {"a": a, "p": p, "one": one, "two": two, "loose": loose, "m": m}


def test_flood_from_one_quad_reaches_its_own_face_and_edges_only():
    from bermake.selection_ops import connected_component

    scene, ids = _two_islands()
    verts, edges, faces = connected_component(scene, {ids["a"]})
    assert faces == {ids["one"]}
    assert len(verts) == 4
    assert len(edges) == 4
    assert ids["loose"] not in edges


def test_flood_does_not_jump_to_a_disjoint_island():
    from bermake.selection_ops import connected_component

    scene, ids = _two_islands()
    _verts, _edges, faces = connected_component(scene, {ids["a"]})
    assert ids["two"] not in faces


def test_flood_crosses_a_shared_edge_between_two_quads():
    from bermake.selection_ops import connected_component

    scene, ids = _quad_pair()
    _verts, _edges, faces = connected_component(scene, {ids["a"]})
    assert faces == {ids["left"], ids["right"]}


def test_flood_crosses_a_single_shared_corner():
    """Connectivity is through shared VERTICES, not shared edges (spec 2.4).
    Two quads meeting at one corner are one component."""
    from bermake.scene import Scene
    from bermake.selection_ops import connected_component

    scene = Scene()
    a = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    corner = scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    d = scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    lower = scene.add_face_from_loop((a, b, corner, d))

    e = scene.add_vertex(np.array([2.0, 1.0, 0.0], dtype=np.float32))
    f = scene.add_vertex(np.array([2.0, 2.0, 0.0], dtype=np.float32))
    g = scene.add_vertex(np.array([1.0, 2.0, 0.0], dtype=np.float32))
    upper = scene.add_face_from_loop((corner, e, f, g))

    _verts, _edges, faces = connected_component(scene, {a})
    assert faces == {lower, upper}


def test_flood_includes_a_loose_edge_hanging_off_a_face():
    from bermake.selection_ops import connected_component

    scene, ids = _quad_pair()
    tip = scene.add_vertex(np.array([0.0, -1.0, 0.0], dtype=np.float32))
    spur = scene.add_edge(ids["a"], tip)
    _verts, edges, _faces = connected_component(scene, {ids["a"]})
    assert spur in edges


def test_flood_result_does_not_depend_on_which_kind_seeded_it():
    """Spec section 4 property 2: seeding from a face's vertex, from an
    endpoint of one of its edges, and from a far corner all agree."""
    from bermake.selection_ops import connected_component

    scene, ids = _quad_pair()
    from_a = connected_component(scene, {ids["a"]})
    from_b = connected_component(scene, {ids["b"]})
    from_f = connected_component(scene, {ids["f"]})
    assert from_a == from_b == from_f


def test_flood_from_no_seed_is_three_empty_sets():
    from bermake.selection_ops import connected_component

    scene, _ids = _quad_pair()
    assert connected_component(scene, set()) == (set(), set(), set())


def test_flood_skips_a_dead_seed_id():
    from bermake.selection_ops import connected_component

    scene, ids = _quad_pair()
    verts, _edges, faces = connected_component(scene, {ids["a"], 9999})
    assert 9999 not in verts
    assert faces == {ids["left"], ids["right"]}


def test_grow_adds_the_face_across_a_shared_edge():
    from bermake.selection_ops import grow

    scene, ids = _quad_pair()
    _edges, faces, _verts = grow(scene, edges=set(), faces={ids["left"]}, vertices=set())
    assert faces == {ids["left"], ids["right"]}


def test_grow_does_not_turn_a_face_selection_into_edges():
    """Spec D10: kinds do not bleed. Growing faces yields faces."""
    from bermake.selection_ops import grow

    scene, ids = _quad_pair()
    edges, _faces, verts = grow(scene, edges=set(), faces={ids["left"]}, vertices=set())
    assert edges == set()
    assert verts == set()


def test_grow_adds_edges_sharing_a_vertex():
    from bermake.selection_ops import grow

    scene, ids = _quad_pair()
    ab = scene.edge_between(ids["a"], ids["b"])
    edges, _faces, _verts = grow(scene, edges={ab}, faces=set(), vertices=set())
    assert ab in edges
    assert len(edges) > 1


def _nine_quad_grid():
    """A 3x3 grid of unit quads sharing edges, lying in the z=0 plane.

        f02 f12 f22
        f01 f11 f21
        f00 f10 f20

    `ids["f{row}{col}"]` names each cell, row and col in 0..2; `ids["center"]`
    aliases f11, the one cell with all four edge-neighbours live inside the
    grid. Vertices are shared across adjacent cells, so face adjacency
    (sharing an edge) is real, unlike a diagonal pair that only shares a
    corner.
    """
    from bermake.scene import Scene

    scene = Scene()
    verts = {}
    for y in range(4):
        for x in range(4):
            verts[(x, y)] = scene.add_vertex(np.array([float(x), float(y), 0.0], dtype=np.float32))

    ids = {}
    for row in range(3):
        for col in range(3):
            a = verts[(col, row)]
            b = verts[(col + 1, row)]
            c = verts[(col + 1, row + 1)]
            d = verts[(col, row + 1)]
            ids[f"f{row}{col}"] = scene.add_face_from_loop((a, b, c, d))
    ids["center"] = ids["f11"]
    return scene, ids


def test_shrink_inverts_grow_on_an_interior_region():
    """Spec section 4 property 1.

    The quad pair (two faces total) cannot exercise this: growing either
    face already reaches the whole universe, so shrinking has nothing on
    the boundary to erode and the round trip passes even if `shrink` is
    broken. A 3x3 grid gives the centre face a genuine interior.

    Hand-derived expectation:
    - `grow({center})` reaches center plus its four edge-adjacent
      neighbours (f01, f10, f12, f21) -- a plus shape of 5 faces. The four
      diagonal corners (f00, f02, f20, f22) share only a vertex with the
      center, not an edge, so one-hop face adjacency does not reach them.
    - Shrinking that plus shape drops every arm: f01's other grid neighbours
      are f00 and f02, neither in the plus shape, so f01 is on the boundary
      and is dropped; f10, f12 and f21 are each boundary for the same
      reason against their own two non-center grid neighbours. The center's
      neighbours are exactly the four arms, all of which ARE in the plus
      shape, so the center is interior and is kept.
    - Net: grow(center) -> shrink -> {center}, the original seed.
    """
    from bermake.selection_ops import grow, shrink

    scene, ids = _nine_quad_grid()
    center = ids["center"]
    _e, grown, _v = grow(scene, edges=set(), faces={center}, vertices=set())
    assert grown == {center, ids["f01"], ids["f10"], ids["f12"], ids["f21"]}

    _e, back_f, _v = shrink(scene, edges=set(), faces=grown, vertices=set())
    assert back_f == {center}


def test_shrink_removes_a_face_whose_neighbour_is_not_selected():
    from bermake.selection_ops import shrink

    scene, ids = _quad_pair()
    _e, faces, _v = shrink(scene, edges=set(), faces={ids["left"]}, vertices=set())
    assert faces == set()


def test_shrink_keeps_a_face_whose_every_neighbour_is_selected():
    from bermake.selection_ops import shrink

    scene, ids = _quad_pair()
    _e, faces, _v = shrink(scene, edges=set(), faces={ids["left"], ids["right"]}, vertices=set())
    assert faces == {ids["left"], ids["right"]}


def test_same_material_matches_on_either_side():
    """Spec D8: a user cannot see which side dictionary the paint came from."""
    from bermake.scene.scene import Side
    from bermake.selection_ops import same_material

    scene, ids = _quad_pair()
    scene.set_face_material(ids["left"], 7, Side.FRONT)
    scene.set_face_material(ids["right"], 7, Side.BACK)
    assert same_material(scene, {ids["left"]}) == {ids["left"], ids["right"]}


def test_same_material_of_an_unpainted_face_finds_the_other_unpainted_faces():
    """Material 0 is Default, which is a real answer rather than a null: two
    unpainted faces do share a material."""
    from bermake.selection_ops import same_material

    scene, ids = _quad_pair()
    assert same_material(scene, {ids["left"]}) == {ids["left"], ids["right"]}


def test_same_material_of_no_seed_is_empty():
    from bermake.selection_ops import same_material

    scene, _ids = _quad_pair()
    assert same_material(scene, set()) == set()


def test_same_material_does_not_pull_in_an_unpainted_face():
    """Controller ruling: Default only seeds when the seed carries no real
    material on either side. A face painted on one side only must not match
    every unpainted face in the model -- that would make "All with Same
    Material" select the whole document off one wall."""
    from bermake.scene.scene import Side
    from bermake.selection_ops import same_material

    scene, ids = _quad_pair()
    scene.set_face_material(ids["left"], 7, Side.FRONT)
    assert same_material(scene, {ids["left"]}) == {ids["left"]}


def test_same_material_of_a_mixed_seed_matches_both_families():
    """Controller ruling refinement: Default participates per seed face, not
    pooled across the whole seed set. Seeding a painted face together with
    an unpainted one must match both families, including the unpainted seed
    matching itself. A global pool (drop Default from the whole seed set
    once ANY seed face has a real material) would silently exclude the
    unpainted seed from its own result -- that was the bug this test pins."""
    from bermake.scene.scene import Side
    from bermake.selection_ops import same_material

    scene, ids = _quad_pair()
    scene.set_face_material(ids["left"], 7, Side.FRONT)
    # ids["right"] stays wholly unpainted (Default on both sides).
    got = same_material(scene, {ids["left"], ids["right"]})
    assert got == {ids["left"], ids["right"]}


def test_invert_returns_everything_not_selected():
    from bermake.model.model import Model
    from bermake.selection import Selection
    from bermake.selection_ops import invert

    model = Model()
    scene = model.active_scene
    a = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    c = scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    one = scene.add_edge(a, b)
    two = scene.add_edge(b, c)
    sel = Selection()
    sel.replace(edges={one})
    edges, _faces, _instances, _verts = invert(model, sel, select_vertices=False)
    assert edges == {two}


def test_invert_leaves_vertices_alone_when_the_mode_is_off():
    from bermake.model.model import Model
    from bermake.selection import Selection
    from bermake.selection_ops import invert

    model = Model()
    scene = model.active_scene
    scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    _e, _f, _i, verts = invert(model, Selection(), select_vertices=False)
    assert verts == set()


def test_invert_includes_vertices_when_the_mode_is_on():
    from bermake.model.model import Model
    from bermake.selection import Selection
    from bermake.selection_ops import invert

    model = Model()
    scene = model.active_scene
    a = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    sel = Selection()
    sel.replace(vertices={a})
    _e, _f, _i, verts = invert(model, sel, select_vertices=True)
    assert verts == {b}


def test_invert_twice_returns_the_original_selection():
    from bermake.model.model import Model
    from bermake.selection import Selection
    from bermake.selection_ops import invert

    model = Model()
    scene = model.active_scene
    a = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    c = scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    one = scene.add_edge(a, b)
    scene.add_edge(b, c)
    sel = Selection()
    sel.replace(edges={one})
    e1, f1, i1, v1 = invert(model, sel, select_vertices=False)
    sel.replace(edges=e1, faces=f1, instances=i1, vertices=v1)
    e2, _f2, _i2, _v2 = invert(model, sel, select_vertices=False)
    assert e2 == {one}


def test_invert_excludes_a_tag_hidden_instance_from_the_universe(model_factory, group_factory):
    """Spec D7: invert's universe is select_all_ids, which already excludes
    instances hidden by tag visibility (see test_model_queries.py's
    test_select_all_excludes_hidden_tag_instances, whose fixture this
    mirrors). A hidden instance must never appear on either side of invert:
    not selected, and not "everything else" either."""
    from bermake.selection import Selection
    from bermake.selection_ops import invert

    model = model_factory()
    scene = model.active_context.mesh
    a = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    c = scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    d = scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    scene.add_face_from_loop((a, b, c, d))
    instance = group_factory(model)
    hidden_tag = model.tags.add("Hidden")
    instance.tag_id = hidden_tag.id
    model.tags.set_visible(hidden_tag.id, False)

    _e, _f, instances, _v = invert(model, Selection(), select_vertices=False)

    assert instance.id not in instances


def test_same_tag_matches_instances_sharing_a_tag(model_factory, group_factory):
    """Written per the brief's instruction to establish the instance-building
    idiom from tests/test_selection_instances.py -- that file turned out to
    hold no such idiom (it never builds a Model), so this instead follows
    test_model_queries.py's model_factory/group_factory fixtures, which are
    the established way elsewhere in this suite to get a real Instance."""
    from bermake.selection_ops import same_tag

    model = model_factory()
    scene = model.active_context.mesh
    a = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    c = scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    d = scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    scene.add_face_from_loop((a, b, c, d))
    one = group_factory(model)

    other_definition = model.new_definition("Other", is_group=True)
    two = model.new_instance(other_definition)
    model.root.children.append(two)

    tag = model.tags.add("Walls")
    one.tag_id = tag.id
    two.tag_id = tag.id

    assert same_tag(model, {one.id}) == {one.id, two.id}


def test_same_tag_does_not_match_a_differently_tagged_instance(model_factory, group_factory):
    from bermake.selection_ops import same_tag

    model = model_factory()
    scene = model.active_context.mesh
    a = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    c = scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    d = scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    scene.add_face_from_loop((a, b, c, d))
    one = group_factory(model)

    other_definition = model.new_definition("Other", is_group=True)
    two = model.new_instance(other_definition)
    model.root.children.append(two)

    walls = model.tags.add("Walls")
    doors = model.tags.add("Doors")
    one.tag_id = walls.id
    two.tag_id = doors.id

    assert same_tag(model, {one.id}) == {one.id}


def test_same_tag_of_no_seed_is_empty(model_factory, group_factory):
    from bermake.selection_ops import same_tag

    model = model_factory()
    scene = model.active_context.mesh
    a = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    c = scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    d = scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    scene.add_face_from_loop((a, b, c, d))
    group_factory(model)

    assert same_tag(model, set()) == set()


# --- Final review I6: every set operation skips a dead id -------------------
#
# The three neighbour queries above each had a "skips a dead id" test; none of
# the five set operations did. That gap is what let `same_material`'s dead
# `except KeyError: continue` survive review: `Scene.face_material` is a dict
# `.get` with a Default fallback and never raises, so a stale seed id
# contributed Default and the match loop returned every unpainted face in the
# document. The module docstring promises "a dead id is skipped rather than
# raised on" for all of them, so all of them are now pinned.


def test_same_material_skips_a_dead_face_id():
    """The regression this group exists for: a dead seed must contribute
    nothing, not Default. Before the fix this returned both faces."""
    from bermake.selection_ops import same_material

    scene, _ids = _quad_pair()
    assert same_material(scene, {9999}) == set()


def test_same_material_skips_a_dead_id_beside_a_live_one():
    """A dead id riding along with a live seed must not widen the result
    either: the live seed is painted, so the answer is its own family."""
    from bermake.scene.scene import Side
    from bermake.selection_ops import same_material

    scene, ids = _quad_pair()
    scene.set_face_material(ids["left"], 7, Side.FRONT)
    assert same_material(scene, {ids["left"], 9999}) == {ids["left"]}


def test_grow_skips_dead_ids_of_every_kind():
    from bermake.selection_ops import grow

    scene, ids = _quad_pair()
    ab = scene.edge_between(ids["a"], ids["b"])
    edges, faces, verts = grow(
        scene, edges={ab, 9999}, faces={ids["left"], 8888}, vertices={ids["a"], 7777}
    )
    assert 9999 not in edges
    assert 8888 not in faces
    assert 7777 not in verts
    # The live ids still grow exactly as they do without the dead company.
    live_edges, live_faces, live_verts = grow(
        scene, edges={ab}, faces={ids["left"]}, vertices={ids["a"]}
    )
    assert (edges, faces, verts) == (live_edges, live_faces, live_verts)


def test_shrink_skips_dead_ids_of_every_kind():
    from bermake.selection_ops import shrink

    scene, ids = _quad_pair()
    ab = scene.edge_between(ids["a"], ids["b"])
    edges, faces, verts = shrink(
        scene, edges={ab, 9999}, faces={ids["left"], 8888}, vertices={ids["a"], 7777}
    )
    assert 9999 not in edges
    assert 8888 not in faces
    assert 7777 not in verts


def test_shrink_does_not_keep_an_edge_only_because_a_dead_neighbour_is_absent():
    """A dead id in the selection must not make a boundary edge look
    interior. Every edge of this quad pair has a live neighbour outside the
    seed, so nothing survives the erosion."""
    from bermake.selection_ops import shrink

    scene, ids = _quad_pair()
    ab = scene.edge_between(ids["a"], ids["b"])
    edges, _faces, _verts = shrink(scene, edges={ab, 9999}, faces=set(), vertices=set())
    assert edges == set()


def test_same_tag_skips_a_dead_instance_id(model_factory, group_factory):
    from bermake.selection_ops import same_tag

    model = model_factory()
    scene = model.active_context.mesh
    a = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    c = scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    d = scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    scene.add_face_from_loop((a, b, c, d))
    group_factory(model)

    assert same_tag(model, {9999}) == set()


def test_invert_skips_a_dead_id_in_the_selection():
    from bermake.model.model import Model
    from bermake.selection import Selection
    from bermake.selection_ops import invert

    model = Model()
    scene = model.active_context.mesh
    a = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    c = scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    d = scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    face = scene.add_face_from_loop((a, b, c, d))

    sel = Selection()
    sel.replace(faces={face, 8888}, edges={9999})
    edges, faces, _instances, _verts = invert(model, sel, select_vertices=False)

    assert 8888 not in faces
    assert 9999 not in edges
    assert faces == set()
    assert edges == {e.id for e in scene.edges_iter()}
