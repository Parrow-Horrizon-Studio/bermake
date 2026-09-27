"""Scene.split_face: sub-loop construction and the kernel call (M7.6a)."""

import numpy as np
from bermake.scene.scene import Scene


def _quad(scene, z=0.0):
    ids = [
        scene.add_vertex(np.array(p, dtype=np.float32))
        for p in [(0, 0, z), (1, 0, z), (1, 1, z), (0, 1, z)]
    ]
    return scene.add_face_from_loop(ids), ids


def test_a_chord_between_opposite_corners_makes_two_triangles():
    s = Scene()
    fid, v = _quad(s)
    s.add_edge(v[0], v[2])

    out = s.split_face(fid, [v[0], v[2]])

    assert out is not None
    a, b = out
    assert sorted([len(s.face_loop(a)), len(s.face_loop(b))]) == [3, 3]
    assert len(list(s.faces_iter())) == 2


def test_a_chord_with_an_interior_vertex_keeps_it_on_both_children():
    """The chain's interior vertices belong to BOTH children's loops, once
    each, which is what makes them a shared boundary rather than a slit."""
    s = Scene()
    fid, v = _quad(s)
    mid = s.add_vertex(np.array((0.5, 0.5, 0.0), dtype=np.float32))
    s.add_edge(v[0], mid)
    s.add_edge(mid, v[2])

    out = s.split_face(fid, [v[0], mid, v[2]])

    assert out is not None
    a, b = out
    assert mid in s.face_loop(a)
    assert mid in s.face_loop(b)
    assert sorted([len(s.face_loop(a)), len(s.face_loop(b))]) == [4, 4]


def test_both_children_keep_the_parents_normal():
    s = Scene()
    fid, v = _quad(s)
    parent_normal = s.face_normal(fid).copy()
    s.add_edge(v[0], v[2])

    a, b = s.split_face(fid, [v[0], v[2]])

    for child in (a, b):
        assert float(np.dot(s.face_normal(child), parent_normal)) > 0.999


def test_a_chain_whose_ends_are_not_both_on_the_loop_is_refused():
    s = Scene()
    fid, v = _quad(s)
    stray = s.add_vertex(np.array((5.0, 5.0, 0.0), dtype=np.float32))
    s.add_edge(v[0], stray)

    assert s.split_face(fid, [v[0], stray]) is None


def test_a_chain_of_fewer_than_two_vertices_is_refused():
    s = Scene()
    fid, v = _quad(s)
    assert s.split_face(fid, [v[0]]) is None


def test_a_refused_split_leaves_the_face_alone():
    s = Scene()
    fid, v = _quad(s)
    assert s.split_face(fid, [v[0], v[1]]) is None  # adjacent corners: no interior
    assert len(list(s.faces_iter())) == 1
    assert len(s.face_loop(fid)) == 4


def test_a_dead_face_is_refused():
    s = Scene()
    fid, v = _quad(s)
    s.remove_face(fid)
    assert s.split_face(fid, [v[0], v[2]]) is None
    assert len(list(s.faces_iter())) == 0


def test_a_chain_with_a_duplicate_interior_vertex_is_refused():
    """A repeated vertex anywhere in the chain, not only at the ends, is
    refused by the same duplicate-vertex check."""
    s = Scene()
    fid, v = _quad(s)
    mid = s.add_vertex(np.array((0.5, 0.5, 0.0), dtype=np.float32))
    assert s.split_face(fid, [v[0], mid, mid, v[2]]) is None
    assert len(list(s.faces_iter())) == 1
    assert len(s.face_loop(fid)) == 4


def test_an_interior_chain_vertex_on_the_parents_loop_is_refused():
    """A chain vertex between the ends must not itself already be a loop
    vertex: that would make it a second chord end, not an interior point of
    a chain, and the two sub-loops built from it would not be simple."""
    s = Scene()
    fid, v = _quad(s)
    s.add_edge(v[0], v[1])
    assert s.split_face(fid, [v[0], v[1], v[2]]) is None
    assert len(list(s.faces_iter())) == 1
    assert len(s.face_loop(fid)) == 4


def test_a_chord_that_would_steal_an_unrelated_faces_edge_is_refused():
    """Whole-branch review Finding 1 (CRITICAL). A 'fin' triangle stands on
    the floor quad's own diagonal (v0-v2), which is not one of the floor's
    four boundary edges -- it only becomes a live edge because the fin's own
    add_face_from_loop call creates it. Splitting the floor along that same
    chord must not steal the edge out from under the fin: refusal is the
    correct answer (the edge would need three incident faces), not a
    fallback. This is exactly the state two ordinary Line-tool gestures
    leave behind: close a loop from v0 up to a point and back to v2, then
    draw v0 to v2 and press Enter."""
    s = Scene()
    fid, v = _quad(s)
    vt = s.add_vertex(np.array((2.0, 2.0, 1.0), dtype=np.float32))
    fin = s.add_face_from_loop([v[0], vt, v[2]])
    assert s._mesh.face_is_live(fin)
    diagonal = s.edge_between(v[0], v[2])
    assert diagonal is not None

    out = s.split_face(fid, [v[0], v[2]])

    assert out is None
    assert len(list(s.faces_iter())) == 2
    assert s._mesh.face_is_live(fid)
    assert s._mesh.face_is_live(fin), "the fin must not be silently detached from its own edge"
    assert len(s.face_loop(fid)) == 4


def test_a_chain_whose_first_and_last_vertices_match_is_refused_as_a_duplicate():
    """chain[0] == chain[-1] is itself a repeated vertex, so this hits the
    same duplicate-vertex check as the interior-duplicate case above rather
    than a separate "ends must differ" guard. `split_face` used to also
    carry a `len(forward) < 2 or len(backward) < 2` check for exactly this
    shape, but that check could only ever fire when first == last, which is
    already a duplicate the check above rejects first; it was dead code and
    has been removed. This test pins the duplicate-vertex check for this
    specific chain shape now that the redundant guard is gone."""
    s = Scene()
    fid, v = _quad(s)
    mid = s.add_vertex(np.array((0.5, 0.5, 0.0), dtype=np.float32))
    s.add_edge(v[0], mid)
    assert s.split_face(fid, [v[0], mid, v[0]]) is None
    assert len(list(s.faces_iter())) == 1
    assert len(s.face_loop(fid)) == 4
