"""Scene-level coverage for M3c dissolve_edge + faces_are_coplanar + query helpers."""

from __future__ import annotations

import numpy as np

from bermake.scene.scene import Scene


def _build_two_quads_sharing_edge(scene: Scene) -> tuple[int, int, int]:
    """Build two adjacent unit quads on the XY plane sharing edge v1—v2.
    Returns (f1_id, f2_id, shared_edge_id)."""
    v0 = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    v1 = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    v2 = scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    v3 = scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    v4 = scene.add_vertex(np.array([2.0, 0.0, 0.0], dtype=np.float32))
    v5 = scene.add_vertex(np.array([2.0, 1.0, 0.0], dtype=np.float32))
    scene.add_edge(v0, v1)
    e_shared = scene.add_edge(v1, v2)
    scene.add_edge(v2, v3)
    scene.add_edge(v3, v0)
    scene.add_edge(v1, v4)
    scene.add_edge(v4, v5)
    scene.add_edge(v5, v2)
    f1 = scene.add_face_from_loop([v0, v1, v2, v3])
    f2 = scene.add_face_from_loop([v1, v4, v5, v2])
    return f1, f2, e_shared


# ---- dissolve_edge wrapper -------------------------------------------------

def test_dissolve_edge_merges_two_quads_into_hexagon():
    scene = Scene()
    f1, f2, e_shared = _build_two_quads_sharing_edge(scene)

    merged = scene.dissolve_edge(e_shared)

    assert merged is not None
    assert len(scene.face(merged).loop_vertex_ids) == 6


def test_dissolve_edge_returns_none_on_boundary_edge():
    scene = Scene()
    v0 = scene.add_vertex(np.array([0, 0, 0], dtype=np.float32))
    v1 = scene.add_vertex(np.array([1, 0, 0], dtype=np.float32))
    e = scene.add_edge(v0, v1)
    # Edge has no faces — it's a "boundary" edge by virtue of no incidence.
    assert scene.dissolve_edge(e) is None


def test_dissolve_edge_refuses_a_pinched_merged_loop():
    """Whole-branch review Finding 2 (IMPORTANT).

    A = (v0, v1, v2); B = (v1, v0, v3, v2, v4). A and B share EXACTLY one
    edge (v0-v1), so the pre-existing "more than one shared edge" guard does
    not fire -- but vertex v2 also appears in both loops away from that
    shared edge, so the naive merge would build the pinched loop
    (v0, v3, v2, v4, v1, v2) with v2 twice. Scene._merge_corner_uvs is
    vertex-keyed, so a pinched loop like this would silently give both
    corners at v2 the same UV, wrong by construction.
    """
    scene = Scene()
    v0 = scene.add_vertex(np.array([0, 0, 0], dtype=np.float32))
    v1 = scene.add_vertex(np.array([1, 0, 0], dtype=np.float32))
    v2 = scene.add_vertex(np.array([1, 1, 0], dtype=np.float32))
    v3 = scene.add_vertex(np.array([0, 2, 0], dtype=np.float32))
    v4 = scene.add_vertex(np.array([2, 2, 0], dtype=np.float32))
    scene.add_edge(v0, v1)  # shared edge
    scene.add_edge(v1, v2)
    scene.add_edge(v2, v0)
    scene.add_edge(v0, v3)
    scene.add_edge(v3, v2)
    scene.add_edge(v2, v4)
    scene.add_edge(v4, v1)
    f1 = scene.add_face_from_loop([v0, v1, v2])
    f2 = scene.add_face_from_loop([v1, v0, v3, v2, v4])
    e_shared = scene.edge_between(v0, v1)
    assert e_shared is not None

    assert scene.dissolve_edge(e_shared) is None
    assert scene._mesh.face_is_live(f1)
    assert scene._mesh.face_is_live(f2)
    assert scene.edge_is_live(e_shared)


# ---- faces_are_coplanar wrapper -------------------------------------------

def test_faces_are_coplanar_with_default_tolerances():
    scene = Scene()
    f1, f2, _ = _build_two_quads_sharing_edge(scene)
    # Both on the XY plane → coplanar with default tolerances.
    assert scene.faces_are_coplanar(f1, f2) is True


def test_faces_are_coplanar_rejects_offset_planes():
    scene = Scene()
    # Two quads on parallel planes 0.001 apart (over 1e-4 dist tol).
    v0 = scene.add_vertex(np.array([0, 0, 0], dtype=np.float32))
    v1 = scene.add_vertex(np.array([1, 0, 0], dtype=np.float32))
    v2 = scene.add_vertex(np.array([0, 1, 0], dtype=np.float32))
    scene.add_edge(v0, v1); scene.add_edge(v1, v2); scene.add_edge(v2, v0)
    f1 = scene.add_face_from_loop([v0, v1, v2])

    v3 = scene.add_vertex(np.array([5, 0, 1e-3], dtype=np.float32))
    v4 = scene.add_vertex(np.array([6, 0, 1e-3], dtype=np.float32))
    v5 = scene.add_vertex(np.array([5, 1, 1e-3], dtype=np.float32))
    scene.add_edge(v3, v4); scene.add_edge(v4, v5); scene.add_edge(v5, v3)
    f2 = scene.add_face_from_loop([v3, v4, v5])

    assert scene.faces_are_coplanar(f1, f2) is False


# ---- query helpers --------------------------------------------------------

def test_face_edges_returns_boundary_edge_ids_in_order():
    scene = Scene()
    f1, _, e_shared = _build_two_quads_sharing_edge(scene)
    edges = scene.face_edges(f1)
    assert len(edges) == 4
    # f1 = quad [v0,v1,v2,v3]; the shared edge (v1,v2) is one of its 4 boundary
    # edges, so it must appear in the returned set.
    assert e_shared in edges
    # All four boundary edge ids must be DISTINCT. This is the assertion that
    # catches an id-mangling bug such as dividing the edge id by 2 (which would
    # collapse distinct ids into duplicates).
    assert len(set(edges)) == 4
    # All returned edges should be live and well-formed.
    for e in edges:
        assert scene.edge(e).v1_id != scene.edge(e).v2_id  # well-formed


def test_edge_faces_returns_both_adjacent_faces():
    scene = Scene()
    f1, f2, e_shared = _build_two_quads_sharing_edge(scene)
    faces = scene.edge_faces(e_shared)
    assert set(faces) == {f1, f2}


def test_edge_faces_returns_none_on_boundary_side():
    scene = Scene()
    f1, f2, e_shared = _build_two_quads_sharing_edge(scene)
    # f1's loop is [v0,v1,v2,v3]; face_edges returns its boundary edges in that
    # order. The shared edge (v1,v2) touches both f1 and f2; every OTHER edge of
    # f1 touches only f1. Pick a guaranteed true-boundary edge by excluding the
    # shared edge explicitly.
    boundary_edges = [e for e in scene.face_edges(f1) if e != e_shared]
    assert boundary_edges, "f1 must have at least one non-shared boundary edge"
    e_boundary = boundary_edges[0]

    faces = scene.edge_faces(e_boundary)
    # A true boundary edge has exactly one incident face (f1) and one None side.
    assert f1 in faces
    assert None in faces


def test_edge_is_boundary_true_for_standalone_face_edges():
    scene = Scene()
    v0 = scene.add_vertex(np.array([0, 0, 0], dtype=np.float32))
    v1 = scene.add_vertex(np.array([1, 0, 0], dtype=np.float32))
    v2 = scene.add_vertex(np.array([0, 1, 0], dtype=np.float32))
    scene.add_edge(v0, v1); scene.add_edge(v1, v2); scene.add_edge(v2, v0)
    f = scene.add_face_from_loop([v0, v1, v2])

    for e in scene.face_edges(f):
        assert scene.edge_is_boundary(e) is True


def test_edge_is_boundary_false_for_shared_edge():
    scene = Scene()
    _, _, e_shared = _build_two_quads_sharing_edge(scene)
    assert scene.edge_is_boundary(e_shared) is False
