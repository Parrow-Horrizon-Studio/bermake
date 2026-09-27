"""Unit tests for the Python scene data model (Vertex, Edge, Face, Scene)."""

from __future__ import annotations

import numpy as np
import pytest


def test_vertex_holds_id_and_position():
    from bermake.scene import Vertex

    v = Vertex(id=7, position=np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert v.id == 7
    np.testing.assert_array_equal(v.position, np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_vertex_is_frozen():
    from bermake.scene import Vertex

    v = Vertex(id=0, position=np.array([0.0, 0.0, 0.0], dtype=np.float32))
    with pytest.raises(Exception):
        v.id = 1  # type: ignore[misc]


def test_edge_holds_id_and_two_vertex_ids():
    from bermake.scene import Edge

    e = Edge(id=3, v1_id=10, v2_id=20)
    assert e.id == 3
    assert e.v1_id == 10
    assert e.v2_id == 20


def test_face_holds_id_loop_normal_triangles():
    from bermake.scene import Face

    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    f = Face(
        id=5,
        loop_vertex_ids=(0, 1, 2, 3),
        plane_normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        triangles=triangles,
    )
    assert f.id == 5
    assert f.loop_vertex_ids == (0, 1, 2, 3)
    np.testing.assert_array_equal(f.plane_normal, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    np.testing.assert_array_equal(f.triangles, triangles)


# ---------------------------------------------------------------------------
# Scene tests (Task 3)
# ---------------------------------------------------------------------------


def test_scene_starts_empty():
    from bermake.scene import Scene

    s = Scene()
    assert len(list(s.vertices_iter())) == 0
    assert len(list(s.edges_iter())) == 0
    assert len(list(s.faces_iter())) == 0
    assert s.dirty is False


def test_add_vertex_returns_new_id_when_position_is_new():
    from bermake.scene import Scene

    s = Scene()
    v0 = s.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    v1 = s.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert v0 != v1
    assert s.dirty is True


def test_add_vertex_is_idempotent_on_exact_match():
    from bermake.scene import Scene

    s = Scene()
    pos = np.array([2.0, 3.0, 0.0], dtype=np.float32)
    v0 = s.add_vertex(pos)
    v1 = s.add_vertex(pos.copy())  # different array object, same exact values
    assert v0 == v1


def test_clear_resets_dirty_flag_and_removes_everything():
    from bermake.scene import Scene

    s = Scene()
    s.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    s.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    s.mark_clean()
    assert s.dirty is False

    s.clear()
    assert len(list(s.vertices_iter())) == 0
    assert len(list(s.edges_iter())) == 0
    assert len(list(s.faces_iter())) == 0
    assert s.dirty is True


def test_vertex_lookup_by_id():
    from bermake.scene import Scene

    s = Scene()
    pos = np.array([5.0, 6.0, 0.0], dtype=np.float32)
    vid = s.add_vertex(pos)
    v = s.vertex(vid)
    assert v.id == vid
    np.testing.assert_array_equal(v.position, pos)


# ---------------------------------------------------------------------------
# Hardening tests (Task 2 carry-overs)
# ---------------------------------------------------------------------------


def test_vertex_is_hashable_by_id():
    from bermake.scene import Vertex

    a = Vertex(id=7, position=np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b = Vertex(id=7, position=np.array([9.0, 9.0, 9.0], dtype=np.float32))
    c = Vertex(id=8, position=np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert hash(a) == hash(b)  # same id → same hash
    assert a == b               # equality by id
    assert a != c               # different id → not equal
    s = {a, b, c}
    assert len(s) == 2          # works in a set


def test_vertex_position_is_immutable():
    from bermake.scene import Vertex

    v = Vertex(id=0, position=np.array([1.0, 2.0, 3.0], dtype=np.float32))
    with pytest.raises(ValueError):
        v.position[0] = 99.0  # array writability locked


def test_face_position_arrays_are_immutable():
    from bermake.scene import Face

    f = Face(
        id=0,
        loop_vertex_ids=(0, 1, 2),
        plane_normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        triangles=np.array([[0, 1, 2]], dtype=np.int32),
    )
    with pytest.raises(ValueError):
        f.plane_normal[0] = 99.0
    with pytest.raises(ValueError):
        f.triangles[0, 0] = 99


def test_add_vertex_collapses_negative_zero():
    """`-0.0` and `0.0` must dedupe to the same vertex (computed coords can produce -0.0)."""
    from bermake.scene import Scene

    s = Scene()
    v0 = s.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    v1 = s.add_vertex(np.array([-0.0, 0.0, 0.0], dtype=np.float32))
    assert v0 == v1
    assert len(list(s.vertices_iter())) == 1


# ---------------------------------------------------------------------------
# add_edge tests (Task 4)
# ---------------------------------------------------------------------------


def test_add_edge_returns_new_id():
    from bermake.scene import Scene

    s = Scene()
    v0 = s.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    v1 = s.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    e = s.add_edge(v0, v1)
    assert isinstance(e, int)
    assert e == 0  # first edge in a fresh scene starts at 0


def test_add_edge_is_idempotent_unordered():
    from bermake.scene import Scene

    s = Scene()
    v0 = s.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    v1 = s.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    a = s.add_edge(v0, v1)
    b = s.add_edge(v1, v0)  # swapped order
    assert a == b


def test_add_edge_rejects_self_loop():
    from bermake.scene import Scene

    s = Scene()
    v0 = s.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    with pytest.raises(ValueError):
        s.add_edge(v0, v0)


def test_add_edge_canonicalises_endpoints():
    from bermake.scene import Scene

    s = Scene()
    v0 = s.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    v1 = s.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    eid = s.add_edge(v1, v0)
    e = next(iter(s.edges_iter()))
    assert e.id == eid
    assert e.v1_id == min(v0, v1)
    assert e.v2_id == max(v0, v1)


def test_add_edge_rejects_unknown_vertex_ids():
    from bermake.scene import Scene

    s = Scene()
    v0 = s.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    with pytest.raises(KeyError):
        s.add_edge(v0, 999)  # 999 doesn't exist
    with pytest.raises(KeyError):
        s.add_edge(999, v0)  # both directions
    # No edge should have been added.
    assert len(list(s.edges_iter())) == 0


# ---------------------------------------------------------------------------
# add_face_from_loop tests (Task 5)
# ---------------------------------------------------------------------------


def test_add_face_from_loop_creates_face_and_triangulates():
    from bermake.scene import Scene

    s = Scene()
    # A unit square on Z=0
    v0 = s.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    v1 = s.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    v2 = s.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    v3 = s.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32))

    fid = s.add_face_from_loop((v0, v1, v2, v3))

    faces = list(s.faces_iter())
    assert len(faces) == 1
    f = faces[0]
    assert f.id == fid
    assert f.loop_vertex_ids == (v0, v1, v2, v3)
    # Ground plane normal is +Z
    np.testing.assert_allclose(f.plane_normal, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    # Square triangulates to 2 triangles
    assert f.triangles.shape == (2, 3)
    # And every triangle vertex must be a known global vertex ID (not a leaked local index).
    assert set(f.triangles.flatten().tolist()).issubset({v0, v1, v2, v3})


def test_add_face_from_loop_rejects_fewer_than_three_vertices():
    from bermake.scene import Scene

    s = Scene()
    v0 = s.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    v1 = s.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))

    with pytest.raises(ValueError):
        s.add_face_from_loop((v0, v1))


def test_add_face_from_loop_triangulates_concave_polygon():
    from bermake.scene import Scene

    s = Scene()
    # An L-shape (6 vertices, concave) on Z=0
    pts = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
    vids = [
        s.add_vertex(np.array([x, y, 0.0], dtype=np.float32)) for (x, y) in pts
    ]

    s.add_face_from_loop(tuple(vids))

    f = next(iter(s.faces_iter()))
    # An L-shape has 6 vertices, so earcut produces 4 triangles
    assert f.triangles.shape == (4, 3)


def test_add_face_from_loop_rejects_unknown_vertex_id():
    """Phantom vertex IDs in the loop must raise — topology coherence."""
    from bermake.scene import Scene

    s = Scene()
    v0 = s.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    v1 = s.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    with pytest.raises(KeyError):
        s.add_face_from_loop((v0, v1, 999))
    assert len(list(s.faces_iter())) == 0


def test_add_face_from_loop_accepts_list_and_stores_as_tuple():
    """Accept any Sequence; the stored loop_vertex_ids must be a tuple."""
    from bermake.scene import Scene

    s = Scene()
    v0 = s.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    v1 = s.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    v2 = s.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32))

    fid = s.add_face_from_loop([v0, v1, v2])  # list, not tuple

    f = next(iter(s.faces_iter()))
    assert isinstance(f.loop_vertex_ids, tuple)
    assert f.loop_vertex_ids == (v0, v1, v2)


# ---------------------------------------------------------------------------
# Scene helper tests (Task 6)
# ---------------------------------------------------------------------------


def test_find_vertex_near_returns_closest_within_tolerance():
    from bermake.scene import Scene

    s = Scene()
    v0 = s.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    v1 = s.add_vertex(np.array([5.0, 0.0, 0.0], dtype=np.float32))

    near_v0 = s.find_vertex_near(np.array([0.1, 0.0, 0.0], dtype=np.float32), tolerance=0.5)
    assert near_v0 == v0

    near_v1 = s.find_vertex_near(np.array([5.05, 0.0, 0.0], dtype=np.float32), tolerance=0.5)
    assert near_v1 == v1


def test_find_vertex_near_returns_none_when_outside_tolerance():
    from bermake.scene import Scene

    s = Scene()
    s.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))

    assert s.find_vertex_near(np.array([10.0, 0.0, 0.0], dtype=np.float32), tolerance=0.5) is None


def test_find_vertex_near_picks_closest_when_multiple_within_tolerance():
    from bermake.scene import Scene

    s = Scene()
    v_far = s.add_vertex(np.array([0.3, 0.0, 0.0], dtype=np.float32))
    v_near = s.add_vertex(np.array([0.05, 0.0, 0.0], dtype=np.float32))

    # Cursor at origin — both within tolerance=1.0, but v_near is closer.
    got = s.find_vertex_near(np.array([0.0, 0.0, 0.0], dtype=np.float32), tolerance=1.0)
    assert got == v_near


def test_edge_line_buffer_shape():
    from bermake.scene import Scene

    s = Scene()
    v0 = s.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    v1 = s.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    v2 = s.add_vertex(np.array([2.0, 0.0, 0.0], dtype=np.float32))
    s.add_edge(v0, v1)
    s.add_edge(v1, v2)

    buf = s.edge_line_buffer()
    assert buf.shape == (4, 3)  # 2 edges * 2 endpoints
    assert buf.dtype == np.float32


def test_edge_line_buffer_is_empty_when_no_edges():
    from bermake.scene import Scene

    s = Scene()
    s.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))  # vertex but no edge

    buf = s.edge_line_buffer()
    assert buf.shape == (0, 3)
    assert buf.dtype == np.float32


def test_face_triangle_buffer_shape():
    from bermake.scene import Scene

    s = Scene()
    v0 = s.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    v1 = s.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    v2 = s.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    v3 = s.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    s.add_face_from_loop((v0, v1, v2, v3))

    positions, normals = s.face_triangle_buffer()
    # 2 triangles * 3 vertices = 6 vertices
    assert positions.shape == (6, 3)
    assert normals.shape == (6, 3)
    # All normals should be +Z for a ground-plane face
    np.testing.assert_allclose(normals, np.tile([0.0, 0.0, 1.0], (6, 1)).astype(np.float32))


def test_face_triangle_buffer_is_empty_when_no_faces():
    from bermake.scene import Scene

    s = Scene()
    s.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    positions, normals = s.face_triangle_buffer()
    assert positions.shape == (0, 3)
    assert normals.shape == (0, 3)


# ---------------------------------------------------------------------------
# remove_* / restore_* tests (Task 12)
# ---------------------------------------------------------------------------


def test_remove_face_leaves_verts_and_edges_alive():
    from bermake.scene import Scene

    s = Scene()
    v0 = s.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    v1 = s.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    v2 = s.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    s.add_edge(v0, v1)
    s.add_edge(v1, v2)
    s.add_edge(v2, v0)
    f = s.add_face_from_loop((v0, v1, v2))

    s.remove_face(f)

    assert len(list(s.faces_iter())) == 0
    # Vertices and edges still alive.
    assert len(list(s.vertices_iter())) == 3
    assert len(list(s.edges_iter())) == 3


def test_remove_edge_rejects_if_face_still_uses_it():
    from bermake.scene import Scene

    s = Scene()
    v0 = s.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    v1 = s.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    v2 = s.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    e0 = s.add_edge(v0, v1)
    s.add_edge(v1, v2)
    s.add_edge(v2, v0)
    s.add_face_from_loop((v0, v1, v2))

    with pytest.raises(ValueError):
        s.remove_edge(e0)


def test_remove_vertex_rejects_if_edge_still_uses_it():
    from bermake.scene import Scene

    s = Scene()
    v0 = s.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    v1 = s.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    s.add_edge(v0, v1)

    with pytest.raises(ValueError):
        s.remove_vertex(v0)


def test_restore_face_round_trip():
    from bermake.scene import Scene

    s = Scene()
    v0 = s.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    v1 = s.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    v2 = s.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    s.add_edge(v0, v1)
    s.add_edge(v1, v2)
    s.add_edge(v2, v0)
    f = s.add_face_from_loop((v0, v1, v2))
    captured_loop = s.face(f).loop_vertex_ids

    s.remove_face(f)
    assert len(list(s.faces_iter())) == 0

    s.restore_face(f, captured_loop)
    assert len(list(s.faces_iter())) == 1
    restored = s.face(f)
    assert restored.loop_vertex_ids == captured_loop


def test_add_vertex_after_tombstone_at_same_position_allocates_new_id():
    """Position-index only tracks live vertices; tombstoned slots stay tombstoned."""
    from bermake.scene import Scene

    s = Scene()
    pos = np.array([5.0, 5.0, 0.0], dtype=np.float32)
    v0 = s.add_vertex(pos)
    s.remove_vertex(v0)
    v1 = s.add_vertex(pos.copy())
    assert v1 != v0  # new ID; old slot stays tombstoned


# ---------------------------------------------------------------------------
# M3b: ray_pick_face / face_loop / face_normal / face_center (Task 4)
# ---------------------------------------------------------------------------


class TestSceneRayPickFace:
    """Scene.ray_pick_face — thin wrapper over bermake._core.ray_intersect_mesh."""

    def test_returns_none_for_empty_scene(self):
        from bermake.scene import Scene

        scene = Scene()
        hit = scene.ray_pick_face(
            origin=np.array([0.0, 0.0, 5.0], dtype=np.float32),
            direction=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        )
        assert hit is None

    def test_returns_face_id_when_ray_hits(self):
        from bermake.scene import Scene

        scene = Scene()
        v0 = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        v1 = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        v2 = scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
        v3 = scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        f = scene.add_face_from_loop([v0, v1, v2, v3])

        hit = scene.ray_pick_face(
            origin=np.array([0.5, 0.5, 5.0], dtype=np.float32),
            direction=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        )
        assert hit is not None
        assert hit.face_id == f
        assert hit.t == pytest.approx(5.0, abs=1e-4)

    def test_returns_none_after_face_removed(self):
        from bermake.scene import Scene

        scene = Scene()
        v0 = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        v1 = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        v2 = scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
        v3 = scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        f = scene.add_face_from_loop([v0, v1, v2, v3])
        scene.remove_face(f)

        hit = scene.ray_pick_face(
            origin=np.array([0.5, 0.5, 5.0], dtype=np.float32),
            direction=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        )
        assert hit is None


class TestSceneFaceLoopNormalCenter:
    """face_loop / face_normal / face_center — extrusion composite needs these."""

    def _make_unit_rect(self):
        from bermake.scene import Scene

        scene = Scene()
        v0 = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        v1 = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        v2 = scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
        v3 = scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        f = scene.add_face_from_loop([v0, v1, v2, v3])
        return scene, f, (v0, v1, v2, v3)

    def test_face_loop_returns_boundary_vertex_ids_in_insertion_order(self):
        scene, f, (v0, v1, v2, v3) = self._make_unit_rect()
        loop = scene.face_loop(f)
        assert loop == [v0, v1, v2, v3]

    def test_face_loop_raises_keyerror_on_invalid_face_id(self):
        from bermake.scene import Scene

        scene = Scene()
        with pytest.raises(KeyError):
            scene.face_loop(99)

    def test_face_normal_on_xy_face_is_plus_z(self):
        scene, f, _ = self._make_unit_rect()
        n = scene.face_normal(f)
        assert n.shape == (3,)
        assert n.dtype == np.float32
        np.testing.assert_allclose(n, [0.0, 0.0, 1.0], atol=1e-6)

    def test_face_center_returns_centroid(self):
        scene, f, _ = self._make_unit_rect()
        c = scene.face_center(f)
        assert c.shape == (3,)
        assert c.dtype == np.float32
        np.testing.assert_allclose(c, [0.5, 0.5, 0.0], atol=1e-6)

    def test_face_normal_planar_face_in_xz_plane_is_plus_y(self):
        """A face with vertices at z varying and y=0 — normal should be ±Y.

        Loop V0(0,0,0)→V1(1,0,0)→V2(1,0,1)→V3(0,0,1): first two edges
        e1=(1,0,0) and e2=(0,0,1). Cross product e1 × e2 = (0·1−0·0,
        0·0−1·1, 1·0−0·0) = (0, −1, 0). So the geometric normal is (0,-1,0).
        """
        from bermake.scene import Scene

        scene = Scene()
        v0 = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        v1 = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        v2 = scene.add_vertex(np.array([1.0, 0.0, 1.0], dtype=np.float32))
        v3 = scene.add_vertex(np.array([0.0, 0.0, 1.0], dtype=np.float32))
        f = scene.add_face_from_loop([v0, v1, v2, v3])
        n = scene.face_normal(f)
        np.testing.assert_allclose(n, [0.0, -1.0, 0.0], atol=1e-6)


# ---------------------------------------------------------------------------
# Regression: vertical face triangulation (M3b side-face wireframe-only bug)
# ---------------------------------------------------------------------------


class TestSceneVerticalFaceTriangulation:
    """Regression: faces lying outside the XY plane must still triangulate.
    Before the fix, Scene.add_face_from_loop only projected XY for earcut,
    so vertical (XZ/YZ-plane) faces produced zero triangles and rendered
    as wireframe-only. This test pins the dominant-axis projection fix."""

    def test_xz_plane_face_produces_triangles(self):
        from bermake.scene import Scene

        scene = Scene()
        v0 = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        v1 = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        v2 = scene.add_vertex(np.array([1.0, 0.0, 1.0], dtype=np.float32))
        v3 = scene.add_vertex(np.array([0.0, 0.0, 1.0], dtype=np.float32))
        f = scene.add_face_from_loop([v0, v1, v2, v3])
        tris = list(scene._mesh.face_triangles(f))
        # 4-vertex face → 2 triangles → 6 vertex IDs in the flat buffer.
        assert len(tris) == 6, f"expected 6 vertex IDs for 2 triangles; got {len(tris)}"

    def test_yz_plane_face_produces_triangles(self):
        from bermake.scene import Scene

        scene = Scene()
        v0 = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        v1 = scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        v2 = scene.add_vertex(np.array([0.0, 1.0, 1.0], dtype=np.float32))
        v3 = scene.add_vertex(np.array([0.0, 0.0, 1.0], dtype=np.float32))
        f = scene.add_face_from_loop([v0, v1, v2, v3])
        tris = list(scene._mesh.face_triangles(f))
        assert len(tris) == 6

    def test_xy_plane_face_still_works(self):
        """Regression sanity: the existing M2 ground-plane rectangle keeps working."""
        from bermake.scene import Scene

        scene = Scene()
        v0 = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        v1 = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        v2 = scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
        v3 = scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        f = scene.add_face_from_loop([v0, v1, v2, v3])
        tris = list(scene._mesh.face_triangles(f))
        assert len(tris) == 6
