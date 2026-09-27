import numpy as np
import pytest

from bermake.scene.scene import Scene, Side


def _quad(scene, z=0.0):
    ids = [
        scene.add_vertex(np.array(p, dtype=np.float32))
        for p in [(0, 0, z), (1, 0, z), (1, 1, z), (0, 1, z)]
    ]
    return scene.add_face_from_loop(ids)


def test_a_face_has_no_stored_uvs_by_default():
    s = Scene()
    f = _quad(s)
    assert s.face_uvs(f, Side.FRONT) is None
    assert s.face_uvs(f, Side.BACK) is None


def test_set_and_read_back():
    s = Scene()
    f = _quad(s)
    s.set_face_uvs(f, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], Side.FRONT)
    got = s.face_uvs(f, Side.FRONT)
    assert got is not None
    assert got.shape == (4, 2)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, [[0, 0], [1, 0], [1, 1], [0, 1]])


def test_the_two_sides_are_independent():
    s = Scene()
    f = _quad(s)
    s.set_face_uvs(f, [(0.0, 0.0)] * 4, Side.FRONT)
    assert s.face_uvs(f, Side.BACK) is None
    s.set_face_uvs(f, [(0.5, 0.5)] * 4, Side.BACK)
    np.testing.assert_allclose(s.face_uvs(f, Side.FRONT), np.zeros((4, 2)))


def test_clear_removes_them():
    s = Scene()
    f = _quad(s)
    s.set_face_uvs(f, [(0.0, 0.0)] * 4, Side.FRONT)
    s.clear_face_uvs(f, Side.FRONT)
    assert s.face_uvs(f, Side.FRONT) is None


def test_faces_with_uvs_lists_both_sides():
    s = Scene()
    f1 = _quad(s)
    f2 = _quad(s, z=1.0)
    s.set_face_uvs(f1, [(0.0, 0.0)] * 4, Side.FRONT)
    s.set_face_uvs(f2, [(0.0, 0.0)] * 4, Side.BACK)
    assert set(s.faces_with_uvs()) == {(f1, Side.FRONT), (f2, Side.BACK)}


def test_setting_uvs_marks_the_scene_dirty():
    s = Scene()
    f = _quad(s)
    s.mark_clean()
    s.set_face_uvs(f, [(0.0, 0.0)] * 4, Side.FRONT)
    assert s.dirty


def test_clearing_absent_uvs_does_not_dirty_the_scene():
    s = Scene()
    f = _quad(s)
    s.mark_clean()
    s.clear_face_uvs(f, Side.FRONT)
    assert not s.dirty


def test_a_wrong_length_array_is_rejected():
    s = Scene()
    f = _quad(s)
    with pytest.raises(ValueError, match="4 corners"):
        s.set_face_uvs(f, [(0.0, 0.0), (1.0, 0.0)], Side.FRONT)


def test_clear_drops_every_stored_array():
    s = Scene()
    f = _quad(s)
    s.set_face_uvs(f, [(0.0, 0.0)] * 4, Side.FRONT)
    s.clear()
    assert s.faces_with_uvs() == []


def test_the_returned_array_is_not_a_live_view():
    s = Scene()
    f = _quad(s)
    s.set_face_uvs(f, [(0.0, 0.0)] * 4, Side.FRONT)
    got = s.face_uvs(f, Side.FRONT)
    got[0, 0] = 9.0
    np.testing.assert_allclose(s.face_uvs(f, Side.FRONT)[0], [0.0, 0.0])


def test_loop_indices_align_with_the_triangle_buffer():
    s = Scene()
    _quad(s)
    positions, _ = s.face_triangle_buffer()
    idx = s.face_triangle_loop_indices()
    assert idx.shape == (positions.shape[0],)
    assert idx.dtype == np.int32
    # A quad triangulates to 2 triangles, 6 corners, all indices within 0..3.
    assert positions.shape[0] == 6
    assert idx.min() >= 0 and idx.max() <= 3


def test_each_corner_index_names_the_vertex_at_that_corner():
    s = Scene()
    f = _quad(s)
    loop = s.face_loop(f)
    positions, _ = s.face_triangle_buffer()
    idx = s.face_triangle_loop_indices()
    for corner in range(positions.shape[0]):
        expected_vertex = loop[int(idx[corner])]
        np.testing.assert_allclose(
            positions[corner], s.vertex(expected_vertex).position, atol=1e-6
        )


def test_loop_indices_span_two_faces_in_walk_order():
    s = Scene()
    f1 = _quad(s)
    f2 = _quad(s, z=1.0)
    positions, _ = s.face_triangle_buffer()
    idx = s.face_triangle_loop_indices()
    assert idx.shape == (positions.shape[0],)
    assert positions.shape[0] == 12
    loops = {f1: s.face_loop(f1), f2: s.face_loop(f2)}
    face_per_tri = s.face_triangle_face_ids()
    for corner in range(positions.shape[0]):
        owning_face = int(face_per_tri[corner // 3])
        expected_vertex = loops[owning_face][int(idx[corner])]
        np.testing.assert_allclose(
            positions[corner], s.vertex(expected_vertex).position, atol=1e-6
        )


def test_an_empty_scene_returns_an_empty_index_array():
    s = Scene()
    idx = s.face_triangle_loop_indices()
    assert idx.shape == (0,)
    assert idx.dtype == np.int32


def test_face_loop_indices_is_the_whole_scene_array_sliced_to_one_face():
    """#117: the overlay needs one face's corners, not every face's.

    Building the whole (3T,) array to read a handful of entries is what
    makes three stored faces among 1,600 cost as much as they do. The
    per-face accessor must agree with the slice it replaces, or the two
    would drift and the overlay would read the wrong loop position.
    """
    s = Scene()
    f1 = _quad(s)
    f2 = _quad(s, z=1.0)
    whole = s.face_triangle_loop_indices()
    face_per_tri = s.face_triangle_face_ids()

    for face in (f1, f2):
        corners = np.flatnonzero(np.repeat(face_per_tri, 3) == face)
        np.testing.assert_array_equal(s.face_loop_indices(face), whole[corners])


def test_face_loop_indices_of_a_lone_quad_covers_its_whole_loop():
    s = Scene()
    f = _quad(s)
    idx = s.face_loop_indices(f)
    assert idx.dtype == np.int32
    assert idx.shape == (6,)
    assert set(int(i) for i in idx) == {0, 1, 2, 3}
