"""Stored UVs replace the projection as the bake's base, with placement on top."""

import numpy as np

from bermake.scene.scene import Scene, Side, TexturePlacement
from bermake.viewport.scene_renderer import build_face_uvs, build_face_uvs_both_sides


def _quad(scene, z=0.0):
    ids = [
        scene.add_vertex(np.array(p, dtype=np.float32))
        for p in [(0, 0, z), (1, 0, z), (1, 1, z), (0, 1, z)]
    ]
    return scene.add_face_from_loop(ids)


def test_without_stored_uvs_the_bake_is_the_plane_projection():
    s = Scene()
    _quad(s)
    uvs = build_face_uvs(s, None, Side.FRONT)
    idx = s.face_triangle_loop_indices()
    assert uvs.shape == (6, 2)  # a quad is 2 triangles

    # Golden pin on the projection path itself, not a self-comparison: the unit
    # quad's centroid is (0.5, 0.5), so each loop vertex projects to a corner of
    # a unit square centred on the origin.
    #
    # Keyed by LOOP index rather than by buffer position, because the
    # triangulation's rotation is not a contract. pyproject.toml asks only for
    # "mapbox-earcut>=2.0", so CI resolved 2.1.0 where this machine had 2.0.0,
    # and 2.1.0 emits the quad's second triangle as [2, 0, 1] where 2.0.0 emits
    # [0, 1, 2]: the same triangle, the same winding, started at a different
    # vertex. Nothing renders differently, but a flat (6, 2) literal captured
    # that rotation as though it were stable, and turned CI red on both
    # platforms while passing locally.
    #
    # Every other test in this file already reads through
    # face_triangle_loop_indices for exactly this reason. Do not reintroduce a
    # positional literal here, and do not pin earcut to paper over one.
    expected_by_loop = {
        0: (0.5, -0.5),
        1: (0.5, 0.5),
        2: (-0.5, 0.5),
        3: (-0.5, -0.5),
    }
    assert set(int(i) for i in idx) == set(expected_by_loop), (
        "every loop vertex should appear in the triangulation"
    )
    for corner in range(6):
        np.testing.assert_allclose(uvs[corner], expected_by_loop[int(idx[corner])], atol=1e-6)


def test_stored_uvs_land_on_the_right_corners():
    s = Scene()
    f = _quad(s)
    # A distinct UV per loop corner so a mis-mapped corner is unmissable.
    stored = [(0.0, 0.0), (0.1, 0.0), (0.2, 0.0), (0.3, 0.0)]
    s.set_face_uvs(f, stored, Side.FRONT)

    uvs = build_face_uvs(s, None, Side.FRONT)
    idx = s.face_triangle_loop_indices()
    assert uvs.shape == (6, 2)
    for corner in range(6):
        np.testing.assert_allclose(uvs[corner], stored[int(idx[corner])], atol=1e-6)


def test_the_other_side_still_projects():
    s = Scene()
    f = _quad(s)
    # Capture the pure projection for BACK before anything is stored anywhere.
    projected_back = build_face_uvs(s, None, Side.BACK)

    s.set_face_uvs(f, [(0.0, 0.0), (0.1, 0.0), (0.2, 0.0), (0.3, 0.0)], Side.FRONT)
    front, back = build_face_uvs_both_sides(s, None)

    np.testing.assert_allclose(back, projected_back, atol=1e-6)
    assert not np.allclose(front, back)


def test_placement_composes_on_top_of_stored_uvs():
    s = Scene()
    f = _quad(s)
    stored = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    s.set_face_uvs(f, stored, Side.FRONT)
    plain = build_face_uvs(s, None, Side.FRONT)

    s.set_face_placement(f, TexturePlacement(0.5, 0.25, 1.0, 0.0), Side.FRONT)
    offset = build_face_uvs(s, None, Side.FRONT)

    # Pin the base itself, not just the delta between plain and offset: without
    # this, offset - plain would cancel the base entirely and only exercise
    # apply_placements' linearity, which this task does not touch.
    idx = s.face_triangle_loop_indices()
    for corner in range(6):
        expected_base = np.array(stored[int(idx[corner])])
        np.testing.assert_allclose(plain[corner], expected_base, atol=1e-6)
        # Offset with unit scale and no rotation is a pure translation of the base.
        np.testing.assert_allclose(offset[corner], expected_base + [0.5, 0.25], atol=1e-6)


def test_one_stored_face_among_many_leaves_the_others_projected():
    s = Scene()
    f1 = _quad(s, z=0.0)
    f2 = _quad(s, z=1.0)
    projected_both = build_face_uvs(s, None, Side.FRONT)

    stored = [(0.0, 0.0), (0.1, 0.0), (0.2, 0.0), (0.3, 0.0)]
    s.set_face_uvs(f1, stored, Side.FRONT)
    mixed = build_face_uvs(s, None, Side.FRONT)

    face_per_tri = s.face_triangle_face_ids()
    idx = s.face_triangle_loop_indices()
    for corner in range(mixed.shape[0]):
        owning = int(face_per_tri[corner // 3])
        if owning == f2:
            np.testing.assert_allclose(mixed[corner], projected_both[corner], atol=1e-6)
        elif owning == f1:
            np.testing.assert_allclose(mixed[corner], stored[int(idx[corner])], atol=1e-6)
    assert not np.allclose(mixed, projected_both)


def test_a_few_stored_faces_do_not_walk_every_face(monkeypatch):
    """#117: the overlay paid a cost proportional to the definition's TOTAL
    face count the moment a single face carried stored UVs.

    `face_triangle_loop_indices()` walks every live face in Python to build
    an array of which the overlay read a handful of entries. Making it raise
    is the discrimination: any implementation that still reaches for the
    whole-scene array to serve two stored faces fails here, and no
    implementation that slices per face can.
    """
    s = Scene()
    faces = [_quad(s, z=float(i)) for i in range(40)]
    s.set_face_uvs(faces[0], [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)], Side.FRONT)
    s.set_face_uvs(faces[7], [(0.0, 0.0), (0.25, 0.0), (0.25, 0.25), (0.0, 0.25)], Side.FRONT)
    expected = build_face_uvs(s, None, Side.FRONT)

    def boom(self):
        raise AssertionError("the overlay walked every face to serve two stored ones")

    monkeypatch.setattr(type(s), "face_triangle_loop_indices", boom)
    got = build_face_uvs(s, None, Side.FRONT)

    np.testing.assert_allclose(got, expected, atol=1e-6)
