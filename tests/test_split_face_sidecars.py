"""Face split carries the parent's face-keyed sidecars onto both children (M7.6a)."""

import numpy as np
from bermake.scene.scene import Scene, Side, TexturePlacement


def _quad_with_chord():
    s = Scene()
    v = [
        s.add_vertex(np.array(p, dtype=np.float32))
        for p in [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)]
    ]
    fid = s.add_face_from_loop(v)
    s.add_edge(v[0], v[2])
    return s, fid, v


def test_both_children_inherit_the_parents_material_on_both_sides():
    s, fid, v = _quad_with_chord()
    s.set_face_material(fid, 7, Side.FRONT)
    s.set_face_material(fid, 9, Side.BACK)

    a, b = s.split_face(fid, [v[0], v[2]])

    for child in (a, b):
        assert s.face_material(child, Side.FRONT) == 7
        assert s.face_material(child, Side.BACK) == 9


def test_both_children_inherit_the_parents_placement():
    s, fid, v = _quad_with_chord()
    s.set_face_placement(fid, TexturePlacement(offset_u=0.25, offset_v=0.5), Side.FRONT)

    a, b = s.split_face(fid, [v[0], v[2]])

    for child in (a, b):
        p = s.face_placement(child, Side.FRONT)
        assert abs(p.offset_u - 0.25) < 1e-6
        assert abs(p.offset_v - 0.5) < 1e-6


def test_both_children_inherit_the_parents_back_placement():
    s, fid, v = _quad_with_chord()
    s.set_face_placement(fid, TexturePlacement(offset_u=0.1, offset_v=0.9), Side.BACK)

    a, b = s.split_face(fid, [v[0], v[2]])

    for child in (a, b):
        p = s.face_placement(child, Side.BACK)
        assert abs(p.offset_u - 0.1) < 1e-6
        assert abs(p.offset_v - 0.9) < 1e-6


def test_a_corner_that_was_on_the_parents_loop_keeps_its_exact_uv():
    s, fid, v = _quad_with_chord()
    stored = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    s.set_face_uvs(fid, stored, Side.FRONT)

    a, b = s.split_face(fid, [v[0], v[2]])

    by_vertex = {}
    for child in (a, b):
        uvs = s.face_uvs(child, Side.FRONT)
        assert uvs is not None
        for vid, uv in zip(s.face_loop(child), uvs, strict=True):
            by_vertex.setdefault(vid, []).append(tuple(round(float(c), 5) for c in uv))
    for vid, expected in zip(v, stored, strict=True):
        for got in by_vertex[vid]:
            assert got == tuple(round(c, 5) for c in expected)


def test_a_corner_that_was_on_the_parents_loop_keeps_its_exact_back_uv():
    s, fid, v = _quad_with_chord()
    stored = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    s.set_face_uvs(fid, stored, Side.BACK)

    a, b = s.split_face(fid, [v[0], v[2]])

    by_vertex = {}
    for child in (a, b):
        uvs = s.face_uvs(child, Side.BACK)
        assert uvs is not None
        for vid, uv in zip(s.face_loop(child), uvs, strict=True):
            by_vertex.setdefault(vid, []).append(tuple(round(float(c), 5) for c in uv))
    for vid, expected in zip(v, stored, strict=True):
        for got in by_vertex[vid]:
            assert got == tuple(round(c, 5) for c in expected)


def test_a_chain_vertex_gets_a_barycentrically_interpolated_uv():
    """The discriminating test. The parent's UV layout here is deliberately
    NOT affine in the plane: opposite corners carry (0,0) and (1,1) but the
    other two both carry (1,0), which no affine map produces. `mid` sits at
    (0.5, 1.0), a point strictly inside the parent's triangulation and off
    its v0..v2 diagonal, so every method that interpolates linearly along
    that diagonal (which is where a point ON the diagonal would make every
    method agree) gives a different answer here: barycentric interpolation
    inside the parent's own triangulation gives (0.5, 0.25); an affine
    least-squares fit of the plane to the UV layout gives (0.625, 0.125);
    fitting a plane through just 3 of the 4 corners gives yet other answers
    depending on which corner is dropped; and nearest-corner gives (0, 0).
    Only the barycentric answer is consistent with the parent's own
    triangulation, which is what the renderer samples.
    """
    s = Scene()
    v = [
        s.add_vertex(np.array(p, dtype=np.float32))
        for p in [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)]
    ]
    fid = s.add_face_from_loop(v)
    s.set_face_uvs(fid, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (1.0, 0.0)], Side.FRONT)

    mid = s.add_vertex(np.array((0.5, 1.0, 0.0), dtype=np.float32))
    s.add_edge(v[0], mid)
    s.add_edge(mid, v[2])
    a, b = s.split_face(fid, [v[0], mid, v[2]])

    got = None
    for child in (a, b):
        uvs = s.face_uvs(child, Side.FRONT)
        assert uvs is not None
        for vid, uv in zip(s.face_loop(child), uvs, strict=True):
            if vid == mid:
                got = np.asarray(uv, dtype=np.float64)
    assert got is not None, "the chain vertex must carry a UV"
    # Measured directly against this fixture (both children agree): the
    # barycentric weights of (0.5, 1.0) in whichever of the parent's two
    # triangles contains it resolve to (0.5, 0.25).
    np.testing.assert_allclose(got, (0.5, 0.25), atol=1e-5)


def test_a_parent_with_no_stored_uvs_gives_children_none():
    s, fid, v = _quad_with_chord()
    a, b = s.split_face(fid, [v[0], v[2]])
    assert s.face_uvs(a, Side.FRONT) is None
    assert s.face_uvs(b, Side.FRONT) is None


def test_a_chain_vertex_outside_the_parents_triangulation_drops_the_whole_uv_array():
    """Intended behaviour, not a bug: `split_face`'s partition check is
    purely topological (chain ends on the loop, interior vertices not on
    it), so it happily accepts a chain whose interior vertex sits in the
    concave notch of an L-shaped face, geometrically outside every triangle
    of the parent's own triangulation. `uv_at_point_in_face` correctly
    reports it can supply no UV there, and `_transfer_subloop_face_attributes`
    drops that side's WHOLE stored-UV array on both children rather than
    leave one corner guessed, while material (which does not depend on
    position) still transfers to both.

    The L-shape below is the hexagon (0,0) (2,0) (2,1) (1,1) (1,2) (0,2),
    reflex at (1,1); the chain vertex sits at (1.5, 1.5), inside the missing
    square the L leaves out.
    """
    s = Scene()
    v = [
        s.add_vertex(np.array(p, dtype=np.float32))
        for p in [(0, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0), (1, 2, 0), (0, 2, 0)]
    ]
    fid = s.add_face_from_loop(v)
    s.set_face_material(fid, 5, Side.FRONT)
    s.set_face_uvs(
        fid,
        [(0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.5, 0.5), (0.5, 1.0), (0.0, 1.0)],
        Side.FRONT,
    )

    notch = s.add_vertex(np.array((1.5, 1.5, 0.0), dtype=np.float32))
    s.add_edge(v[2], notch)
    s.add_edge(notch, v[4])
    out = s.split_face(fid, [v[2], notch, v[4]])
    assert out is not None
    a, b = out

    for child in (a, b):
        assert s.face_material(child, Side.FRONT) == 5
        assert s.face_uvs(child, Side.FRONT) is None
