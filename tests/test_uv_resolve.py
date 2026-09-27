import numpy as np

from bermake.model.material import MaterialLibrary
from bermake.scene.scene import Scene, Side, TexturePlacement
from bermake.viewport.scene_renderer import build_face_uvs
from bermake.viewport.uv_resolve import resolve_face_uvs


def _quad(scene):
    ids = [
        scene.add_vertex(np.array(p, dtype=np.float32))
        for p in [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    ]
    return scene.add_face_from_loop(ids)


def test_a_face_with_no_stored_uvs_resolves_to_the_projection():
    s = Scene()
    f = _quad(s)
    got = resolve_face_uvs(s, None, f, Side.FRONT)
    assert got.shape == (4, 2)
    # Centroid-relative projection of a unit quad, corners at +/- 0.5.
    assert np.allclose(np.abs(got), 0.5)


def test_stored_uvs_replace_the_projection():
    s = Scene()
    f = _quad(s)
    stored = [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (0.75, 0.0)]
    s.set_face_uvs(f, stored, Side.FRONT)
    np.testing.assert_allclose(resolve_face_uvs(s, None, f, Side.FRONT), stored, atol=1e-6)


def test_placement_composes_on_top_of_stored_uvs():
    s = Scene()
    f = _quad(s)
    stored = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    s.set_face_uvs(f, stored, Side.FRONT)
    s.set_face_placement(f, TexturePlacement(0.5, 0.25, 1.0, 0.0), Side.FRONT)
    expected = np.asarray(stored) + np.array([0.5, 0.25])
    np.testing.assert_allclose(resolve_face_uvs(s, None, f, Side.FRONT), expected, atol=1e-6)


def test_the_two_sides_resolve_independently():
    s = Scene()
    f = _quad(s)
    s.set_face_uvs(f, [(0.0, 0.0)] * 4, Side.FRONT)
    front = resolve_face_uvs(s, None, f, Side.FRONT)
    back = resolve_face_uvs(s, None, f, Side.BACK)
    assert np.allclose(front, 0.0)
    assert not np.allclose(back, 0.0)


def test_it_agrees_with_the_renderer_corner_for_corner():
    """Spec D9: the renderer's batched bake and this per-face resolver must not
    drift. They are separate implementations for performance reasons, so this
    test is the only thing holding them together."""
    s = Scene()
    f = _quad(s)
    s.set_face_uvs(f, [(0.0, 0.0), (0.3, 0.1), (0.6, 0.4), (0.9, 0.7)], Side.FRONT)
    s.set_face_placement(f, TexturePlacement(0.2, -0.1, 2.0, 0.4), Side.FRONT)

    per_face = resolve_face_uvs(s, None, f, Side.FRONT)
    baked = build_face_uvs(s, None, Side.FRONT)
    idx = s.face_triangle_loop_indices()
    for corner in range(baked.shape[0]):
        np.testing.assert_allclose(baked[corner], per_face[int(idx[corner])], atol=1e-5)


def test_it_agrees_with_the_renderer_for_a_projected_face_too():
    s = Scene()
    f = _quad(s)
    s.set_face_placement(f, TexturePlacement(0.1, 0.2, 1.5, 0.35), Side.FRONT)
    per_face = resolve_face_uvs(s, None, f, Side.FRONT)
    baked = build_face_uvs(s, None, Side.FRONT)
    idx = s.face_triangle_loop_indices()
    for corner in range(baked.shape[0]):
        np.testing.assert_allclose(baked[corner], per_face[int(idx[corner])], atol=1e-5)


def test_it_agrees_with_the_renderer_for_the_material_path_too():
    """Both agreement tests above pass materials=None on both sides, so the
    texture_size lookup itself is never cross-checked: resolve_face_uvs reads
    it via `materials.get(...).texture_size` while the renderer's _side_uvs
    does its own separate per-face gather. A drift there (wrong side, stale
    snapshot, a swapped width and height) would sail past every other test in
    this file. Use a non-square texture_size so a swapped width/height would
    be caught, and a non-identity placement so base and placement both
    matter."""
    s = Scene()
    f = _quad(s)
    lib = MaterialLibrary()
    mat = lib.add_custom("tall", (1.0, 1.0, 1.0))
    lib.edit(mat.id, texture_size=(2.0, 4.0))
    s.set_face_material(f, mat.id, Side.FRONT)
    s.set_face_placement(f, TexturePlacement(0.2, -0.1, 1.5, 0.3), Side.FRONT)

    class _ModelStub:
        materials = lib

    per_face = resolve_face_uvs(s, lib, f, Side.FRONT)
    baked = build_face_uvs(s, _ModelStub(), Side.FRONT)
    idx = s.face_triangle_loop_indices()
    for corner in range(baked.shape[0]):
        np.testing.assert_allclose(baked[corner], per_face[int(idx[corner])], atol=1e-5)


def test_a_materials_texture_size_scales_the_projection():
    s = Scene()
    f = _quad(s)
    lib = MaterialLibrary()
    mat = lib.add_custom("big", (1.0, 1.0, 1.0))
    lib.edit(mat.id, texture_size=(2.0, 2.0))
    s.set_face_material(f, mat.id, Side.FRONT)
    got = resolve_face_uvs(s, lib, f, Side.FRONT)
    # Twice the texture size means half the UV extent.
    assert np.allclose(np.abs(got), 0.25)
