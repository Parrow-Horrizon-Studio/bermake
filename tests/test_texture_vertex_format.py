"""M7.5b Task 4: per-corner UVs in the face vertex buffer."""

from __future__ import annotations

import numpy as np
import pytest
from bermake._core import make_box
from bermake.model.model import Model
from bermake.scene.mesh_builder import build_mesh_into_scene
from bermake.scene.scene import Scene, Side, TexturePlacement
from bermake.viewport import scene_renderer
from bermake.viewport.scene_renderer import (
    _FACE_VERTEX_FLOATS,
    _DefBuffers,
    SceneRenderer,
    build_face_uvs,
    uv_material_key,
)
from bermake.viewport.uv_projection import project_corners


def _boxed():
    model = Model()
    scene = model.root.mesh
    build_mesh_into_scene(make_box(2.0, 2.0, 2.0), scene)
    return model, scene


def _painted_box():
    model, scene = _boxed()
    mat = model.materials.add_custom("Brick", (1.0, 1.0, 1.0))
    for f in scene.faces_iter():
        scene.set_face_material(f.id, mat.id)
    return model, scene, mat


def _quad_and_triangle(scene: Scene) -> tuple[int, int]:
    """A 2-triangle quad and a 1-triangle face, coplanar but separate.

    Unequal triangle counts per face, which a box does not have: any drift
    between the UV walk and the position buffer slides a whole corner block.
    """

    def v(x, y):
        return scene.add_vertex(np.array([x, y, 0.0], dtype=np.float32))

    quad = [v(0.0, 0.0), v(1.0, 0.0), v(1.0, 1.0), v(0.0, 1.0)]
    for a, b in zip(quad, quad[1:] + quad[:1], strict=True):
        scene.add_edge(a, b)
    q = scene.add_face_from_loop(quad)

    tri = [v(3.0, 0.0), v(4.0, 0.0), v(3.0, 1.0)]
    for a, b in zip(tri, tri[1:] + tri[:1], strict=True):
        scene.add_edge(a, b)
    t = scene.add_face_from_loop(tri)
    return q, t


def _split_edge_square(scene: Scene) -> int:
    """A square whose loop STARTS at a mid-edge vertex.

    Its first three boundary vertices are collinear, which is the shape any
    edge split produces. Scene.face_normal used to raise on exactly this
    (issue #110, fixed by switching it to Newell's method); face_triangle_buffer
    always triangulated it correctly.
    """
    pts = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    v = [scene.add_vertex(np.array([x, y, 0.0], dtype=np.float32)) for x, y in pts]
    for a, b in zip(v, v[1:] + v[:1], strict=True):
        scene.add_edge(a, b)
    return scene.add_face_from_loop(v)


def test_the_face_vertex_carries_position_normal_and_both_side_uvs():
    assert _FACE_VERTEX_FLOATS == 10


def test_face_ids_align_one_per_triangle_with_the_buffer():
    _, scene = _boxed()
    pos, _ = scene.face_triangle_buffer()
    ids = scene.face_triangle_face_ids()
    assert ids.dtype == np.int64
    assert ids.shape[0] * 3 == np.asarray(pos).shape[0]


def test_every_reported_face_id_is_a_live_face():
    _, scene = _boxed()
    live = {f.id for f in scene.faces_iter()}
    assert set(scene.face_triangle_face_ids().tolist()) <= live


def test_face_ids_match_the_material_walk_exactly():
    # The two arrays MUST come from the same next_live_face walk. If they drift,
    # UVs land on the wrong triangles and the result looks like scrambled
    # texturing rather than an error anyone can trace.
    _, scene = _boxed()
    assert scene.face_triangle_face_ids().shape == scene.face_triangle_materials().shape


def test_face_ids_repeat_by_triangle_count_not_once_per_face():
    # Kills a walk that emits one id per face instead of one per triangle: on a
    # quad that is 1 entry where the buffer holds 2 triangles.
    model = Model()
    quad, tri = _quad_and_triangle(model.root.mesh)
    ids = model.root.mesh.face_triangle_face_ids().tolist()
    assert ids.count(quad) == 2
    assert ids.count(tri) == 1


def test_uvs_are_one_per_vertex_not_one_per_triangle():
    model, scene = _boxed()
    pos, _ = scene.face_triangle_buffer()
    uvs = build_face_uvs(scene, model)
    assert uvs.shape == (np.asarray(pos).shape[0], 2)
    assert uvs.dtype == np.float32


def test_a_two_unit_face_spans_two_tiles_of_a_unit_texture():
    # The whole point of texture_size: a 1x1 image on a 2x2 box face tiles
    # twice rather than stretching to fit.
    model, scene = _boxed()
    mat = model.materials.add_custom("Brick", (1.0, 1.0, 1.0))
    model.materials.edit(mat.id, texture_size=(1.0, 1.0))
    for f in scene.faces_iter():
        scene.set_face_material(f.id, mat.id)
    uvs = build_face_uvs(scene, model)
    assert float(uvs[:, 0].max() - uvs[:, 0].min()) >= 1.9


def test_a_larger_texture_size_tiles_fewer_times():
    model, scene = _boxed()
    mat = model.materials.add_custom("Brick", (1.0, 1.0, 1.0))
    for f in scene.faces_iter():
        scene.set_face_material(f.id, mat.id)

    model.materials.edit(mat.id, texture_size=(1.0, 1.0))
    small = build_face_uvs(scene, model)
    model.materials.edit(mat.id, texture_size=(4.0, 4.0))
    large = build_face_uvs(scene, model)

    def span(a):
        return float(a[:, 0].max() - a[:, 0].min())

    assert span(large) < span(small)


def test_a_face_placement_moves_only_that_face_s_uvs():
    model, scene = _boxed()
    mat = model.materials.add_custom("Brick", (1.0, 1.0, 1.0))
    for f in scene.faces_iter():
        scene.set_face_material(f.id, mat.id)
    before = build_face_uvs(scene, model)

    target = next(iter(scene.faces_iter())).id
    scene.set_face_placement(target, TexturePlacement(offset_u=10.0))
    after = build_face_uvs(scene, model)

    ids = scene.face_triangle_face_ids()
    moved = np.repeat(ids == target, 3)
    assert not np.allclose(before[moved], after[moved])
    assert np.allclose(before[~moved], after[~moved])


def test_each_face_s_corner_block_holds_that_face_s_own_vertices():
    # The alignment invariant checked against the MESH rather than against the
    # walk. Every other alignment test here compares face_triangle_face_ids to
    # something that also came from face_triangle_face_ids, so a walk that
    # disagreed with the C++ face_triangle_buffer would satisfy all of them
    # consistently and still scramble texturing. This one asks the geometry:
    # the corners a face id claims must be that face's own vertices. The quad
    # sits at x in [0, 1] and the triangle at x in [3, 4], so no corner of one
    # can be mistaken for a corner of the other.
    model = Model()
    scene = model.root.mesh
    _quad_and_triangle(scene)

    positions, _ = scene.face_triangle_buffer()
    ids = scene.face_triangle_face_ids()
    assert len(set(ids.tolist())) == 2

    for f_id in set(ids.tolist()):
        own = {tuple(np.round(scene.vertex(v).position, 5)) for v in scene.face_loop(int(f_id))}
        claimed = {tuple(np.round(p, 5)) for p in positions[np.repeat(ids == f_id, 3)]}
        assert claimed <= own, f"face {f_id} claims corners it does not own"


def test_the_uvs_of_a_face_land_on_that_face_s_own_corner_block():
    # Catches a BLOCK-OFFSET error, on a mesh whose faces have UNEQUAL triangle
    # counts: a builder that sized one face's corner run by another face's
    # triangle count slides every later block and fails here.
    #
    # It is NOT a second independent guard on the walk order. It re-projects
    # through the walk's own face assignment, so a walk that disagreed with the
    # C++ face_triangle_buffer would satisfy it consistently. That is what
    # test_each_face_s_corner_block_holds_that_face_s_own_vertices is for.
    model = Model()
    scene = model.root.mesh
    _quad_and_triangle(scene)

    positions, _ = scene.face_triangle_buffer()
    ids = scene.face_triangle_face_ids()
    uvs = build_face_uvs(scene, model)

    for f_id in dict.fromkeys(int(i) for i in ids):
        corner = np.repeat(ids == f_id, 3)
        expected = project_corners(
            positions[corner].astype(np.float64),
            np.asarray(scene.face_normal(f_id), dtype=np.float64),
            np.asarray(scene.face_center(f_id), dtype=np.float64),
            (1.0, 1.0),
        )
        assert np.allclose(uvs[corner], expected, atol=1e-5)


def test_the_back_side_reads_the_back_placement():
    model, scene = _boxed()
    mat = model.materials.add_custom("Brick", (1.0, 1.0, 1.0))
    for f in scene.faces_iter():
        scene.set_face_material(f.id, mat.id, Side.BACK)
    target = next(iter(scene.faces_iter())).id
    scene.set_face_placement(target, TexturePlacement(offset_u=5.0), Side.BACK)

    front = build_face_uvs(scene, model, Side.FRONT)
    back = build_face_uvs(scene, model, Side.BACK)
    assert not np.allclose(front, back)


def test_an_empty_scene_yields_an_empty_uv_array():
    model = Model()
    uvs = build_face_uvs(model.root.mesh, model)
    assert uvs.shape == (0, 2)
    assert uvs.dtype == np.float32


def test_a_face_whose_loop_starts_on_a_split_edge_still_gets_uvs():
    # Scene.face_normal used to raise ValueError when a face's first three loop
    # vertices were collinear, and splitting an edge produces exactly that.
    # Nothing in the render or upload path called face_normal before M7.5b —
    # every caller was interactive tool code working on one picked face — so
    # taking the normal from it would have let one ordinary face stop the entire
    # document from rendering. face_triangle_buffer's own normals block is
    # correct for this face, so the projection reads it instead.
    #
    # That raise was a pre-existing defect with six interactive callers, fixed
    # since as issue #110 (face_normal now uses Newell's method over the whole
    # loop; see tests/test_face_normal_newell.py). It was deliberately never
    # asserted here, so this test kept passing across the fix. What is asserted
    # is the property that had to hold either way — a geometrically ordinary
    # face gets finite UVs. The renderer still reads the buffer's normals block
    # rather than calling face_normal per face, which remains the right split.
    model = Model()
    scene = model.root.mesh
    _split_edge_square(scene)

    _, normals = scene.face_triangle_buffer()
    assert np.allclose(normals[0], [0.0, 0.0, 1.0])

    uvs = build_face_uvs(scene, model)
    assert uvs.shape == (9, 2)
    assert np.all(np.isfinite(uvs))


def test_uploading_a_definition_with_a_split_edge_face_does_not_raise(monkeypatch):
    # The same trap at the level where it actually bit: the exception
    # propagated _upload_definition -> _ensure_buffers -> paint, so the whole
    # viewport went dark rather than one face looking wrong.
    model = Model()
    _split_edge_square(model.root.mesh)
    monkeypatch.setattr(scene_renderer.GL, "glBindBuffer", lambda *a: None)
    monkeypatch.setattr(scene_renderer.GL, "glBufferData", lambda *a: None)
    r = SceneRenderer()
    monkeypatch.setattr(r, "_alloc_def_buffers", lambda: _DefBuffers(face_vbo=1, edge_vbo=2))

    buf = r._ensure_buffers(model.root, frozenset(), model)
    assert buf.face_count == 9


def test_an_untextured_scene_still_produces_finite_uvs():
    # Untextured faces still get UVs, because the attribute exists for every
    # vertex. They are simply never sampled. NaN or inf here would corrupt the
    # buffer for textured neighbours in the same definition.
    model, scene = _boxed()
    uvs = build_face_uvs(scene, model)
    assert np.all(np.isfinite(uvs))


# --- Step 6: the UVs reach the vertex buffer, under the batch permutation ----


def test_the_uploaded_vertex_rows_carry_uvs_under_the_batch_permutation(monkeypatch):
    # Two invariants at once, both silent when broken. The row must be 10 wide
    # (a concatenate that forgot a UV block uploads narrow data against a
    # 40-byte stride), and BOTH UV blocks must be permuted by plan.vertex_order
    # just like positions and normals — leaving either unsorted scrambles
    # texturing only on definitions that contain translucent faces.
    model, scene = _boxed()
    # texture_id is set as well as texture_size because the bake is gated on
    # the model carrying a texture at all (final review, item 2) -- without it
    # this definition would correctly upload zero-filled UVs and the
    # permutation this test exists to check would be invisible.
    glass = model.materials.add_custom("Glass", (0.3, 0.5, 0.9))
    model.materials.edit(glass.id, alpha=0.4, texture_id=11, texture_size=(2.0, 2.0))
    lining = model.materials.add_custom("Lining", (0.9, 0.4, 0.1))
    model.materials.edit(lining.id, texture_id=12, texture_size=(5.0, 5.0))
    for f in list(scene.faces_iter())[:3]:
        scene.set_face_material(f.id, glass.id)
        scene.set_face_material(f.id, lining.id, Side.BACK)

    uploads: list[np.ndarray] = []
    monkeypatch.setattr(scene_renderer.GL, "glBindBuffer", lambda *a: None)
    monkeypatch.setattr(
        scene_renderer.GL,
        "glBufferData",
        lambda target, size, data, usage: uploads.append(np.asarray(data, dtype=np.float32).copy()),
    )
    r = SceneRenderer()
    monkeypatch.setattr(r, "_alloc_def_buffers", lambda: _DefBuffers(face_vbo=1, edge_vbo=2))

    buf = r._upload_definition(model.root, frozenset({glass.id}), model)
    rows = uploads[0].reshape(-1, _FACE_VERTEX_FLOATS)

    order = np.asarray(buf.plan.vertex_order)
    assert not np.array_equal(order, np.arange(rows.shape[0])), (
        "the plan did not reorder anything, so this test could not detect an "
        "unpermuted UV block"
    )
    positions, normals = scene.face_triangle_buffer()
    assert np.allclose(rows[:, 0:3], positions[order])
    assert np.allclose(rows[:, 3:6], normals[order])
    front = build_face_uvs(scene, model, Side.FRONT)
    back = build_face_uvs(scene, model, Side.BACK)
    assert not np.allclose(front, back), (
        "front and back UVs are identical here, so this test could not tell the "
        "two blocks apart"
    )
    assert np.allclose(rows[:, 6:8], front[order], atol=1e-6)
    assert np.allclose(rows[:, 8:10], back[order], atol=1e-6)


def test_the_upload_records_the_uv_key_it_baked(monkeypatch):
    # _ensure_buffers compares buf.uv_key against a freshly computed one, so an
    # upload that forgot to record it would re-upload every single frame.
    model, _, mat = _painted_box()
    monkeypatch.setattr(scene_renderer.GL, "glBindBuffer", lambda *a: None)
    monkeypatch.setattr(scene_renderer.GL, "glBufferData", lambda *a: None)
    r = SceneRenderer()
    monkeypatch.setattr(r, "_alloc_def_buffers", lambda: _DefBuffers(face_vbo=1, edge_vbo=2))

    buf = r._upload_definition(model.root, frozenset(), model)
    assert buf.uv_key == uv_material_key(model.root.mesh, model)
    assert any(entry[0] == mat.id for entry in buf.uv_key)


# --- Step 6b: re-upload when a material's UV-affecting fields change ---------


def test_changing_texture_size_changes_the_uv_key():
    model, _, mat = _painted_box()
    model.materials.edit(mat.id, texture_size=(1.0, 1.0))
    before = uv_material_key(model.root.mesh, model)
    model.materials.edit(mat.id, texture_size=(4.0, 4.0))
    assert uv_material_key(model.root.mesh, model) != before


def test_changing_texture_id_changes_the_uv_key():
    model, _, mat = _painted_box()
    before = uv_material_key(model.root.mesh, model)
    model.materials.edit(mat.id, texture_id=7)
    assert uv_material_key(model.root.mesh, model) != before


def test_changing_a_colour_does_not_change_the_uv_key():
    # A colour edit must NOT force a re-upload: uniforms are read per batch at
    # draw time. Re-uploading on every colour tweak would rebuild every visible
    # definition's vertex buffer while the user drags a colour picker.
    model, _, mat = _painted_box()
    before = uv_material_key(model.root.mesh, model)
    model.materials.edit(mat.id, base_color=(0.2, 0.9, 0.3), alpha=0.5, roughness=0.1)
    assert uv_material_key(model.root.mesh, model) == before


def _back_painted_box():
    """A box painted only on the BACK, so the front sidecar names nothing."""
    model, scene = _boxed()
    mat = model.materials.add_custom("Lining", (0.9, 0.4, 0.1))
    for f in scene.faces_iter():
        scene.set_face_material(f.id, mat.id, Side.BACK)
    return model, scene, mat


def test_resizing_a_back_only_material_changes_the_uv_key():
    # Both sides are baked into the vertex buffer, so the key has to span both.
    # A key that only walked Side.FRONT would never name this material and a
    # back texture resize would go on tiling at the old size for ever.
    model, _, mat = _back_painted_box()
    before = uv_material_key(model.root.mesh, model)
    assert any(entry[0] == mat.id for entry in before), "the back material is not in the key"
    model.materials.edit(mat.id, texture_size=(4.0, 4.0))
    assert uv_material_key(model.root.mesh, model) != before


def test_recolouring_a_back_only_material_does_not_change_the_uv_key():
    # The paired half for the back side, for the same reason as the front pair:
    # a key that hashed the whole material would pass the test above alone.
    model, _, mat = _back_painted_box()
    before = uv_material_key(model.root.mesh, model)
    model.materials.edit(mat.id, base_color=(0.1, 0.8, 0.4), alpha=0.6, metallic=0.9)
    assert uv_material_key(model.root.mesh, model) == before


@pytest.fixture
def uv_stubbed_renderer(monkeypatch):
    """A SceneRenderer whose _upload_definition is a counted stub.

    Same pattern as tests/test_translucent_pass.py: lets the reuse-vs-rebuild
    decision in _ensure_buffers be observed directly, with no GL context.
    """
    r = SceneRenderer()
    calls: list[tuple] = []

    def fake_upload(definition, translucent_ids, model):
        calls.append((translucent_ids, model))
        return _DefBuffers(
            translucent_ids=translucent_ids,
            uv_key=uv_material_key(definition.mesh, model),
            edge_color=r._environment.edge_color,
        )

    monkeypatch.setattr(r, "_upload_definition", fake_upload)
    return r, calls


def test_an_unchanged_uv_key_reuses_the_buffer(uv_stubbed_renderer):
    # Kills a renderer that re-uploads unconditionally every frame, which would
    # pass every uv_material_key test above while doing no caching at all.
    r, calls = uv_stubbed_renderer
    model, _, _ = _painted_box()
    first = r._ensure_buffers(model.root, frozenset(), model)
    again = r._ensure_buffers(model.root, frozenset(), model)
    assert again is first
    assert len(calls) == 1


def test_a_changed_texture_size_rebuilds_the_buffer(uv_stubbed_renderer):
    # The bug Step 6b exists for: texture_size lives on the MaterialLibrary,
    # not the Scene, so editing it dirties no mesh. Without the uv_key check
    # the baked UVs keep tiling at the old size and the edit looks inert.
    r, calls = uv_stubbed_renderer
    model, _, mat = _painted_box()
    first = r._ensure_buffers(model.root, frozenset(), model)
    model.materials.edit(mat.id, texture_size=(4.0, 4.0))
    changed = r._ensure_buffers(model.root, frozenset(), model)
    assert changed is not first
    assert len(calls) == 2


def test_the_per_frame_staleness_check_does_not_walk_the_triangles(uv_stubbed_renderer):
    # Why uv_key_still_matches exists rather than re-deriving the key each
    # frame. face_triangle_materials walks every live face, measured at ~3.5 ms
    # for a 10k-face definition — per definition, per frame. The cached key
    # already names the materials that matter, so re-reading just those is
    # enough. Kills a revert to uv_material_key(scene, model) in the draw loop.
    r, _ = uv_stubbed_renderer
    model, scene, _ = _painted_box()
    r._ensure_buffers(model.root, frozenset(), model)

    walks = []
    real = scene.face_triangle_materials

    def counted(*args, **kwargs):
        walks.append(1)
        return real(*args, **kwargs)

    scene.face_triangle_materials = counted
    r._ensure_buffers(model.root, frozenset(), model)
    assert walks == []


def test_a_changed_back_texture_size_rebuilds_the_buffer(uv_stubbed_renderer):
    # The back UVs are baked into the same buffer, so a back-only material's
    # resize must invalidate it exactly as a front one does. A key or a
    # staleness check that only spanned Side.FRONT reuses the stale buffer.
    r, calls = uv_stubbed_renderer
    model, _, mat = _back_painted_box()
    first = r._ensure_buffers(model.root, frozenset(), model)
    model.materials.edit(mat.id, texture_size=(4.0, 4.0))
    changed = r._ensure_buffers(model.root, frozenset(), model)
    assert changed is not first
    assert len(calls) == 2


def test_a_colour_edit_does_not_rebuild_the_buffer(uv_stubbed_renderer):
    # The paired half at buffer level: a colour edit must leave the VBO alone.
    r, calls = uv_stubbed_renderer
    model, _, mat = _painted_box()
    first = r._ensure_buffers(model.root, frozenset(), model)
    model.materials.edit(mat.id, base_color=(0.1, 0.2, 0.3))
    again = r._ensure_buffers(model.root, frozenset(), model)
    assert again is first
    assert len(calls) == 1


def test_painting_a_material_the_key_does_not_name_rebuilds_the_buffer(uv_stubbed_renderer):
    # uv_key_still_matches re-reads only the materials the cached key already
    # names. That is safe ONLY because every path that can introduce a new
    # material id also marks the scene render-dirty (Scene.set_face_material).
    # Nothing else pins that cross-module invariant: a refactor that stopped
    # dirtying would leave the renderer baking UVs from the wrong material with
    # every other test in this file green — the exact shape of failure Step 6b
    # exists to prevent. Driven through the real dirty flag, not the stub.
    r, calls = uv_stubbed_renderer
    model, scene, _ = _painted_box()
    first = r._ensure_buffers(model.root, frozenset(), model)
    assert not scene.dirty, "the upload should have marked the scene clean"

    fresh = model.materials.add_custom("Tile", (0.2, 0.2, 0.2))
    model.materials.edit(fresh.id, texture_size=(3.0, 3.0))
    assert all(entry[0] != fresh.id for entry in first.uv_key), (
        "the new material is already in the cached key, so this test would pass "
        "without the dirty flag doing any work"
    )

    scene.set_face_material(next(iter(scene.faces_iter())).id, fresh.id)
    changed = r._ensure_buffers(model.root, frozenset(), model)
    assert changed is not first
    assert len(calls) == 2
    assert any(entry[0] == fresh.id for entry in changed.uv_key)


# --- M7.5b final review, item 2: the bake is gated on the model having a texture


def _capture_uploads(monkeypatch):
    """A GL-less SceneRenderer plus the FACE buffer uploads it issues.

    Edges go through the same glBufferData with a 6-float row, so the bound
    VBO id is tracked and only face_vbo (1) uploads are kept -- reshaping an
    edge upload to 10 floats is a ValueError, not a wrong answer, but the
    filter is what makes `uploads[-1]` mean "the latest face upload".
    """
    uploads: list[np.ndarray] = []
    bound = [0]

    def bind(_target, vbo):
        bound[0] = vbo

    def buffer_data(_target, _size, data, _usage):
        if bound[0] == 1:
            uploads.append(np.asarray(data, dtype=np.float32).copy())

    monkeypatch.setattr(scene_renderer.GL, "glBindBuffer", bind)
    monkeypatch.setattr(scene_renderer.GL, "glBufferData", buffer_data)
    r = SceneRenderer()
    monkeypatch.setattr(r, "_alloc_def_buffers", lambda: _DefBuffers(face_vbo=1, edge_vbo=2))
    return r, uploads


def _count_bakes(monkeypatch):
    """Count calls to build_face_uvs_both_sides without suppressing them."""
    calls: list[int] = []
    real = scene_renderer.build_face_uvs_both_sides

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(scene_renderer, "build_face_uvs_both_sides", counted)
    return calls


def test_an_untextured_model_uploads_zero_uvs_without_walking_the_faces(monkeypatch):
    # The bake is ~61 ms per re-upload on a 9,600-face definition and produced
    # coordinates nothing in an untextured document can ever sample. Both
    # assertions are needed: a version that still walked the faces and then
    # zeroed the result would pass the UV check while paying the whole cost,
    # and a version that skipped the walk but left the blocks unwritten would
    # upload garbage. The row width is asserted too -- the vertex format stays
    # unconditionally 10 floats; only the arithmetic is gated, never the layout.
    model, _, mat = _painted_box()
    assert mat.texture_id is None
    calls = _count_bakes(monkeypatch)
    r, uploads = _capture_uploads(monkeypatch)

    buf = r._upload_definition(model.root, frozenset(), model)

    assert calls == [], "an untextured model must not pay for the UV bake"
    rows = uploads[0].reshape(-1, _FACE_VERTEX_FLOATS)
    assert rows.shape[1] == 10
    assert rows.shape[0] == buf.face_count
    assert np.all(rows[:, 6:10] == 0.0)


def test_a_model_that_gains_a_texture_rebuilds_real_uvs(monkeypatch):
    # The other half of the pair, and the one that makes the gate safe to
    # ship: assigning a texture to a material the scene already carries
    # changes uv_material_key, so _ensure_buffers stops reusing the
    # zero-filled buffer and the bake finally runs. A gate that latched
    # "this document has no textures" once and never re-checked would pass
    # the test above and fail here with textures rendering as a single texel.
    model, scene, mat = _painted_box()
    model.materials.edit(mat.id, texture_size=(2.0, 2.0))
    r, uploads = _capture_uploads(monkeypatch)
    r._ensure_buffers(model.root, frozenset(), model)
    assert np.all(uploads[0].reshape(-1, _FACE_VERTEX_FLOATS)[:, 6:10] == 0.0)

    model.materials.edit(mat.id, texture_id=5)
    calls = _count_bakes(monkeypatch)
    buf = r._ensure_buffers(model.root, frozenset(), model)

    assert len(calls) == 1, "gaining a texture must re-bake, not reuse the zeros"
    rows = uploads[-1].reshape(-1, _FACE_VERTEX_FLOATS)
    order = np.asarray(buf.plan.vertex_order)
    front = build_face_uvs(scene, model, Side.FRONT)
    assert np.any(front != 0.0), "this box projects to all-zero UVs, so the test is blind"
    assert np.allclose(rows[:, 6:8], front[order], atol=1e-6)


def test_clearing_the_last_texture_goes_back_to_the_cheap_path(monkeypatch):
    # Symmetry check on the gate itself: it is re-evaluated per upload rather
    # than sampled once at construction, so a document whose only texture is
    # cleared stops paying again.
    model, _, mat = _painted_box()
    model.materials.edit(mat.id, texture_id=5)
    r, uploads = _capture_uploads(monkeypatch)
    r._ensure_buffers(model.root, frozenset(), model)

    model.materials.edit(mat.id, texture_id=None)
    calls = _count_bakes(monkeypatch)
    r._ensure_buffers(model.root, frozenset(), model)

    assert calls == []
    assert np.all(uploads[-1].reshape(-1, _FACE_VERTEX_FLOATS)[:, 6:10] == 0.0)


def test_a_definition_carrying_no_textured_material_skips_the_bake(monkeypatch):
    """#112: the M7.5b gate asked the whole MaterialLibrary, so a document
    containing one image anywhere made EVERY definition pay the full bake,
    including definitions where nothing is textured.

    Measured on a 9,600-face definition, that was 33.8 ms against about
    95 ms per re-upload, and uploads fire on every mesh edit that dirties a
    definition, so it is a stall per push/pull on the ordinary document
    that has some textured surfaces and many untextured ones.

    The gate asks about THIS definition now: the materials its own faces
    carry, which uv_material_key already collects for both sides. A texture
    on a material no face here carries samples nothing here.
    """
    model, _, mat = _painted_box()
    assert mat.texture_id is None
    elsewhere = model.materials.add_custom("Textured", (1.0, 1.0, 1.0))
    model.materials.edit(elsewhere.id, texture_id=5)

    calls = _count_bakes(monkeypatch)
    r, uploads = _capture_uploads(monkeypatch)

    buf = r._upload_definition(model.root, frozenset(), model)

    assert calls == [], "a definition whose own materials carry no texture must not bake"
    rows = uploads[0].reshape(-1, _FACE_VERTEX_FLOATS)
    assert rows.shape[1] == 10
    assert rows.shape[0] == buf.face_count
    assert np.all(rows[:, 6:10] == 0.0)


def test_painting_this_definition_with_the_textured_material_bakes_it(monkeypatch):
    """The other half of the pair. A per-definition gate that latched off
    would pass the test above and leave a genuinely textured definition
    rendering a single texel."""
    model, scene, mat = _painted_box()
    model.materials.edit(mat.id, texture_id=5)

    calls = _count_bakes(monkeypatch)
    r, uploads = _capture_uploads(monkeypatch)
    buf = r._upload_definition(model.root, frozenset(), model)

    assert len(calls) == 1
    rows = uploads[0].reshape(-1, _FACE_VERTEX_FLOATS)
    order = np.asarray(buf.plan.vertex_order)
    front = build_face_uvs(scene, model, Side.FRONT)
    assert np.any(front != 0.0), "this box projects to all-zero UVs, so the test is blind"
    assert np.allclose(rows[:, 6:8], front[order], atol=1e-6)
