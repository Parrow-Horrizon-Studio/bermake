import numpy as np

from bermake.io.obj_codec import parse_obj
from bermake.io.obj_io import build_obj_into_model
from bermake.model.model import Model
from bermake.scene.scene import Side

QUAD = (
    "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
    "vt 0.1 0.2\nvt 0.3 0.4\nvt 0.5 0.6\nvt 0.7 0.8\n"
    "f 1/1 2/2 3/3 4/4\n"
)


def _import(obj_text):
    model = Model()
    doc = parse_obj(obj_text, None)
    build_obj_into_model(doc, model, model.root)
    return model.root.mesh


def test_the_kernel_loop_matches_the_order_the_importer_passed():
    # The positional UV mapping depends on this. Verified, not assumed.
    mesh = _import(QUAD)
    fid = next(iter(mesh.faces_iter())).id
    loop = mesh.face_loop(fid)
    positions = [tuple(round(float(c), 6) for c in mesh.vertex(v).position) for v in loop]
    assert positions == [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]


def test_imported_uvs_land_in_loop_order():
    mesh = _import(QUAD)
    fid = next(iter(mesh.faces_iter())).id
    got = mesh.face_uvs(fid, Side.FRONT)
    assert got is not None
    np.testing.assert_allclose(got, [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], rtol=1e-6)


def test_both_sides_receive_the_array():
    mesh = _import(QUAD)
    fid = next(iter(mesh.faces_iter())).id
    np.testing.assert_allclose(
        mesh.face_uvs(fid, Side.FRONT), mesh.face_uvs(fid, Side.BACK), rtol=1e-6
    )


def test_a_face_without_vt_gets_no_stored_array():
    mesh = _import("v 0 0 0\nv 1 0 0\nv 1 1 0\nf 1 2 3\n")
    fid = next(iter(mesh.faces_iter())).id
    assert mesh.face_uvs(fid, Side.FRONT) is None


def test_a_uv_seam_gives_two_faces_different_uvs_on_one_shared_vertex():
    # Two triangles sharing the edge (1,0,0)-(1,1,0). Both `f` lines reference
    # the same OBJ vertex row for (1,0,0): this is index reuse, not welding
    # (no two distinct `v` rows collapse here). It still proves the thing that
    # matters for storage: one shared kernel corner carries a different UV per
    # face, which per-vertex storage could not express but per-corner storage
    # does. See test_real_v_row_duplicates_weld_to_one_kernel_vertex below for
    # the case where two distinct `v` rows actually collapse.
    obj = (
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 2 0 0\n"
        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0.25 0.75\nvt 0.5 0.5\n"
        "f 1/1 2/2 3/3\n"
        "f 2/4 4/5 3/1\n"
    )
    mesh = _import(obj)
    faces = [f.id for f in mesh.faces_iter()]
    assert len(faces) == 2
    uv_a = mesh.face_uvs(faces[0], Side.FRONT)
    uv_b = mesh.face_uvs(faces[1], Side.FRONT)
    np.testing.assert_allclose(uv_a[1], [1.0, 0.0], rtol=1e-6)
    np.testing.assert_allclose(uv_b[0], [0.25, 0.75], rtol=1e-6)


def test_real_v_row_duplicates_weld_to_one_kernel_vertex():
    # Two DISTINCT `v` rows (2 and 4) carry identical coordinates (1, 0, 0).
    # This is deliberate, not an oversight: it is exactly what an OBJ exporter
    # emits when it splits vertices at a UV seam, which is the single most
    # common way a textured OBJ arrives. Do not "tidy" these into one `v` row;
    # that removes the case this test exists to cover, namely add_vertex's
    # position idempotence actually collapsing two rows into one kernel id.
    obj = (
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 1 0 0\nv 2 0 0\n"
        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0.25 0.75\nvt 0.5 0.5\n"
        "f 1/1 2/2 3/3\n"
        "f 4/4 5/5 3/1\n"
    )
    doc = parse_obj(obj, None)
    assert len(doc.vertices) == 5

    model = Model()
    build_obj_into_model(doc, model, model.root)
    mesh = model.root.mesh

    kernel_vertex_count = len(list(mesh.vertices_iter()))
    assert kernel_vertex_count < len(doc.vertices)

    faces = [f.id for f in mesh.faces_iter()]
    assert len(faces) == 2
    loop_a = mesh.face_loop(faces[0])
    loop_b = mesh.face_loop(faces[1])
    # Row 2 (in face A) and row 4 (in face B) must collapse to the same
    # kernel vertex id, asserted directly so this fails loudly if add_vertex
    # ever stops being position-idempotent.
    assert loop_a[1] == loop_b[0]

    uv_a = mesh.face_uvs(faces[0], Side.FRONT)
    uv_b = mesh.face_uvs(faces[1], Side.FRONT)
    np.testing.assert_allclose(uv_a[1], [1.0, 0.0], rtol=1e-6)
    np.testing.assert_allclose(uv_b[0], [0.25, 0.75], rtol=1e-6)


def test_a_pinched_face_imports_geometry_but_gets_no_stored_uvs():
    # Finding 2 (M7.5c-2 review): unlike the two-face case above, here the
    # SAME face names two distinct `v` rows (2 and 4) that weld to one
    # kernel vertex, e.g. loop [0, 1, 2, 1, 3]. Scene.face_triangle_loop_
    # indices builds a {vertex_id: loop_index} map, so the duplicate's later
    # occurrence silently overwrites the earlier one there, while
    # resolve_face_uvs and the exporter walk the loop in order and disagree
    # with the renderer. The face must still import (geometry + material),
    # it just falls back to the plane projection like any face whose UV data
    # cannot be honored. Do NOT fix this by guarding it in
    # Scene.set_face_uvs instead: that is stage-1 code, and import is the
    # only path that can create such a face with stored UVs today.
    obj = (
        "v 0 0 0\nv 1 0 0\nv 2 0 0\nv 1 0 0\nv 0 1 0\n"
        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0.5 0.9\nvt 0 1\n"
        "f 1/1 2/2 3/3 4/4 5/5\n"
    )
    doc = parse_obj(obj, None)
    model = Model()
    result = build_obj_into_model(doc, model, model.root)

    assert result.summary.faces_imported == 1
    assert result.summary.faces_skipped == 0
    fid = next(iter(model.root.mesh.faces_iter())).id
    loop = model.root.mesh.face_loop(fid)
    assert len(set(loop)) < len(loop)  # confirms the pinch actually happened
    assert model.root.mesh.face_uvs(fid, Side.FRONT) is None
    assert model.root.mesh.face_uvs(fid, Side.BACK) is None


def test_a_mixed_document_stores_uvs_only_where_they_exist():
    obj = (
        "v 0 0 0\nv 1 0 0\nv 1 1 0\n"
        "v 3 0 0\nv 4 0 0\nv 4 1 0\n"
        "vt 0 0\nvt 1 0\nvt 1 1\n"
        "f 1/1 2/2 3/3\n"
        "f 4 5 6\n"
    )
    mesh = _import(obj)
    stored = [f for f, s in mesh.faces_with_uvs() if s is Side.FRONT]
    assert len(stored) == 1


def test_uvs_survive_a_grouped_import():
    obj = "o thing\n" + QUAD
    model = Model()
    doc = parse_obj(obj, None)
    build_obj_into_model(doc, model, model.root)
    defn = model.root.children[0].definition
    fid = next(iter(defn.mesh.faces_iter())).id
    got = defn.mesh.face_uvs(fid, Side.FRONT)
    assert got is not None
    np.testing.assert_allclose(got, [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], rtol=1e-6)
