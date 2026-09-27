import subprocess
import sys

import numpy as np
import pytest
from bermake.io.obj_codec import parse_obj
from bermake.io.obj_io import export_obj, model_to_objdoc
from bermake.model.model import Model
from bermake.scene.scene import Side
from bermake.viewport.uv_resolve import resolve_face_uvs

FAKE_PNG = b"\x89PNG\r\n\x1a\n-fake-bytes"


def _model_with_stored_uvs():
    model = Model()
    mesh = model.root.mesh
    ids = [
        mesh.add_vertex(np.array(p, dtype=np.float32))
        for p in [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    ]
    fid = mesh.add_face_from_loop(ids)
    mesh.set_face_uvs(fid, [(0.0, 0.0), (0.25, 0.0), (0.5, 0.5), (0.75, 1.0)], Side.FRONT)
    return model, fid


def test_export_writes_a_vt_per_corner():
    model, _ = _model_with_stored_uvs()
    doc = model_to_objdoc(model, resolver=resolve_face_uvs)
    face = doc.objects[0].faces[0]
    assert face.uv_indices is not None
    assert len(face.uv_indices) == len(face.vertex_indices)


def test_exported_uvs_are_the_stored_values():
    model, _ = _model_with_stored_uvs()
    doc = model_to_objdoc(model, resolver=resolve_face_uvs)
    face = doc.objects[0].faces[0]
    got = [doc.uvs[t] for t in face.uv_indices]
    np.testing.assert_allclose(got, [(0.0, 0.0), (0.25, 0.0), (0.5, 0.5), (0.75, 1.0)], atol=1e-5)


def test_a_face_with_no_stored_uvs_and_no_texture_exports_no_uvs():
    # UVs are written only when a face genuinely needs them: it has its own
    # stored front-side UVs, or its front material carries a texture.
    # Baking a projection onto a face with neither freezes it for no benefit
    # (a later texture_size/placement edit can no longer reproject it) and
    # bloats untextured exports with a UV map nothing downstream needs.
    model = Model()
    mesh = model.root.mesh
    ids = [
        mesh.add_vertex(np.array(p, dtype=np.float32)) for p in [(0, 0, 0), (1, 0, 0), (1, 1, 0)]
    ]
    mesh.add_face_from_loop(ids)
    doc = model_to_objdoc(model, resolver=resolve_face_uvs)
    assert doc.objects[0].faces[0].uv_indices is None
    assert doc.uvs == ()


def test_a_face_with_no_stored_uvs_but_a_textured_material_exports_uvs():
    model = Model()
    mesh = model.root.mesh
    ids = [
        mesh.add_vertex(np.array(p, dtype=np.float32)) for p in [(0, 0, 0), (1, 0, 0), (1, 1, 0)]
    ]
    fid = mesh.add_face_from_loop(ids)
    tex = model.textures.add("brick", FAKE_PNG, "png", 4, 8, False)
    mat = model.materials.add_custom("brick", (1.0, 1.0, 1.0))
    model.materials.edit(mat.id, texture_id=tex.id)
    mesh.set_face_material(fid, mat.id, Side.FRONT)

    doc = model_to_objdoc(model, resolver=resolve_face_uvs)
    face = doc.objects[0].faces[0]
    assert face.uv_indices is not None
    assert len(face.uv_indices) == len(face.vertex_indices)


def test_export_without_a_resolver_writes_no_uvs():
    model, _ = _model_with_stored_uvs()
    doc = model_to_objdoc(model)
    assert doc.uvs == ()
    assert doc.objects[0].faces[0].uv_indices is None


def test_the_texture_image_is_written_beside_the_obj(tmp_path):
    model, _ = _model_with_stored_uvs()
    tex = model.textures.add("brick", FAKE_PNG, "png", 4, 8, False)
    mat = model.materials.add_custom("brick", (1.0, 1.0, 1.0))
    model.materials.edit(mat.id, texture_id=tex.id)
    model.root.mesh.set_face_material(
        next(iter(model.root.mesh.faces_iter())).id, mat.id, Side.FRONT
    )

    out = tmp_path / "m.obj"
    export_obj(out, model)
    image = tmp_path / "brick.png"
    assert image.is_file()
    assert image.read_bytes() == FAKE_PNG
    assert "map_Kd brick.png" in (tmp_path / "m.mtl").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "material_name",
    ["../../pwned", "a:b", "a#b", "!!!!!!"],
)
def test_a_hostile_material_name_exports_a_safe_sidecar(tmp_path, material_name):
    """Finding 1 (M7.5c-3 whole-branch review): sanitize_material_name feeds
    a `Path.with_name` filename for the texture sidecar, and a material name
    is untrusted text from a file someone else authored (the glTF side of
    this same defect is pinned in test_gltf_export_uvs.py). The export must
    complete and the written image filename must be safe."""
    model, fid = _model_with_stored_uvs()
    tex = model.textures.add("brick", FAKE_PNG, "png", 4, 8, False)
    mat = model.materials.add_custom(material_name, (1.0, 1.0, 1.0))
    model.materials.edit(mat.id, texture_id=tex.id)
    model.root.mesh.set_face_material(fid, mat.id, Side.FRONT)

    out = tmp_path / "m.obj"
    export_obj(out, model)

    mtl_text = (tmp_path / "m.mtl").read_text(encoding="utf-8")
    (map_kd_line,) = [line for line in mtl_text.splitlines() if line.startswith("map_Kd ")]
    image_name = map_kd_line.removeprefix("map_Kd ").strip()
    assert "/" not in image_name and "\\" not in image_name and ":" not in image_name
    assert "#" not in image_name and "?" not in image_name and "%" not in image_name
    written = tmp_path / image_name
    assert written.is_file()
    assert written.read_bytes() == FAKE_PNG


def test_a_round_trip_preserves_the_uvs(tmp_path):
    model, _ = _model_with_stored_uvs()
    out = tmp_path / "m.obj"
    export_obj(out, model)
    back = parse_obj(out.read_text(encoding="utf-8"), None)
    face = back.objects[0].faces[0]
    got = [back.uvs[t] for t in face.uv_indices]
    np.testing.assert_allclose(got, [(0.0, 0.0), (0.25, 0.0), (0.5, 0.5), (0.75, 1.0)], atol=1e-5)


def test_obj_io_imports_without_qt():
    """bermake.io must never reach Qt, even transitively.

    export_obj imports resolve_face_uvs lazily so bermake/io stays Qt-free at
    module scope. This runs the check in a clean subprocess interpreter so it
    is not polluted by anything the test session has already imported.
    """
    code = (
        "import sys\n"
        "import bermake.io.obj_io\n"
        "qt = [m for m in sys.modules if 'PySide6' in m]\n"
        "assert qt == [], qt\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_two_materials_that_sanitize_alike_export_as_two_materials(tmp_path):
    """#119: `Brick Wall` and `Brick_Wall` both sanitize to `Brick_Wall`.

    The OBJ material dict was keyed on that sanitized name, so the two
    collapsed onto one entry and the loser's colour and image vanished from
    the export with nothing raised and nothing warned. OBJ names are minted
    once per material id now, so a collision becomes `Brick_Wall.001`.
    """
    model = Model()
    mesh = model.root.mesh
    tri_a = [
        mesh.add_vertex(np.array(p, dtype=np.float32)) for p in [(0, 0, 0), (1, 0, 0), (1, 1, 0)]
    ]
    tri_b = [
        mesh.add_vertex(np.array(p, dtype=np.float32)) for p in [(2, 0, 0), (3, 0, 0), (3, 1, 0)]
    ]
    face_a = mesh.add_face_from_loop(tri_a)
    face_b = mesh.add_face_from_loop(tri_b)

    image_a, image_b = FAKE_PNG + b"-A", FAKE_PNG + b"-B"
    tex_a = model.textures.add("a", image_a, "png", 4, 8, False)
    tex_b = model.textures.add("b", image_b, "png", 4, 8, False)
    mat_a = model.materials.add_custom("Brick Wall", (1.0, 0.0, 0.0))
    mat_b = model.materials.add_custom("Brick_Wall", (0.0, 0.0, 1.0))
    model.materials.edit(mat_a.id, texture_id=tex_a.id)
    model.materials.edit(mat_b.id, texture_id=tex_b.id)
    mesh.set_face_material(face_a, mat_a.id, Side.FRONT)
    mesh.set_face_material(face_b, mat_b.id, Side.FRONT)

    out = tmp_path / "m.obj"
    export_obj(out, model)

    mtl_text = (tmp_path / "m.mtl").read_text(encoding="utf-8")
    lines = mtl_text.splitlines()
    names = [ln.removeprefix("newmtl ").strip() for ln in lines if ln.startswith("newmtl ")]
    assert len(names) == 2, mtl_text
    assert len(set(names)) == 2, mtl_text

    colors = {ln.strip() for ln in lines if ln.startswith("Kd ")}
    assert len(colors) == 2, mtl_text

    images = {ln.removeprefix("map_Kd ").strip() for ln in lines if ln.startswith("map_Kd ")}
    assert len(images) == 2, mtl_text
    assert {(tmp_path / n).read_bytes() for n in images} == {image_a, image_b}

    back = parse_obj(out.read_text(encoding="utf-8"), mtl_text)
    used = sorted(f.material for o in back.objects for f in o.faces)
    assert used == sorted(names)
