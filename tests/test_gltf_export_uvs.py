"""glTF export bakes resolved UVs into per-primitive vertex pools."""

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from bermake.io.gltf_export import export_gltf, model_to_gltf
from bermake.model.model import Model
from bermake.scene.scene import Side
from bermake.viewport.uv_resolve import resolve_face_uvs as _resolver


def _quad_definition(model):
    """One unit quad in the XY plane as two triangle faces sharing edge v0-v2."""
    defn = model.new_definition("Quad", is_group=False)
    mesh = defn.mesh
    v = [
        mesh.add_vertex(np.array(p, dtype=np.float32))
        for p in [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    ]
    f0 = mesh.add_face_from_loop([v[0], v[1], v[2]])
    f1 = mesh.add_face_from_loop([v[0], v[2], v[3]])
    model.root.children.append(model.new_instance(defn))
    return defn, [f0, f1]


def _two_material_model():
    """That quad with its two triangles painted differently, so the grouping
    into two separate primitives is genuinely exercised."""
    model = Model()
    defn, fids = _quad_definition(model)
    red = model.materials.add_custom("Red", (1.0, 0.0, 0.0))
    blue = model.materials.add_custom("Blue", (0.0, 0.0, 1.0))
    defn.mesh.set_face_material(fids[0], red.id)
    defn.mesh.set_face_material(fids[1], blue.id)
    return model


def _two_faces_sharing_an_edge():
    """The same quad with both triangles on ONE material, so they land in one
    primitive. A seam is only observable inside a single primitive."""
    model = Model()
    defn, fids = _quad_definition(model)
    mat = model.materials.add_custom("Skin", (1.0, 1.0, 1.0))
    for fid in fids:
        defn.mesh.set_face_material(fid, mat.id)
    return model, fids


def _single_quad_with_uvs(uvs):
    """ONE four-corner face carrying `uvs`. Model has no definitions()
    accessor, so callers reach the mesh through _only_definition."""
    model = Model()
    defn = model.new_definition("Quad", is_group=False)
    mesh = defn.mesh
    ids = [
        mesh.add_vertex(np.array(p, dtype=np.float32))
        for p in [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    ]
    fid = mesh.add_face_from_loop(ids)
    mesh.set_face_uvs(fid, uvs, Side.FRONT)
    model.root.children.append(model.new_instance(defn))
    return model, fid


def _only_definition(model):
    for defn, _world in model.traverse():
        if list(defn.mesh.faces_iter()):
            return defn
    raise AssertionError("no definition with faces")


def _decode(asset):
    """(json doc, read(accessor_index)) decoding out of the asset's own buffer.

    Goes through write_gltf rather than touching GltfAsset._buffer, so the
    test reads exactly the bytes a consumer would.
    """
    json_text, blob, _sidecars = asset.write_gltf("m.bin")
    doc = json.loads(json_text)

    def read(index):
        acc = doc["accessors"][index]
        base = doc["bufferViews"][acc["bufferView"]].get("byteOffset", 0)
        if acc["type"] == "VEC3":
            return [struct.unpack_from("<3f", blob, base + i * 12) for i in range(acc["count"])]
        if acc["type"] == "VEC2":
            return [struct.unpack_from("<2f", blob, base + i * 8) for i in range(acc["count"])]
        return list(struct.unpack_from(f"<{acc['count']}I", blob, base))

    return doc, read


def _canon(tri):
    """Rotate a triangle to start at its smallest corner.

    Rotation is NOT a contract: mapbox_earcut 2.0.0 and 2.1.0 emit the same
    triangle started at a different vertex, and pinning a rotation turned CI
    red during stage 2. Winding IS a contract, so rotate rather than sort.
    """
    i = min(range(3), key=lambda k: tri[k])
    return (tri[i], tri[(i + 1) % 3], tri[(i + 2) % 3])


def _triangle_soup(asset):
    """Every exported triangle as (material, canonical corner positions)."""
    doc, read = _decode(asset)
    out = []
    for mesh in doc["meshes"]:
        for prim in mesh["primitives"]:
            pos = read(prim["attributes"]["POSITION"])
            idx = read(prim["indices"])
            for a, b, c in zip(idx[0::3], idx[1::3], idx[2::3], strict=True):
                corners = tuple(tuple(round(float(v), 5) for v in pos[i]) for i in (a, b, c))
                out.append((prim.get("material"), _canon(corners)))
    return sorted(out)


def test_the_pool_rewrite_does_not_change_the_geometry_it_exports():
    """The rewrite must export the same triangles, in the same material
    groups, at the same coordinates.

    Deliberately NOT an assertion about accessor counts. With per-primitive
    pools a primitive stops carrying the definition vertices it never indexes,
    so this untextured two-material quad legitimately drops from 8 pooled
    positions (4 shared vertices published twice) to 6. Triangles are what a
    consumer sees, and those must not move.
    """
    model = _two_material_model()
    asset = model_to_gltf(model)  # no resolver: UVs off, exactly as today
    doc, _read = _decode(asset)
    for mesh in doc["meshes"]:
        for prim in mesh["primitives"]:
            assert "TEXCOORD_0" not in prim["attributes"]
    assert _triangle_soup(asset) == sorted(
        [
            (0, _canon(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)))),
            (1, _canon(((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)))),
        ]
    )


def test_a_uv_seam_duplicates_the_position():
    """Two faces meeting at a welded vertex with different UVs there.

    glTF indexes POSITION and TEXCOORD_0 with ONE index buffer, so the shared
    vertex must appear twice. This is the whole reason the pool is keyed on
    (vertex, uv) and not on vertex alone, and OBJ needed none of it because
    `f v/vt` indexes the two pools independently.
    """
    model, fids = _two_faces_sharing_an_edge()
    mesh = _only_definition(model).mesh
    # Both faces touch v0 and v2. Give them DIFFERENT UVs there: that is a
    # seam, and it is exactly what a single shared pool cannot express.
    mesh.set_face_uvs(fids[0], [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)], Side.FRONT)
    mesh.set_face_uvs(fids[1], [(0.5, 0.5), (0.25, 0.9), (0.0, 1.0)], Side.FRONT)

    asset = model_to_gltf(model, resolver=_resolver)
    doc, _read = _decode(asset)
    prims = doc["meshes"][0]["primitives"]
    assert len(prims) == 1, "one material must give one primitive"
    pos_count = doc["accessors"][prims[0]["attributes"]["POSITION"]]["count"]
    uv_count = doc["accessors"][prims[0]["attributes"]["TEXCOORD_0"]]["count"]
    assert pos_count == uv_count, "glTF indexes both through one index buffer"
    assert pos_count == 6, "v0 and v2 disagree across the two faces, so neither pools"


def test_matching_uvs_at_a_shared_vertex_still_pool():
    """The control for the seam test.

    Keying on (vertex, uv) must NOT duplicate a vertex whose UV genuinely
    agrees across both faces. Without this, the seam test above passes against
    an implementation that simply never pools anything, which would inflate
    every textured export.
    """
    model, fids = _two_faces_sharing_an_edge()
    mesh = _only_definition(model).mesh
    mesh.set_face_uvs(fids[0], [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)], Side.FRONT)
    mesh.set_face_uvs(fids[1], [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0)], Side.FRONT)

    asset = model_to_gltf(model, resolver=_resolver)
    doc, _read = _decode(asset)
    prim = doc["meshes"][0]["primitives"][0]
    assert doc["accessors"][prim["attributes"]["POSITION"]]["count"] == 4


def test_exported_v_is_flipped_back_into_gltf_convention():
    """Pinned as literals, not as 1 - v recomputed the way the exporter
    computes it, which would agree with itself while both were wrong.

    u and v are chosen so that the exported u range (0.6 to 0.9) and the
    exported v range (0.05 to 0.35) are disjoint. A missing flip, a doubled
    flip and a u/v swap therefore each produce a distinct, visible failure.
    """
    model, _fid = _single_quad_with_uvs([(0.6, 0.65), (0.7, 0.75), (0.8, 0.85), (0.9, 0.95)])
    asset = model_to_gltf(model, resolver=_resolver)
    doc, read = _decode(asset)
    uvs = read(doc["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_0"])
    assert sorted({round(v, 5) for _u, v in uvs}) == [0.05, 0.15, 0.25, 0.35]
    assert sorted({round(u, 5) for u, _v in uvs}) == [0.6, 0.7, 0.8, 0.9]


def test_an_untextured_primitive_with_no_stored_uvs_gets_no_texcoord():
    """Stage 2's gate, lifted to the primitive (D16).

    Baking a projection onto a face with neither stored UVs nor a texture
    would freeze it at the export-time projection, so a later texture_size or
    placement edit could no longer reproject it, and would bloat the file for
    no benefit.
    """
    model = _two_material_model()
    asset = model_to_gltf(model, resolver=_resolver)
    doc, _read = _decode(asset)
    for prim in doc["meshes"][0]["primitives"]:
        assert "TEXCOORD_0" not in prim["attributes"]


def test_a_primitive_where_only_one_face_has_stored_uvs_emits_uvs_for_all():
    """D16: TEXCOORD_0 count must equal POSITION count, so UVs are
    all-or-nothing per primitive. The face without stored UVs gets its plane
    projection, which is what the renderer draws for it, rather than zeros,
    which would be a UV map with no meaning."""
    model, fids = _two_faces_sharing_an_edge()
    mesh = _only_definition(model).mesh
    mesh.set_face_uvs(fids[0], [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)], Side.FRONT)
    # fids[1] deliberately left with nothing stored.

    asset = model_to_gltf(model, resolver=_resolver)
    doc, read = _decode(asset)
    prim = doc["meshes"][0]["primitives"][0]
    uvs = read(prim["attributes"]["TEXCOORD_0"])
    assert len(uvs) == doc["accessors"][prim["attributes"]["POSITION"]]["count"]

    # The projected face's corners must carry its REAL projection, not (0, 0).
    projected = _resolver(mesh, model.materials, fids[1], Side.FRONT)
    exported = {(round(u, 4), round(1.0 - v, 4)) for u, v in uvs}
    for u, v in projected:
        assert (round(float(u), 4), round(float(v), 4)) in exported
    assert exported != {(0.0, 0.0)}


def test_a_textured_material_gates_uvs_on_even_with_nothing_stored():
    """A textured face needs UVs whether or not an importer gave it any:
    without them a consumer has no way to place the image at all."""
    model, fids = _two_faces_sharing_an_edge()
    mesh = _only_definition(model).mesh
    mid = mesh.face_material(fids[0])
    tex = model.textures.add("brick", b"\x89PNG\r\n\x1a\nx", "png", 4, 4, False)
    model.materials.edit(mid, texture_id=tex.id)

    asset = model_to_gltf(model, resolver=_resolver)
    doc, _read = _decode(asset)
    assert "TEXCOORD_0" in doc["meshes"][0]["primitives"][0]["attributes"]


def test_glb_embeds_the_texture_image_in_the_buffer(tmp_path):
    model = _textured_model()
    export_gltf(model, tmp_path / "out.glb")
    assert list(tmp_path.iterdir()) == [tmp_path / "out.glb"], "a GLB must be self-contained"
    doc = _glb_json(tmp_path / "out.glb")
    assert "bufferView" in doc["images"][0]
    assert doc["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"] == 0


def test_gltf_writes_the_image_as_a_sibling(tmp_path):
    model = _textured_model()
    export_gltf(model, tmp_path / "out.gltf")
    doc = json.loads((tmp_path / "out.gltf").read_text(encoding="utf-8"))
    uri = doc["images"][0]["uri"]
    assert (tmp_path / uri).is_file()
    assert (tmp_path / uri).read_bytes() == _the_texture_bytes(model), "original bytes, no re-encode"
    assert not list(tmp_path.glob("*.tmp")), "atomic write left a temp file behind"


def test_an_untextured_model_writes_no_image_files(tmp_path):
    export_gltf(_plain_model(), tmp_path / "out.gltf")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.bin", "out.gltf"]


def test_only_materials_actually_used_by_an_exported_face_get_images(tmp_path):
    """A library texture on a material nothing is painted with must not be
    written. OBJ's exporter already filters this way, and gltf_material_for is
    only ever called for a material some face carries, so this pins that the
    filter really is free rather than accidentally absent."""
    model = _textured_model()
    unused = model.materials.add_custom("Unused", (0.0, 1.0, 0.0))
    stray = model.textures.add("stray", b"\x89PNG\r\n\x1a\nzzz", "png", 2, 2, False)
    model.materials.edit(unused.id, texture_id=stray.id)

    export_gltf(model, tmp_path / "out.gltf")
    doc = json.loads((tmp_path / "out.gltf").read_text(encoding="utf-8"))
    assert len(doc["images"]) == 1
    assert not any(p.name.startswith("stray") for p in tmp_path.iterdir())


def test_gltf_export_is_importable_with_no_qt_loaded():
    """D7. A string scan for 'PySide6' passes against a lazy import that still
    fires at call time, so this actually runs one."""
    import subprocess
    import sys

    code = (
        "import sys; import bermake.io.gltf_export; "
        "assert not [m for m in sys.modules if m.startswith('PySide6')], "
        "sorted(m for m in sys.modules if m.startswith('PySide6'))"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


_PNG = b"\x89PNG\r\n\x1a\nnot-really-decoded-here"


def _textured_model():
    """A quad on one material carrying one texture."""
    model, fids = _two_faces_sharing_an_edge()
    mid = _only_definition(model).mesh.face_material(fids[0])
    tex = model.textures.add("brick", _PNG, "png", 4, 4, False)
    model.materials.edit(mid, texture_id=tex.id)
    return model


def _the_texture_bytes(model):
    return next(t for t in model.textures.textures() if t.name == "brick").data


def _plain_model():
    """Geometry with no textures and no stored UVs anywhere."""
    return _two_material_model()


def _glb_json(path):
    b = Path(path).read_bytes()
    length, _kind = struct.unpack_from("<II", b, 12)
    return json.loads(b[20 : 20 + length])


def test_two_materials_sharing_one_texture_embed_it_once():
    """The memoisation contract this task's brief calls out: keyed on Bermake
    texture id, not material id, so two DIFFERENT materials painted with the
    SAME texture must still produce exactly one glTF image/texture pair, both
    materials' baseColorTexture pointing at it. Nothing else in this file
    pins this: a future refactor that memoised per-material instead would
    silently double the embedded bytes on every multi-material textured
    export, and the suite would stay green."""
    model, fids = _two_faces_sharing_an_edge()
    mesh = _only_definition(model).mesh
    mat_a = model.materials.add_custom("A", (1.0, 0.0, 0.0))
    mat_b = model.materials.add_custom("B", (0.0, 0.0, 1.0))
    mesh.set_face_material(fids[0], mat_a.id)
    mesh.set_face_material(fids[1], mat_b.id)
    tex = model.textures.add("shared", _PNG, "png", 4, 4, False)
    model.materials.edit(mat_a.id, texture_id=tex.id)
    model.materials.edit(mat_b.id, texture_id=tex.id)

    asset = model_to_gltf(model, resolver=_resolver)
    doc, _read = _decode(asset)

    assert len(doc["images"]) == 1, "one shared texture must embed exactly once"
    assert len(doc["materials"]) == 2, "both materials must still be exported"
    for mat in doc["materials"]:
        assert mat["pbrMetallicRoughness"]["baseColorTexture"]["index"] == 0


def test_two_distinct_textures_sharing_a_sanitized_name_do_not_collide(tmp_path):
    """The exact scenario that motivated appending the texture id to the
    sidecar filename: two DIFFERENT textures (different bytes, different
    Bermake ids) that both sanitize to "diffuse", each painted on its own
    material, both materials used by an exported face. Without the id
    suffix, `add_image(embed=False)` would write both under the filename
    "diffuse.png" in `sidecars`, the second call silently overwriting the
    first -- the OBJ #119 bug, one level down (texture names rather than
    material names). This is written to fail against that pre-fix scheme:
    see the task report for the RED run with the suffix removed."""
    model, fids = _two_faces_sharing_an_edge()
    mesh = _only_definition(model).mesh
    mat_a = model.materials.add_custom("A", (1.0, 0.0, 0.0))
    mat_b = model.materials.add_custom("B", (0.0, 0.0, 1.0))
    mesh.set_face_material(fids[0], mat_a.id)
    mesh.set_face_material(fids[1], mat_b.id)
    tex_a = model.textures.add("diffuse", _PNG, "png", 4, 4, False)
    tex_b = model.textures.add("diffuse", _PNG + b"-not-the-same-bytes", "png", 4, 4, False)
    model.materials.edit(mat_a.id, texture_id=tex_a.id)
    model.materials.edit(mat_b.id, texture_id=tex_b.id)

    export_gltf(model, tmp_path / "out.gltf")
    doc = json.loads((tmp_path / "out.gltf").read_text(encoding="utf-8"))

    assert len(doc["images"]) == 2
    uris = [img["uri"] for img in doc["images"]]
    assert len(set(uris)) == 2, "two distinct textures must not share one sidecar filename"

    written = {uri: (tmp_path / uri).read_bytes() for uri in uris}
    assert len(written) == 2, "two distinct sidecar files must exist on disk"
    assert set(written.values()) == {tex_a.data, tex_b.data}, "each file keeps its own bytes"


@pytest.mark.parametrize(
    "texture_name",
    ["../../pwned", "a:b", "a#b", "!!!!!!"],
)
def test_a_hostile_texture_name_exports_a_safe_sidecar(tmp_path, texture_name):
    """Finding 1 (M7.5c-3 whole-branch review): a texture name is untrusted
    text -- on import a texture is minted with the name of the source
    document's own material -- and it becomes both a `Path.with_name`
    filename and a glTF `image.uri`. Before the fix, '../../pwned' raised
    ValueError('Invalid name ...') from `Path.with_name` AFTER out.gltf and
    out.bin were already written, leaving a half-exported document with an
    unescaped '../../pwned_1.png' URI on disk (see the task report's RED
    evidence). Every one of these names must now produce a complete export
    with a safe sidecar filename instead."""
    model, fids = _two_faces_sharing_an_edge()
    mesh = _only_definition(model).mesh
    mid = mesh.face_material(fids[0])
    tex = model.textures.add(texture_name, _PNG, "png", 4, 4, False)
    model.materials.edit(mid, texture_id=tex.id)

    out = tmp_path / "out.gltf"
    export_gltf(model, out)

    doc = json.loads(out.read_text(encoding="utf-8"))
    uri = doc["images"][0]["uri"]
    assert "/" not in uri and "\\" not in uri and ":" not in uri
    assert "#" not in uri and "?" not in uri and "%" not in uri
    written = tmp_path / uri
    assert written.is_file()
    assert written.read_bytes() == _PNG
    assert not list(tmp_path.glob("*.tmp")), "atomic write left a temp file behind"


def test_a_resolver_failure_on_one_face_degrades_that_face_not_the_export():
    """Finding 6 (M7.5c-3 whole-branch review): OBJ's `_uv_indices_for` wraps
    its resolver call in `except (KeyError, ValueError)` and degrades that
    one face to no UV contribution. The glTF path called the resolver bare,
    so any resolver failure -- including the `zip(..., strict=True)` above
    turning a stored-UV/loop-length mismatch into a ValueError -- aborted the
    WHOLE export. `model_to_gltf` itself never touches disk, so this pins the
    in-memory shape directly: positions and uvs must stay in lockstep and
    both faces must still be triangulated."""
    model, fids = _two_faces_sharing_an_edge()
    mesh = _only_definition(model).mesh
    mid = mesh.face_material(fids[0])
    tex = model.textures.add("brick", _PNG, "png", 4, 4, False)
    model.materials.edit(mid, texture_id=tex.id)
    failing_fid = fids[0]

    def flaky_resolver(mesh_, materials_, fid, side):
        if fid == failing_fid:
            raise ValueError("simulated resolver failure")
        return _resolver(mesh_, materials_, fid, side)

    asset = model_to_gltf(model, resolver=flaky_resolver)
    doc, read = _decode(asset)
    prim = doc["meshes"][0]["primitives"][0]
    positions = read(prim["attributes"]["POSITION"])
    uvs = read(prim["attributes"]["TEXCOORD_0"])
    assert len(positions) == len(uvs), "positions and uvs pools must stay in lockstep"
    indices = read(prim["indices"])
    assert len(indices) == 6, "both triangles (both faces) must still be exported"


def test_a_resolver_failure_still_produces_a_complete_export_on_disk(tmp_path, monkeypatch):
    """The disk-write shape of the same fix: `export_gltf` has no resolver
    parameter of its own (it lazily imports the real `resolve_face_uvs`), so
    this monkeypatches that import target -- the same lazy `from X import Y`
    pattern `export_obj` uses, re-resolved on every call -- to prove a
    resolver failure still yields a complete, valid .gltf + .bin rather than
    an uncaught exception with partial output already on disk."""
    import bermake.viewport.uv_resolve as uv_resolve_mod

    model, fids = _two_faces_sharing_an_edge()
    mesh = _only_definition(model).mesh
    mid = mesh.face_material(fids[0])
    tex = model.textures.add("brick", _PNG, "png", 4, 4, False)
    model.materials.edit(mid, texture_id=tex.id)
    failing_fid = fids[0]
    real_resolver = uv_resolve_mod.resolve_face_uvs

    def flaky_resolver(mesh_, materials_, fid, side):
        if fid == failing_fid:
            raise ValueError("simulated resolver failure")
        return real_resolver(mesh_, materials_, fid, side)

    monkeypatch.setattr(uv_resolve_mod, "resolve_face_uvs", flaky_resolver)

    out = tmp_path / "out.gltf"
    export_gltf(model, out)

    assert out.is_file()
    assert (tmp_path / "out.bin").is_file()
    doc = json.loads(out.read_text(encoding="utf-8"))
    prim = doc["meshes"][0]["primitives"][0]
    pos_count = doc["accessors"][prim["attributes"]["POSITION"]]["count"]
    uv_count = doc["accessors"][prim["attributes"]["TEXCOORD_0"]]["count"]
    assert pos_count == uv_count
    assert not list(tmp_path.glob("*.tmp")), "atomic write left a temp file behind"
