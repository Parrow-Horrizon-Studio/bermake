"""End-to-end glTF bridge integration (needs the compiled kernel + Assimp).

Includes the PERMANENT Draco CI gate — do NOT add skip/xfail markers here. If
a vcpkg assimp bump drops Draco, this must fail CI rather than degrade silently.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import bermake._core as core
from bermake.io.gltf_export import export_gltf
from bermake.model.model import Model

DATA = Path(__file__).parent / "data" / "gltf"


def test_plain_box_decodes():
    s = core.import_gltf(str(DATA / "plain_box.glb"))
    assert len(s.meshes) >= 1
    assert len(s.meshes[0].triangles) > 0


def test_draco_box_decodes_CI_GATE():  # noqa: N802 (name is the permanent CI gate marker)
    """PERMANENT GATE: Assimp must decode KHR_draco_mesh_compression. Never skip."""
    s = core.import_gltf(str(DATA / "draco_box.glb"))
    assert len(s.meshes) >= 1
    assert len(s.meshes[0].triangles) > 0, "Draco decode produced no geometry"


def test_avocado_draco_uvs_decode_CI_GATE():  # noqa: N802 (permanent CI gate marker)
    """PERMANENT GATE: Assimp must decode KHR_draco_mesh_compression INCLUDING
    TEXCOORD_0. Never skip, never xfail.

    draco_box.glb stays as the geometry-only gate. It declares NORMAL and
    POSITION only, so it passes perfectly while UV extraction is completely
    broken, which is exactly the hole this test closes.
    """
    s = core.import_gltf(str(DATA / "avocado_draco.glb"))
    assert len(s.meshes) >= 1
    assert len(s.meshes[0].triangles) > 0, "Draco decode produced no geometry"
    assert len(s.meshes[0].uvs) == len(s.meshes[0].positions), "Draco decoded no TEXCOORD_0"
    assert len(s.images) == 1 and s.images[0].data, "the embedded base colour image is missing"


def test_assimp_already_flips_v_CI_GATE():  # noqa: N802 (permanent CI gate marker)
    """PERMANENT GATE: Assimp's glTF2 importer applies 1 - v itself. Never skip.

    Bermake's v = 0 is the image's BOTTOM (viewport/texture_cache.gl_row_order
    reverses Qt's top-down rows for OpenGL's bottom-left origin); glTF's v = 0
    is the image's TOP. Assimp does that conversion before the bridge sees a
    coordinate, so import deliberately performs NO flip of its own and export
    performs one (D14).

    If an Assimp upgrade ever stops flipping, every imported texture renders
    upside down with nothing else failing. This compares the bridge's output
    against the fixture's RAW BUFFER BYTES so that change fails here, loudly
    and by name, instead of silently in the viewport.
    """
    import json
    import struct

    raw = (DATA / "textured_box.glb").read_bytes()
    json_len, _kind = struct.unpack_from("<II", raw, 12)
    doc = json.loads(raw[20 : 20 + json_len])
    bin_start = 20 + json_len + 8
    accessor = doc["accessors"][doc["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_0"]]
    view = doc["bufferViews"][accessor["bufferView"]]
    # This gate only reads bufferView.byteOffset below; it does not account
    # for accessor.byteOffset or an interleaved bufferView.byteStride. Both
    # are absent from the current fixture, so ignoring them is harmless
    # today, but a silent regeneration that changed the layout would make
    # this gate compare the wrong bytes instead of failing. Assert the
    # layout this gate actually knows how to read, so a future fixture that
    # needs either one fails here by name instead of corrupting the
    # comparison below.
    assert accessor.get("byteOffset", 0) == 0, (
        "gate does not account for accessor.byteOffset; update it before trusting this fixture"
    )
    assert "byteStride" not in view, (
        "gate does not account for an interleaved bufferView.byteStride; update it first"
    )
    base = bin_start + view.get("byteOffset", 0)
    file_uvs = [struct.unpack_from("<2f", raw, base + i * 8) for i in range(accessor["count"])]

    bridge_uvs = core.import_gltf(str(DATA / "textured_box.glb")).meshes[0].uvs
    assert len(bridge_uvs) == len(file_uvs)

    # Compare as sorted multisets: JoinIdenticalVertices may reorder vertices,
    # but the SET of v values must be the file's set with 1 - v applied.
    expected_v = sorted(round(1.0 - v, 5) for _u, v in file_uvs)
    actual_v = sorted(round(float(uv[1]), 5) for uv in bridge_uvs)
    assert actual_v == expected_v, (
        f"Assimp's V convention changed: expected {expected_v}, got {actual_v}. "
        "If Assimp stopped flipping, import must start flipping (see D14)."
    )
    # u must be untouched in either convention.
    assert sorted(round(u, 5) for u, _v in file_uvs) == sorted(
        round(float(uv[0]), 5) for uv in bridge_uvs
    )


def _shared_component_model():
    """A model with one component instanced twice (mesh-level instancing)."""
    model = Model()
    comp = model.new_definition("Widget", is_group=False)
    ids = [
        comp.mesh.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        comp.mesh.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        comp.mesh.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32)),
    ]
    comp.mesh.add_face_from_loop(ids)
    i1 = model.new_instance(comp)
    i2 = model.new_instance(comp, transform=np.eye(4, dtype=np.float64))
    model.root.children.extend([i1, i2])
    return model


def test_export_preserves_mesh_level_instancing(tmp_path):
    export_gltf(_shared_component_model(), str(tmp_path / "inst.glb"))
    s = core.import_gltf(str(tmp_path / "inst.glb"))
    # one shared mesh, referenced by two nodes
    mesh_refs = [n for n in s.nodes if len(n.mesh_indices) > 0]
    used = {mi for n in s.nodes for mi in n.mesh_indices}
    assert len(used) == 1                      # exactly one distinct mesh
    assert len(mesh_refs) == 2                 # referenced by two nodes


def test_gltf_sidecar_roundtrips(tmp_path):
    export_gltf(_shared_component_model(), str(tmp_path / "h.gltf"))
    assert (tmp_path / "h.bin").exists()
    s = core.import_gltf(str(tmp_path / "h.gltf"))
    assert len(s.meshes) >= 1
