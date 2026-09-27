from __future__ import annotations

import json
import struct

import pytest

from bermake.io.gltf_codec import GltfAsset

TRI_POS = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
TRI_IDX = [0, 1, 2]
IDENT16 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def _parse_glb(blob: bytes):
    magic, version, total = struct.unpack_from("<III", blob, 0)
    assert magic == 0x46546C67 and version == 2 and total == len(blob)
    jlen, jtype = struct.unpack_from("<II", blob, 12)
    assert jtype == 0x4E4F534A
    json_bytes = blob[20:20 + jlen]
    blen, btype = struct.unpack_from("<II", blob, 20 + jlen)
    assert btype == 0x004E4942
    return json.loads(json_bytes), blen


def _asset_with_triangle():
    a = GltfAsset()
    mat = a.add_material("Red", (1.0, 0.0, 0.0))
    mesh = a.add_mesh([(TRI_POS, None, TRI_IDX, mat)])
    node = a.add_node(name="tri", matrix=IDENT16, mesh=mesh, children=None)
    a.scene_roots.append(node)
    return a


def test_glb_framing_and_structure():
    blob = _asset_with_triangle().write_glb()
    assert len(blob) % 4 == 0
    doc, blen = _parse_glb(blob)
    assert doc["asset"]["version"] == "2.0"
    assert doc["scenes"][0]["nodes"] == [0]
    assert len(doc["meshes"]) == 1
    assert len(doc["materials"]) == 1
    assert doc["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"] == [1.0, 0.0, 0.0, 1.0]
    # POSITION accessor has correct min/max
    pos_acc = doc["accessors"][doc["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    assert pos_acc["type"] == "VEC3" and pos_acc["componentType"] == 5126
    assert pos_acc["count"] == 3
    assert pos_acc["min"] == [0.0, 0.0, 0.0] and pos_acc["max"] == [1.0, 1.0, 0.0]
    assert doc["buffers"][0]["byteLength"] == blen


def test_gltf_has_external_buffer_uri():
    a = _asset_with_triangle()
    json_text, bin_bytes, sidecars = a.write_gltf("model.bin")
    doc = json.loads(json_text)
    assert doc["buffers"][0]["uri"] == "model.bin"
    assert doc["buffers"][0]["byteLength"] == len(bin_bytes)
    assert sidecars == {}

    glb_doc, _ = _parse_glb(a.write_glb())
    assert "uri" not in glb_doc["buffers"][0]


def test_uv_accessor_is_vec2_float_with_a_matching_count():
    a = GltfAsset()
    uvs = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
    a.add_mesh([(TRI_POS, uvs, TRI_IDX, None)])
    prim = a.meshes[0]["primitives"][0]
    uv_acc = a.accessors[prim["attributes"]["TEXCOORD_0"]]
    pos_acc = a.accessors[prim["attributes"]["POSITION"]]
    assert uv_acc["type"] == "VEC2"
    assert uv_acc["componentType"] == 5126
    assert uv_acc["count"] == pos_acc["count"], "glTF requires one index buffer for both"


def test_a_primitive_without_uvs_declares_no_texcoord():
    a = GltfAsset()
    a.add_mesh([(TRI_POS, None, TRI_IDX, None)])
    assert "TEXCOORD_0" not in a.meshes[0]["primitives"][0]["attributes"]


def test_an_embedded_image_lands_in_the_buffer_with_a_mime_type():
    a = GltfAsset()
    idx = a.add_image(b"\x89PNG\r\n\x1a\nxxxx", "png", "brick", embed=True)
    assert a.images[idx]["mimeType"] == "image/png"
    assert "bufferView" in a.images[idx]
    assert "uri" not in a.images[idx]
    assert a.sidecars == {}


def test_a_sidecar_image_gets_a_uri_and_is_handed_back_to_the_caller():
    a = GltfAsset()
    idx = a.add_image(b"\x89PNG\r\n\x1a\nxxxx", "png", "brick", embed=False)
    assert a.images[idx]["uri"] == "brick.png"
    assert "bufferView" not in a.images[idx]
    assert a.sidecars == {"brick.png": b"\x89PNG\r\n\x1a\nxxxx"}


def test_a_textured_material_carries_base_color_texture_and_factor():
    """D18: glTF multiplies the two, matching OBJ's Kd plus map_Kd."""
    a = GltfAsset()
    img = a.add_image(b"\x89PNG\r\n\x1a\nx", "png", "brick", embed=True)
    tex = a.add_texture(img)
    mat = a.add_material("Brick", (0.5, 0.25, 0.125), texture=tex)
    pbr = a.materials[mat]["pbrMetallicRoughness"]
    assert pbr["baseColorTexture"] == {"index": tex}
    assert pbr["baseColorFactor"][:3] == [0.5, 0.25, 0.125]
    assert a.textures[tex] == {"source": img}


def test_json_omits_empty_image_and_texture_arrays():
    """glTF 2.0 forbids an empty array for these; an untextured export must
    produce exactly the document it produces today."""
    a = GltfAsset()
    a.add_mesh([(TRI_POS, None, TRI_IDX, None)])
    a.scene_roots.append(a.add_node(name="n", mesh=0))
    doc = json.loads(a.write_gltf("m.bin")[0])
    assert "images" not in doc
    assert "textures" not in doc
    assert "samplers" not in doc


def test_mismatched_uv_and_position_counts_are_refused():
    """A glTF primitive indexes POSITION and TEXCOORD_0 through ONE index
    buffer, so their accessor counts must be equal. Unequal counts produce a
    file that is invalid by spec but that no validator in this codebase runs.

    Without this guard the only defence lives in the single caller
    (gltf_export._definition_primitives, which appends to both pools inside
    one branch), and a reviewer had to hand-trace that control flow to prove
    the invariant. A future caller will not have that trace. Refusing here
    turns an unnoticed invalid export into an exception at the point of the
    mistake.
    """
    a = GltfAsset()
    with pytest.raises(ValueError, match="TEXCOORD_0"):
        a.add_mesh([(TRI_POS, [(0.0, 0.0), (1.0, 0.0)], TRI_IDX, None)])


def test_a_matching_uv_count_is_still_accepted():
    """The guard must not reject the legitimate case it sits next to."""
    a = GltfAsset()
    uvs = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
    assert a.add_mesh([(TRI_POS, uvs, TRI_IDX, None)]) == 0
