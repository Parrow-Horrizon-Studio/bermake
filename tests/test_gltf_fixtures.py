"""The fixtures must be able to see the bugs they exist to catch."""

import json
import struct
from pathlib import Path

DATA = Path(__file__).parent / "data" / "gltf"


def _glb_json(path):
    b = path.read_bytes()
    assert b[:4] == b"glTF", f"{path.name} is not a GLB"
    length, kind = struct.unpack_from("<II", b, 12)
    assert kind == 0x4E4F534A, "first chunk is not JSON"
    return json.loads(b[20 : 20 + length])


def test_uvgrid_is_asymmetric_in_both_axes():
    """Four distinct corner colours, so no flip or mirror is invisible."""
    import sys

    sys.path.insert(0, str(DATA))
    from make_fixtures import uvgrid_png
    from bermake.viewport.texture_cache import decode_image

    img = decode_image(uvgrid_png())
    assert img is not None and (img.width, img.height) == (4, 4)
    px = img.pixels  # top-down rows, RGBA
    corners = {
        "top_left": tuple(px[0, 0][:3]),
        "top_right": tuple(px[0, 3][:3]),
        "bottom_left": tuple(px[3, 0][:3]),
        "bottom_right": tuple(px[3, 3][:3]),
    }
    assert corners == {
        "top_left": (255, 0, 0),
        "top_right": (0, 255, 0),
        "bottom_left": (0, 0, 255),
        "bottom_right": (255, 255, 255),
    }
    assert len(set(corners.values())) == 4, "a repeated corner colour hides a flip"


def test_textured_box_glb_declares_uvs_and_an_embedded_image():
    d = _glb_json(DATA / "textured_box.glb")
    attrs = d["meshes"][0]["primitives"][0]["attributes"]
    assert "TEXCOORD_0" in attrs, "the fixture must carry UVs or it proves nothing"
    assert d["images"][0].get("bufferView") is not None, "image must be embedded"
    assert d["images"][0]["mimeType"] == "image/png"
    assert d["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"] == 0


def test_textured_box_gltf_references_a_sibling_image():
    d = json.loads((DATA / "textured_box.gltf").read_text(encoding="utf-8"))
    assert d["images"][0]["uri"] == "uvgrid.png"
    assert "bufferView" not in d["images"][0]
    assert (DATA / "uvgrid.png").is_file()
    assert (DATA / "textured_box.bin").is_file()


def test_avocado_draco_is_compressed_textured_and_uv_bearing():
    """The UV-bearing replacement for the Draco CI gate.

    draco_box.glb declares NORMAL and POSITION only, so it can decode
    perfectly while UV extraction is broken. This one cannot.
    """
    d = _glb_json(DATA / "avocado_draco.glb")
    assert "KHR_draco_mesh_compression" in d.get("extensionsRequired", d["extensionsUsed"])
    prim = d["meshes"][0]["primitives"][0]
    draco_attrs = prim["extensions"]["KHR_draco_mesh_compression"]["attributes"]
    assert "TEXCOORD_0" in draco_attrs, "UVs must be inside the Draco payload"
    assert d["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"] == 0
    assert d["images"][0].get("bufferView") is not None
    assert len(d["images"]) == 1, "the three Khronos PBR maps must have been dropped"


def test_the_old_fixtures_still_have_no_uvs():
    """Guards the claim above: these two are why new fixtures were needed."""
    for name in ("plain_box.glb", "draco_box.glb"):
        d = _glb_json(DATA / name)
        for mesh in d["meshes"]:
            for prim in mesh["primitives"]:
                inner = prim.get("extensions", {}).get("KHR_draco_mesh_compression", {})
                declared = set(prim["attributes"]) | set(inner.get("attributes", {}))
                assert "TEXCOORD_0" not in declared, f"{name} gained UVs; update this test"
