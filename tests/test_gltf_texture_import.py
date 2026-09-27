"""Resolving a glTF's external image, with the same containment rule as OBJ."""

from pathlib import Path

import pytest

from bermake.io.gltf_scene import GltfImage, GltfMaterial, GltfMesh, GltfSceneData
from bermake.io.gltf_import import read_gltf_texture_bytes

DATA = Path(__file__).parent / "data" / "gltf"


def _scene(materials, images=()):
    return GltfSceneData(nodes=(), meshes=(), materials=tuple(materials), images=tuple(images))


def test_embedded_image_is_taken_from_the_scene():
    img = GltfImage(name="tex", data=b"\x89PNG\r\n\x1a\nfake", format_hint="png")
    scene = _scene([GltfMaterial("M", (1.0, 1.0, 1.0), texture_index=0, texture_uri="")], [img])
    assert read_gltf_texture_bytes(DATA / "textured_box.glb", scene) == {0: img.data}


def test_raw_texel_image_is_skipped_not_invented():
    """D17: an image with no encoded bytes has nothing Texture can store."""
    img = GltfImage(name="raw", data=b"", format_hint="rgba8888")
    scene = _scene([GltfMaterial("M", (1.0, 1.0, 1.0), texture_index=0, texture_uri="")], [img])
    assert read_gltf_texture_bytes(DATA / "textured_box.glb", scene) == {}


def test_external_image_is_read_from_beside_the_document():
    scene = _scene([GltfMaterial("M", (1.0, 1.0, 1.0), texture_index=-1, texture_uri="uvgrid.png")])
    got = read_gltf_texture_bytes(DATA / "textured_box.gltf", scene)
    assert got[0] == (DATA / "uvgrid.png").read_bytes()


@pytest.mark.parametrize(
    "uri",
    [
        "../uvgrid.png",
        "../../uvgrid.png",
        "C:/Windows/win.ini",
        "/etc/passwd",
        "sub/../../uvgrid.png",
    ],
)
def test_an_image_outside_the_document_directory_is_refused(uri):
    """A .gltf someone emailed you must not be able to read an arbitrary file
    off disk and embed it into the model the user then saves and shares."""
    scene = _scene([GltfMaterial("M", (1.0, 1.0, 1.0), texture_index=-1, texture_uri=uri)])
    assert read_gltf_texture_bytes(DATA / "textured_box.gltf", scene) == {}


def test_a_missing_image_is_best_effort_not_fatal():
    scene = _scene([GltfMaterial("M", (1.0, 1.0, 1.0), texture_index=-1, texture_uri="nope.png")])
    assert read_gltf_texture_bytes(DATA / "textured_box.gltf", scene) == {}
