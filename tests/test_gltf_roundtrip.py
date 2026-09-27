from __future__ import annotations

from pathlib import Path

import numpy as np
from bermake.io.gltf_export import export_gltf
from bermake.io.gltf_import import build_gltf_into_model, read_gltf_scene, read_gltf_texture_bytes
from bermake.model.model import Model

DATA = Path(__file__).parent / "data" / "gltf"


def _decoder(data):
    """The Qt-free decoder shape bermake/io expects, for a 4x4 opaque PNG."""
    return ("png", 4, 4, False)


def _up_quad_model():
    model = Model()
    mesh = model.root.mesh
    ids = [
        mesh.add_vertex(np.array([0.0, 0.0, 1.0], dtype=np.float32)),
        mesh.add_vertex(np.array([1.0, 0.0, 1.0], dtype=np.float32)),
        mesh.add_vertex(np.array([1.0, 1.0, 1.0], dtype=np.float32)),
        mesh.add_vertex(np.array([0.0, 1.0, 1.0], dtype=np.float32)),
    ]
    fid = mesh.add_face_from_loop(ids)
    red = model.materials.add_custom("Red", (1.0, 0.0, 0.0))
    mesh.set_face_material(fid, red.id)
    return model


def test_glb_roundtrip_preserves_geometry_orientation_and_color(tmp_path):
    export_gltf(_up_quad_model(), str(tmp_path / "rt.glb"))
    scene = read_gltf_scene(str(tmp_path / "rt.glb"))
    assert len(scene.meshes) >= 1
    assert any(m.color == (1.0, 0.0, 0.0) or np.allclose(m.color, (1.0, 0.0, 0.0), atol=1e-4)
               for m in scene.materials)

    # Rebuild into a fresh model; the up face must land back on Bermake z ~ 1.
    model = Model()
    build_gltf_into_model(scene, model, model.active_context)
    zs = [w @ np.append(v.position, 1.0)
          for d, w in model.traverse()
          for v in d.mesh.vertices_iter()]
    assert zs, "no vertices imported"
    assert max(pt[2] for pt in zs) > 0.9        # up preserved (Z-up)


def test_gltf_roundtrip_writes_and_reads(tmp_path):
    export_gltf(_up_quad_model(), str(tmp_path / "rt.gltf"))
    scene = read_gltf_scene(str(tmp_path / "rt.gltf"))
    assert len(scene.meshes) >= 1


def test_a_textured_glb_survives_import_then_export_then_import(tmp_path):
    """The one test that fails if only one side of the V flip landed.

    Import applies v -> 1 - v and export applies it again, so the two must
    cancel. Comparing the re-imported UVs to the ORIGINAL file's, not to the
    in-model ones, is what makes that visible.
    """
    original = read_gltf_scene(DATA / "textured_box.glb")
    model = Model()
    build_gltf_into_model(
        original,
        model,
        model.root,
        texture_bytes=read_gltf_texture_bytes(DATA / "textured_box.glb", original),
        decoder=_decoder,
    )
    out = tmp_path / "rt.glb"
    export_gltf(model, out)
    again = read_gltf_scene(out)

    before = sorted((round(u, 5), round(v, 5)) for u, v in original.meshes[0].uvs)
    after = sorted((round(u, 5), round(v, 5)) for u, v in again.meshes[0].uvs)
    assert before == after, "a single missing or doubled v flip shows up here"
    assert again.images and again.images[0].data == original.images[0].data
