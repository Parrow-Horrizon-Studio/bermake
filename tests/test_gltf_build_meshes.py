from __future__ import annotations

from bermake.io.gltf_import import _build_mesh_components, _ensure_gltf_materials
from bermake.io.gltf_scene import GltfMaterial, GltfMesh, GltfSceneData
from bermake.model.model import Model

TRI = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


def _scene(meshes, materials=()):
    return GltfSceneData(nodes=(), meshes=tuple(meshes), materials=tuple(materials))


def test_valid_mesh_builds_one_component_with_one_face():
    scene = _scene([GltfMesh(positions=TRI, triangles=((0, 1, 2),), material_index=-1)])
    model = Model()
    mats, _images_skipped = _ensure_gltf_materials(scene.materials, model)
    meshdefs, imported, skipped, without_uvs, built = _build_mesh_components(scene, model, mats)
    assert built == 1
    assert imported == 1 and skipped == 0 and without_uvs == 0
    assert meshdefs[0] is not None
    assert not meshdefs[0].is_group  # a Component
    assert len(list(meshdefs[0].mesh.faces_iter())) == 1


def test_degenerate_triangle_is_skipped_and_component_dropped():
    scene = _scene([GltfMesh(positions=TRI, triangles=((0, 0, 1),), material_index=-1)])
    model = Model()
    mats, _images_skipped = _ensure_gltf_materials(scene.materials, model)
    meshdefs, imported, skipped, without_uvs, built = _build_mesh_components(scene, model, mats)
    assert imported == 0 and skipped == 1
    assert meshdefs[0] is None and built == 0


def test_material_is_deduped_and_applied():
    mat = GltfMaterial(name="Red", color=(1.0, 0.0, 0.0))
    scene = _scene(
        [GltfMesh(positions=TRI, triangles=((0, 1, 2),), material_index=0)],
        materials=[mat],
    )
    model = Model()
    mats, _images_skipped = _ensure_gltf_materials(scene.materials, model)
    assert mats[0] is not None
    meshdefs, *_ = _build_mesh_components(scene, model, mats)
    fid = next(iter(meshdefs[0].mesh.faces_iter())).id
    assert meshdefs[0].mesh.face_material(fid) == mats[0]


def test_default_material_maps_to_none():
    mat = GltfMaterial(name="DefaultMaterial", color=(0.8, 0.8, 0.8))
    model = Model()
    assert _ensure_gltf_materials((mat,), model) == ([None], 0)
