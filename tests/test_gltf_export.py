from __future__ import annotations

import struct

import numpy as np
from bermake.io.gltf_export import export_gltf, model_to_gltf
from bermake.model.model import Model


def _painted_quad_model():
    """A model with one quad face at Bermake z=1 (up), painted red."""
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


def test_model_to_gltf_has_mesh_material_and_root():
    asset = model_to_gltf(_painted_quad_model())
    assert len(asset.meshes) == 1
    assert len(asset.scene_roots) == 1
    assert len(asset.materials) == 1
    assert asset.materials[0]["pbrMetallicRoughness"]["baseColorFactor"] == [1.0, 0.0, 0.0, 1.0]
    # a quad -> 2 triangles -> 6 indices in the (single-material) primitive
    prim = asset.meshes[0]["primitives"][0]
    idx_acc = asset.accessors[prim["indices"]]
    assert idx_acc["count"] == 6


def test_root_matrix_is_zup_to_yup_column_major():
    asset = model_to_gltf(_painted_quad_model())
    root = asset.nodes[asset.scene_roots[0]]
    m = np.array(root["matrix"], dtype=np.float64).reshape(4, 4, order="F")  # column-major
    # Z-up point (0,0,1) -> Y-up (0,1,0)
    assert np.allclose((m @ np.array([0.0, 0.0, 1.0, 1.0]))[:3], [0.0, 1.0, 0.0], atol=1e-6)


def test_export_glb_writes_magic(tmp_path):
    p = tmp_path / "out.glb"
    export_gltf(_painted_quad_model(), str(p))
    data = p.read_bytes()
    assert struct.unpack_from("<I", data, 0)[0] == 0x46546C67


def test_export_gltf_writes_sidecar_bin(tmp_path):
    p = tmp_path / "out.gltf"
    export_gltf(_painted_quad_model(), str(p))
    assert p.exists()
    assert (tmp_path / "out.bin").exists()


def test_shared_definition_exports_one_mesh():
    """Mesh-level instancing: a Definition instanced twice -> one glTF mesh."""
    model = Model()
    comp = model.new_definition("Widget", is_group=False)
    ids = [
        comp.mesh.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        comp.mesh.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        comp.mesh.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32)),
    ]
    comp.mesh.add_face_from_loop(ids)
    model.root.children.append(model.new_instance(comp))
    model.root.children.append(model.new_instance(comp, transform=np.eye(4, dtype=np.float64)))
    asset = model_to_gltf(model)
    assert len(asset.meshes) == 1


def test_child_instance_matrix_is_column_major():
    """A non-identity child-instance transform round-trips via an order='F' reshape."""
    model = Model()
    comp = model.new_definition("Widget", is_group=False)
    ids = [
        comp.mesh.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        comp.mesh.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        comp.mesh.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32)),
    ]
    comp.mesh.add_face_from_loop(ids)
    transform_mat = np.array([[1.0, 0.0, 0.0, 5.0],
                             [0.0, 1.0, 0.0, 2.0],
                             [0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    model.root.children.append(model.new_instance(comp, transform=transform_mat))
    asset = model_to_gltf(model)
    root_node = asset.nodes[asset.scene_roots[0]]
    child = asset.nodes[root_node["children"][0]]
    m = np.array(child["matrix"], dtype=np.float64).reshape(4, 4, order="F")
    assert np.allclose(m, transform_mat)


def test_export_uses_kernel_triangulation_for_concave_face():
    from bermake.io.gltf_export import _definition_primitives
    model = Model()
    mesh = model.root.mesh
    # A concave U-shape in the z=0 plane (a fan from vertex 0 would triangulate it differently).
    pts = [(0, 0, 0), (3, 0, 0), (3, 2, 0), (2, 2, 0),
           (2, 1, 0), (1, 1, 0), (1, 2, 0), (0, 2, 0)]
    ids = [mesh.add_vertex(np.array(p, dtype=np.float32)) for p in pts]
    fid = mesh.add_face_from_loop(ids)
    vpos = {v.id: tuple(round(float(c), 3) for c in v.position) for v in mesh.vertices_iter()}
    face = next(f for f in mesh.faces_iter() if f.id == fid)
    kernel_tris = {frozenset(vpos[int(v)] for v in tri) for tri in face.triangles}
    positions, _uvs, indices, _mat = _definition_primitives(model.root, lambda mid: None)[0]
    export_tris = set()
    for i in range(0, len(indices), 3):
        export_tris.add(frozenset(
            tuple(round(float(c), 3) for c in positions[indices[i + k]]) for k in range(3)))
    assert export_tris == kernel_tris                 # exporter uses the kernel triangulation
    assert len(export_tris) == len(pts) - 2           # n-2 triangles for a simple polygon
