from __future__ import annotations

import numpy as np
from bermake.io.gltf_import import build_gltf_into_model
from bermake.io.gltf_scene import GltfMesh, GltfNode, GltfSceneData
from bermake.model.model import Model

TRI = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
UP = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))  # vertex 2 is at glTF +Y
IDENT = (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)


def _one_mesh(nodes, mesh=None):
    mesh = mesh or GltfMesh(positions=TRI, triangles=((0, 1, 2),), material_index=-1)
    return GltfSceneData(nodes=tuple(nodes), meshes=(mesh,), materials=())


def test_shared_mesh_makes_one_component_two_instances():
    # two root leaf nodes both referencing mesh 0
    scene = _one_mesh([
        GltfNode(name="A", parent=-1, transform=IDENT, mesh_indices=(0,)),
        GltfNode(name="B", parent=-1, transform=IDENT, mesh_indices=(0,)),
    ])
    model = Model()
    result = build_gltf_into_model(scene, model, model.active_context, root_name="scene")
    wrapper = result.root_instance.definition
    kids = wrapper.children
    assert len(kids) == 2
    # collapsed leaves reference the SAME Component definition (instancing)
    assert kids[0].definition is kids[1].definition
    assert not kids[0].definition.is_group


def test_leaf_collapses_group_when_node_has_children():
    # node A has a mesh AND a child B -> A must be a group, not collapsed
    scene = _one_mesh([
        GltfNode(name="A", parent=-1, transform=IDENT, mesh_indices=(0,)),
        GltfNode(name="B", parent=0, transform=IDENT, mesh_indices=()),
    ])
    model = Model()
    result = build_gltf_into_model(scene, model, model.active_context)
    a_inst = result.root_instance.definition.children[0]
    assert a_inst.definition.is_group                     # A is a group
    # A's group holds: an instance of the mesh Component + the child node B
    assert len(a_inst.definition.children) == 2


def test_axis_yup_to_zup_puts_up_vertex_on_z():
    scene = _one_mesh(
        [GltfNode(name="A", parent=-1, transform=IDENT, mesh_indices=(0,))],
        mesh=GltfMesh(positions=UP, triangles=((0, 1, 2),), material_index=-1),
    )
    model = Model()
    build_gltf_into_model(scene, model, model.active_context)
    # find the mesh Component + its world transform via traverse
    world_of = {id(d): w for d, w in model.traverse()}
    meshdef = next(d for d, _ in model.traverse()
                   if not d.is_group and len(list(d.mesh.vertices_iter())) == 3)
    w = world_of[id(meshdef)]
    up_local = np.array([0.0, 1.0, 0.0, 1.0])          # glTF +Y
    world = w @ up_local
    assert np.allclose(world[:3], [0.0, 0.0, 1.0], atol=1e-6)  # -> Bermake +Z


def test_summary_counts():
    scene = _one_mesh([
        GltfNode(name="A", parent=-1, transform=IDENT, mesh_indices=(0,)),
        GltfNode(name="B", parent=-1, transform=IDENT, mesh_indices=(0,)),
    ])
    model = Model()
    result = build_gltf_into_model(scene, model, model.active_context)
    assert result.summary.nodes == 2
    assert result.summary.meshes == 1
    # faces_imported counts faces built into Component meshes (per distinct mesh):
    # both nodes share the SAME mesh (one triangle), so only 1 face is built,
    # not 2 -- this is a build/instancing count, not a per-instance count.
    assert result.summary.faces_imported == 1
