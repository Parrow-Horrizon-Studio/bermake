from __future__ import annotations

from bermake.commands.gltf_commands import ImportGltfCommand
from bermake.io.gltf_scene import GltfMesh, GltfNode, GltfSceneData
from bermake.model.model import Model

TRI = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
IDENT = (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)


def _scene():
    return GltfSceneData(
        nodes=(GltfNode(name="A", parent=-1, transform=IDENT, mesh_indices=(0,)),),
        meshes=(GltfMesh(positions=TRI, triangles=((0, 1, 2),), material_index=-1),),
        materials=(),
    )


def test_do_adds_one_wrapper_child_then_undo_removes_it():
    model = Model()
    target = model.active_context
    before = len(target.children)
    cmd = ImportGltfCommand(_scene(), target, root_name="scene")
    cmd.do(model)
    assert len(target.children) == before + 1
    assert cmd.summary.faces_imported == 1
    cmd.undo(model)
    assert len(target.children) == before          # fully removed


def test_redo_rebuilds():
    model = Model()
    target = model.active_context
    cmd = ImportGltfCommand(_scene(), target)
    cmd.do(model)
    cmd.undo(model)
    cmd.do(model)                                   # stack re-runs do() for redo
    assert len(target.children) == 1


def test_double_undo_is_noop():
    model = Model()
    target = model.active_context
    cmd = ImportGltfCommand(_scene(), target)
    cmd.do(model)
    cmd.undo(model)
    cmd.undo(model)                                 # guarded, no crash
    assert len(target.children) == 0
