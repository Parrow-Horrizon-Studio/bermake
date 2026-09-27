from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from bermake.io.errors import BermakeIOError
from bermake.io.gltf_scene import GltfMaterial, GltfMesh, GltfNode, GltfSceneData

DATA = Path(__file__).parent / "data" / "gltf"


def test_ir_dataclasses_are_frozen():
    m = GltfMaterial(name="Red", color=(1.0, 0.0, 0.0))
    with pytest.raises(FrozenInstanceError):
        m.name = "Blue"  # frozen


def test_read_gltf_scene_populates_ir():
    from bermake.io.gltf_import import read_gltf_scene

    scene = read_gltf_scene(str(DATA / "plain_box.glb"))
    assert isinstance(scene, GltfSceneData)
    assert len(scene.meshes) >= 1
    assert isinstance(scene.meshes[0], GltfMesh)
    assert len(scene.meshes[0].positions[0]) == 3
    assert isinstance(scene.nodes[0], GltfNode)
    assert scene.nodes[0].parent == -1
    assert len(scene.nodes[0].transform) == 16


def test_read_missing_file_raises_bermake_error():
    from bermake.io.gltf_import import read_gltf_scene

    with pytest.raises((BermakeIOError, OSError)):
        read_gltf_scene(str(DATA / "does_not_exist.glb"))
