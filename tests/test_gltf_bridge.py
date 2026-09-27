"""_core.import_gltf bridge smoke tests (needs the compiled kernel)."""
from __future__ import annotations

from pathlib import Path

import bermake._core as core
import pytest

DATA = Path(__file__).parent / "data" / "gltf"


def test_import_plain_box_exposes_full_struct():
    s = core.import_gltf(str(DATA / "plain_box.glb"))
    assert len(s.meshes) >= 1
    mesh = s.meshes[0]
    assert len(mesh.positions) > 0
    assert len(mesh.positions[0]) == 3          # (x, y, z)
    assert len(mesh.triangles) > 0
    assert len(mesh.triangles[0]) == 3          # index triple
    assert isinstance(mesh.material_index, int)
    assert len(s.nodes) >= 1
    node = s.nodes[0]
    assert node.parent == -1
    assert len(node.transform) == 16
    # isinstance(list(x), list) is always true regardless of x's contents —
    # assert the element type instead, which is the thing actually worth
    # pinning (mesh_indices must be a sequence of ints into s.meshes).
    mesh_indices = list(node.mesh_indices)
    assert all(isinstance(i, int) for i in mesh_indices)
    assert len(s.materials) >= 1
    assert len(s.materials[0].base_color) == 4


def test_import_missing_file_raises():
    with pytest.raises(RuntimeError):
        core.import_gltf(str(DATA / "nope.glb"))
