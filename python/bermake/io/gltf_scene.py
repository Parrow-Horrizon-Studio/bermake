"""Neutral glTF import IR (M6c).

Pure dataclasses mirroring the C++ bridge structs 1:1 — no Model, no _core, no
Assimp. This lets the import-mapping layer be unit-tested with hand-built
fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GltfImage:
    name: str
    data: bytes  # the image file's ENCODED bytes; empty means raw texels (D17)
    format_hint: str


@dataclass(frozen=True)
class GltfMaterial:
    name: str
    color: tuple[float, float, float]  # RGB; alpha dropped
    texture_index: int = -1  # index into GltfSceneData.images, -1 = none
    texture_uri: str = ""  # an external image filename, unresolved


@dataclass(frozen=True)
class GltfMesh:
    positions: tuple[tuple[float, float, float], ...]
    triangles: tuple[tuple[int, int, int], ...]
    material_index: int  # -1 = none
    # Empty, or one entry per position. ALREADY in Bermake's convention (v = 0
    # is the image bottom): Assimp's glTF2 importer applies 1 - v before the
    # bridge sees a coordinate, so import performs no flip of its own. Export
    # is asymmetric and does flip, because it writes through Bermake's own
    # codec rather than Assimp. See D14 and test_assimp_already_flips_v_CI_GATE.
    uvs: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class GltfNode:
    name: str
    parent: int  # -1 = root
    transform: tuple[float, ...]  # 16 floats, row-major
    mesh_indices: tuple[int, ...]


@dataclass(frozen=True)
class GltfSceneData:
    nodes: tuple[GltfNode, ...]
    meshes: tuple[GltfMesh, ...]
    materials: tuple[GltfMaterial, ...]
    images: tuple[GltfImage, ...] = ()
