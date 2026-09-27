"""Pure glTF 2.0 buffer/JSON assembly (M6c export codec).

No Model, no filesystem. Assemble a GltfAsset (nodes/meshes/materials, with
accessors packed into one binary buffer) and serialize to .glb bytes or
(.gltf json, .bin bytes). Positions are VEC3/FLOAT (with min/max); indices are
SCALAR/UNSIGNED_INT. Node matrices are glTF column-major 16-float arrays
(the caller supplies column-major order).
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field

_FLOAT = 5126
_UINT = 5125
_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963

_GLB_MAGIC = 0x46546C67
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942

_IMAGE_MIME = {"png": "image/png", "jpeg": "image/jpeg", "jpg": "image/jpeg"}


def _pad4(n: int) -> int:
    return (4 - (n % 4)) % 4


@dataclass
class GltfAsset:
    _buffer: bytearray = field(default_factory=bytearray)
    accessors: list = field(default_factory=list)
    buffer_views: list = field(default_factory=list)
    materials: list = field(default_factory=list)
    meshes: list = field(default_factory=list)
    nodes: list = field(default_factory=list)
    scene_roots: list = field(default_factory=list)
    images: list = field(default_factory=list)
    textures: list = field(default_factory=list)
    sidecars: dict = field(default_factory=dict)  # filename -> bytes, .gltf only

    def add_material(self, name, color, texture=None) -> int:
        pbr = {
            "baseColorFactor": [float(color[0]), float(color[1]), float(color[2]), 1.0],
            "metallicFactor": 0.0,
            "roughnessFactor": 1.0,
        }
        if texture is not None:
            pbr["baseColorTexture"] = {"index": texture}
        self.materials.append({"name": name, "pbrMetallicRoughness": pbr})
        return len(self.materials) - 1

    def _add_buffer_view(self, data: bytes, target: int | None) -> int:
        self._buffer.extend(b"\x00" * _pad4(len(self._buffer)))
        offset = len(self._buffer)
        self._buffer.extend(data)
        bv = {
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(data),
        }
        if target is not None:
            bv["target"] = target
        self.buffer_views.append(bv)
        return len(self.buffer_views) - 1

    def _add_position_accessor(self, positions) -> int:
        data = bytearray()
        for x, y, z in positions:
            data += struct.pack("<3f", x, y, z)
        bv = self._add_buffer_view(bytes(data), _ARRAY_BUFFER)
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        zs = [p[2] for p in positions]
        self.accessors.append(
            {
                "bufferView": bv,
                "componentType": _FLOAT,
                "count": len(positions),
                "type": "VEC3",
                "min": [min(xs), min(ys), min(zs)],
                "max": [max(xs), max(ys), max(zs)],
            }
        )
        return len(self.accessors) - 1

    def _add_index_accessor(self, indices) -> int:
        data = struct.pack(f"<{len(indices)}I", *indices)
        bv = self._add_buffer_view(data, _ELEMENT_ARRAY_BUFFER)
        self.accessors.append(
            {
                "bufferView": bv,
                "componentType": _UINT,
                "count": len(indices),
                "type": "SCALAR",
            }
        )
        return len(self.accessors) - 1

    def _add_uv_accessor(self, uvs) -> int:
        """A VEC2/FLOAT accessor. No min/max: glTF requires those for POSITION
        only, and a bounding box over texture coordinates means nothing."""
        data = bytearray()
        for u, v in uvs:
            data += struct.pack("<2f", u, v)
        bv = self._add_buffer_view(bytes(data), _ARRAY_BUFFER)
        self.accessors.append(
            {"bufferView": bv, "componentType": _FLOAT, "count": len(uvs), "type": "VEC2"}
        )
        return len(self.accessors) - 1

    def add_mesh(self, primitives) -> int:
        """Each primitive is (positions, uvs|None, indices, material|None).

        A primitive with UVs must supply exactly one per position: glTF indexes
        POSITION and TEXCOORD_0 through the SAME index buffer, so their accessor
        counts have to match. Unequal counts assemble happily here and produce a
        file that is invalid by spec but that nothing in this codebase would
        reject, so the mistake would only surface in whatever tool opened it.
        Refusing at the assembly layer puts the error where the caller made it.
        """
        prims = []
        for positions, uvs, indices, mat in primitives:
            if uvs is not None and len(uvs) != len(positions):
                raise ValueError(
                    f"TEXCOORD_0 has {len(uvs)} entries but POSITION has {len(positions)}; "
                    "a glTF primitive indexes both through one index buffer"
                )
            p = {
                "attributes": {"POSITION": self._add_position_accessor(positions)},
                "indices": self._add_index_accessor(indices),
            }
            if uvs is not None:
                p["attributes"]["TEXCOORD_0"] = self._add_uv_accessor(uvs)
            if mat is not None:
                p["material"] = mat
            prims.append(p)
        self.meshes.append({"primitives": prims})
        return len(self.meshes) - 1

    def add_image(self, data: bytes, image_format: str, name: str, embed: bool) -> int:
        """Register an image. `embed` puts the bytes in the buffer (GLB); the
        alternative records a sibling filename in `sidecars` for the caller to
        write beside the .gltf, matching how geometry already splits between
        the two containers."""
        fmt = str(image_format).lower()
        mime = _IMAGE_MIME.get(fmt, "image/png")
        if embed:
            bv = self._add_buffer_view(bytes(data), None)
            self.images.append({"name": name, "bufferView": bv, "mimeType": mime})
        else:
            filename = f"{name}.{'jpg' if mime == 'image/jpeg' else 'png'}"
            self.sidecars[filename] = bytes(data)
            self.images.append({"name": name, "uri": filename})
        return len(self.images) - 1

    def add_texture(self, image_index: int) -> int:
        """No sampler: glTF's defaults are REPEAT wrapping and automatic
        filtering, which is exactly what Bermake's projected UVs need when they
        run outside 0..1."""
        self.textures.append({"source": int(image_index)})
        return len(self.textures) - 1

    def add_node(self, name=None, matrix=None, mesh=None, children=None) -> int:
        node: dict = {}
        if name:
            node["name"] = name
        if matrix is not None:
            node["matrix"] = [float(v) for v in matrix]
        if mesh is not None:
            node["mesh"] = mesh
        if children:
            node["children"] = list(children)
        self.nodes.append(node)
        return len(self.nodes) - 1

    def _json(self, buffer_obj) -> dict:
        doc = {
            "asset": {"version": "2.0", "generator": "Bermake"},
            "scene": 0,
            "scenes": [{"nodes": list(self.scene_roots)}],
            "nodes": self.nodes,
            "meshes": self.meshes,
            "accessors": self.accessors,
            "bufferViews": self.buffer_views,
            "buffers": [buffer_obj],
        }
        if self.materials:
            doc["materials"] = self.materials
        if self.images:
            doc["images"] = self.images
        if self.textures:
            doc["textures"] = self.textures
        return doc

    def write_glb(self) -> bytes:
        bin_blob = bytes(self._buffer) + b"\x00" * _pad4(len(self._buffer))
        doc = self._json({"byteLength": len(bin_blob)})
        json_bytes = json.dumps(doc, separators=(",", ":")).encode("utf-8")
        json_bytes += b" " * _pad4(len(json_bytes))
        total = 12 + 8 + len(json_bytes) + 8 + len(bin_blob)
        out = bytearray()
        out += struct.pack("<III", _GLB_MAGIC, 2, total)
        out += struct.pack("<II", len(json_bytes), _CHUNK_JSON) + json_bytes
        out += struct.pack("<II", len(bin_blob), _CHUNK_BIN) + bin_blob
        return bytes(out)

    def write_gltf(self, bin_name: str):
        bin_blob = bytes(self._buffer)
        doc = self._json({"byteLength": len(bin_blob), "uri": bin_name})
        return json.dumps(doc, indent=2), bin_blob, dict(self.sidecars)
