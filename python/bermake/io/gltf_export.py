"""glTF export: Model -> GltfAsset mapping + atomic filesystem write (M6c).

Mirrors the import mapping: shared Definitions -> one shared glTF mesh
(mesh-level instancing), each Instance -> a glTF node with its transform,
n-gon faces triangulated (via the kernel's earcut, concave-safe) and grouped
by material into primitives, and a Z-up -> Y-up conversion baked at the
export root. This is the only glTF export module that knows about
Model/Scene.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from bermake.io.gltf_codec import GltfAsset
from bermake.io.image_paths import sanitize_filename_stem
from bermake.scene.scene import Side


def _sanitize_image_name(name: str) -> str:
    """A filename- and URI-safe stem for a texture name (image_paths.
    sanitize_filename_stem): the name becomes both a `Path.with_name`
    filename and a glTF `image.uri`, and is untrusted -- on import a texture
    is minted with the name of the source document's own material, arbitrary
    text from a file someone else authored (see gltf_import.
    _find_or_decode_texture). This alone does NOT guarantee uniqueness -- two
    textures can share a name, or sanitize to the same one -- so callers that
    write a sidecar file must still disambiguate (see `gltf_material_for`)."""
    return sanitize_filename_stem(str(name), "image")


def _zup_to_yup() -> np.ndarray:
    """Rx(-90°): Bermake Z-up -> glTF Y-up. (x, y, z) -> (x, z, -y)."""
    return np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _definition_primitives(defn, gltf_material_for, resolver=None, materials=None):
    """Triangulate the definition's faces (kernel earcut, concave-safe), grouped
    by material into (positions, uvs|None, indices, gltf_material_index|None).

    Unlike OBJ, each primitive gets its OWN vertex pool keyed on (vertex id,
    uv): a glTF primitive indexes POSITION and TEXCOORD_0 through one index
    buffer whose accessors must have equal counts, so a UV seam duplicates the
    position. With no resolver the key degenerates to the vertex id alone and
    the pools are exactly what they were before UVs existed.
    """
    mesh = defn.mesh
    if not list(mesh.vertices_iter()):
        return []
    pos_by_id = {
        v.id: (float(v.position[0]), float(v.position[1]), float(v.position[2]))
        for v in mesh.vertices_iter()
    }
    faces = list(mesh.faces_iter())

    # Pass 1: which material groups need UVs at all. Stage 2 gated this per
    # face; a primitive is a material group and cannot carry a partial array,
    # so the gate is per group (D16). is_textured is uniform across a group
    # by construction; only has_stored_uvs varies.
    group_of: dict = {}
    needs_uvs: dict = {}
    for f in faces:
        mid = mesh.face_material(f.id)
        gmat = gltf_material_for(mid)
        group_of[f.id] = gmat
        if resolver is None:
            continue
        mat = materials.get(mid) if materials is not None else None
        if mesh.face_uvs(f.id, Side.FRONT) is not None or (
            mat is not None and mat.texture_id is not None
        ):
            needs_uvs[gmat] = True

    # Pass 2: build one pool per group.
    pools: dict = defaultdict(lambda: ([], [], [], {}))  # positions, uvs, indices, key->slot
    for f in faces:
        gmat = group_of[f.id]
        positions, uvs, indices, slot_of = pools[gmat]
        uv_for_vid: dict = {}
        if needs_uvs.get(gmat):
            try:
                resolved = np.asarray(
                    resolver(mesh, materials, f.id, Side.FRONT), dtype=np.float64
                ).reshape(-1, 2)
                for lv, (u, v) in zip(f.loop_vertex_ids, resolved, strict=True):
                    # A pinched face repeats a vertex in its loop; first
                    # corner wins, matching what the renderer samples for it.
                    # Stage 1's set_face_uvs refuses to store for such a face
                    # at all, so this only arises for a projected base.
                    #
                    # glTF's TEXCOORD_0 origin is the image's UPPER left and
                    # Bermake's v = 0 is its BOTTOM, so the flip belongs here.
                    # Export writes through Bermake's own codec rather than
                    # Assimp, unlike import: see gltf_import._corner_uvs,
                    # whose docstring explains why import does NOT flip. D14.
                    uv_for_vid.setdefault(int(lv), (float(u), 1.0 - float(v)))
            except (KeyError, ValueError):
                # Matches obj_io._uv_indices_for's contract: a face whose UVs
                # cannot be resolved (or whose resolved array's length does
                # not match its loop, the strict=True zip above) degrades to
                # no UV contribution for THIS face, rather than raising out
                # of an export that has already written partial output
                # (.gltf + .bin, and any earlier sidecar images) to disk.
                # uv_for_vid stays empty, so every corner below falls through
                # to the (0.0, 0.0) placeholder, exactly like an untextured
                # face with nothing stored -- the positions and uvs pools
                # stay in lockstep because pass 2 unconditionally appends one
                # uv entry per new position slot whenever needs_uvs[gmat] is
                # true, whether or not uv_for_vid has anything for that
                # vertex.
                uv_for_vid = {}
        for tri in f.triangles:  # kernel earcut triangulation (concave-safe)
            for vid in tri:
                vid = int(vid)
                uv = uv_for_vid.get(vid)
                key = (vid, uv)
                slot = slot_of.get(key)
                if slot is None:
                    slot = len(positions)
                    slot_of[key] = slot
                    positions.append(pos_by_id[vid])
                    if needs_uvs.get(gmat):
                        uvs.append(uv if uv is not None else (0.0, 0.0))
                indices.append(slot)

    return [
        (positions, uvs if needs_uvs.get(gmat) else None, indices, gmat)
        for gmat, (positions, uvs, indices, _) in pools.items()
    ]


def model_to_gltf(model, resolver=None, embed_images=True) -> GltfAsset:
    """`resolver` is `uv_resolve.resolve_face_uvs`-shaped; None keeps today's
    UV-free export. `embed_images` selects the container a textured material's
    image lands in: True embeds it in the buffer (GLB), False records it as a
    sidecar for the caller to write beside a .gltf."""
    asset = GltfAsset()
    default_id = model.materials.DEFAULT_ID
    mat_index: dict = {}
    mesh_index: dict = {}
    texture_index: dict = {}  # Bermake texture id -> gltf texture index

    def gltf_material_for(mid):
        # Called only for a material some face actually carries (Pass 1 of
        # _definition_primitives), so an untextured or unused material's
        # texture is never touched: the "used by an exported face" filter
        # falls out of this for free.
        if mid == default_id:
            return None
        if mid not in mat_index:
            m = model.materials.get(mid)
            texture_idx = None
            if m.texture_id is not None:
                texture_idx = texture_index.get(m.texture_id)
                if texture_idx is None:
                    tex = model.textures.get(m.texture_id)
                    if tex is not None:
                        # Memoised by Bermake texture id, so two materials that
                        # share one texture embed/write it once. Two DIFFERENT
                        # textures can still sanitize to the same name (both
                        # imported as "diffuse.png", say); appending the
                        # texture's own id -- unique by construction in
                        # TextureLibrary -- keeps the sidecar filename unique
                        # too, so a later image never silently overwrites an
                        # earlier one under the same key (the OBJ exporter's
                        # #119, for the same family of bug over material
                        # names rather than texture names).
                        image_name = f"{_sanitize_image_name(tex.name)}_{tex.id}"
                        image_idx = asset.add_image(
                            tex.data, tex.image_format, image_name, embed=embed_images
                        )
                        texture_idx = asset.add_texture(image_idx)
                        texture_index[m.texture_id] = texture_idx
            mat_index[mid] = asset.add_material(m.name, m.base_color, texture=texture_idx)
        return mat_index[mid]

    def mesh_for(defn):
        if defn.id in mesh_index:
            return mesh_index[defn.id]
        prims = _definition_primitives(
            defn, gltf_material_for, resolver=resolver, materials=model.materials
        )
        if not prims:
            return None
        idx = asset.add_mesh(prims)
        mesh_index[defn.id] = idx
        return idx

    def emit(inst):
        defn = inst.definition
        m = mesh_for(defn)
        children = [emit(child) for child in defn.children]
        matrix = np.asarray(inst.transform, dtype=np.float64).flatten(order="F")
        return asset.add_node(name=defn.name, matrix=matrix, mesh=m, children=children or None)

    root_mesh = mesh_for(model.root)
    root_children = [emit(inst) for inst in model.root.children]
    root_node = asset.add_node(
        name="Bermake",
        matrix=_zup_to_yup().flatten(order="F"),
        mesh=root_mesh,
        children=root_children or None,
    )
    asset.scene_roots.append(root_node)
    return asset


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def export_gltf(model, path) -> None:
    """Write the whole model to `path`. `.gltf` -> JSON + a sibling `.bin` +
    one sibling image per textured material; any other suffix (incl. `.glb`)
    -> a single self-contained binary GLB. Every file atomic.

    `resolve_face_uvs` is imported here, lazily, for the reason `export_obj`
    documents: `bermake/io` must not import `bermake.viewport` at module scope
    because that package reaches Qt through the GPU-side texture loader.
    `uv_resolve.py` is itself Qt-free, so importing it inside this function
    keeps `bermake.io.gltf_export` importable with no Qt loaded while still
    giving every caller real UVs with no extra argument.
    """
    from bermake.viewport.uv_resolve import resolve_face_uvs

    path = Path(path)
    is_gltf = path.suffix.lower() == ".gltf"
    asset = model_to_gltf(model, resolver=resolve_face_uvs, embed_images=not is_gltf)
    if is_gltf:
        bin_name = path.stem + ".bin"
        json_text, bin_bytes, sidecars = asset.write_gltf(bin_name)
        _atomic_write_bytes(path, json_text.encode("utf-8"))
        _atomic_write_bytes(path.with_name(bin_name), bin_bytes)
        for filename, data in sidecars.items():
            _atomic_write_bytes(path.with_name(filename), data)
    else:
        _atomic_write_bytes(path, asset.write_glb())
