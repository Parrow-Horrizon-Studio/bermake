"""OBJ filesystem + model mapping (M6b).

export_obj / read_obj_document touch the filesystem; model_to_objdoc and
build_obj_into_model map between the model and the pure ObjDocument IR. This is
the only OBJ module that knows about Model/Scene.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bermake.io.image_paths import read_sibling_image_bytes
from bermake.io.obj_codec import (
    ObjDocument,
    ObjFace,
    ObjObject,
    parse_obj,
    sanitize_material_name,
    write_obj,
)
from bermake.scene.scene import Side


def _unique_name(base: str, used: set[str]) -> str:
    name = base if base else "object"
    candidate = name
    n = 1
    while candidate in used:
        candidate = f"{name}.{n:03d}"
        n += 1
    used.add(candidate)
    return candidate


def _painted_material_ids(model) -> list[int]:
    """Material ids at least one exported face carries, in first-painted order.

    Only these reach the `.mtl`, so only these should compete for a name: a
    library material nobody painted with must not take `Brick_Red` and push
    the material actually being exported to `Brick_Red.001`. The seeded
    palette makes that the common case, not a corner one.
    """
    default_id = model.materials.DEFAULT_ID
    seen: dict[int, None] = {}
    for definition, _ in model.traverse():
        mesh = definition.mesh
        for f in mesh.faces_iter():
            mat_id = mesh.face_material(f.id)
            if mat_id != default_id:
                seen.setdefault(mat_id, None)
    return list(seen)


def _material_obj_names(model) -> dict[int, str]:
    """One OBJ material name per material **id**, minted through `_unique_name`.

    `sanitize_material_name` is many-to-one: `Brick Wall` and `Brick_Wall`
    both come out `Brick_Wall`. Keying the exported material dict on that
    name merged the two, dropping one material's colour and overwriting its
    image file, silently, on a user-initiated export (#119). Minting per id
    gives the second one `Brick_Wall.001` instead.

    Both `model_to_objdoc` and `export_obj` derive names from this, so the
    `.mtl` block, the `usemtl` on each face and the image file on disk cannot
    disagree about what a material is called.
    """
    used: set[str] = set()
    names: dict[int, str] = {}
    for mat_id in _painted_material_ids(model):
        mat = model.materials.get(mat_id)
        if mat is None:
            continue
        names[mat_id] = _unique_name(sanitize_material_name(mat.name), used)
    return names


def model_to_objdoc(model, resolver=None) -> ObjDocument:
    """Flatten the scene graph to a world-space ObjDocument (one object per node
    with geometry). Never mutates the model.

    `resolver` resolves a face's front-side UVs (Task 6's `resolve_face_uvs`
    shape: `resolver(mesh, materials, face_id, side) -> np.ndarray`). It is
    injected rather than imported here because `bermake/io` may not reach Qt,
    and `bermake.viewport` does (through the Qt-backed texture loader it uses
    for GPU upload). With no resolver, no `vt`s are produced and every face's
    `uv_indices` stays None.

    UVs are written only for a face that genuinely needs them: one with its
    own stored front-side UVs, or whose front material carries a texture.
    Baking a projection onto a face with neither would freeze it at the
    export-time projection with no benefit (a later texture_size or placement
    edit could no longer reproject it), bloat untextured exports for nothing,
    and hand downstream tools a UV map with no meaning.
    """
    vertices: list[tuple[float, float, float]] = []
    objects: list[ObjObject] = []
    materials: dict[str, tuple[float, float, float]] = {}
    used_names: set[str] = set()
    default_id = model.materials.DEFAULT_ID
    obj_material_names = _material_obj_names(model)

    uv_pool: list[tuple[float, float]] = []
    uv_lookup: dict[tuple[float, float], int] = {}

    normal_pool: list[tuple[float, float, float]] = []
    normal_lookup: dict[tuple[float, float, float], int] = {}

    def _normal_index_for(mesh, fid, normal_world):
        """Intern this face's world-space normal, or None if it has none.

        The kernel stores one normal per face, so the whole face gets one
        index (#113). A degenerate face has no defensible normal and simply
        goes out without one, which is what an exporter that never wrote
        normals at all did for every face: best-effort, the way a rejected
        material or a missing .mtl already degrades here.
        """
        if normal_world is None:
            return None
        try:
            local = np.asarray(mesh.face_normal(fid), dtype=np.float64)
        except (KeyError, ValueError):
            return None
        world = normal_world @ local
        length = float(np.linalg.norm(world))
        if not np.isfinite(length) or length < 1e-12:
            return None
        key = tuple(round(float(c), 6) for c in world / length)
        idx = normal_lookup.get(key)
        if idx is None:
            idx = len(normal_pool)
            normal_lookup[key] = idx
            normal_pool.append(key)
        return idx

    def _uv_indices_for(mesh, fid, needs_uvs):
        """Resolve this face's UVs and intern them into the shared vt pool."""
        if resolver is None or not needs_uvs:
            return None
        try:
            resolved = resolver(mesh, getattr(model, "materials", None), fid, Side.FRONT)
            resolved = np.asarray(resolved, dtype=np.float64).reshape(-1, 2)
        except (KeyError, ValueError):
            return None
        out = []
        for u, v in resolved:
            key = (round(float(u), 6), round(float(v), 6))
            idx = uv_lookup.get(key)
            if idx is None:
                idx = len(uv_pool)
                uv_lookup[key] = idx
                uv_pool.append(key)
            out.append(idx)
        return tuple(out)

    for definition, world in model.traverse():
        mesh = definition.mesh
        verts = list(mesh.vertices_iter())
        if not verts:
            continue  # skip empty definitions (e.g. an empty root)
        # Normals transform by the inverse-transpose of the linear block, not
        # by the block itself (#92): under a non-uniform scale the block tilts
        # a normal off its surface, and export is exactly where a scaled group
        # gets flattened into world space. A collapsed transform has no
        # inverse and so no normals; its geometry still exports.
        try:
            normal_world = np.linalg.inv(world[:3, :3]).T
        except np.linalg.LinAlgError:
            normal_world = None
        idmap: dict[int, int] = {}
        for v in verts:
            local = np.array([v.position[0], v.position[1], v.position[2], 1.0], dtype=np.float64)
            wp = world @ local
            idmap[v.id] = len(vertices)
            vertices.append((float(wp[0]), float(wp[1]), float(wp[2])))
        faces: list[ObjFace] = []
        for f in mesh.faces_iter():
            loop = tuple(idmap[vid] for vid in f.loop_vertex_ids)
            mat_id = mesh.face_material(f.id)
            mat = model.materials.get(mat_id)
            has_stored_uvs = mesh.face_uvs(f.id, Side.FRONT) is not None
            is_textured = mat is not None and mat.texture_id is not None
            uv_indices = _uv_indices_for(mesh, f.id, has_stored_uvs or is_textured)
            normal_index = _normal_index_for(mesh, f.id, normal_world)
            if mat_id != default_id:
                mname = obj_material_names[mat_id]
                materials[mname] = mat.base_color
                faces.append(ObjFace(loop, mname, uv_indices=uv_indices, normal_index=normal_index))
            else:
                faces.append(ObjFace(loop, None, uv_indices=uv_indices, normal_index=normal_index))
        objects.append(ObjObject(_unique_name(definition.name, used_names), tuple(faces)))

    material_textures: dict[str, str] = {}
    for m in model.materials.materials():
        mname = obj_material_names.get(m.id)
        if mname is None or mname not in materials:
            continue  # only materials actually used by an exported face
        if m.texture_id is None:
            continue
        tex = model.textures.get(m.texture_id)
        if tex is None:
            continue
        material_textures[mname] = f"{mname}.{tex.image_format}"

    return ObjDocument(
        vertices=tuple(vertices),
        objects=tuple(objects),
        materials=materials,
        material_textures=material_textures,
        uvs=tuple(uv_pool),
        normals=tuple(normal_pool),
        has_object_tags=bool(objects),
    )


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def export_obj(path, model) -> None:
    """Write the model to `path` as OBJ, with a sibling `<stem>.mtl` if it has
    painted materials, plus a sibling image file for each textured material.
    Each file is written atomically (temp + os.replace).

    `resolve_face_uvs` is imported here, lazily, rather than at module scope:
    `bermake/io` must not import `bermake.viewport` at import time, since that
    package reaches Qt through the GPU-side texture loader it uses for upload.
    `uv_resolve.py` itself is Qt-free (numpy and bermake.scene only), so
    importing it inside this function keeps `bermake.io.obj_io` importable
    with no Qt loaded, while still giving every caller of `export_obj` real
    UVs with no extra argument.
    """
    from bermake.viewport.uv_resolve import resolve_face_uvs

    path = Path(path)
    doc = model_to_objdoc(model, resolver=resolve_face_uvs)
    mtl_name = path.stem + ".mtl"
    obj_text, mtl_text = write_obj(doc, mtl_name)
    _atomic_write_text(path, obj_text)
    if mtl_text is not None:
        _atomic_write_text(path.with_name(mtl_name), mtl_text)

    obj_material_names = _material_obj_names(model)
    for m in model.materials.materials():
        if m.texture_id is None:
            continue
        tex = model.textures.get(m.texture_id)
        if tex is None:
            continue
        obj_name = obj_material_names.get(m.id)
        if obj_name is None:
            continue
        name = f"{obj_name}.{tex.image_format}"
        if name in doc.material_textures.values():
            _atomic_write_bytes(path.with_name(name), tex.data)


def read_obj_document(path) -> ObjDocument:
    """Read a .obj (and its `mtllib` sidecar, if present next to it) and parse to
    an ObjDocument. Raises BermakeFormatError on malformed content; OSError on a
    missing/unreadable .obj propagates. A missing sidecar .mtl is non-fatal."""
    path = Path(path)
    obj_text = path.read_text(encoding="utf-8")
    mtl_text: str | None = None
    for raw in obj_text.splitlines():
        s = raw.strip()
        if s.startswith("mtllib"):
            parts = s.split()
            if len(parts) > 1:
                try:
                    mtl_path = path.with_name(parts[1])
                    if mtl_path.exists():
                        mtl_text = mtl_path.read_text(encoding="utf-8")
                except (ValueError, OSError):
                    mtl_text = None
            break
    return parse_obj(obj_text, mtl_text)


def read_obj_texture_bytes(path, doc) -> dict[str, bytes]:
    """Material name to image bytes, for every `map_Kd` resolvable beside the .obj.

    Best-effort in the same shape as the .mtl sibling lookup: an image that is
    missing, unreadable or outside the document's directory is skipped rather
    than failing the import. The caller decides what to do with an empty result.
    """
    base = Path(path).parent
    out: dict[str, bytes] = {}
    for name, rel in doc.material_textures.items():
        data = read_sibling_image_bytes(base, rel)
        if data is not None:
            out[name] = data
    return out


@dataclass(frozen=True)
class ImportSummary:
    objects: int  # groups created (0 for the merge case)
    faces_imported: int
    faces_skipped: int


@dataclass
class BuildResult:
    summary: ImportSummary
    created_instances: list  # Instances added to target_context (group case)
    created_geometry: tuple  # (vertex_ids, edge_ids, face_ids) added to the scene (merge)


def _ensure_materials(materials, model, texture_bytes=None, decoder=None) -> dict:
    """Add each OBJ material to the library, reusing an existing one when name AND
    color already match. Returns name -> material_id.

    When `texture_bytes` and `decoder` are both given, each material with image
    bytes gains a Texture in the model's library and points at it. Following the
    same rule as material adds, texture adds are NOT undone by ImportObjCommand:
    no library add is undoable anywhere in bermake.

    A reused material (name+color match) that already carries a *different*
    texture is never repointed at the import's image: that library edit is
    not undone by ImportObjCommand, so it would silently retexture every
    pre-existing face already painted with it. Such an import gets its own
    new material instead, named via `_unique_name` like every other
    collision in this module - unless an earlier colliding import already
    created exactly the material+texture this one needs, in which case that
    one is reused so repeated imports of the same colliding document do not
    pile up duplicate materials and duplicate image blobs.
    """
    name_to_id: dict[str, int] = {}
    existing = {m.name: m for m in model.materials.materials()}
    for name, color in materials.items():
        m = existing.get(name)
        if m is not None and tuple(m.base_color) == tuple(color):
            name_to_id[name] = m.id
        else:
            new = model.materials.add_custom(name, color)
            name_to_id[name] = new.id
            existing[new.name] = new

    if not texture_bytes or decoder is None:
        return name_to_id

    existing_textures = list(model.textures.textures())

    def _find_or_decode_texture(tex_name: str, data: bytes):
        """A Texture for `data`: any library entry whose bytes already match
        it (name never participates in that match: identical bytes are the
        same image whatever it is called), else a freshly decoded+added one
        named `tex_name`. None if the bytes are undecodable."""
        data = bytes(data)
        for cached in existing_textures:
            if cached.data == data:
                return cached
        decoded = decoder(data)
        if decoded is None:
            return None
        image_format, width, height, has_transparency = decoded
        tex = model.textures.add(tex_name, data, image_format, width, height, has_transparency)
        existing_textures.append(tex)
        return tex

    for name, data in texture_bytes.items():
        mid = name_to_id.get(name)
        if mid is None:
            continue
        mat = model.materials.get(mid)
        current = model.textures.get(mat.texture_id) if mat.texture_id is not None else None
        if current is not None and current.data == bytes(data):
            continue  # already textured with exactly these bytes

        if mat.texture_id is not None:
            # `mat` is a reused material that already carries a different
            # texture (finding 1). Give the import its own material so the
            # pre-existing faces painted with `mat` are left alone.
            tex_name = _unique_name(name, {t.name for t in existing_textures})
            tex = _find_or_decode_texture(tex_name, data)
            if tex is None:
                continue  # unreadable image: faces stay on the shared material
            reused = next(
                (
                    cand
                    for cand in model.materials.materials()
                    if cand.texture_id == tex.id and tuple(cand.base_color) == tuple(mat.base_color)
                ),
                None,
            )
            if reused is not None:
                name_to_id[name] = reused.id
                continue
            new_name = _unique_name(name, set(existing))
            new_mat = model.materials.add_custom(new_name, mat.base_color)
            existing[new_mat.name] = new_mat
            model.materials.edit(new_mat.id, texture_id=tex.id)
            name_to_id[name] = new_mat.id
            continue

        tex = _find_or_decode_texture(name, data)
        if tex is None:
            continue  # unreadable image: the material stays untextured
        model.materials.edit(mid, texture_id=tex.id)
    return name_to_id


def _add_faces(mesh, faces, localmap, name_to_id, uv_pool=()) -> tuple[int, int]:
    """Best-effort: build each face, skipping+counting any the kernel rejects
    or any that references an unknown/out-of-range vertex index.

    Imported per-corner UVs are stored on BOTH sides (spec D6): the source has
    one set and bermake has two, so writing both means the model reads correctly
    from either side. The mapping is positional, because add_face_from_loop
    stores the loop in the order it was given.
    """
    imported = skipped = 0
    for face in faces:
        try:
            loop = [localmap[gi] for gi in face.vertex_indices]
            if len(set(loop)) < 3:  # degenerate: < 3 unique verts
                skipped += 1
                continue
            fid = mesh.add_face_from_loop(loop)
        except (KeyError, ValueError, IndexError, RuntimeError):
            skipped += 1
            continue
        if face.material is not None:
            mid = name_to_id.get(face.material)
            if mid is not None:
                mesh.set_face_material(fid, mid)
        # A duplicate vertex in the built loop (two `v` rows welded to the
        # same mesh vertex by Scene.add_vertex's exact-position dedupe, e.g.
        # loop [0, 1, 2, 1, 3]) must never get stored UVs. face_uvs is
        # parallel to the loop, but Scene.face_triangle_loop_indices maps
        # vertex_id -> loop_index, so the duplicate's later occurrence
        # silently overwrites the earlier one there while resolve_face_uvs
        # and the exporter walk the loop in order: the renderer would then
        # sample a different corner than everything else agrees on (finding
        # 2). Do not "fix" this by guarding it in Scene.set_face_uvs
        # instead: that is stage-1 code and import is the only path that can
        # create such a face with stored UVs today.
        pinched = len(set(loop)) != len(loop)
        if face.uv_indices is not None and uv_pool and not pinched:
            try:
                corner_uvs = [uv_pool[t] for t in face.uv_indices]
                for side in (Side.FRONT, Side.BACK):
                    mesh.set_face_uvs(fid, corner_uvs, side)
            except (IndexError, ValueError):
                # A UV problem never costs the face: it falls back to the
                # plane projection, which is what an untextured face uses.
                pass
        imported += 1
    return imported, skipped


def _snapshot_ids(mesh):
    return (
        {v.id for v in mesh.vertices_iter()},
        {e.id for e in mesh.edges_iter()},
        {f.id for f in mesh.faces_iter()},
    )


def build_obj_into_model(
    doc: ObjDocument, model, target_context, texture_bytes=None, decoder=None
) -> BuildResult:
    """Build an ObjDocument into the model. Adaptive: has_object_tags -> one group
    per object in target_context; else merge into target_context.mesh. Best-effort
    face building. Returns the created ids for undo."""
    name_to_id = _ensure_materials(doc.materials, model, texture_bytes, decoder)

    if doc.has_object_tags:
        created_instances: list = []
        imported = skipped = 0
        for obj in doc.objects:
            used = sorted(
                {gi for f in obj.faces for gi in f.vertex_indices if 0 <= gi < len(doc.vertices)}
            )
            defn = model.new_definition(obj.name or "Imported", is_group=True)
            localmap = {}
            for gi in used:
                x, y, z = doc.vertices[gi]
                localmap[gi] = defn.mesh.add_vertex(np.array([x, y, z], dtype=np.float32))
            i, s = _add_faces(defn.mesh, obj.faces, localmap, name_to_id, doc.uvs)
            imported += i
            skipped += s
            inst = model.new_instance(defn)
            target_context.children.append(inst)
            created_instances.append(inst)
        summary = ImportSummary(len(doc.objects), imported, skipped)
        return BuildResult(summary, created_instances, ([], [], []))

    # merge case
    mesh = target_context.mesh
    before = _snapshot_ids(mesh)
    localmap = {}
    for gi, (x, y, z) in enumerate(doc.vertices):
        localmap[gi] = mesh.add_vertex(np.array([x, y, z], dtype=np.float32))
    all_faces = [f for o in doc.objects for f in o.faces]
    imported, skipped = _add_faces(mesh, all_faces, localmap, name_to_id, doc.uvs)
    after = _snapshot_ids(mesh)
    created = tuple(sorted(after[i] - before[i]) for i in range(3))  # (vids, eids, fids)
    return BuildResult(ImportSummary(0, imported, skipped), [], created)
