"""glTF import: bridge adapter + model builder (M6c).

read_gltf_scene adapts the _core.import_gltf bridge into the neutral IR;
build_gltf_into_model (Tasks 4-5) maps the IR into the Model. This is the only
glTF module that imports Model/Scene.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from bermake.io.errors import BermakeFormatError
from bermake.io.gltf_scene import GltfImage, GltfMaterial, GltfMesh, GltfNode, GltfSceneData
from bermake.io.image_paths import read_sibling_image_bytes
from bermake.io.obj_io import _unique_name
from bermake.scene.scene import Side

if TYPE_CHECKING:
    from bermake.model.instance import Instance

# glTF import is an untrusted-input path (Assimp has a real CVE history): a
# file someone emailed you is a normal thing to import. Everything below runs
# BEFORE the file reaches Assimp, so a hostile file is rejected deterministically
# rather than risking a native crash/hang/OOM (#84).

# On-disk size ceiling. Protects against simply reading/parsing a gigantic
# file (whether or not its *declared* counts are absurd) before Assimp is
# even invoked. Checked via stat(), never by reading the file into memory.
# A few hundred MiB comfortably covers any legitimate architectural asset.
_MAX_GLTF_BYTES = 512 * 1024 * 1024  # 512 MiB

# Ceiling for any declared "how many of these" number in the glTF JSON:
# accessors[].count, and the number of accessors/meshes/primitives. Protects
# against the #84 CVE class where a tiny JSON payload declares an
# astronomical count (e.g. 10**12 vertices) that a naive parser tries to
# honor, driving unbounded allocation in native code. 50M is far beyond any
# real mesh (millions of vertices is already an extreme architectural model)
# while comfortably rejecting hostile counts.
_MAX_ELEMENT_COUNT = 50_000_000

# GLB (binary glTF) container layout: magic + version + total length, then
# one or more length-prefixed chunks; the first chunk is conventionally JSON.
_GLB_MAGIC = b"glTF"
_GLB_HEADER_SIZE = 12
_GLB_CHUNK_HEADER_SIZE = 8
_GLB_JSON_CHUNK_TYPE = b"JSON"


def read_gltf_scene(path) -> GltfSceneData:
    """Read a .glb/.gltf via the Assimp bridge into a GltfSceneData.

    Raises BermakeFormatError if the file cannot be decoded, is larger than
    _MAX_GLTF_BYTES, or declares element counts larger than
    _MAX_ELEMENT_COUNT. OSError from a genuinely missing/unreadable path
    propagates.
    """
    _validate_gltf_file_before_parse(path)

    import bermake._core as core

    try:
        raw = core.import_gltf(str(path))
    except RuntimeError as e:  # bridge raises std::runtime_error -> RuntimeError
        raise BermakeFormatError(f"Could not import glTF: {e}") from e

    materials = tuple(
        GltfMaterial(
            name=m.name,
            color=(m.base_color[0], m.base_color[1], m.base_color[2]),
            texture_index=int(m.texture_index),
            texture_uri=str(m.texture_uri),
        )
        for m in raw.materials
    )
    meshes = tuple(
        GltfMesh(
            positions=tuple((p[0], p[1], p[2]) for p in m.positions),
            triangles=tuple((t[0], t[1], t[2]) for t in m.triangles),
            material_index=m.material_index,
            uvs=tuple((u[0], u[1]) for u in m.uvs),
        )
        for m in raw.meshes
    )
    nodes = tuple(
        GltfNode(
            name=n.name,
            parent=n.parent,
            transform=tuple(n.transform),
            mesh_indices=tuple(n.mesh_indices),
        )
        for n in raw.nodes
    )
    images = tuple(
        GltfImage(name=i.name, data=bytes(i.data), format_hint=str(i.format_hint))
        for i in raw.images
    )
    return GltfSceneData(nodes=nodes, meshes=meshes, materials=materials, images=images)


def read_gltf_texture_bytes(path, scene) -> dict[int, bytes]:
    """Material index to base-colour image bytes, for every material that has one.

    Two shapes, both best-effort in the same way a missing .mtl is: an
    EMBEDDED image comes from the scene's own image table, and an EXTERNAL one
    is read from beside `path` under the containment rule in image_paths. An
    image that is missing, unreadable, outside the document's directory, or
    (D17) arrived as raw texels with no encoded form, is skipped. The caller
    decides what an empty result means.
    """
    base = Path(path).parent
    out: dict[int, bytes] = {}
    for index, material in enumerate(scene.materials):
        data: bytes | None = None
        if 0 <= material.texture_index < len(scene.images):
            data = scene.images[material.texture_index].data or None
        elif material.texture_uri:
            data = read_sibling_image_bytes(base, material.texture_uri)
        if data:
            out[index] = data
    return out


def _validate_gltf_file_before_parse(path) -> None:
    """Pre-parse validation that runs before Assimp ever sees the file.

    Checks size first (cheap, no file content read), then for JSON-based
    .gltf files parses and validates the document; for binary .glb files at
    minimum the size ceiling applies, plus a best-effort JSON-chunk check
    when the GLB header is well-formed enough to extract one.
    """
    p = Path(path)
    # os.path.getsize/.stat() reads only filesystem metadata, not file
    # content, so this is cheap even for a maliciously huge file.
    size = p.stat().st_size  # OSError propagates for missing/unreadable path
    if size > _MAX_GLTF_BYTES:
        raise BermakeFormatError(
            f"glTF file is {size} bytes, exceeding the {_MAX_GLTF_BYTES} byte ceiling"
        )

    if p.suffix.lower() == ".glb":
        doc = _read_glb_json_chunk(p)
    else:
        doc = _read_gltf_json(p)

    if doc is not None:
        _validate_gltf_element_counts(doc)


def _read_gltf_json(p: Path) -> dict:
    """Read+parse a .gltf (JSON) file. A truncated or non-JSON file surfaces
    as BermakeFormatError deterministically, rather than being handed to
    Assimp's native parser first."""
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise BermakeFormatError(f"glTF file is not valid UTF-8 text: {e}") from e
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as e:
        raise BermakeFormatError(f"glTF file is not valid JSON: {e}") from e
    if not isinstance(doc, dict):
        raise BermakeFormatError("glTF file JSON root is not an object")
    return doc


def _read_glb_json_chunk(p: Path) -> dict | None:
    """Best-effort extraction of the JSON chunk from a .glb container.

    Returns None (deferring to Assimp + the existing exception wrapping)
    when the header doesn't look like a well-formed GLB we can safely parse
    ourselves -- we never want our own parsing here to be a *new* source of
    crashes on malformed input.
    """
    with p.open("rb") as f:
        header = f.read(_GLB_HEADER_SIZE)
        if len(header) < _GLB_HEADER_SIZE or header[:4] != _GLB_MAGIC:
            return None
        chunk_header = f.read(_GLB_CHUNK_HEADER_SIZE)
        if len(chunk_header) < _GLB_CHUNK_HEADER_SIZE:
            return None
        chunk_length = int.from_bytes(chunk_header[0:4], "little")
        chunk_type = chunk_header[4:8]
        if chunk_type != _GLB_JSON_CHUNK_TYPE:
            return None
        if chunk_length > _MAX_GLTF_BYTES:
            raise BermakeFormatError(
                f"glTF GLB JSON chunk declares {chunk_length} bytes, "
                f"exceeding the {_MAX_GLTF_BYTES} byte ceiling"
            )
        json_bytes = f.read(chunk_length)
        if len(json_bytes) < chunk_length:
            raise BermakeFormatError("glTF GLB JSON chunk is truncated")

    try:
        doc = json.loads(json_bytes)
    except json.JSONDecodeError as e:
        raise BermakeFormatError(f"glTF GLB JSON chunk is not valid JSON: {e}") from e
    if not isinstance(doc, dict):
        raise BermakeFormatError("glTF GLB JSON chunk root is not an object")
    return doc


def _validate_gltf_element_counts(doc: dict) -> None:
    """Reject declared accessor/mesh/primitive counts (and out-of-range
    accessor/bufferView/attribute references) that exceed a sane ceiling,
    before Assimp allocates anything on their behalf. This is the guard for
    the #84 CVE class: a tiny JSON payload can declare an astronomically
    large accessor.count that a parser may try to honor."""
    accessors = doc.get("accessors") or []
    if not isinstance(accessors, list):
        raise BermakeFormatError("glTF 'accessors' is not an array")
    if len(accessors) > _MAX_ELEMENT_COUNT:
        raise BermakeFormatError(
            f"glTF declares {len(accessors)} accessors, exceeding the {_MAX_ELEMENT_COUNT} ceiling"
        )
    buffer_views = doc.get("bufferViews") or []
    if not isinstance(buffer_views, list):
        raise BermakeFormatError("glTF 'bufferViews' is not an array")
    for i, accessor in enumerate(accessors):
        if not isinstance(accessor, dict):
            raise BermakeFormatError(f"glTF accessors[{i}] is not an object")
        count = accessor.get("count")
        is_int_count = isinstance(count, int) and not isinstance(count, bool)
        if not (is_int_count and 0 <= count <= _MAX_ELEMENT_COUNT):
            raise BermakeFormatError(
                f"glTF accessors[{i}].count={count!r} exceeds the "
                f"{_MAX_ELEMENT_COUNT} element ceiling"
            )
        buffer_view = accessor.get("bufferView")
        if buffer_view is not None and not (
            isinstance(buffer_view, int)
            and not isinstance(buffer_view, bool)
            and 0 <= buffer_view < len(buffer_views)
        ):
            raise BermakeFormatError(
                f"glTF accessors[{i}] references out-of-range bufferView {buffer_view!r}"
            )

    meshes = doc.get("meshes") or []
    if not isinstance(meshes, list):
        raise BermakeFormatError("glTF 'meshes' is not an array")
    if len(meshes) > _MAX_ELEMENT_COUNT:
        raise BermakeFormatError(
            f"glTF declares {len(meshes)} meshes, exceeding the {_MAX_ELEMENT_COUNT} ceiling"
        )
    for mi, mesh in enumerate(meshes):
        if not isinstance(mesh, dict):
            raise BermakeFormatError(f"glTF meshes[{mi}] is not an object")
        primitives = mesh.get("primitives") or []
        if not isinstance(primitives, list):
            raise BermakeFormatError(f"glTF meshes[{mi}].primitives is not an array")
        if len(primitives) > _MAX_ELEMENT_COUNT:
            raise BermakeFormatError(
                f"glTF meshes[{mi}] declares {len(primitives)} primitives, "
                f"exceeding the {_MAX_ELEMENT_COUNT} ceiling"
            )
        for pi, prim in enumerate(primitives):
            if not isinstance(prim, dict):
                raise BermakeFormatError(f"glTF meshes[{mi}].primitives[{pi}] is not an object")
            attributes = prim.get("attributes") or {}
            if not isinstance(attributes, dict):
                raise BermakeFormatError(
                    f"glTF meshes[{mi}].primitives[{pi}].attributes is not an object"
                )
            for attr_name, accessor_index in attributes.items():
                if not (
                    isinstance(accessor_index, int)
                    and not isinstance(accessor_index, bool)
                    and 0 <= accessor_index < len(accessors)
                ):
                    raise BermakeFormatError(
                        f"glTF meshes[{mi}].primitives[{pi}] attribute {attr_name!r} "
                        f"references out-of-range accessor {accessor_index!r}"
                    )
            indices = prim.get("indices")
            if indices is not None and not (
                isinstance(indices, int)
                and not isinstance(indices, bool)
                and 0 <= indices < len(accessors)
            ):
                raise BermakeFormatError(
                    f"glTF meshes[{mi}].primitives[{pi}] references out-of-range "
                    f"indices accessor {indices!r}"
                )


_DEFAULT_MATERIAL_NAMES = {"", "DefaultMaterial"}


def _is_default_material(m) -> bool:
    return m.name in _DEFAULT_MATERIAL_NAMES


def _ensure_gltf_materials(materials, model, texture_bytes=None, decoder=None) -> tuple[list, int]:
    """Material id per glTF material index (None for default/unpainted), and the
    count of images that were referenced but could not be applied.

    Real materials deduped by (name, color); add_custom otherwise.

    When `texture_bytes` (material index -> image bytes) and `decoder` are
    both given, this mirrors obj_io._ensure_materials' texture rules exactly
    (stage 2 hardened those against three real defects, so this does not
    invent a second policy):

    1. A texture is deduped by bytes, never by name: identical bytes are the
       same image whatever it is called.
    2. A reused material that already carries a *different* texture is never
       repointed at the import's image -- that library edit is not undone by
       ImportGltfCommand, so repointing would silently retexture every
       pre-existing face already painted with it. Such an import gets its own
       new material via `_unique_name` instead.
    3. A repeat import of the same colliding document reuses the material and
       texture it minted last time rather than piling up `Brick.001`,
       `Brick.002`, and duplicate byte-identical blobs.
    """
    result: list = []
    existing = {(m.name, tuple(m.base_color)): m for m in model.materials.materials()}
    used_names = {m.name for m in model.materials.materials()}
    for gm in materials:
        if _is_default_material(gm):
            result.append(None)
            continue
        key = (gm.name, tuple(gm.color))
        m = existing.get(key)
        if m is not None:
            result.append(m.id)
        else:
            new = model.materials.add_custom(gm.name, tuple(gm.color))
            existing[key] = new
            used_names.add(new.name)
            result.append(new.id)

    # A material can lose its texture before it ever reaches the loop below:
    # read_gltf_texture_bytes drops an entry entirely when the image is
    # missing, unreadable, outside the containment directory, or (D17)
    # raw texels with no encoded form. That makes `texture_bytes` an
    # already-filtered map, so a dropped image must be counted here, against
    # the full material list, before the `not texture_bytes` early return
    # below -- otherwise a document that lost every one of its images (the
    # single most common real-world break: a .gltf shipped without its
    # texture folder) hands back an EMPTY dict, `not {}` is True, and this
    # function returns 0 skipped for a model that imports fully untextured.
    # Gated on an attempt having actually been made (`is not None` on both),
    # so a caller that never wired up texture import (several existing tests)
    # is not scored as having skipped every textured material.
    dropped_upstream = 0
    if texture_bytes is not None and decoder is not None:
        for index, gm in enumerate(materials):
            has_texture_ref = gm.texture_index >= 0 or bool(gm.texture_uri)
            if has_texture_ref and not texture_bytes.get(index):
                dropped_upstream += 1

    images_skipped = dropped_upstream
    if not texture_bytes or decoder is None:
        return result, images_skipped

    existing_textures = list(model.textures.textures())

    def _find_or_decode_texture(tex_name: str, data: bytes):
        """A Texture for `data`: any library entry whose bytes already match
        it (name never participates in that match), else a freshly
        decoded+added one named `tex_name`. None if the bytes are
        undecodable."""
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

    for index, data in texture_bytes.items():
        if not (0 <= index < len(result)):
            continue
        mid = result[index]
        if mid is None:
            continue
        mat = model.materials.get(mid)
        current = model.textures.get(mat.texture_id) if mat.texture_id is not None else None
        if current is not None and current.data == bytes(data):
            continue  # already textured with exactly these bytes

        gm_name = materials[index].name

        if mat.texture_id is not None:
            # `mat` is a reused material that already carries a different
            # texture (rule 2). Give the import its own material so the
            # pre-existing faces painted with `mat` are left alone.
            tex_name = _unique_name(gm_name, {t.name for t in existing_textures})
            tex = _find_or_decode_texture(tex_name, data)
            if tex is None:
                images_skipped += 1
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
                result[index] = reused.id
                continue
            new_name = _unique_name(gm_name, used_names)
            new_mat = model.materials.add_custom(new_name, mat.base_color)
            existing[(new_mat.name, tuple(new_mat.base_color))] = new_mat
            used_names.add(new_mat.name)
            model.materials.edit(new_mat.id, texture_id=tex.id)
            result[index] = new_mat.id
            continue

        tex = _find_or_decode_texture(gm_name, data)
        if tex is None:
            images_skipped += 1
            continue  # unreadable image: the material stays untextured
        model.materials.edit(mid, texture_id=tex.id)
    return result, images_skipped


def _corner_uvs(uvs, tri):
    """One glTF triangle's UVs in loop order.

    NO vertical flip happens here, deliberately. glTF's TEXCOORD_0 origin is
    the image's UPPER left and Bermake's v = 0 is its BOTTOM, but Assimp's
    glTF2 importer already applies 1 - v before the bridge sees a coordinate,
    so what arrives is ALREADY in Bermake's convention. Measured on
    textured_box.glb: the file holds v = [1.0, 1.0, 0.25, 0.25] and
    import_gltf returns [0.0, 0.0, 0.75, 0.75]. Flipping again here would
    render every imported texture upside down.

    Export is NOT symmetric with this: model_to_gltf does flip, because it
    writes through Bermake's own codec rather than through Assimp. See D14.
    test_assimp_already_flips_v_CI_GATE pins the dependency.

    Returns None when any corner index is out of range, which drops that face
    to projection rather than failing the import.
    """
    out = []
    for gi in tri:
        if not 0 <= gi < len(uvs):
            return None
        u, v = uvs[gi]
        out.append((float(u), float(v)))
    return out


def _add_triangles(mesh, triangles, localmap, material_id, uvs=()) -> tuple[int, int, int]:
    """Best-effort: build each triangle, skipping+counting kernel rejects.

    Returns (imported, skipped, without_uvs). A glTF triangle's corners map
    onto the kernel loop positionally and in order, so no matching logic is
    needed. Both sides get the array (spec 1.5): a source format has one UV
    set per corner and Bermake has two, and writing both means the model reads
    correctly from either side without import holding an opinion about
    sidedness.
    """
    imported = skipped = without_uvs = 0
    for tri in triangles:
        try:
            loop = [localmap[gi] for gi in tri]
            if len(set(loop)) < 3:
                skipped += 1
                continue
            fid = mesh.add_face_from_loop(loop)
        except (KeyError, ValueError, IndexError, RuntimeError):
            skipped += 1
            continue
        if material_id is not None:
            mesh.set_face_material(fid, material_id)
        if uvs:
            corners = _corner_uvs(uvs, tri)
            if corners is None:
                without_uvs += 1
            else:
                try:
                    mesh.set_face_uvs(fid, corners, Side.FRONT)
                    mesh.set_face_uvs(fid, corners, Side.BACK)
                except ValueError:
                    # The kernel welds coincident positions, so a triangle
                    # whose corners collapsed has fewer loop corners than
                    # glTF vertices. Projection is the honest answer.
                    without_uvs += 1
        imported += 1
    return imported, skipped, without_uvs


def _build_mesh_components(scene, model, mat_id_by_index):
    """Build each GltfMesh into a shared Component Definition (built once, later
    instanced). Returns (meshdefs, imported, skipped, without_uvs, built).
    meshdefs[i] is a Definition or None (empty or all-faces-skipped mesh)."""
    meshdefs: list = []
    imported = skipped = without_uvs = built = 0
    for i, gmesh in enumerate(scene.meshes):
        if not gmesh.positions:
            meshdefs.append(None)
            continue
        defn = model.new_definition(f"Mesh.{i:03d}", is_group=False)
        localmap = {}
        for gi, (x, y, z) in enumerate(gmesh.positions):
            localmap[gi] = defn.mesh.add_vertex(np.array([x, y, z], dtype=np.float32))
        mid = None
        if 0 <= gmesh.material_index < len(mat_id_by_index):
            mid = mat_id_by_index[gmesh.material_index]
        imp, skp, novu = _add_triangles(defn.mesh, gmesh.triangles, localmap, mid, gmesh.uvs)
        imported += imp
        skipped += skp
        without_uvs += novu
        if imp == 0:
            meshdefs.append(None)  # unreferenced def is GC'd
            continue
        meshdefs.append(defn)
        built += 1
    return meshdefs, imported, skipped, without_uvs, built


@dataclass(frozen=True)
class GltfImportSummary:
    nodes: int  # glTF nodes mapped (one object each)
    meshes: int  # distinct Component meshes built
    faces_imported: int  # faces built into Component meshes (per distinct mesh)
    faces_skipped: int
    faces_without_uvs: int = 0  # imported, but their UV data was unusable
    images_skipped: int = 0  # referenced but missing, unreadable or raw texels


@dataclass
class GltfBuildResult:
    summary: GltfImportSummary
    root_instance: Instance  # appended to target_context.children


def _yup_to_zup() -> np.ndarray:
    """Rx(+90°): glTF Y-up -> Bermake Z-up. (x, y, z) -> (x, -z, y)."""
    return np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def build_gltf_into_model(
    scene, model, target_context, root_name="glTF", texture_bytes=None, decoder=None
) -> GltfBuildResult:
    """Build a GltfSceneData into the model under target_context. Preserves the
    node hierarchy (each node -> one object; single-mesh childless nodes collapse
    to a direct shared-Component instance), converts Y-up -> Z-up at the file
    wrapper, and is best-effort. Returns the single wrapper Instance for undo.

    `texture_bytes` (material index -> image bytes, from read_gltf_texture_bytes)
    and `decoder` are both optional and default to None so every existing
    caller is unaffected; when both are given, materials gain textures per
    `_ensure_gltf_materials`."""
    mat_id_by_index, images_skipped = _ensure_gltf_materials(
        scene.materials, model, texture_bytes, decoder
    )
    meshdefs, imported, skipped, without_uvs, built = _build_mesh_components(
        scene, model, mat_id_by_index
    )

    wrapper = model.new_definition(root_name or "glTF", is_group=True)
    has_children = {n.parent for n in scene.nodes if n.parent >= 0}
    container_def: dict = {}

    for idx, node in enumerate(scene.nodes):
        local = np.array(node.transform, dtype=np.float64).reshape(4, 4)
        mesh_idxs = [
            mi for mi in node.mesh_indices if 0 <= mi < len(meshdefs) and meshdefs[mi] is not None
        ]
        collapsible = (len(mesh_idxs) == 1) and (idx not in has_children)
        if collapsible:
            inst = model.new_instance(meshdefs[mesh_idxs[0]], transform=local)
        else:
            g = model.new_definition(node.name or "Node", is_group=True)
            for mi in mesh_idxs:
                g.children.append(model.new_instance(meshdefs[mi]))
            inst = model.new_instance(g, transform=local)
            container_def[idx] = g
        # Precondition: scene.nodes lists each node's parent at a lower index
        # than the node itself (glTF's own node-array ordering guarantee), so
        # container_def[node.parent] is always already populated here.
        parent = wrapper if node.parent == -1 else container_def[node.parent]
        parent.children.append(inst)

    root_instance = model.new_instance(wrapper, transform=_yup_to_zup())
    target_context.children.append(root_instance)

    summary = GltfImportSummary(
        nodes=len(scene.nodes),
        meshes=built,
        faces_imported=imported,
        faces_skipped=skipped,
        faces_without_uvs=without_uvs,
        images_skipped=images_skipped,
    )
    return GltfBuildResult(summary=summary, root_instance=root_instance)
