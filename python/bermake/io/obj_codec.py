"""Pure OBJ/MTL text <-> ObjDocument IR (M6b).

No Qt, GL, filesystem, or model here — just text <-> a neutral intermediate
representation, so the whole round-trip is headlessly unit-testable. OBJ is a
world-space polygon soup with a shared global vertex pool (`v` global, `f`
references 1-based global indices) + optional `o`/`g` objects + a sidecar `.mtl`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bermake.io.errors import BermakeFormatError
from bermake.io.image_paths import sanitize_filename_stem


@dataclass(frozen=True)
class ObjFace:
    vertex_indices: tuple[int, ...]  # 0-based, into ObjDocument.vertices
    material: str | None = None  # sanitized material name, or None (unpainted)
    uv_indices: tuple[int, ...] | None = None  # 0-based into ObjDocument.uvs, or None
    # One normal for the WHOLE face, not one per corner: the kernel stores a
    # single normal[3] per face (Newell's method over the boundary loop), so a
    # per-corner index could only ever repeat it. Honouring imported per-corner
    # normals is the smooth-shading half of #113 and needs storage bermake does
    # not have yet.
    normal_index: int | None = None  # 0-based into ObjDocument.normals, or None


@dataclass(frozen=True)
class ObjObject:
    name: str
    faces: tuple[ObjFace, ...]


@dataclass(frozen=True)
class ObjDocument:
    vertices: tuple[tuple[float, float, float], ...]
    objects: tuple[ObjObject, ...]
    materials: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    material_textures: dict[str, str] = field(default_factory=dict)
    uvs: tuple[tuple[float, float], ...] = ()
    normals: tuple[tuple[float, float, float], ...] = ()
    has_object_tags: bool = False


def sanitize_material_name(name: str) -> str:
    """OBJ names are whitespace-delimited tokens; collapse whitespace to '_',
    and restrict to filesystem-safe characters (image_paths.sanitize_filename_stem):
    this name flows into a `Path.with_name` filename for a texture sidecar
    (obj_io.export_obj), and an OBJ material name is untrusted text from a
    file someone else authored."""
    return sanitize_filename_stem(name, "material")


def write_obj(doc: ObjDocument, mtl_filename: str = "model.mtl") -> tuple[str, str | None]:
    """Serialize an ObjDocument to (obj_text, mtl_text|None). mtl_text is None
    when there are no materials (and no `mtllib` line is written)."""
    obj: list[str] = ["# Bermake OBJ export"]
    if doc.materials:
        obj.append(f"mtllib {mtl_filename}")
    for vx, vy, vz in doc.vertices:
        obj.append(f"v {vx:.6f} {vy:.6f} {vz:.6f}")
    for tu, tv in doc.uvs:
        obj.append(f"vt {tu:.6f} {tv:.6f}")
    for nx, ny, nz in doc.normals:
        obj.append(f"vn {nx:.6f} {ny:.6f} {nz:.6f}")
    for o in doc.objects:
        obj.append(f"o {o.name}")
        # Sort faces so unpainted (None) come first, then grouped by material,
        # giving clean `usemtl` runs (OBJ has no "unset material" directive).
        faces_sorted = sorted(o.faces, key=lambda f: (f.material is not None, f.material or ""))
        current = None
        for face in faces_sorted:
            if face.material is not None and face.material != current:
                obj.append(f"usemtl {face.material}")
            current = face.material
            # OBJ corner forms: `v`, `v/vt`, `v//vn`, `v/vt/vn`. The empty
            # middle field is load-bearing -- `v/vn` would read the normal
            # index as a texture-coordinate index.
            n = "" if face.normal_index is None else str(face.normal_index + 1)
            if face.uv_indices is None:
                corners = [f"{v + 1}//{n}" if n else str(v + 1) for v in face.vertex_indices]
            else:
                corners = [
                    f"{v + 1}/{t + 1}/{n}" if n else f"{v + 1}/{t + 1}"
                    for v, t in zip(face.vertex_indices, face.uv_indices, strict=True)
                ]
            obj.append("f " + " ".join(corners))
    obj_text = "\n".join(obj) + "\n"

    mtl_text: str | None = None
    if doc.materials:
        m: list[str] = ["# Bermake MTL export"]
        for name, (r, g, b) in doc.materials.items():
            m.append(f"newmtl {name}")
            m.append(f"Kd {r:.6f} {g:.6f} {b:.6f}")
            m.append("Ka 0.000000 0.000000 0.000000")
            m.append("Ns 10.000000")
            m.append("d 1.000000")
            tex = doc.material_textures.get(name)
            if tex:
                m.append(f"map_Kd {tex}")
        mtl_text = "\n".join(m) + "\n"
    return obj_text, mtl_text


def _map_kd_filename(parts: list[str]) -> str | None:
    """The filename from a `map_Kd` line, skipping its option arguments.

    OBJ map lines may carry options before the filename (`-s 1 1 1 brick.png`)
    and filenames may contain spaces (`my brick.png`), so neither parts[1] nor
    parts[-1] is right on its own. Drop a leading run of option tokens, each
    consuming itself and any numeric values that follow it, then join what is
    left.
    """
    i = 1
    while i < len(parts) and parts[i].startswith("-"):
        i += 1
        while i < len(parts):
            try:
                float(parts[i])
            except ValueError:
                break
            i += 1
    name = " ".join(parts[i:]).strip()
    return name or None


def _parse_mtl(mtl_text: str) -> tuple[dict[str, tuple[float, float, float]], dict[str, str]]:
    """Parse an MTL to (name -> Kd colour, name -> map_Kd filename)."""
    materials: dict[str, tuple[float, float, float]] = {}
    textures: dict[str, str] = {}
    current: str | None = None
    for raw in mtl_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "newmtl":
            current = " ".join(parts[1:]) if len(parts) > 1 else "material"
            materials.setdefault(current, (0.8, 0.8, 0.8))
        elif parts[0] == "Kd" and current is not None:
            try:
                materials[current] = (float(parts[1]), float(parts[2]), float(parts[3]))
            except (IndexError, ValueError):
                pass  # keep the default grey
        elif parts[0] == "map_Kd" and current is not None:
            name = _map_kd_filename(parts)
            if name is not None:
                textures[current] = name
    return materials, textures


def parse_obj(obj_text: str, mtl_text: str | None) -> ObjDocument:
    """Parse OBJ (+ optional MTL) text to an ObjDocument. Raises BermakeFormatError
    on a structurally invalid face index."""
    materials, material_textures = _parse_mtl(mtl_text) if mtl_text else ({}, {})
    vertices: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    objects: list[ObjObject] = []
    has_object_tags = False
    current_name: str | None = None
    current_faces: list[ObjFace] = []
    current_material: str | None = None

    def flush() -> None:
        nonlocal current_faces
        if current_faces or current_name is not None:
            name = current_name if current_name is not None else "default"
            objects.append(ObjObject(name=name, faces=tuple(current_faces)))
            current_faces = []

    for raw in obj_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        tag = parts[0]
        if tag == "v":
            try:
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            except (IndexError, ValueError) as e:
                raise BermakeFormatError(f"bad vertex line: {line!r}") from e
        elif tag == "vt":
            try:
                u = float(parts[1])
                v = float(parts[2]) if len(parts) > 2 else 0.0
            except (IndexError, ValueError):
                continue  # a malformed vt is skipped, not fatal: faces fall back
            uvs.append((u, v))
        elif tag in ("o", "g"):
            flush()
            has_object_tags = True
            current_name = " ".join(parts[1:]) if len(parts) > 1 else "object"
            current_material = None
        elif tag == "usemtl":
            current_material = parts[1] if len(parts) > 1 else None
        elif tag == "f":
            idx: list[int] = []
            uv_idx: list[int] = []
            uv_ok = True
            for token in parts[1:]:
                fields = token.split("/")
                try:
                    vi = int(fields[0])
                except ValueError as e:
                    raise BermakeFormatError(f"bad face index {token!r}") from e
                vi = len(vertices) + vi if vi < 0 else vi - 1  # relative or 1-based
                if not (0 <= vi < len(vertices)):
                    raise BermakeFormatError(f"face index out of range: {token!r}")
                idx.append(vi)

                # A UV is optional per corner. Anything wrong with it drops the
                # whole face's UVs rather than raising: the geometry is still
                # good and that face can fall back to the plane projection.
                if not uv_ok or len(fields) < 2 or not fields[1]:
                    uv_ok = False
                    continue
                try:
                    ti = int(fields[1])
                except ValueError:
                    uv_ok = False
                    continue
                ti = len(uvs) + ti if ti < 0 else ti - 1
                if not (0 <= ti < len(uvs)):
                    uv_ok = False
                    continue
                uv_idx.append(ti)
            current_faces.append(
                ObjFace(
                    vertex_indices=tuple(idx),
                    material=current_material,
                    uv_indices=tuple(uv_idx) if uv_ok and len(uv_idx) == len(idx) else None,
                )
            )
        # mtllib / vn / vt / s / everything else: ignored
    flush()
    if not objects:
        objects.append(ObjObject(name="default", faces=()))
    return ObjDocument(
        vertices=tuple(vertices),
        objects=tuple(objects),
        materials=materials,
        material_textures=material_textures,
        uvs=tuple(uvs),
        has_object_tags=has_object_tags,
    )
