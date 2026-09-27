"""Pure dict <-> document codec for the native .berm format (M6a).

No Qt, GL, zip, or filesystem here — just dict <-> in-memory objects, so the
whole round-trip is headlessly unit-testable. Geometry is index-based (edges and
faces reference a vertex's POSITION in `vertices[]`, not its kernel id), which
compacts id gaps and lets load replay add_vertex/add_edge/add_face_from_loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from bermake.io.errors import BermakeFormatError
from bermake.model.annotation import Dimension, Guide, GuidePoint, Label
from bermake.model.definition import Definition
from bermake.model.instance import Instance
from bermake.model.material import MaterialLibrary
from bermake.model.model import Model
from bermake.model.tag import TagLibrary
from bermake.model.texture import TextureLibrary
from bermake.scene.scene import DEFAULT_PLACEMENT, Scene, Side, TexturePlacement
from bermake.units import Units, units_from_dict, units_to_dict
from bermake.viewport.environment import LEGACY_ENVIRONMENT, Environment
from bermake.viewport.face_batches import MAX_MATERIAL_ID
from bermake.viewport.render_style import FaceStyle, RenderStyle

_DEFAULT_MATERIAL_ID = 0  # mirrors MaterialLibrary.DEFAULT_ID


def geometry_to_dict(scene: Scene) -> dict:
    """Serialize a Scene's geometry with index-based edges/faces."""
    idmap: dict[int, int] = {}
    vertices: list[list[float]] = []
    for v in scene.vertices_iter():
        idmap[v.id] = len(vertices)
        vertices.append([float(v.position[0]), float(v.position[1]), float(v.position[2])])

    edges = [[idmap[e.v1_id], idmap[e.v2_id]] for e in scene.edges_iter()]

    faces: list[list[int]] = []
    face_materials: dict[str, int] = {}
    face_materials_back: dict[str, int] = {}
    face_placements: dict[str, list[float]] = {}
    face_placements_back: dict[str, list[float]] = {}
    face_uvs: dict[str, list[float]] = {}
    face_uvs_back: dict[str, list[float]] = {}
    for face_index, f in enumerate(scene.faces_iter()):
        faces.append([idmap[vid] for vid in f.loop_vertex_ids])
        front = scene.face_material(f.id, Side.FRONT)
        if front != _DEFAULT_MATERIAL_ID:
            face_materials[str(face_index)] = int(front)
        back = scene.face_material(f.id, Side.BACK)
        if back != _DEFAULT_MATERIAL_ID:
            face_materials_back[str(face_index)] = int(back)
        for side, target in (
            (Side.FRONT, face_placements),
            (Side.BACK, face_placements_back),
        ):
            p = scene.face_placement(f.id, side)
            if p != DEFAULT_PLACEMENT:
                target[str(face_index)] = [p.offset_u, p.offset_v, p.scale, p.rotation]

        for side, target in (
            (Side.FRONT, face_uvs),
            (Side.BACK, face_uvs_back),
        ):
            stored = scene.face_uvs(f.id, side)
            if stored is not None:
                target[str(face_index)] = [float(x) for x in stored.reshape(-1)]

    return {
        "vertices": vertices,
        "edges": edges,
        "faces": faces,
        "face_materials": face_materials,
        "face_materials_back": face_materials_back,
        "face_placements": face_placements,
        "face_placements_back": face_placements_back,
        "face_uvs": face_uvs,
        "face_uvs_back": face_uvs_back,
    }


def geometry_from_dict(scene: Scene, data: dict) -> None:
    """Replay geometry into an empty `scene`. Raises BermakeFormatError on bad indices."""
    new_vids: list[int] = []
    for pos in data["vertices"]:
        new_vids.append(scene.add_vertex(np.asarray(pos, dtype=np.float32)))
    n = len(new_vids)

    def _vid(i: int) -> int:
        if not (0 <= i < n):
            raise BermakeFormatError(f"vertex index {i} out of range (0..{n - 1})")
        return new_vids[i]

    for a, b in data["edges"]:
        scene.add_edge(_vid(int(a)), _vid(int(b)))

    new_fids: list[int] = []
    for loop in data["faces"]:
        new_fids.append(scene.add_face_from_loop([_vid(int(i)) for i in loop]))

    def _apply_face_materials(materials: dict, side: Side) -> None:
        for face_index_str, mat in materials.items():
            fi = int(face_index_str)
            if not (0 <= fi < len(new_fids)):
                raise BermakeFormatError(f"face index {fi} out of range (0..{len(new_fids) - 1})")
            mid = int(mat)
            # The material id is validated here, beside the face index, because
            # M7.5a's plan_face_batches rejects ids outside [0, MAX_MATERIAL_ID]
            # and it runs inside render(). Without this a document carrying
            # {"0": -3} loaded clean and then raised every single frame, where
            # nothing can report it; before M7.5a the same file simply rendered
            # as Default. Failing at load turns a corrupt file into one error
            # message the UI already knows how to show.
            #
            # Only the RANGE is checked. An id that is in range but names no
            # material in the library is a different case and stays tolerated:
            # MaterialLibrary.get falls back to Default for it, which is the
            # forgiving behaviour a file that lost a material should get.
            if not (0 <= mid <= MAX_MATERIAL_ID):
                raise BermakeFormatError(
                    f"face material id {mid} out of range (0..{MAX_MATERIAL_ID})"
                )
            scene.set_face_material(new_fids[fi], mid, side)

    _apply_face_materials(data.get("face_materials", {}), Side.FRONT)
    _apply_face_materials(data.get("face_materials_back", {}), Side.BACK)

    def _apply_face_placements(placements: dict, side: Side) -> None:
        for face_index_str, values in placements.items():
            fi = int(face_index_str)
            # Structural, like the face-index check in _apply_face_materials above:
            # an out-of-range index means the geometry and the placement sidecar
            # have drifted apart. Staying silent here would drop the user's
            # adjustment on load, and the very next save would then write that
            # loss back out permanently.
            if not (0 <= fi < len(new_fids)):
                raise BermakeFormatError(f"face index {fi} out of range (0..{len(new_fids) - 1})")
            if len(values) != 4:
                raise BermakeFormatError(f"face placement must have 4 numbers, got {len(values)}")
            scene.set_face_placement(new_fids[fi], TexturePlacement(*values), side)

    _apply_face_placements(data.get("face_placements", {}), Side.FRONT)
    _apply_face_placements(data.get("face_placements_back", {}), Side.BACK)

    def _apply_face_uvs(stored: dict, side: Side) -> None:
        for face_index_str, values in stored.items():
            try:
                fi = int(face_index_str)
            except (TypeError, ValueError):
                raise BermakeFormatError(
                    f"face_uvs: face index {face_index_str!r} is not an integer"
                ) from None
            if fi < 0 or fi >= len(new_fids):
                raise BermakeFormatError(f"face_uvs: face index {fi} is out of range")
            flat = list(values)
            if len(flat) % 2 != 0:
                raise BermakeFormatError(
                    f"face_uvs: face {fi} has an odd coordinate count ({len(flat)})"
                )
            pairs = [(float(flat[i]), float(flat[i + 1])) for i in range(0, len(flat), 2)]
            expected = len(scene.face_loop(new_fids[fi]))
            # Deliberately duplicated in Scene.set_face_uvs below: that check exists to
            # catch a programming error and raises ValueError, this one exists to catch
            # a corrupt file and raises BermakeFormatError. Same arithmetic, different
            # contract, so neither call site can be removed in favour of the other.
            if len(pairs) != expected:
                raise BermakeFormatError(
                    f"face_uvs: face {fi} has {expected} corners, file gives {len(pairs)}"
                )
            scene.set_face_uvs(new_fids[fi], pairs, side)

    _apply_face_uvs(data.get("face_uvs", {}), Side.FRONT)
    _apply_face_uvs(data.get("face_uvs_back", {}), Side.BACK)


def annotation_to_dict(ann: Dimension | Label | Guide | GuidePoint) -> dict:
    """Serialize an annotation, discriminated by `ann.kind`.

    Dispatch is explicit on every kind. The previous shape fell through to
    'label' for anything that was not a dimension, which would have silently
    written a Guide out as a malformed label.
    """
    if ann.kind == "dimension":
        return {
            "kind": "dimension",
            "id": ann.id,
            "p1": list(ann.p1),
            "p2": list(ann.p2),
            "offset": list(ann.offset),
        }
    if ann.kind == "label":
        return {
            "kind": "label",
            "id": ann.id,
            "anchor": list(ann.anchor),
            "text_pos": list(ann.text_pos),
            "text": ann.text,
        }
    if ann.kind == "guide":
        return {
            "kind": "guide",
            "id": ann.id,
            "origin": list(ann.origin),
            "direction": list(ann.direction),
        }
    if ann.kind == "guide_point":
        return {"kind": "guide_point", "id": ann.id, "position": list(ann.position)}
    raise BermakeFormatError(f"annotation has unknown kind: {ann.kind!r}")


def annotation_from_dict(record: dict) -> Dimension | Label | Guide | GuidePoint:
    """Rebuild the annotation produced by `annotation_to_dict`.

    Raises BermakeFormatError if 'kind' is missing or unrecognized. A v8 file
    read by an older build therefore fails loudly rather than dropping guides.

    A structurally well-formed record whose VALUES an entity rejects (today:
    `Guide.__post_init__` on a zero-length direction) raises ValueError, not
    BermakeFormatError. `document_from_dict`'s blanket catch does translate
    that, but only because this function happens to be called from inside it;
    a corrupt record is this module's own business to report, so it is
    translated here too, with the record named the way every other malformed
    record in this module is named.
    """
    kind = record.get("kind")
    try:
        if kind == "dimension":
            return Dimension(
                record["id"], tuple(record["p1"]), tuple(record["p2"]), tuple(record["offset"])
            )
        if kind == "label":
            return Label(
                record["id"], tuple(record["anchor"]), tuple(record["text_pos"]), record["text"]
            )
        if kind == "guide":
            return Guide(record["id"], tuple(record["origin"]), tuple(record["direction"]))
        if kind == "guide_point":
            return GuidePoint(record["id"], tuple(record["position"]))
    except ValueError as e:
        raise BermakeFormatError(f"malformed {kind} annotation: {e}") from e
    raise BermakeFormatError(f"annotation has unknown kind: {kind!r}")


def model_to_dict(model: Model) -> dict:
    """Serialize the scene graph. Definitions reachable from root are emitted once."""
    # Walks only definitions reachable from root via placed instances — an unplaced
    # / orphan definition would not persist. Fine today (every definition is placed);
    # relevant once a future component browser (M7+) allows library-only definitions.
    defs_by_id: dict[int, Definition] = {}
    stack = [model.root]
    while stack:
        d = stack.pop()
        if d.id in defs_by_id:
            continue
        defs_by_id[d.id] = d
        for inst in d.children:
            stack.append(inst.definition)

    definitions = []
    for d in defs_by_id.values():
        definitions.append(
            {
                "id": d.id,
                "name": d.name,
                "is_group": d.is_group,
                "geometry": geometry_to_dict(d.mesh),
                "annotations": [annotation_to_dict(a) for a in d.annotations],
                "children": [
                    {
                        "id": inst.id,
                        "definition_id": inst.definition.id,
                        "transform": [float(x) for x in inst.transform.flatten()],
                        "tag_id": int(inst.tag_id),
                        "name": str(inst.name),
                        "hidden": bool(inst.hidden),
                    }
                    for inst in d.children
                ],
            }
        )

    return {
        "next_def_id": model._next_def_id,
        "next_inst_id": model._next_inst_id,
        "root_id": model.root.id,
        "definitions": definitions,
    }


def model_from_dict(data: dict) -> Model:
    """Rebuild a Model (two-pass: skeleton, then geometry + children)."""
    model = Model()  # throwaway root/libraries get replaced below

    # Pass 1: empty definitions, keyed by id.
    defs_by_id: dict[int, Definition] = {}
    for rec in data["definitions"]:
        d = Definition(int(rec["id"]), str(rec["name"]), bool(rec["is_group"]))
        d.annotations = [annotation_from_dict(r) for r in rec.get("annotations", [])]
        defs_by_id[d.id] = d

    # Pass 2: geometry + child instances.
    for rec in data["definitions"]:
        d = defs_by_id[rec["id"]]
        geometry_from_dict(d.mesh, rec["geometry"])
        for crec in rec["children"]:
            def_id = int(crec["definition_id"])
            if def_id not in defs_by_id:
                raise BermakeFormatError(f"child references unknown definition {def_id}")
            transform = crec["transform"]
            if len(transform) != 16:
                raise BermakeFormatError(f"transform must have 16 numbers, got {len(transform)}")
            inst = Instance(
                int(crec["id"]),
                defs_by_id[def_id],
                np.asarray(transform, dtype=np.float64).reshape(4, 4),
            )
            inst.tag_id = int(crec["tag_id"])
            # M7.3 (schema 4). Read with defaults: a schema-3 file has neither
            # key, and must still open.
            inst.name = str(crec.get("name", ""))
            inst.hidden = bool(crec.get("hidden", False))
            d.children.append(inst)
            inst.definition.instances.append(inst)

    root_id = int(data["root_id"])
    if root_id not in defs_by_id:
        raise BermakeFormatError(f"root_id {root_id} not found among definitions")
    model.root = defs_by_id[root_id]
    model.active_path = []
    model._next_def_id = int(data["next_def_id"])
    model._next_inst_id = int(data["next_inst_id"])

    max_ann_id = -1
    for d in defs_by_id.values():
        for ann in d.annotations:
            max_ann_id = max(max_ann_id, ann.id)
    model._next_annotation_id = max_ann_id + 1

    return model


def render_style_to_dict(style: RenderStyle) -> dict:
    """Serialize the document's render style (face style, X-Ray, Color-by-Tag).

    Every field of RenderStyle belongs here: a display mode the user turned on
    and then saved must be on again when the document reopens. color_by_tag
    (M7.5a) was missed on its first pass, so a document saved in the mode came
    back with it off.
    """
    return {
        "face_style": style.face_style.name,
        "xray": bool(style.xray),
        "color_by_tag": bool(style.color_by_tag),
    }


def render_style_from_dict(d: dict | None) -> RenderStyle:
    """Rebuild a RenderStyle; missing/empty data yields the default (SHADED).

    Each optional key is read with a default, so a schema-5 file written before
    color_by_tag was persisted (and every older file) still loads, with the
    mode off.
    """
    if not d:
        return RenderStyle()
    return RenderStyle(
        face_style=FaceStyle[d["face_style"]],
        xray=bool(d.get("xray", False)),
        color_by_tag=bool(d.get("color_by_tag", False)),
    )


def environment_to_dict(env: Environment) -> dict:
    """Serialize the document's environment (background, sky, ground, ink).

    Every field of Environment belongs here, for the reason
    render_style_to_dict's docstring gives: a look the user chose and then
    saved must come back when the document reopens. Colours go out as lists
    because JSON has no tuple.
    """
    return {
        "background": list(env.background),
        "sky_enabled": bool(env.sky_enabled),
        "sky_color": list(env.sky_color),
        "ground_enabled": bool(env.ground_enabled),
        "ground_color": list(env.ground_color),
        "ground_opacity": float(env.ground_opacity),
        "edge_color": list(env.edge_color),
        "grid_color": list(env.grid_color),
        "grid_centerline_color": list(env.grid_centerline_color),
    }


def _environment_color(value, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    """One colour triple from stored data, or `fallback` when absent.

    A stored value of the wrong length or type raises ValueError or TypeError,
    which document_from_dict normalizes into BermakeFormatError.

    Each channel is clamped to [0, 1] for the same reason _environment_opacity
    clamps: GL clamps these values silently and at a different stage (upload
    versus the shader's own mix()), so an out-of-range channel from a
    hand-edited file would look like a renderer defect instead of a bad file.
    """
    if value is None:
        return fallback
    r, g, b = (min(1.0, max(0.0, float(x))) for x in value)
    return (r, g, b)


def _environment_opacity(value, fallback: float) -> float:
    """Ground opacity from stored data, or `fallback` when absent, clamped to
    [0, 1].

    GLSL mix() extrapolates outside [0, 1] rather than clamping, so an
    out-of-range value from a hand-edited document would drive the ground colour
    past the sky and look like a renderer defect instead of a bad file.

    An explicit JSON null reaches here as None, exactly like every colour
    field passed to _environment_color, and is defaulted the same way rather
    than reaching float(None) and raising.
    """
    if value is None:
        return fallback
    return min(1.0, max(0.0, float(value)))


def environment_from_dict(d: dict | None) -> Environment:
    """Rebuild an Environment; missing or empty data yields LEGACY_ENVIRONMENT.

    Studio, not the DocumentSettings default (spec D5): every file written
    before M7.7 was authored against the dark background, so anchoring the
    fallback there is what makes an old document reopen unchanged. A brand-new
    document gets DEFAULT_ENVIRONMENT from DocumentSettings instead.

    Each key is read with the legacy value as its fallback, matching
    render_style_from_dict's per-key defaulting, so a partially written document
    still loads. That "matching" does not extend to the falsy guard below: the
    type check runs first, on purpose, so every non-dict value raises, not just
    the truthy ones.

    The isinstance guard is load-bearing. document_from_dict catches KeyError,
    TypeError, ValueError and IndexError; without this, a JSON string here would
    reach `.get` and raise AttributeError, which that handler does not catch and
    no caller expects (#116). It runs before the `not d` check so that a falsy
    non-dict such as "", 0, False or [] raises too, rather than silently being
    treated the same as an absent key.
    """
    if d is not None and not isinstance(d, dict):
        raise TypeError(f"'environment' must be an object, got {type(d).__name__}")
    if not d:
        return LEGACY_ENVIRONMENT
    base = LEGACY_ENVIRONMENT
    return Environment(
        background=_environment_color(d.get("background"), base.background),
        sky_enabled=bool(d.get("sky_enabled", base.sky_enabled)),
        sky_color=_environment_color(d.get("sky_color"), base.sky_color),
        ground_enabled=bool(d.get("ground_enabled", base.ground_enabled)),
        ground_color=_environment_color(d.get("ground_color"), base.ground_color),
        ground_opacity=_environment_opacity(d.get("ground_opacity"), base.ground_opacity),
        edge_color=_environment_color(d.get("edge_color"), base.edge_color),
        grid_color=_environment_color(d.get("grid_color"), base.grid_color),
        grid_centerline_color=_environment_color(
            d.get("grid_centerline_color"), base.grid_centerline_color
        ),
    )


@dataclass(frozen=True)
class CameraState:
    """Snapshot of the viewport Camera's user-facing state (not aspect/near/far,
    which are re-derived from the viewport on load)."""

    position: tuple
    target: tuple
    up: tuple
    fov_y_deg: float

    @classmethod
    def from_camera(cls, cam) -> CameraState:
        return cls(
            position=tuple(float(x) for x in cam.position),
            target=tuple(float(x) for x in cam.target),
            up=tuple(float(x) for x in cam.up),
            fov_y_deg=float(cam.fov_y_deg),
        )

    def to_dict(self) -> dict:
        return {
            "position": list(self.position),
            "target": list(self.target),
            "up": list(self.up),
            "fov_y_deg": self.fov_y_deg,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CameraState:
        return cls(
            position=tuple(float(x) for x in d["position"]),
            target=tuple(float(x) for x in d["target"]),
            up=tuple(float(x) for x in d["up"]),
            fov_y_deg=float(d["fov_y_deg"]),
        )

    def apply_to(self, cam) -> None:
        cam.position = np.array(self.position, dtype=np.float32)
        cam.target = np.array(self.target, dtype=np.float32)
        cam.up = np.array(self.up, dtype=np.float32)
        cam.fov_y_deg = float(self.fov_y_deg)


class LoadedDocument(NamedTuple):
    """Result of loading a .berm document: model + camera + units + style + environment."""

    model: Model
    camera_state: CameraState
    units: Units
    style: RenderStyle
    environment: Environment


def document_to_dict(model: Model, camera, doc, render_style) -> dict:
    """Serialize the top-level document: units, camera, libraries, scenes, style, model."""
    return {
        "units": units_to_dict(doc.units),
        "camera": CameraState.from_camera(camera).to_dict(),
        "materials": {"next_id": model.materials.next_id, "items": model.materials.to_records()},
        "textures": {"next_id": model.textures.next_id, "items": model.textures.to_records()},
        "tags": {"next_id": model.tags.next_id, "items": model.tags.to_records()},
        "scenes": {"next_id": model.views.next_id, "items": model.views.to_records()},
        "style": render_style_to_dict(render_style),
        "environment": environment_to_dict(doc.environment),
        "model": model_to_dict(model),
    }


def document_from_dict(data: dict, blobs: dict[int, bytes] | None = None) -> LoadedDocument:
    """Rebuild a LoadedDocument. Any structural malformation anywhere in the
    document (including in nested geometry/model data) is normalized into
    BermakeFormatError — the only exception callers need to catch.

    `blobs` carries texture bytes the container stored as sibling entries,
    keyed by texture id; `.get("textures", {})` below is what lets a schema 5
    file (no `"textures"` key at all) still load, and also accepts a bare
    list of records (no `next_id`) for a hand-written document.
    """
    from bermake.views.view_library import ViewLibrary  # function-level: breaks import cycle

    try:
        model = model_from_dict(data["model"])
        model.materials = MaterialLibrary.from_records(
            data["materials"]["items"], data["materials"]["next_id"]
        )
        textures = data.get("textures", {})
        # A schema-6 file with no textures wrote {} in place of {"items": [...],
        # "next_id": n} above, and a bare list is also accepted (spec: "keep the
        # .get default so a hand-written bare list still loads").
        tex_items = textures.get("items", []) if isinstance(textures, dict) else textures
        tex_next_id = textures.get("next_id") if isinstance(textures, dict) else None
        model.textures = TextureLibrary.from_records(tex_items, blobs or {}, tex_next_id)
        model.tags = TagLibrary.from_records(data["tags"]["items"], data["tags"]["next_id"])
        scenes = data.get("scenes", {})
        model.views = ViewLibrary.from_records(scenes.get("items", []), scenes.get("next_id", 0))
        camera_state = CameraState.from_dict(data["camera"])
        units = units_from_dict(data["units"])
        style = render_style_from_dict(data.get("style"))
        environment = environment_from_dict(data.get("environment"))
    except (KeyError, TypeError, ValueError, IndexError) as e:
        raise BermakeFormatError(f"malformed document: {e}") from e
    return LoadedDocument(
        model=model,
        camera_state=camera_state,
        units=units,
        style=style,
        environment=environment,
    )
