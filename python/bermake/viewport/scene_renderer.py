"""Owns GL resources for the M2 scene: grid + axes + user geometry + tool overlay.

Lifecycle is driven by QOpenGLWidget:
  initialize_gl() -> first paintGL() call sets up VBOs and shader programs.
  resize(w, h)    -> called from resizeGL.
  render(camera, model, tool_overlay, selection) -> called from paintGL each frame.
"""

from __future__ import annotations

import ctypes
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from importlib.resources import files
from typing import NamedTuple

import numpy as np
from OpenGL import GL

from bermake.geometry.transforms import apply_mat, is_identity_transform
from bermake.scene.scene import Side
from bermake.viewport.camera import Camera
from bermake.viewport.environment import DEFAULT_ENVIRONMENT, Environment, environment_pass_needed
from bermake.viewport.face_batches import BatchPlan, FaceBatch, plan_face_batches
from bermake.viewport.render_style import (
    BACK_DEFAULT_COLOR,
    FACE_STYLE_TABLE,
    FaceShading,
    PhongMaterial,
    RenderStyle,
    ResolvedFacePass,
    phong_material_for,
    resolve_face_pass,
)
from bermake.viewport.snap_engine import SnapKind
from bermake.viewport.texture_cache import TextureCache
from bermake.viewport.translucency import (
    order_back_to_front,
    transform_points,
    triangle_centroids,
    triangle_order_to_vertex_order,
)
from bermake.viewport.uv_projection import apply_placements, plane_bases, project_onto_bases


def definition_is_dimmed(definition, model) -> bool:
    """True when `definition` should render dimmed (recede) — i.e. you are
    inside a group (active_path is non-empty) and this definition is not the
    active editing context. At the root context, nothing is dimmed.

    Thin wrapper kept for back-compat with existing callers/tests; the single
    source of truth is now Model.definition_is_dimmed (#95), so annotations
    (draw_plan.collect_annotation_plans) dim by exactly the same rule as
    geometry."""
    return model.definition_is_dimmed(definition)


# --- AABB helpers (Task 15) -------------------------------------------------


def aabb_world_edges(lo, hi, world_transform) -> np.ndarray:
    """Return (24, 3) float32 array of 12 AABB edge endpoint pairs in world space.

    Builds the 8 corners of the axis-aligned box [lo, hi] and transforms each
    with ``world_transform`` using ``apply_mat``.  The 12 edges are returned as
    24 endpoints (two per edge, interleaved) suitable for GL_LINES.

    Args:
        lo: (3,) array-like — minimum corner in local space.
        hi: (3,) array-like — maximum corner in local space.
        world_transform: (4, 4) array-like — local-to-world matrix.

    Returns:
        np.ndarray of shape (24, 3), dtype float32.
    """
    from bermake.geometry.transforms import apply_mat

    lx, ly, lz = float(lo[0]), float(lo[1]), float(lo[2])
    hx, hy, hz = float(hi[0]), float(hi[1]), float(hi[2])

    # 8 corners of the box
    corners = np.array(
        [
            [lx, ly, lz],  # 0 — low-low-low
            [hx, ly, lz],  # 1 — high-low-low
            [hx, hy, lz],  # 2 — high-high-low
            [lx, hy, lz],  # 3 — low-high-low
            [lx, ly, hz],  # 4 — low-low-high
            [hx, ly, hz],  # 5 — high-low-high
            [hx, hy, hz],  # 6 — high-high-high
            [lx, hy, hz],  # 7 — low-high-high
        ],
        dtype=np.float64,
    )

    # Transform all corners at once
    world_corners = apply_mat(corners, world_transform)  # (8, 3) float32

    # 12 edges — each edge is a pair of corner indices
    _edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),  # bottom face
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),  # top face
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # vertical pillars
    ]

    out = np.empty((24, 3), dtype=np.float32)
    for i, (a, b) in enumerate(_edges):
        out[2 * i] = world_corners[a]
        out[2 * i + 1] = world_corners[b]
    return out


# --- Constants for the scene -----------------------------------------------

_GRID_HALF_EXTENT = 5.0  # meters, so grid is 10x10
_GRID_SPACING = 1.0

_AXIS_LENGTH = 5.0
_AXIS_X_COLOR = (0.90, 0.20, 0.20)
_AXIS_Y_COLOR = (0.20, 0.90, 0.20)
_AXIS_Z_COLOR = (0.20, 0.40, 0.90)

# Phong material + light — used for user geometry.
_LIGHT_DIR = (-1.0, +1.0, -2.0)
_LIGHT_COLOR = (1.00, 0.97, 0.92)
_MATERIAL_AMBIENT = (0.40, 0.40, 0.42)
_MATERIAL_DIFFUSE = (0.65, 0.65, 0.70)
_MATERIAL_SPECULAR = (0.10, 0.10, 0.10)
_MATERIAL_SHININESS = 16.0
_DEFAULT_MATERIAL = PhongMaterial(
    ambient=_MATERIAL_AMBIENT,
    diffuse=_MATERIAL_DIFFUSE,
    specular=_MATERIAL_SPECULAR,
    shininess=_MATERIAL_SHININESS,
)

# The background is no longer a constant: it is Environment.background, held by
# DocumentSettings and reached through SceneRenderer._environment. Studio's
# value is the one this line used to hold (see viewport/environment.py).

# Edge / overlay colors (per-vertex, packed into the VBO alongside positions).
_SELECTION_FILL_COLOR = (0.20, 0.50, 0.95, 0.25)  # selected faces (blue, 25% alpha)
_SELECTION_EDGE_COLOR = (0.20, 0.55, 1.00)  # selected edges (bright blue)

# Task 15 — dim pass + instance bbox colors.
_DIM_AMBIENT = (0.30, 0.30, 0.31)  # desaturated ambient for dimmed definitions
_DIM_DIFFUSE = (0.40, 0.40, 0.42)  # desaturated diffuse for dimmed definitions
_DIM_ALPHA_BLEND = 0.35  # alpha for dimmed geometry (blended toward bg)
# M7.5b (#107, refined): floor under the dim*material_alpha product so a
# deeply translucent material dimmed to near-zero alpha does not vanish.
_DIM_ALPHA_FLOOR = 0.25
_INSTANCE_BBOX_COLOR = (0.30, 0.55, 0.95)  # selection-blue bbox for selected instances
_INSTANCE_BBOX_WIDTH = 2.0
# Uniform names looked up once per program in initialize_gl().
_PHONG_UNIFORMS = (
    "u_view",
    "u_projection",
    "u_model",
    "u_camera_pos",
    "u_light_dir",
    "u_light_color",
    "u_material_ambient",
    "u_material_diffuse",
    "u_material_specular",
    "u_material_shininess",
    "u_alpha",
    # M7.5a Task 6 — the back side's material set, selected in the fragment
    # shader by gl_FrontFacing. Must stay in lockstep with phong.frag's
    # declarations; tests/test_two_sided_shading.py asserts the two agree.
    "u_material_ambient_back",
    "u_material_diffuse_back",
    "u_material_specular_back",
    "u_material_shininess_back",
    "u_alpha_back",
    # M7.5b Task 6 — the material's image, per side. The samplers carry a unit
    # index and are set once after linking (_bind_sampler_units); the two flags
    # are per-batch draw state.
    "u_texture",
    "u_texture_back",
    "u_has_texture",
    "u_has_texture_back",
)
# The subset of _PHONG_UNIFORMS that is program state rather than per-draw
# state. tests/test_two_sided_shading.py exempts exactly these from its
# "every cached uniform is written during a face draw" guard, and
# tests/test_textured_shading.py guards them at their own link-time setter.
_PHONG_LINK_TIME_UNIFORMS = ("u_texture", "u_texture_back")

# Two texture units, because both sides of a face are shaded in one draw call
# and each side can carry its own image. One unit would make the back silently
# sample the front's texture -- a bug that looks perfectly correct from the
# front. The GL enums are derived once at import: the draw path must not do
# arithmetic on a GL constant, since a recording stand-in for the GL module
# (used by several renderer tests) supplies callables, not ints.
_TEXTURE_UNIT_FRONT = 0
_TEXTURE_UNIT_BACK = 1
_TEXTURE_UNIT_ENUM = (
    GL.GL_TEXTURE0 + _TEXTURE_UNIT_FRONT,
    GL.GL_TEXTURE0 + _TEXTURE_UNIT_BACK,
)
_LINE_UNIFORMS = ("u_view", "u_projection")
_GHOST_FILL_UNIFORMS = ("u_view", "u_projection", "u_color")
_ENVIRONMENT_UNIFORMS = (
    "u_inv_view_proj",
    "u_camera_pos",
    "u_background",
    "u_sky_color",
    "u_ground_color",
    "u_ground_opacity",
    "u_sky_enabled",
    "u_ground_enabled",
)


def _empty_batch_plan() -> BatchPlan:
    """The plan for a definition with no triangles."""
    return plan_face_batches([], [])


# The face VBO's interleaved layout: (pos.xyz, normal.xyz, front_uv.xy,
# back_uv.xy) float32 per vertex. _alloc_def_buffers' vertex-attribute stride
# and _sort_translucent_slice's glBufferSubData byte offset are the same number,
# so they share one constant rather than each hard-coding 40 and drifting apart.
#
# M7.5b: every face vertex carries BOTH sides' UVs whether or not anything in
# the document is textured. Both, because spec D7 and 1.4 make front and back
# placement independent while two-sided shading draws both sides in one pass
# selected by gl_FrontFacing — a single UV attribute would make back placement
# physically unrepresentable, leaving Task 8's command, Task 10's fields and
# Task 12's drag with a dead `side` control. Unconditionally, because the
# alternative — a vertex format chosen per definition — puts a branch in the
# renderer's hottest path, so the untextured case pays the 16 bytes deliberately.
_FACE_VERTEX_FLOATS = 10
_FACE_VERTEX_BYTES = _FACE_VERTEX_FLOATS * 4


def _empty_face_interleaved() -> np.ndarray:
    return np.zeros((0, _FACE_VERTEX_FLOATS), dtype=np.float32)


@dataclass
class _DefBuffers:
    """Per-definition GL buffer handles and vertex counts."""

    face_vao: int = 0
    face_vbo: int = 0
    face_count: int = 0  # number of triangle vertices
    edge_vao: int = 0
    edge_vbo: int = 0
    edge_count: int = 0  # number of line-segment vertices
    # The batch plan the face VBO was uploaded under: its vertex_order is
    # already baked into the buffer, so `opaque` / `translucent` /
    # `translucent_first` index straight into it.
    plan: BatchPlan = field(default_factory=_empty_batch_plan)
    # The translucent ids the plan above was computed under. Editing a
    # material's alpha dirties no mesh, so this is the only thing that can tell
    # the renderer its opaque/translucent partition has gone stale (spec 1.7).
    translucent_ids: frozenset[int] = frozenset()
    # M7.5b: the same asymmetry one level down. UVs are baked into the face VBO
    # from each material's texture_size, which lives on the MaterialLibrary, so
    # resizing a texture dirties no mesh either. See uv_material_key.
    uv_key: tuple = ()

    # M7.7: the same asymmetry once more. The edge colour is tiled into the edge
    # VBO from the document's environment, which no mesh dirty flag covers, so
    # switching environment has to be able to invalidate this buffer. None means
    # "never uploaded", which reads as stale.
    edge_color: tuple[float, float, float] | None = None

    # --- Task 7: state the translucent pass re-sorts each frame -------------
    #
    # The interleaved vertex data in its upload order (already permuted by
    # plan.vertex_order), so the translucent suffix can be re-permuted from a
    # stable base every time the view changes rather than accumulating
    # permutations. Kept ONLY for definitions that actually have translucent
    # faces — _sort_translucent_slice is the sole reader, and holding a CPU
    # copy of every opaque definition's faces alongside its VBO would roughly
    # double scene face memory for a pass those definitions never enter.
    face_interleaved: np.ndarray = field(default_factory=_empty_face_interleaved)
    # Local-space centroid of each translucent triangle, in upload order.
    translucent_centroids: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float64)
    )
    # This definition's own sort key for pass 2 — the mean of the above.
    translucent_local_centroid: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    # (front, back) material ids per translucent triangle, in upload order, so
    # the depth-sorted suffix can be re-cut into batches that still name the
    # right materials.
    translucent_pairs: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int64))
    # The batches that currently describe the face VBO's translucent suffix.
    # Equal to plan.translucent until the first sort, then re-cut by
    # rebuild_translucent_batches to follow the permuted order.
    translucent_draw_batches: list[FaceBatch] = field(default_factory=list)
    # (camera_pos, world_transform) the suffix was last sorted for; None when
    # it has never been sorted, or when a re-upload invalidated the sort.
    translucent_sort_key: tuple | None = None

    def release(self) -> None:
        """Delete this definition's GL objects. Guarded so a zero handle
        (never-uploaded / already-released / headless test stand-in) never
        reaches a GL call — safe to call without a current context in that
        case, and correct GL hygiene in all cases.

        The FaceBatch entries in `plan` are metadata slices (front_material_id,
        back_material_id, first, count) into face_vbo — they hold no GL
        handles of their own, so there is nothing per-batch to release.
        """
        if self.face_vao:
            GL.glDeleteVertexArrays(1, [self.face_vao])
        if self.face_vbo:
            GL.glDeleteBuffers(1, [self.face_vbo])
        if self.edge_vao:
            GL.glDeleteVertexArrays(1, [self.edge_vao])
        if self.edge_vbo:
            GL.glDeleteBuffers(1, [self.edge_vbo])


# --- M7.5a Task 6: two-sided material resolution ----------------------------
#
# Module-level and GL-free on purpose: the resolution logic is where the
# per-side behaviour actually lives, and the fragment shader it feeds cannot
# be unit-tested. Keeping this importable without a GL context or a
# QApplication is what makes the shading testable at all.


# --- M7.5b Task 4: the UV vertex attribute ----------------------------------
#
# GL-free and module-level for the same reason as the block above: a face's UVs
# are computed entirely on the CPU at upload time, and a misaligned UV array
# produces scrambled texturing rather than an exception, so it has to be
# testable without a GL context or a QApplication.


def _face_corner_runs(face_ids: np.ndarray) -> list[tuple[int, int, int]]:
    """(face_id, first_corner, corner_count) for each face, in walk order.

    face_ids comes from Scene.face_triangle_face_ids, which repeats each id by
    the face's triangle count in next_live_face order — so a face's triangles
    are always one contiguous run. Reading the runs off directly keeps the walk
    order (which is what the position buffer is in) and is linear, where
    de-duplicating the ids and counting matches per face would be quadratic.
    """
    n = int(face_ids.shape[0])
    if n == 0:
        return []
    cuts = np.flatnonzero(face_ids[1:] != face_ids[:-1]) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [n]))
    return [
        (int(face_ids[s]), int(s) * 3, (int(e) - int(s)) * 3)
        for s, e in zip(starts, ends, strict=True)
    ]


class _FaceUvGeometry(NamedTuple):
    """The half of the UV bake that does not depend on which side is baked.

    The face walk, each face's plane basis and each face's centroid are the same
    for the front and the back; only the texture_size and placement gathers
    differ. Computing this once is what stops baking both sides costing twice
    what baking one does.

    `positions` and `origins` are per corner, and `counts` says how many corners
    each entry of `face_ids` owns, which is how a per-face value is gathered onto
    its corners.
    """

    face_ids: list[int]
    counts: np.ndarray
    positions: np.ndarray
    origins: np.ndarray
    u_axes: np.ndarray
    v_axes: np.ndarray


def _face_uv_geometry(scene, face_buffer=None) -> _FaceUvGeometry | None:
    """Build the side-independent context, or None when there is nothing to bake."""
    positions, normals = scene.face_triangle_buffer() if face_buffer is None else face_buffer
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if positions.shape[0] == 0:
        return None
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)

    runs = _face_corner_runs(scene.face_triangle_face_ids())
    face_ids = [r[0] for r in runs]
    counts = np.array([r[2] for r in runs], dtype=np.int64)

    # The basis comes from the triangle buffer's normals, NOT from
    # Scene.face_normal. This runs for every face of every upload, and
    # face_normal is a per-face Python routine that walks the loop and can
    # raise; anything it raises propagates out through paint() and blanks the
    # whole viewport, so one face would cost the entire document. The buffer's
    # normals block is already computed, already correct, and already here.
    #
    # face_normal used to raise on a merely awkward loop — first three vertices
    # collinear, which is what splitting an edge leaves behind — which made this
    # urgent. That defect is fixed (issue #110: it uses Newell's method over the
    # whole loop now, and only a zero-area face raises). The reason above does
    # not depend on that and still stands: keep reading the buffer.
    u_axes, v_axes = plane_bases(normals)

    centers = np.array([scene.face_center(f) for f in face_ids], dtype=np.float64)
    return _FaceUvGeometry(
        face_ids=face_ids,
        counts=counts,
        positions=positions,
        origins=np.repeat(centers, counts, axis=0),
        u_axes=u_axes,
        v_axes=v_axes,
    )


class _OverlayCache:
    """The two lookups the stored-UV overlay needs, built lazily and shared.

    Both sides want the same corner-span map and the same loop indices, and
    neither side wants either unless it has a stored face, so each is built
    on first use and then reused. Building them per side is part of what made
    stored UVs cost what they did (#117).
    """

    __slots__ = ("_geom", "_loops", "_scene", "_spans")

    def __init__(self, scene, geom: _FaceUvGeometry) -> None:
        self._scene = scene
        self._geom = geom
        self._spans: dict[int, tuple[int, int]] | None = None
        self._loops: np.ndarray | None = None

    def spans(self) -> dict[int, tuple[int, int]]:
        """The whole map, not one lookup at a time: the caller reads it once
        per stored face, and a bound-method call per face is measurable at
        the counts a fully imported model reaches."""
        if self._spans is None:
            geom = self._geom
            starts = np.concatenate(([0], np.cumsum(geom.counts)[:-1]))
            self._spans = {
                int(f): (int(start), int(count))
                for f, start, count in zip(geom.face_ids, starts, geom.counts, strict=True)
            }
        return self._spans

    def prefers_whole_walk(self, stored_count: int) -> bool:
        """Is one walk over every face cheaper than `stored_count` per-face calls?

        The whole-scene array is a single Python walk of every live face; a
        per-face call repeats that walk's body for one face with a call's
        overhead on top. Measured on a 1,600-quad definition, the whole walk
        costs about 1.4 ms and the per-face route about 1.4 us per face, so
        they cross near two thirds. Half is the rule because it is a rule
        someone can check, and the arithmetic either side of it is flat.
        """
        return stored_count * 2 >= len(self._geom.face_ids)

    def loop_indices(self) -> np.ndarray:
        if self._loops is None:
            self._loops = self._scene.face_triangle_loop_indices()
        return self._loops


def _overlay_stored_uvs(
    scene, side: Side, geom: _FaceUvGeometry, uvs: np.ndarray, cache: _OverlayCache
) -> np.ndarray:
    """Replace projected UVs with stored ones, for the faces that have them.

    Stored UVs are parallel to a face's boundary loop; `uvs` is per triangle
    corner. face_triangle_loop_indices is the bridge: corner i of a stored face
    takes stored[loop_indices[i]].

    Iterates the sidecar's entries rather than every face, so a definition with
    three stored faces among ten thousand touches three. The loop indices come
    per face for the same reason: the whole-scene array walks every live face
    in Python to produce entries only these faces read (#117).
    """
    stored_here = [f for f, s in scene.faces_with_uvs() if s is side]
    if not stored_here:
        return uvs
    whole = cache.loop_indices() if cache.prefers_whole_walk(len(stored_here)) else None
    spans = cache.spans()
    per_face_loop_indices = scene.face_loop_indices

    out = uvs
    for face_id in stored_here:
        where = spans.get(int(face_id))
        if where is None:
            continue  # stored on a face this definition's buffer does not carry
        first, count = where
        arr = scene.face_uvs(face_id, side)
        idx = whole[first : first + count] if whole is not None else per_face_loop_indices(face_id)
        if int(idx.max()) >= arr.shape[0]:
            # Defensive: a stored array shorter than the face's loop, which the
            # public API (set_face_uvs' length validation) prevents.
            continue
        if out is uvs:
            out = uvs.copy()
        out[first : first + count] = arr[idx]
    return out


def _side_uvs(
    scene, materials, side: Side, geom: _FaceUvGeometry, cache: _OverlayCache | None = None
) -> np.ndarray:
    """One side's (3T, 2) UVs from the shared geometry.

    `cache` lets a two-sided bake share the overlay's lookups; a single-sided
    caller can leave it out and get its own.
    """
    if cache is None:
        cache = _OverlayCache(scene, geom)
    if materials is None:
        sizes = np.ones((geom.positions.shape[0], 2), dtype=np.float64)
    else:
        per_face = np.array(
            [materials.get(scene.face_material(f, side)).texture_size for f in geom.face_ids],
            dtype=np.float64,
        )
        sizes = np.repeat(per_face, geom.counts, axis=0)

    uvs = project_onto_bases(geom.positions, geom.u_axes, geom.v_axes, geom.origins, sizes)

    uvs = _overlay_stored_uvs(scene, side, geom, uvs, cache)

    # The placement sidecars hold only adjusted faces, so a definition nobody has
    # placed a texture on skips the gather entirely rather than reading the
    # identity back for every face.
    if not any(s is side for _, s in scene.faces_with_placement()):
        return uvs

    placements = [scene.face_placement(f, side) for f in geom.face_ids]
    return apply_placements(
        uvs,
        np.repeat(np.array([(p.offset_u, p.offset_v) for p in placements]), geom.counts, axis=0),
        np.repeat(np.array([p.scale for p in placements]), geom.counts),
        np.repeat(np.array([p.rotation for p in placements]), geom.counts),
    )


def build_face_uvs(scene, model, side: Side = Side.FRONT, face_buffer=None) -> np.ndarray:
    """(3T, 2) per-corner UVs for one side of a definition's faces.

    Walks faces in the same next_live_face order face_triangle_buffer uses, so
    the result is aligned corner for corner with the position buffer. Faces with
    no texture still get UVs; they are simply never sampled.

    `face_buffer` is an already-fetched (positions, normals) pair. Baking one
    side at a time rebuilds the shared geometry each call; _upload_definition
    needs both sides and uses build_face_uvs_both_sides instead.
    """
    geom = _face_uv_geometry(scene, face_buffer)
    if geom is None:
        return np.zeros((0, 2), dtype=np.float32)
    return _side_uvs(scene, getattr(model, "materials", None), side, geom)


def build_face_uvs_both_sides(scene, model, face_buffer=None) -> tuple[np.ndarray, np.ndarray]:
    """(front, back) per-corner UVs, sharing one pass over the faces."""
    geom = _face_uv_geometry(scene, face_buffer)
    if geom is None:
        empty = np.zeros((0, 2), dtype=np.float32)
        return empty, empty
    materials = getattr(model, "materials", None)
    cache = _OverlayCache(scene, geom)
    return (
        _side_uvs(scene, materials, Side.FRONT, geom, cache),
        _side_uvs(scene, materials, Side.BACK, geom, cache),
    )


def uv_key_needs_bake(key: tuple) -> bool:
    """Can anything the definition behind `key` sample a UV?

    The per-corner UV bake is pure cost where nothing can read it. M7.5b
    measured ~61 ms added to every re-upload of a 9,600-face definition in a
    document with no image at all, and gated on the whole MaterialLibrary.
    That gate was too coarse: one image anywhere made every definition pay,
    including definitions with nothing textured, so the ordinary document
    with some textured surfaces and many untextured ones got no relief at
    all (#112, about 95 ms against 33.8 ms per upload).

    `key` comes from uv_material_key, which already names exactly the
    materials this definition's own faces carry, on both sides, and which
    _upload_definition has to compute anyway. Asking it instead costs
    nothing and scopes the answer to one definition.

    The invalidation argument is the one that made the coarse gate safe, and
    it is unchanged: a texture reaches a fragment only through a material's
    texture_id, and texture_id is what the key snapshots, so assigning one
    to a material this definition carries breaks uv_key_still_matches and
    forces a re-upload with real UVs. A material entering the scene can only
    happen by painting a face, which dirties the mesh and re-uploads anyway.

    This gates the CPU bake ONLY. The vertex format stays unconditionally 10
    floats (spec 1.5 as superseded during execution): a per-definition layout
    branch in the renderer's hot path is expressly not wanted, and the memory
    cost of the two zero-filled UV blocks is accepted.
    """
    return any(texture_id is not None for _, texture_id, _ in key)


def uv_material_key(scene, model) -> tuple:
    """The material state the baked UVs depend on, for change detection.

    UVs are baked into the vertex buffer, but texture_id and texture_size live
    on the MaterialLibrary rather than the Scene, so editing either dirties no
    mesh. Without this snapshot the buffer would keep tiling at the old size
    and the edit would look inert — the same asymmetry that spec 1.7's
    translucent_ids exists for.

    Spans BOTH sides, because both sides' UVs are baked into the same buffer. A
    key that only walked the front would never name a back-only material, and
    resizing its texture would go on tiling at the old size for ever.

    Only texture_id and texture_size matter: a colour, alpha, metallic or
    roughness edit changes uniforms read per batch at draw time and needs no
    re-upload. Including them would re-upload every definition on every colour
    tweak, rebuilding visible geometry while the user drags a colour picker.
    """
    return uv_material_key_for(
        scene.face_triangle_materials(Side.FRONT),
        scene.face_triangle_materials(Side.BACK),
        model,
    )


def uv_material_key_for(front_materials, back_materials, model) -> tuple:
    """uv_material_key from per-triangle material arrays already in hand.

    _upload_definition fetches both arrays for plan_face_batches regardless,
    and face_triangle_materials measures ~3.5 ms on a 10k-face definition, so
    letting the key re-fetch them doubled that walk for nothing.
    """
    materials = getattr(model, "materials", None)
    if materials is None:
        return ()
    ids: set[int] = set()
    for arr in (front_materials, back_materials):
        ids.update(int(m) for m in arr.tolist())
    key = []
    for mid in sorted(ids):
        mat = materials.get(mid)
        key.append((mid, mat.texture_id, tuple(mat.texture_size)))
    return tuple(key)


def uv_key_still_matches(key: tuple, model) -> bool:
    """Does `key` still describe the library's texture state?

    The per-frame half of uv_material_key, and deliberately not a second
    mechanism: it re-reads the very materials the key already names. Re-deriving
    the whole key each frame would mean walking every triangle of every visible
    definition on every frame — face_triangle_materials measures ~3.5 ms for a
    10k-face definition, a fifth of a 60 fps budget for one definition alone.

    A material entering the scene that the key does not name can only happen by
    painting a face, which marks the scene render-dirty and re-uploads anyway,
    so scanning only the named materials misses nothing.
    """
    materials = getattr(model, "materials", None)
    if materials is None:
        return not key
    for mid, texture_id, texture_size in key:
        mat = materials.get(mid)
        if mat.texture_id != texture_id or tuple(mat.texture_size) != texture_size:
            return False
    return True


def _edge_buffer_is_stale(buf: _DefBuffers, edge_color: tuple[float, float, float]) -> bool:
    """True when this definition's edge VBO was uploaded under a different ink.

    Pulled out as a function rather than inlined into the _ensure_buffers
    condition so it can be tested headlessly, including the zero-edge case that
    otherwise re-uploads every frame forever.
    """
    return buf.edge_color != edge_color


def _translucent_ids(materials, textures) -> frozenset[int]:
    """Ids of every material that must draw in the sorted translucent pass.

    Two ways in, not one. The material's own alpha is the M7.5a rule. M7.5b
    adds a second: a texture with transparent texels. A cutout PNG on an
    alpha-1.0 material is fully opaque by the first test and still belongs in
    the second pass -- drawn opaque it writes depth through its own holes and
    shows the background colour instead of what is behind it.

    `textures` is positional and has no default on purpose: a caller that
    forgot it would quietly drop every cutout back into the opaque pass, which
    looks like nothing happened. Passing None is still allowed and means the
    model has no texture library, so nothing can enter by the second route.

    The Default material (id 0) is excluded from the TEXTURE route, and only
    that route. MaterialLibrary.edit permits editing id 0, but the renderer
    never textures it -- _texture_for_material short-circuits mid == 0, because
    id 0 means "unpainted" and every unpainted back face carries it. Without
    this guard a cutout on Default would put 0 into the set, which would blend
    and stop depth writes for essentially EVERY batch in the model while
    nothing rendered textured at all. Its `alpha < 1.0` behaviour is
    pre-existing M7.5a behaviour and is deliberately left alone.
    """
    if materials is None:
        return frozenset()
    ids: set[int] = set()
    for m in materials.materials():
        if m.is_translucent:
            ids.add(m.id)
            continue
        if m.id == 0 or m.texture_id is None or textures is None:
            continue
        tex = textures.get(m.texture_id)
        if tex is not None and tex.has_transparency:
            ids.add(m.id)
    return frozenset(ids)


def textures_visible(style: RenderStyle, *, tag_color: tuple | None) -> bool:
    """Does this style show material textures?

    A texture is part of a material's own colour contribution, so it applies
    exactly where that colour applies. Hidden Line fills with the background
    colour and Monochrome with MONO_COLOR, both deliberately discarding the
    painted colour; Color-by-Tag replaces the material outright before
    resolution. Multiplying any of those by a texel would put the image back
    into a mode whose entire point is that it has none -- and Hidden Line's
    flat fill would come out modulated by the image rather than flat.

    Deliberately NOT consulted by _translucent_ids: the opaque/translucent
    partition is baked into the vertex buffer, so making it style-dependent
    would re-upload every textured definition on every style toggle. A cutout
    under any of the three therefore still draws in the sorted pass, but with
    nothing sampled its alpha stays 1.0 and it fills solid, which is what the
    style asks for.

    Color-by-Tag is the third of those three and is easy to miss, because it is
    a MODE rather than a face style: `tag_color is not None` turns textures off
    under any face style at all, including a lit one. So a cutout batch under
    Color-by-Tag still carries blend=True and depth_write=False while sampling
    nothing, and two such surfaces overlapping resolve by draw order rather
    than by depth. That is bounded -- the sorted pass already orders the whole
    translucent suffix back to front -- and it is accepted, not overlooked: the
    alternative is making the buffer partition style-dependent, which is the
    re-upload this paragraph exists to refuse.
    """
    return tag_color is None and FACE_STYLE_TABLE[style.face_style].shading is FaceShading.LIT


def _material_terms(materials, mid: int, side: Side) -> tuple[PhongMaterial, float]:
    """(PhongMaterial, alpha) for one side's material id.

    An unpainted FRONT keeps using the hand-tuned _DEFAULT_MATERIAL exactly as
    before, so unpainted front faces look unchanged. An unpainted BACK is new:
    it resolves to the distinct blue-grey BACK_DEFAULT_COLOR (spec D3) so a
    reversed face is obvious on screen.
    """
    if mid != 0 and materials is not None:
        m = materials.get(mid)
        return (
            phong_material_for(m.base_color, metallic=m.metallic, roughness=m.roughness),
            m.alpha,
        )
    if side is Side.BACK:
        return phong_material_for(BACK_DEFAULT_COLOR), 1.0
    return _DEFAULT_MATERIAL, 1.0


def resolve_batch_sides(
    batch: FaceBatch,
    materials,
    render_style: RenderStyle,
    *,
    bg: tuple[float, float, float],
    dimmed: bool,
    translucent_ids: frozenset[int],
    tag_color: tuple[float, float, float] | None = None,
) -> tuple[ResolvedFacePass, ResolvedFacePass]:
    """Resolve one batch into its front and back face passes.

    blend and depth_write are per-batch rather than per-side (correction 4):
    both sides of a face reach the shader in one draw call, and blending and
    the depth mask are draw-call state, so a batch blends if EITHER side is
    translucent and both returned passes carry the same flags.

    `tag_color` is Color-by-Tag's override (None when the mode is off). It
    replaces the MATERIAL both sides resolve from, before resolution, rather
    than patching the resolved result afterwards. That distinction is the whole
    behaviour: a post-hoc diffuse patch leaves the material's own ambient (and,
    for a metal, its specular) reaching the shader, so three faces on one tag
    painted three ways still render three different colours, and it overwrites
    the face style's own diffuse decision -- Hidden Line's deliberate flat
    (0, 0, 0) fill included. Substituting the material first routes the tag
    colour through face_uniforms, so every style keeps its own rule: Hidden
    Line stays a flat background fill, Monochrome stays MONO_COLOR, the dim
    pass still dims, and both sides get an ambient and specular derived from
    the tag colour instead of from whatever was painted.

    Each side keeps its OWN alpha. Alpha is not a colour: a translucent
    material under Color-by-Tag must still blend and still sort into the
    translucent pass, so opacity is deliberately not bypassed.

    `translucent_ids` is the same set the buffer partition was built from (see
    _translucent_ids). It is needed here because a CUTOUT texture is
    translucent in a way no other argument can see: the material's own alpha is
    1.0, so resolve_face_pass would leave blending off, the fragment shader's
    sampled alpha would be written into a buffer that ignores it, and the holes
    would come out solid. Required rather than defaulted, matching
    _translucent_ids' own discipline: a caller that omitted it would get solid
    cutouts and no error, which looks exactly like the feature not existing.
    Pass `frozenset()` to mean "nothing here is translucent".

    `bg` is the document environment's background. Hidden Line fills faces with
    it, so it is the colour of the page rather than of any material. Required
    rather than defaulted for the same reason translucent_ids is: a caller that
    omitted it would get the old dark fill with no error, and resolve_face_pass
    (which this wraps) already requires it.
    """
    front_mat, front_alpha = _material_terms(materials, batch.front_material_id, Side.FRONT)
    back_mat, back_alpha = _material_terms(materials, batch.back_material_id, Side.BACK)
    if tag_color is not None:
        front_mat = back_mat = phong_material_for(tag_color)

    def _resolve(mat: PhongMaterial, alpha: float) -> ResolvedFacePass:
        return resolve_face_pass(
            render_style,
            dimmed=dimmed,
            bg=bg,
            material=mat,
            dim_ambient=_DIM_AMBIENT,
            dim_diffuse=_DIM_DIFFUSE,
            dim_alpha=_DIM_ALPHA_BLEND,
            dim_alpha_floor=_DIM_ALPHA_FLOOR,
            material_alpha=alpha,
        )

    front = _resolve(front_mat, front_alpha)
    back = _resolve(back_mat, back_alpha)
    # Joins the same batch-wide or/and combine as the per-side flags: a
    # material whose texture carries transparency is in translucent_ids but has
    # alpha 1.0, so neither resolved side asked to blend.
    sorted_pass = (
        batch.front_material_id in translucent_ids or batch.back_material_id in translucent_ids
    )
    blend = front.blend or back.blend or sorted_pass
    depth_write = front.depth_write and back.depth_write and not sorted_pass
    return (
        replace(front, blend=blend, depth_write=depth_write),
        replace(back, blend=blend, depth_write=depth_write),
    )


# --- M7.5a Task 11: Color-by-Tag --------------------------------------------
#
# Module-level and GL-free for the same reason as resolve_batch_sides above.
# Tags are per-INSTANCE while a batch is per-definition, so WHICH colour applies
# is decided here, in the render loop, from the traversal's tag id. Applying it
# is resolve_batch_sides' job: the colour has to replace the batch's materials
# before resolution, not patch the resolved result, so it cannot live here.


def traverse_visible_tagged(model):
    """Yield (definition, world, tag_id) once per visible OCCURRENCE.

    The same tree, the same order and the same pruning as
    Model.traverse_visible() -- which already yields one entry per
    occurrence, so a component placed twice appears twice -- with the tag of
    the Instance that placed *this* occurrence carried alongside.

    Per occurrence, not per definition, because a tag is a property of the
    Instance: one definition placed by two instances on two tags must draw
    in two colours, and collapsing that into an id(definition) -> tag_id map
    lets the last instance visited win. Walking the pruned tree (rather than
    all of model.root) matters for the same reason: a hidden instance, or
    one on a hidden tag, must not supply the colour a visible placement
    draws with.

    model.root is placed by no Instance, so it is reported as UNTAGGED_ID.

    Lives here rather than on Model so traverse()/traverse_visible() keep
    their two-tuple shape; test_tag_color.py pins this walk to
    traverse_visible()'s so the repeated pruning rule cannot drift.
    """
    active_ids = {inst.id for inst in model.active_path}
    tags = model.tags

    def walk(definition, world, tag_id):
        yield definition, world, tag_id
        for inst in definition.children:
            if inst.id not in active_ids and (inst.hidden or not tags.is_visible(inst.tag_id)):
                continue
            yield from walk(inst.definition, world @ inst.transform, inst.tag_id)

    yield from walk(model.root, np.eye(4, dtype=np.float64), tags.UNTAGGED_ID)


def resolve_tag_color(
    tag_id: int, tags, render_style: RenderStyle
) -> tuple[float, float, float] | None:
    """The tag-colour override for one occurrence's draw, or None when
    Color-by-Tag is off.

    `tag_id` comes from traverse_visible_tagged, so it is the tag of the
    Instance that placed the occurrence being drawn; this function decides
    only whether the mode is on and what colour results, so it is testable
    without a scene graph or a GL context.

    The colour it returns is handed to resolve_batch_sides as `tag_color`,
    which substitutes it for the batch's materials BEFORE resolution -- see
    that function for why the override cannot be applied afterwards.
    """
    if not render_style.color_by_tag:
        return None
    return tags.get(tag_id).color


# --- M7.5a Task 7: the translucent pass -------------------------------------
#
# Module-level and GL-free for the same reason as the block above: every
# ordering decision the pass makes is testable, and none of the GL calls are.


def order_definitions_for_translucent_pass(entries, *, camera_pos, centroid_of) -> list[int]:
    """Indices of (definition, world) entries, farthest definition first.

    Correction 1: face buffers are per-definition, so a scene-global face sort
    is not available. Faces sort within a definition; definitions sort among
    themselves here. Two interpenetrating translucent groups therefore sort by
    centroid rather than per face.
    """
    if not entries:
        return []
    world_centroids = np.stack(
        [
            transform_points(np.asarray(centroid_of(d), dtype=np.float64).reshape(1, 3), w)[0]
            for d, w in entries
        ]
    )
    return order_back_to_front(world_centroids, camera_pos).tolist()


def rebuild_translucent_batches(sorted_pairs, first_vertex: int) -> list[FaceBatch]:
    """Re-cut the depth-sorted translucent suffix into (front, back) batches.

    `sorted_pairs` is (T, 2) int64 — the (front, back) material id of each
    translucent triangle in the order it now occupies in the VBO. Runs of the
    same pair become one batch.

    This is what makes a whole-suffix depth sort safe. plan_face_batches groups
    the suffix by material pair, so a definition carrying two translucent
    materials has two batches inside it; sorting the suffix by depth moves
    triangles across those boundaries, and drawing with the original ranges
    would shade a triangle with the other material's uniforms. Re-cutting keeps
    the depth order AND the materials, at the cost of fragmentation: materials
    that alternate in depth degenerate to one batch per triangle. That bound is
    accepted rather than capped — a cap would reintroduce the wrong colours.
    """
    pairs = np.asarray(sorted_pairs, dtype=np.int64).reshape(-1, 2)
    if pairs.shape[0] == 0:
        return []
    changed = np.any(pairs[1:] != pairs[:-1], axis=1)
    starts = np.concatenate([np.zeros(1, dtype=np.int64), np.nonzero(changed)[0] + 1])
    ends = np.concatenate([starts[1:], np.array([pairs.shape[0]], dtype=np.int64)])
    return [
        FaceBatch(
            front_material_id=int(pairs[s, 0]),
            back_material_id=int(pairs[s, 1]),
            first=first_vertex + int(s) * 3,
            count=(int(e) - int(s)) * 3,
        )
        for s, e in zip(starts, ends, strict=True)
    ]


def _translucent_pairs(plan: BatchPlan) -> np.ndarray:
    """(T, 2) int64 (front, back) material id per translucent triangle, in the
    upload order `plan.vertex_order` baked into the VBO."""
    total = sum(b.count for b in plan.translucent) // 3
    pairs = np.zeros((total, 2), dtype=np.int64)
    for batch in plan.translucent:
        lo = (batch.first - plan.translucent_first) // 3
        pairs[lo : lo + batch.count // 3] = (batch.front_material_id, batch.back_material_id)
    return pairs


def _reset_translucent_state(buf: _DefBuffers, plan: BatchPlan, interleaved: np.ndarray) -> None:
    """Point `buf`'s translucent-pass state at freshly uploaded vertex data.

    Called from _upload_definition for both the has-triangles and no-triangles
    cases. Every field the pass caches is (re)set together here — in particular
    `translucent_sort_key` is cleared, because the buffer that a surviving key
    described has just been replaced under it. _DefBuffers instances are reused
    across re-uploads, so leaving a stale key would keep the previous frame's
    permutation and batch ranges over entirely different vertex data.
    """
    # Only the translucent pass reads face_interleaved, so an all-opaque
    # definition keeps nothing.
    buf.face_interleaved = interleaved if plan.translucent else _empty_face_interleaved()
    buf.translucent_centroids = triangle_centroids(interleaved[plan.translucent_first :, :3])
    buf.translucent_local_centroid = (
        buf.translucent_centroids.mean(axis=0)
        if buf.translucent_centroids.shape[0]
        else np.zeros(3, dtype=np.float64)
    )
    buf.translucent_pairs = _translucent_pairs(plan)
    buf.translucent_draw_batches = list(plan.translucent)
    buf.translucent_sort_key = None


def _load_shader_source(name: str) -> str:
    return (files("bermake.viewport") / "shaders" / name).read_text(encoding="utf-8")


def _compile_shader(source: str, shader_type: int) -> int:
    shader = GL.glCreateShader(shader_type)
    GL.glShaderSource(shader, source)
    GL.glCompileShader(shader)
    if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
        log = GL.glGetShaderInfoLog(shader).decode("utf-8", errors="replace")
        kind = "vertex" if shader_type == GL.GL_VERTEX_SHADER else "fragment"
        raise RuntimeError(f"{kind} shader compile failed:\n{log}")
    return shader


def _link_program(vert_src: str, frag_src: str) -> int:
    vs = _compile_shader(vert_src, GL.GL_VERTEX_SHADER)
    fs = _compile_shader(frag_src, GL.GL_FRAGMENT_SHADER)
    program = GL.glCreateProgram()
    GL.glAttachShader(program, vs)
    GL.glAttachShader(program, fs)
    GL.glLinkProgram(program)
    if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
        log = GL.glGetProgramInfoLog(program).decode("utf-8", errors="replace")
        raise RuntimeError(f"shader program link failed:\n{log}")
    GL.glDeleteShader(vs)
    GL.glDeleteShader(fs)
    return program


def _cache_uniform_locations(program: int, names: Sequence[str]) -> dict[str, int]:
    """Look up uniform locations once per program. Returned dict is read by
    every subsequent draw to avoid glGetUniformLocation in the hot path."""
    return {name: GL.glGetUniformLocation(program, name) for name in names}


def _build_grid_vertex_array(
    grid_color: tuple[float, float, float],
    centerline_color: tuple[float, float, float],
) -> np.ndarray:
    """Return a (N, 6) float32 array of grid-line vertices: x,y,z, r,g,b.

    N is derived from _GRID_HALF_EXTENT and _GRID_SPACING (44 at the current
    values); reshape(-1, 6) absorbs either changing, so this docstring does
    not hardcode the count.

    The colours are arguments rather than module constants because they belong
    to the document's environment now. They are baked into the vertex data, so
    changing environment re-uploads this array; a tint uniform cannot express
    it, because the two colours move in opposite directions between presets.
    """
    verts: list[float] = []
    n = int(2 * _GRID_HALF_EXTENT / _GRID_SPACING) + 1
    for i in range(n):
        v = -_GRID_HALF_EXTENT + i * _GRID_SPACING
        is_centerline = abs(v) < 1e-5
        c = centerline_color if is_centerline else grid_color
        # Line parallel to X (varying x at fixed y)
        verts.extend([-_GRID_HALF_EXTENT, v, 0.0, *c])
        verts.extend([+_GRID_HALF_EXTENT, v, 0.0, *c])
        # Line parallel to Y (varying y at fixed x)
        verts.extend([v, -_GRID_HALF_EXTENT, 0.0, *c])
        verts.extend([v, +_GRID_HALF_EXTENT, 0.0, *c])
    return np.array(verts, dtype=np.float32).reshape(-1, 6)


def _build_axes_vertex_array() -> np.ndarray:
    """Return a (6, 6) float32 array: 3 colored line segments through origin."""
    return np.array(
        [
            # X axis (red)
            [0.0, 0.0, 0.0, *_AXIS_X_COLOR],
            [_AXIS_LENGTH, 0.0, 0.0, *_AXIS_X_COLOR],
            # Y axis (green)
            [0.0, 0.0, 0.0, *_AXIS_Y_COLOR],
            [0.0, _AXIS_LENGTH, 0.0, *_AXIS_Y_COLOR],
            # Z axis (blue)
            [0.0, 0.0, 0.0, *_AXIS_Z_COLOR],
            [0.0, 0.0, _AXIS_LENGTH, *_AXIS_Z_COLOR],
        ],
        dtype=np.float32,
    )


def _snap_marker_vertices(kind: int, p) -> np.ndarray:
    """GL_LINES vertices (N, 3) for a snap marker centered at world point p.

    Shape per kind: triangle (Midpoint), diamond (On-Edge), X (Intersection),
    square (Endpoint / On-Face / Grid / Axis / default). Drawn flat in the XY
    plane at p.z (a billboard approximation, matching M2/M3b markers).
    """
    s = 0.05
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    if kind == int(SnapKind.MIDPOINT):
        return np.array(
            [
                [x - s, y - s, z],
                [x + s, y - s, z],
                [x + s, y - s, z],
                [x, y + s, z],
                [x, y + s, z],
                [x - s, y - s, z],
            ],
            dtype=np.float32,
        )
    if kind == int(SnapKind.ON_EDGE):  # diamond
        return np.array(
            [
                [x, y + s, z],
                [x + s, y, z],
                [x + s, y, z],
                [x, y - s, z],
                [x, y - s, z],
                [x - s, y, z],
                [x - s, y, z],
                [x, y + s, z],
            ],
            dtype=np.float32,
        )
    if kind == int(SnapKind.INTERSECTION):  # X
        return np.array(
            [[x - s, y - s, z], [x + s, y + s, z], [x - s, y + s, z], [x + s, y - s, z]],
            dtype=np.float32,
        )
    # default: square
    return np.array(
        [
            [x - s, y - s, z],
            [x + s, y - s, z],
            [x + s, y - s, z],
            [x + s, y + s, z],
            [x + s, y + s, z],
            [x - s, y + s, z],
            [x - s, y + s, z],
            [x - s, y - s, z],
        ],
        dtype=np.float32,
    )


def _snap_marker_halo(pos: np.ndarray, p, scale: float) -> np.ndarray:
    """The marker's vertices scaled about the snap point, for the contrast halo.

    Drawn behind the marker in the environment's edge colour so every marker
    kind reads against every background. Scaling about `p` rather than the
    origin keeps the halo centred on the snap point at any world position.
    """
    centre = np.array([float(p[0]), float(p[1]), float(p[2])], dtype=np.float32)
    return ((pos - centre) * float(scale) + centre).astype(np.float32)


def _selection_face_polygons(scene, selection) -> list[np.ndarray]:
    """World-space loops (N,3 float32) for each LIVE selected face."""
    polys: list[np.ndarray] = []
    for f_id in selection.faces:
        try:
            loop = scene.face_loop(f_id)
        except KeyError:
            continue  # dead/stale id
        pts = np.array([scene.vertex(v).position for v in loop], dtype=np.float32)
        polys.append(pts)
    return polys


def _selection_edge_segments(scene, selection) -> np.ndarray:
    """(2E,3) float32 endpoint pairs for each LIVE selected edge."""
    out: list[np.ndarray] = []
    for e_id in selection.edges:
        try:
            e = scene.edge(e_id)
        except KeyError:
            continue
        out.append(np.asarray(scene.vertex(e.v1_id).position, dtype=np.float32))
        out.append(np.asarray(scene.vertex(e.v2_id).position, dtype=np.float32))
    if not out:
        return np.zeros((0, 3), dtype=np.float32)
    return np.array(out, dtype=np.float32)


_BOX_DASH_PX = 6.0  # on and off length, screen pixels


def _box_rect_ndc_segments(box_rect, viewport_w, viewport_h, *, dashed=False) -> np.ndarray:
    """Convert a pixel-space rect (x0,y0,x1,y1) to NDC GL_LINES segments
    (z=0) tracing its outline. y is flipped (screen y-down -> NDC y-up).

    `dashed` subdivides each side into _BOX_DASH_PX on/off runs, which is
    SketchUp's crossing-mode style (#39). The period is measured in PIXELS
    before the NDC conversion, so the dashes look the same at every rectangle
    size; defined as a fraction of the side, they would grow with the drag
    and read as a different line style at each size.
    """
    x0, y0, x1, y1 = box_rect
    w = max(int(viewport_w), 1)
    h = max(int(viewport_h), 1)

    def ndc(px, py):
        return ((2.0 * px / w) - 1.0, 1.0 - (2.0 * py / h))

    corners_px = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    out: list[list[float]] = []
    for i in range(4):
        ax, ay = corners_px[i]
        bx, by = corners_px[(i + 1) % 4]
        if not dashed:
            nax, nay = ndc(ax, ay)
            nbx, nby = ndc(bx, by)
            out.append([nax, nay, 0.0])
            out.append([nbx, nby, 0.0])
            continue
        length = math.hypot(bx - ax, by - ay)
        if length <= 1e-6:
            continue
        step = _BOX_DASH_PX * 2.0
        travelled = 0.0
        while travelled < length:
            t0 = travelled / length
            t1 = min(travelled + _BOX_DASH_PX, length) / length
            sax, say = ndc(ax + (bx - ax) * t0, ay + (by - ay) * t0)
            sbx, sby = ndc(ax + (bx - ax) * t1, ay + (by - ay) * t1)
            out.append([sax, say, 0.0])
            out.append([sbx, sby, 0.0])
            travelled += step
    if not out:
        return np.zeros((0, 3), dtype=np.float32)
    return np.array(out, dtype=np.float32)


def _screen_marker_ndc_quad(
    sx: float, sy: float, size_px: float, width: int, height: int
) -> np.ndarray:
    """4 corner NDC points of a `size_px` square centred at pixel (sx, sy)."""
    w = max(int(width), 1)
    h = max(int(height), 1)
    half = size_px * 0.5
    corners_px = [
        (sx - half, sy - half),
        (sx + half, sy - half),
        (sx + half, sy + half),
        (sx - half, sy + half),
    ]
    out = np.empty((4, 2), dtype=np.float32)
    for i, (px, py) in enumerate(corners_px):
        out[i, 0] = (2.0 * px / w) - 1.0
        out[i, 1] = 1.0 - (2.0 * py / h)
    return out


class SceneRenderer:
    """Owns GL resources for the grid + axes + user geometry + tool overlay."""

    def __init__(self) -> None:
        self._initialized = False
        # Programs
        self._phong_program: int = 0
        self._line_program: int = 0
        # Uniform location caches (populated in initialize_gl)
        self._phong_locs: dict[str, int] = {}
        self._line_locs: dict[str, int] = {}
        # Grid + axes buffers
        self._grid_vao: int = 0
        self._grid_vbo: int = 0
        self._grid_vertex_count: int = 0
        # set_environment can run with no current GL context (a View menu click
        # is not inside paintGL), so a colour change queues the re-upload and
        # render() performs it -- the same discipline as the pending texture
        # evictions.
        self._grid_dirty: bool = False
        self._axes_vao: int = 0
        self._axes_vbo: int = 0
        self._axes_vertex_count: int = 0
        # Per-definition GL buffer cache (M4e Task 11): keyed by id(definition).
        # Each entry is a _DefBuffers holding face/edge VAO+VBO+count.
        self._def_buffers: dict[int, _DefBuffers] = {}
        # M7.5b Task 6: GL texture objects keyed by the model's texture id.
        # Constructed here rather than in initialize_gl because it makes no GL
        # call until something is actually uploaded.
        self._texture_cache = TextureCache()
        # M7.5b Task 9: staleness reported by ordinary Qt slots (undo/redo,
        # document New/Open) has to wait for render() to actually free GL
        # objects -- see _flush_pending_texture_evictions().
        self._pending_stale_textures: set[int] = set()
        self._pending_release_all_textures: bool = False

        # Tool overlay buffers (rebuilt every frame)
        self._overlay_line_vao: int = 0
        self._overlay_line_vbo: int = 0
        self._overlay_marker_vao: int = 0
        self._overlay_marker_vbo: int = 0

        # Ghost-fill overlay (M3b)
        self._ghost_fill_program: int = 0
        self._ghost_fill_locs: dict[str, int] = {}
        self._ghost_fill_vao: int = 0
        self._ghost_fill_vbo: int = 0
        # View / projection matrices captured each frame so draw_face_fill_overlays
        # (called by tool overlays) can reuse them without re-deriving from camera.
        self._current_view_matrix: np.ndarray | None = None
        self._current_projection_matrix: np.ndarray | None = None

        # Viewport pixel size — updated in resize(); used by _draw_box_rect.
        self._viewport_w: int = 1
        self._viewport_h: int = 1

        # Active display style (Shaded/Wireframe/HiddenLine + X-Ray toggle).
        # Updated by set_render_style(), called by the View menu via the viewport.
        self._render_style = RenderStyle()

        # Active document environment (background + sky/ground + ink). Updated
        # by set_environment(), called by the View menu via the viewport.
        self._environment: Environment = DEFAULT_ENVIRONMENT

        # Sky/ground pass (M7.7)
        self._environment_program: int = 0
        self._environment_locs: dict[str, int] = {}
        self._environment_vao: int = 0
        self._environment_vbo: int = 0

    # --- Lifecycle --------------------------------------------------------

    def initialize_gl(self) -> None:
        if self._initialized:
            return
        GL.glClearColor(*self._environment.background, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)

        self._phong_program = _link_program(
            _load_shader_source("phong.vert"),
            _load_shader_source("phong.frag"),
        )
        self._line_program = _link_program(
            _load_shader_source("line.vert"),
            _load_shader_source("line.frag"),
        )

        # Cache uniform locations once per program — uniform locations are
        # stable for the lifetime of a linked program, so doing the lookup
        # in the per-frame draw path is pure waste.
        self._phong_locs = _cache_uniform_locations(self._phong_program, _PHONG_UNIFORMS)
        self._line_locs = _cache_uniform_locations(self._line_program, _LINE_UNIFORMS)
        self._bind_sampler_units()

        self._ghost_fill_program = _link_program(
            _load_shader_source("ghost_fill.vert"),
            _load_shader_source("ghost_fill.frag"),
        )
        self._ghost_fill_locs = _cache_uniform_locations(
            self._ghost_fill_program, _GHOST_FILL_UNIFORMS
        )

        self._environment_program = _link_program(
            _load_shader_source("environment.vert"),
            _load_shader_source("environment.frag"),
        )
        self._environment_locs = _cache_uniform_locations(
            self._environment_program, _ENVIRONMENT_UNIFORMS
        )

        self._init_grid_buffers()
        self._init_axes_buffers()
        self._init_overlay_buffers()
        self._init_ghost_fill_buffers()
        self._init_environment_buffers()

        self._initialized = True

    def _bind_sampler_units(self) -> None:
        """Point each phong sampler at its own texture unit.

        A sampler's value is program state, not per-draw state, so this is done
        once after linking. Both samplers default to unit 0, so skipping it
        makes the back side sample the FRONT's texture with no error anywhere.
        """
        GL.glUseProgram(self._phong_program)
        GL.glUniform1i(self._phong_locs["u_texture"], _TEXTURE_UNIT_FRONT)
        GL.glUniform1i(self._phong_locs["u_texture_back"], _TEXTURE_UNIT_BACK)
        GL.glUseProgram(0)

    def resize(self, w: int, h: int) -> None:
        self._viewport_w = int(w)
        self._viewport_h = int(h)
        if not self._initialized:
            return
        GL.glViewport(0, 0, w, h)

    def set_render_style(self, style: RenderStyle) -> None:
        """Set the active display style (called by the viewport from the View menu)."""
        self._render_style = replace(style)

    def set_environment(self, environment: Environment) -> None:
        """Set the active environment (called by the viewport from the View menu).

        The grid's colours are baked into its vertex buffer, so a change queues a
        rebuild for the next frame rather than touching GL here: this runs
        outside paintGL and may have no current context.
        """
        if environment == self._environment:
            return
        self._environment = environment
        self._grid_dirty = True

    def render(self, camera: Camera, model=None, tool_overlay=None, selection=None) -> None:
        """Draw the full scene: grid + axes + user geometry (all definitions) + tool overlay.

        Iterates model.traverse() to draw each definition's geometry with its
        accumulated world transform as the model matrix. Per-definition GL
        buffers are cached in _def_buffers and re-uploaded only when the
        definition's mesh is dirty.
        """
        if not self._initialized:
            return

        # M7.5b Task 9: flush any texture-cache releases queued by a slot
        # that ran with no current GL context (undo/redo, document New/Open)
        # -- BEFORE anything below looks up a GL texture for a material, so a
        # stale upload is never bound this frame either.
        self._flush_pending_texture_evictions()
        if self._grid_dirty:
            self._init_grid_buffers()

        GL.glClearColor(*self._environment.background, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)

        view = camera.view_matrix()
        projection = camera.projection_matrix()
        self._current_view_matrix = view
        self._current_projection_matrix = projection

        # 0. Environment (M7.7): sky and ground, behind everything.
        self._draw_environment(camera, view, projection)

        # 1. Grid (M1)
        self._draw_lines(self._grid_vao, self._grid_vertex_count, view, projection)
        # 2. Axes (M1)
        self._draw_lines(self._axes_vao, self._axes_vertex_count, view, projection)

        # 3 & 4. Draw all definitions in the scene graph with their world transforms.
        if model is not None:
            # Task 16 (#59): reconcile _def_buffers against the live model before
            # drawing so buffers for deleted/exploded definitions don't leak for
            # the rest of the session. Requires a current GL context, so this can
            # only run here in the render path (not from an arbitrary caller).
            self.evict_unreachable(model)
            materials = getattr(model, "materials", None)
            textures = getattr(model, "textures", None)
            translucent_ids = _translucent_ids(materials, textures)
            # Task 11: traverse_visible_tagged is traverse_visible() plus the
            # tag of the Instance that placed each occurrence, so `visible`
            # entries are (definition, world, tag_id) triples. The tag rides
            # along in the traversal itself -- rather than being looked up in
            # a per-definition map -- because one definition placed twice on
            # two tags must draw in two colours. Carried unconditionally
            # (like translucent_ids above) rather than gated on the style
            # flag: resolve_tag_color itself is the None-when-off gate.
            visible = list(traverse_visible_tagged(model))

            # Correction 2: translucent faces must be drawn after ALL opaque
            # geometry across every definition, not merely after their own
            # definition's, so the traversal is split in two rather than
            # gaining a second inner loop.
            #
            # Pass 1: every definition's opaque batches, plus its edges.
            for definition, world, tag_id in visible:
                buf = self._ensure_buffers(definition, translucent_ids, model)
                model_mat = world.astype(np.float32)
                # Task 15: dim pass — dim anything that is NOT the active context.
                # At root (active_path is empty), nothing is dimmed.
                dimmed = definition_is_dimmed(definition, model)
                # Task 11: resolved once per occurrence, not per batch. Not the
                # same shape as `dimmed` above despite the similar line: dimming
                # is a property of the definition relative to the active path,
                # while the tag belongs to the Instance that placed THIS
                # occurrence -- hence tag_id comes from the traversal entry.
                tag_color = resolve_tag_color(tag_id, model.tags, self._render_style)
                show_textures = textures_visible(self._render_style, tag_color=tag_color)
                for batch in buf.plan.opaque:
                    front, back = resolve_batch_sides(
                        batch,
                        materials,
                        self._render_style,
                        bg=self._environment.background,
                        dimmed=dimmed,
                        tag_color=tag_color,
                        translucent_ids=translucent_ids,
                    )
                    # draw_faces comes from the face style alone, so it is the
                    # same on both sides; front is representative.
                    if front.draw_faces and batch.count > 0:
                        front_tex, back_tex = self._batch_textures(
                            materials, textures, batch, show_textures=show_textures
                        )
                        self._draw_definition_faces(
                            buf,
                            view,
                            projection,
                            camera.position,
                            model_mat,
                            front=front,
                            back=back,
                            first=batch.first,
                            count=batch.count,
                            front_texture=front_tex,
                            back_texture=back_tex,
                        )
                # Edges stay in pass 1 — they are opaque.
                if buf.edge_count > 0:
                    self._draw_definition_edges(buf, view, projection, model_mat, dimmed=dimmed)

            # Pass 2: translucent batches only, definitions back to front.
            # The filter names the same list the loop below draws, so the two
            # cannot disagree. Filtering on `plan.translucent` instead happens
            # to work only because _reset_translucent_state seeds both together.
            # Task 11: the tag id rides along in these triples, so `order`
            # (which indexes into translucent_entries) keeps naming the same
            # occurrence's tag. order_definitions_for_translucent_pass sorts
            # (definition, world) pairs, so it gets the projection -- built in
            # the same order, so the indices still line up.
            translucent_entries = [
                (d, w, t)
                for d, w, t in visible
                if self._def_buffers[id(d)].translucent_draw_batches
            ]
            order = order_definitions_for_translucent_pass(
                [(d, w) for d, w, _ in translucent_entries],
                camera_pos=camera.position,
                centroid_of=lambda d: self._def_buffers[id(d)].translucent_local_centroid,
            )
            for i in order:
                definition, world, tag_id = translucent_entries[i]
                buf = self._def_buffers[id(definition)]
                self._sort_translucent_slice(buf, world, camera.position)
                model_mat = world.astype(np.float32)
                dimmed = definition_is_dimmed(definition, model)
                tag_color = resolve_tag_color(tag_id, model.tags, self._render_style)
                show_textures = textures_visible(self._render_style, tag_color=tag_color)
                for batch in buf.translucent_draw_batches:
                    front, back = resolve_batch_sides(
                        batch,
                        materials,
                        self._render_style,
                        bg=self._environment.background,
                        dimmed=dimmed,
                        tag_color=tag_color,
                        translucent_ids=translucent_ids,
                    )
                    if front.draw_faces and batch.count > 0:
                        front_tex, back_tex = self._batch_textures(
                            materials, textures, batch, show_textures=show_textures
                        )
                        self._draw_definition_faces(
                            buf,
                            view,
                            projection,
                            camera.position,
                            model_mat,
                            front=front,
                            back=back,
                            first=batch.first,
                            count=batch.count,
                            front_texture=front_tex,
                            back_texture=back_tex,
                        )

            # 4.5 Selection highlight (persistent, drawn on top of geometry).
            # Operates on the active context (root at identity, or entered context).
            if selection is not None:
                active_scene = model.active_scene
                self._draw_selection(
                    active_scene,
                    selection,
                    view,
                    projection,
                    world_transform=model.active_world_transform,
                )

            # 4.6 Task 15: Selected-instance bounding boxes (selection-blue).
            if selection is not None and selection.instances:
                active_world = model.active_world_transform
                for inst in model.active_context.children:
                    if inst.id in selection.instances and model.tags.is_visible(inst.tag_id):
                        aabb = inst.definition.local_aabb()
                        if aabb is not None:
                            lo, hi = aabb
                            world_t = active_world @ inst.transform
                            segs = aabb_world_edges(lo, hi, world_t)
                            self._draw_world_segments(
                                segs, _INSTANCE_BBOX_COLOR, _INSTANCE_BBOX_WIDTH, view, projection
                            )

        # 5. Tool overlay (NEW) — drawn on top with depth-test disabled
        if tool_overlay is not None:
            self._draw_tool_overlay(tool_overlay, view, projection)

        # 6. Face fills (M3b) — drawn last (on top of edges/markers).
        # Tools emit face_fill_polygons in WORLD coords already (each tool lifts
        # its own geometry), so we draw at identity — no world_transform here.
        if tool_overlay is not None and tool_overlay.face_fill_polygons:
            self.draw_face_fill_overlays(
                polygons=tool_overlay.face_fill_polygons,
                color=tool_overlay.face_fill_color,
            )

        # 7. Box-select rectangle (M4b) — screen space, on top.
        if tool_overlay is not None and tool_overlay.box_rect is not None:
            self._draw_box_rect(
                tool_overlay.box_rect, tool_overlay.box_rect_color, tool_overlay.box_rect_dashed
            )

        # 8. Generic gizmo primitives (M4c) — world polylines + screen markers.
        if tool_overlay is not None:
            if getattr(tool_overlay, "world_polylines", None):
                self._draw_world_polylines(tool_overlay.world_polylines, view, projection)
            if getattr(tool_overlay, "screen_markers", None):
                self._draw_screen_markers(
                    camera, tool_overlay.screen_markers, self._viewport_w, self._viewport_h
                )

    def evict_unreachable(self, model) -> None:
        """Drop + release GL buffers for definitions no longer reachable from `model`.

        Reachability is computed with model.traverse() (ALL instantiated
        definitions, root included) rather than traverse_visible(), so a
        definition that is merely on a hidden tag keeps its buffers —
        eviction only targets definitions with no instance anywhere
        (deleted or exploded), the actual leak described in #59.

        No-op-safe without a GL context: does not require self._initialized,
        and _DefBuffers.release() itself guards every delete call behind a
        zero-handle check, so an all-zero stand-in record releases cleanly.
        """
        reachable = {id(definition) for definition, _world in model.traverse()}
        reachable.add(id(model.root))  # defensive; traverse() already yields root first
        for key in [k for k in self._def_buffers if k not in reachable]:
            buf = self._def_buffers.pop(key)
            buf.release()

    def evict_stale_textures(self, textures) -> None:
        """Queue GL uploads for texture ids no longer in `textures` for release.

        M7.5b Task 9: an undone import (AddTextureCommand.undo) or a deleted
        texture (DeleteTextureCommand) removes a Texture record from the
        model's TextureLibrary, but leaves whatever this cache uploaded for
        that id sitting around until release_all() -- normally document
        close. Reconciling here, the same way evict_unreachable() reconciles
        `_def_buffers` against the model, frees it promptly instead of
        letting a long session of import/undo cycles accumulate orphaned GL
        textures.

        This is called from MainWindow._on_after_undo_redo(), an ordinary Qt
        slot -- NOT the render path -- so there is no current GL context to
        call TextureCache.invalidate() (which issues glDeleteTextures) with;
        ViewportWidget never calls makeCurrent() outside
        initializeGL/resizeGL/paintGL. So this only records which ids are
        stale; render() flushes them (_flush_pending_texture_evictions),
        mirroring how evict_unreachable() is itself only ever called from
        inside render(), where a context is guaranteed current.
        """
        live = {t.id for t in textures.textures()}
        self._pending_stale_textures |= self._texture_cache.cached_ids() - live

    def release_all_textures(self) -> None:
        """Queue every cached GL texture upload for release on the next frame.

        Without this, a new or reloaded Model's TextureLibrary restarts its
        id numbering from 1 -- the same ids the PREVIOUS document's textures
        used -- so texture_for() would happily hand back the old document's
        GL texture for the new document's id 1 (wrong image bound, not just a
        leak) until this cache was cleared.

        Same GL-context constraint as evict_stale_textures(): this is called
        from MainWindow._reset_document(), an ordinary Qt slot, so the actual
        glDeleteTextures calls are deferred to the next render()
        (_flush_pending_texture_evictions), which runs before that frame
        binds any texture for the just-loaded model.
        """
        self._pending_release_all_textures = True
        self._pending_stale_textures.clear()  # superseded by the full release

    def _flush_pending_texture_evictions(self) -> None:
        """Apply queued texture-cache releases. Only safe with a current GL
        context, so this must only be called from render()."""
        if self._pending_release_all_textures:
            self._texture_cache.release_all()
            self._pending_release_all_textures = False
            self._pending_stale_textures.clear()
            return
        for tid in self._pending_stale_textures:
            self._texture_cache.invalidate(tid)
        self._pending_stale_textures.clear()

    # --- Init helpers -----------------------------------------------------

    def _init_grid_buffers(self) -> None:
        if self._grid_vao:
            GL.glDeleteVertexArrays(1, [self._grid_vao])
        if self._grid_vbo:
            GL.glDeleteBuffers(1, [self._grid_vbo])
        env = self._environment
        verts = _build_grid_vertex_array(env.grid_color, env.grid_centerline_color)
        self._grid_vertex_count = int(verts.shape[0])
        self._grid_vao, self._grid_vbo = self._upload_interleaved_lines(verts)
        self._grid_dirty = False

    def _init_axes_buffers(self) -> None:
        verts = _build_axes_vertex_array()
        self._axes_vertex_count = int(verts.shape[0])
        self._axes_vao, self._axes_vbo = self._upload_interleaved_lines(verts)

    def _alloc_def_buffers(self) -> _DefBuffers:
        """Allocate a new _DefBuffers: create empty VAO+VBO pairs for faces and edges."""
        buf = _DefBuffers()

        # Face buffers — interleaved (pos.xyz, normal.xyz, front_uv.xy,
        # back_uv.xy), 40 bytes per vertex
        buf.face_vao = int(GL.glGenVertexArrays(1))
        buf.face_vbo = int(GL.glGenBuffers(1))
        GL.glBindVertexArray(buf.face_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buf.face_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 0, None, GL.GL_DYNAMIC_DRAW)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, _FACE_VERTEX_BYTES, None)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(
            1, 3, GL.GL_FLOAT, GL.GL_FALSE, _FACE_VERTEX_BYTES, ctypes.c_void_p(12)
        )
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(
            2, 2, GL.GL_FLOAT, GL.GL_FALSE, _FACE_VERTEX_BYTES, ctypes.c_void_p(24)
        )
        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(
            3, 2, GL.GL_FLOAT, GL.GL_FALSE, _FACE_VERTEX_BYTES, ctypes.c_void_p(32)
        )
        GL.glEnableVertexAttribArray(3)
        GL.glBindVertexArray(0)

        # Edge buffers — interleaved (pos.xyz, color.rgb), 24 bytes per vertex
        buf.edge_vao = int(GL.glGenVertexArrays(1))
        buf.edge_vbo = int(GL.glGenBuffers(1))
        GL.glBindVertexArray(buf.edge_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buf.edge_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 0, None, GL.GL_DYNAMIC_DRAW)
        stride = 6 * ctypes.sizeof(ctypes.c_float)  # 24 bytes
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(
            1,
            3,
            GL.GL_FLOAT,
            GL.GL_FALSE,
            stride,
            ctypes.c_void_p(3 * ctypes.sizeof(ctypes.c_float)),
        )
        GL.glEnableVertexAttribArray(1)
        GL.glBindVertexArray(0)

        return buf

    def _init_overlay_buffers(self) -> None:
        """Create empty VBOs for tool-overlay lines and snap-marker quads.

        Both buffers use the line shader's (pos.xyz, color.rgb) layout.
        """
        # Rubber-band lines
        self._overlay_line_vao = int(GL.glGenVertexArrays(1))
        self._overlay_line_vbo = int(GL.glGenBuffers(1))
        GL.glBindVertexArray(self._overlay_line_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._overlay_line_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 0, None, GL.GL_DYNAMIC_DRAW)
        stride = 6 * ctypes.sizeof(ctypes.c_float)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(
            1,
            3,
            GL.GL_FLOAT,
            GL.GL_FALSE,
            stride,
            ctypes.c_void_p(3 * ctypes.sizeof(ctypes.c_float)),
        )
        GL.glEnableVertexAttribArray(1)
        GL.glBindVertexArray(0)

        # Snap marker — small world-aligned wireframe square re-uploaded per frame.
        self._overlay_marker_vao = int(GL.glGenVertexArrays(1))
        self._overlay_marker_vbo = int(GL.glGenBuffers(1))
        GL.glBindVertexArray(self._overlay_marker_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._overlay_marker_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 0, None, GL.GL_DYNAMIC_DRAW)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(
            1,
            3,
            GL.GL_FLOAT,
            GL.GL_FALSE,
            stride,
            ctypes.c_void_p(3 * ctypes.sizeof(ctypes.c_float)),
        )
        GL.glEnableVertexAttribArray(1)
        GL.glBindVertexArray(0)

    @staticmethod
    def _upload_interleaved_lines(verts: np.ndarray) -> tuple[int, int]:
        """Upload an (N, 6) float32 array (x,y,z, r,g,b per vertex). Returns (vao, vbo)."""
        vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(vao)
        vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, verts.nbytes, verts, GL.GL_STATIC_DRAW)
        stride = 6 * ctypes.sizeof(ctypes.c_float)
        # position (vec3)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0))
        # color (vec3) at offset 3 floats
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(
            1,
            3,
            GL.GL_FLOAT,
            GL.GL_FALSE,
            stride,
            ctypes.c_void_p(3 * ctypes.sizeof(ctypes.c_float)),
        )
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)
        return vao, vbo

    # --- Draw helpers -----------------------------------------------------

    def _draw_lines(self, vao: int, count: int, view: np.ndarray, projection: np.ndarray) -> None:
        GL.glUseProgram(self._line_program)
        locs = self._line_locs
        _set_mat4(locs["u_view"], view)
        _set_mat4(locs["u_projection"], projection)
        GL.glBindVertexArray(vao)
        GL.glDrawArrays(GL.GL_LINES, 0, count)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)

    def _draw_environment(self, camera: Camera, view: np.ndarray, projection: np.ndarray) -> None:
        """Fill the frame with the environment's sky and ground.

        Runs after glClear and before the grid, with the depth test off and
        depth writes masked, so it can never occlude geometry and leaves no
        depth behind for the passes after it.

        Returns immediately when both halves are disabled: glClearColor has
        already painted every pixel this would paint, so Plain White and Studio
        pay nothing for a feature they do not use.
        """
        env = self._environment
        if not environment_pass_needed(env):
            return

        inv_view_proj = np.linalg.inv(
            np.asarray(projection, dtype=np.float64) @ np.asarray(view, dtype=np.float64)
        )

        GL.glUseProgram(self._environment_program)
        locs = self._environment_locs
        _set_mat4(locs["u_inv_view_proj"], inv_view_proj)
        _set_vec3(locs["u_camera_pos"], camera.position)
        _set_vec3(locs["u_background"], env.background)
        _set_vec3(locs["u_sky_color"], env.sky_color)
        _set_vec3(locs["u_ground_color"], env.ground_color)
        _set_float(locs["u_ground_opacity"], env.ground_opacity)
        _set_int(locs["u_sky_enabled"], 1 if env.sky_enabled else 0)
        _set_int(locs["u_ground_enabled"], 1 if env.ground_enabled else 0)

        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glBindVertexArray(self._environment_vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
        GL.glBindVertexArray(0)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glUseProgram(0)

    def _ensure_buffers(self, definition, translucent_ids: frozenset[int], model) -> _DefBuffers:
        """The cached buffers for `definition`, re-uploading them if stale.

        Lifted out of the draw loop because pass 1 and pass 2 both need it and
        duplicating it would let the two drift.

        Staleness has two causes, not one. A dirty mesh is the familiar one.
        The other is spec 1.7's asymmetry: uniforms are read from the library
        per batch at draw time, so editing a material's COLOUR shows up with no
        buffer work at all — but the opaque/translucent partition is computed
        here, and editing a material's ALPHA dirties no mesh. Without the set
        comparison a material turned translucent would keep drawing in the
        opaque pass, writing depth and never sorting, which looks like the edit
        did nothing. Comparing the whole set rather than tracking a per-material
        boundary crossing is deliberate: it catches a material being added,
        removed, or edited in either direction, and the set is a handful of ints.

        M7.5b adds a third cause of the same shape: the per-corner UVs are baked
        into the vertex buffer from each material's texture_size, which also
        lives on the library, so resizing a texture dirties no mesh and would
        otherwise go on tiling at the old size.

        M7.7 adds a fourth cause of the same shape: the edge colour is baked
        into the edge vertex buffer from the document's environment, which no
        mesh dirty flag covers either, so switching environment would otherwise
        leave every definition drawing its old ink.
        """
        buf = self._def_buffers.get(id(definition))
        if (
            buf is None
            or definition.mesh.dirty
            or buf.translucent_ids != translucent_ids
            or not uv_key_still_matches(buf.uv_key, model)
            or _edge_buffer_is_stale(buf, self._environment.edge_color)
        ):
            buf = self._upload_definition(definition, translucent_ids, model)
            definition.mesh.mark_clean()
            self._def_buffers[id(definition)] = buf
        return buf

    def _sort_translucent_slice(self, buf: _DefBuffers, world, camera_pos) -> None:
        """Re-order this definition's translucent suffix back to front.

        No-ops when neither the camera nor the geometry has moved since the
        last sort, so a static view costs nothing. Only the suffix is touched;
        the opaque prefix is never re-uploaded. The permutation is always
        computed against `face_interleaved`, which stays in upload order, so
        repeated sorts do not compose.

        The suffix may span several material pairs, so the batches that
        describe it are re-cut from the sorted order — see
        rebuild_translucent_batches for why drawing with the original ranges
        would be wrong.
        """
        if buf.translucent_centroids.shape[0] == 0:
            return
        key = (tuple(np.asarray(camera_pos, dtype=np.float64).tolist()), world.tobytes())
        if key == buf.translucent_sort_key:
            return

        world_centroids = transform_points(buf.translucent_centroids, world)
        tri_order = order_back_to_front(world_centroids, camera_pos)
        vertex_order = triangle_order_to_vertex_order(tri_order)

        first = buf.plan.translucent_first
        data = np.ascontiguousarray(buf.face_interleaved[first:][vertex_order], dtype=np.float32)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buf.face_vbo)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, first * _FACE_VERTEX_BYTES, data.nbytes, data)

        buf.translucent_draw_batches = rebuild_translucent_batches(
            buf.translucent_pairs[tri_order], first
        )
        buf.translucent_sort_key = key

    def _upload_definition(self, definition, translucent_ids: frozenset[int], model) -> _DefBuffers:
        """Build or update GL buffers for a single definition's mesh.

        Looks up (or allocates) a _DefBuffers entry for this definition,
        uploads fresh geometry data, and returns the updated _DefBuffers.
        The caller is responsible for calling definition.mesh.mark_clean()
        and storing the result back into self._def_buffers[id(definition)].

        `translucent_ids` is the set the batch planner needs in order to
        classify a batch as translucent. Passed in rather than derived from the
        model here because _ensure_buffers also compares it against the set the
        buffer was last built under, and both must see exactly the same value.

        `model` supplies the material library the per-corner UVs are projected
        through. Required, not defaulted: a missing model silently bakes
        identity-sized UVs AND leaves uv_key_still_matches unable to validate a
        non-empty key, so every frame would re-upload wrong UVs rather than
        failing. A caller that genuinely has no model must say so explicitly.
        """
        # Re-use existing GL objects if we already have them; allocate if not.
        buf = self._def_buffers.get(id(definition))
        if buf is None:
            buf = self._alloc_def_buffers()

        scene = definition.mesh

        # Faces: (3*T, 3) positions + (3*T, 3) normals + (3*T, 2) front UVs +
        # (3*T, 2) back UVs -> interleaved (3*T, 10). build_face_uvs walks faces
        # in the same next_live_face order face_triangle_buffer does, so all
        # four blocks are aligned corner for corner before plan.vertex_order
        # permutes them — and the permutation must reach every one of them.
        face_buffer = scene.face_triangle_buffer()
        positions, normals = face_buffer
        # Fetched once and used three times: the batch plan, the UV key, and
        # through the key the bake gate. Each fetch walks every triangle.
        front_mats = scene.face_triangle_materials(Side.FRONT)
        back_mats = scene.face_triangle_materials(Side.BACK)
        uv_key = uv_material_key_for(front_mats, back_mats, model)
        if positions.shape[0] > 0:
            if uv_key_needs_bake(uv_key):
                front_uvs, back_uvs = build_face_uvs_both_sides(scene, model, face_buffer)
            else:
                # Nothing THIS definition carries can sample a UV, so the walk
                # is skipped and the two blocks go out zero-filled -- the LAYOUT
                # is unchanged, only the arithmetic is. See uv_key_needs_bake
                # for why gaining a texture still re-bakes correctly.
                zeros = np.zeros((positions.shape[0], 2), dtype=np.float64)
                front_uvs = back_uvs = zeros
            interleaved = np.concatenate([positions, normals, front_uvs, back_uvs], axis=1).astype(
                np.float32
            )
            # Group triangles by (front, back) material so each pair draws as one
            # contiguous batch, with the translucent pairs as a contiguous suffix
            # that render()'s second pass depth-sorts.
            plan = plan_face_batches(front_mats, back_mats, translucent_ids)
            interleaved = np.ascontiguousarray(interleaved[plan.vertex_order])
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buf.face_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, interleaved.nbytes, interleaved, GL.GL_DYNAMIC_DRAW)
            buf.face_count = int(positions.shape[0])
            buf.plan = plan
        else:
            interleaved = _empty_face_interleaved()
            plan = _empty_batch_plan()
            buf.face_count = 0
            buf.plan = plan
        buf.translucent_ids = translucent_ids
        buf.uv_key = uv_key
        _reset_translucent_state(buf, plan, interleaved)

        # Edges: (2*E, 3) positions — pack constant color per vertex so the
        # line shader's attribute 1 (in_color) is always satisfied.
        edge_color = self._environment.edge_color
        edges = scene.edge_line_buffer()
        if edges.shape[0] > 0:
            n = int(edges.shape[0])
            colors = np.tile(np.array(edge_color, dtype=np.float32), (n, 1))
            data = np.ascontiguousarray(np.concatenate([edges.astype(np.float32), colors], axis=1))
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buf.edge_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_DYNAMIC_DRAW)
            buf.edge_count = n
        else:
            buf.edge_count = 0
        # Recorded on BOTH branches. A definition with no edges that skipped
        # this would stay stale forever and re-upload on every frame.
        buf.edge_color = edge_color

        return buf

    def _texture_for_material(self, materials, textures, mid: int) -> int | None:
        """The GL texture for one side's material, or None to draw untextured.

        Every step here is allowed to come up empty and none of them is an
        error: material 0 is the unpainted Default, a plain material has
        texture_id None, and TextureLibrary.get returns None for an id pointing
        at nothing -- deliberately unlike MaterialLibrary.get, because spec 1.8
        requires a document referencing a missing container entry to open with
        that material UNTEXTURED rather than being refused or shown a
        placeholder image. TextureCache.texture_for returns None on the same
        terms for bytes that will not decode.
        """
        if mid == 0 or materials is None or textures is None:
            return None
        tid = materials.get(mid).texture_id
        if tid is None:
            return None
        tex = textures.get(tid)
        if tex is None:
            return None
        return self._texture_cache.texture_for(tex)

    def _batch_textures(
        self, materials, textures, batch: FaceBatch, *, show_textures: bool
    ) -> tuple[int | None, int | None]:
        """The (front, back) GL textures for one batch.

        A batch keys on the (front_material, back_material) pair and a texture
        belongs to a material, so one batch has exactly one of each.
        """
        if not show_textures:
            return (None, None)
        return (
            self._texture_for_material(materials, textures, batch.front_material_id),
            self._texture_for_material(materials, textures, batch.back_material_id),
        )

    def _draw_definition_faces(
        self,
        buf: _DefBuffers,
        view: np.ndarray,
        projection: np.ndarray,
        camera_pos: np.ndarray,
        model_mat: np.ndarray,
        *,
        front: ResolvedFacePass,
        back: ResolvedFacePass,
        first: int = 0,
        count: int | None = None,
        front_texture: int | None = None,
        back_texture: int | None = None,
    ) -> None:
        """Draw a definition's faces using resolved face-passes (style + dim + X-Ray).

        ``front`` and ``back`` carry the two sides' material uniforms and alphas;
        the fragment shader selects between them with gl_FrontFacing. Blend and
        depth-mask state is draw-call state rather than per-side, so it is taken
        from ``front`` — resolve_batch_sides gives both sides the same flags.
        X-Ray turns depth writes off so geometry behind shows through; the dim
        pass and X-Ray both arrive here pre-composed as a reduced alpha. State
        (blend, depth mask) is restored after the draw — see the M4e Task-15
        blend-leak fix for why this hygiene is mandatory.

        ``front_texture`` / ``back_texture`` are GL texture ids, or None for an
        untextured side. None binds 0 AND clears the side's u_has_texture flag:
        the flag is what actually turns sampling off, since a GLSL sampler
        always reads whatever object sits in its unit.
        """
        GL.glUseProgram(self._phong_program)
        locs = self._phong_locs
        _set_mat4(locs["u_view"], view)
        _set_mat4(locs["u_projection"], projection)
        _set_mat4(locs["u_model"], model_mat)
        _set_vec3(locs["u_camera_pos"], camera_pos)
        _set_vec3(locs["u_light_dir"], _LIGHT_DIR)
        _set_vec3(locs["u_light_color"], _LIGHT_COLOR)
        _set_vec3(locs["u_material_ambient"], front.ambient)
        _set_vec3(locs["u_material_diffuse"], front.diffuse)
        _set_vec3(locs["u_material_specular"], front.specular)
        _set_float(locs["u_material_shininess"], front.shininess)
        _set_float(locs["u_alpha"], front.alpha)
        _set_vec3(locs["u_material_ambient_back"], back.ambient)
        _set_vec3(locs["u_material_diffuse_back"], back.diffuse)
        _set_vec3(locs["u_material_specular_back"], back.specular)
        _set_float(locs["u_material_shininess_back"], back.shininess)
        _set_float(locs["u_alpha_back"], back.alpha)

        # `is not None`, not truthiness: the two sentinels are different
        # things. None means "this side resolved to no texture"; 0 is the GL
        # name for "no texture object", which is what gets BOUND in that case.
        # Conflating them would make a resolver that wrongly returned 0 look
        # indistinguishable from one that correctly returned None.
        _set_float(locs["u_has_texture"], 1.0 if front_texture is not None else 0.0)
        _set_float(locs["u_has_texture_back"], 1.0 if back_texture is not None else 0.0)
        GL.glActiveTexture(_TEXTURE_UNIT_ENUM[_TEXTURE_UNIT_BACK])
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0 if back_texture is None else back_texture)
        # The front is bound LAST so unit 0 is the active unit when this
        # returns. Anything that binds a texture without selecting a unit first
        # — TextureCache's own upload included — would otherwise land in unit 1
        # and leave the back side reading a stale object.
        GL.glActiveTexture(_TEXTURE_UNIT_ENUM[_TEXTURE_UNIT_FRONT])
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0 if front_texture is None else front_texture)

        if front.blend:
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        if not front.depth_write:
            GL.glDepthMask(GL.GL_FALSE)

        GL.glBindVertexArray(buf.face_vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, first, buf.face_count if count is None else count)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)

        if not front.depth_write:
            GL.glDepthMask(GL.GL_TRUE)
        if front.blend:
            GL.glDisable(GL.GL_BLEND)

    def _draw_definition_edges(
        self,
        buf: _DefBuffers,
        view: np.ndarray,
        projection: np.ndarray,
        model_mat: np.ndarray,
        *,
        dimmed: bool = False,
    ) -> None:
        """Draw a definition's edge geometry with the given model matrix.

        Note: the line shader does not have a u_model uniform (it uses only
        u_view and u_projection), so edges are always drawn in local space
        transformed to world space by the line vertex shader via the view
        matrix. For the root definition at identity this is equivalent to
        the previous hardcoded behaviour. For child definitions the geometry
        is already in local space; the model matrix must be folded into the
        view or the edges drawn in world space. Since the line shader has no
        u_model, we pre-multiply view by model_mat for this draw call only.

        When ``dimmed`` is True (Task 15), lines are drawn at reduced opacity
        using the constant-alpha blend so the geometry recedes.

        Note (#61): the face dim/transparency path migrated to the
        `u_alpha` phong uniform + standard GL_SRC_ALPHA blending back in
        M5a (commit 5a3bc65); this edge path still uses the older
        glBlendColor + GL_CONSTANT_ALPHA idiom. That's intentional-on-record
        rather than an oversight: the line shader has no alpha uniform, and
        every other blended pass already sets its own glBlendFunc to
        GL_SRC_ALPHA before drawing, so the divergence has no visible
        effect today. Giving the line shader a u_alpha uniform to unify the
        two paths would mean auditing every other draw call that shares
        `self._line_program` (_draw_lines, _draw_tool_overlay,
        _draw_world_segments, _draw_screen_space_lines) so none of them
        silently render at GLSL's zero-initialized alpha; left as a
        follow-up rather than risking a blast-radius change here.
        """
        # Pre-multiply: view_for_edges = view @ model_mat (transforms local→clip).
        # Both are float32; the result is float32.
        view_x_model = (view.astype(np.float64) @ model_mat.astype(np.float64)).astype(np.float32)

        GL.glUseProgram(self._line_program)
        locs = self._line_locs
        _set_mat4(locs["u_view"], view_x_model)
        _set_mat4(locs["u_projection"], projection)

        if dimmed:
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendColor(0.0, 0.0, 0.0, _DIM_ALPHA_BLEND)
            GL.glBlendFunc(GL.GL_CONSTANT_ALPHA, GL.GL_ONE_MINUS_CONSTANT_ALPHA)

        GL.glBindVertexArray(buf.edge_vao)
        GL.glLineWidth(1.5)
        GL.glDrawArrays(GL.GL_LINES, 0, buf.edge_count)
        GL.glLineWidth(1.0)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)

        if dimmed:
            GL.glDisable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            # Restore the blend color too, not just the blend func — a leftover
            # non-default GL_CONSTANT_ALPHA color is harmless today only because
            # every subsequent blended pass overwrites glBlendFunc before use;
            # reset it so that invariant isn't required to hold.
            GL.glBlendColor(0.0, 0.0, 0.0, 1.0)

    def _draw_tool_overlay(
        self,
        overlay,
        view: np.ndarray,
        projection: np.ndarray,
    ) -> None:
        GL.glUseProgram(self._line_program)
        locs = self._line_locs
        _set_mat4(locs["u_view"], view)
        _set_mat4(locs["u_projection"], projection)

        # Disable depth test so the overlay always wins.
        GL.glDisable(GL.GL_DEPTH_TEST)
        try:
            # Rubber-band segments — each segment is two vertices with pos + color.
            segs = overlay.rubber_band_segments
            if segs.shape[0] > 0:
                n = int(segs.shape[0])
                colors = np.tile(np.array(overlay.rubber_band_color, dtype=np.float32), (n, 1))
                data = np.ascontiguousarray(
                    np.concatenate([segs.astype(np.float32), colors], axis=1)
                )
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._overlay_line_vbo)
                GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_DYNAMIC_DRAW)
                GL.glBindVertexArray(self._overlay_line_vao)
                GL.glLineWidth(2.0)
                GL.glDrawArrays(GL.GL_LINES, 0, n)
                GL.glLineWidth(1.0)
                GL.glBindVertexArray(0)

            # Snap marker — small wireframe shape at the snap point.
            # Shape depends on snap kind: see _snap_marker_vertices.
            if overlay.snap_marker_position is not None:
                p = overlay.snap_marker_position
                pos = _snap_marker_vertices(overlay.snap_marker_kind, p)
                n = pos.shape[0]

                # Contrast halo (M7.7 Task 6b): a larger copy of the marker in
                # the environment's own edge colour, drawn first so every
                # marker kind reads against every background. edge_color is
                # reused rather than computed because the contrast floor in
                # tests/test_environment.py already guarantees it differs from
                # every enabled backdrop of its own preset.
                halo_scale = 1.8
                halo_pos = _snap_marker_halo(pos, p, halo_scale)
                hr, hg, hb = self._environment.edge_color
                halo_colors = np.tile(np.array([hr, hg, hb], dtype=np.float32), (n, 1))
                halo_data = np.ascontiguousarray(
                    np.concatenate([halo_pos, halo_colors], axis=1).astype(np.float32)
                )

                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._overlay_marker_vbo)
                GL.glBufferData(GL.GL_ARRAY_BUFFER, halo_data.nbytes, halo_data, GL.GL_DYNAMIC_DRAW)
                GL.glBindVertexArray(self._overlay_marker_vao)
                GL.glLineWidth(4.0)
                GL.glDrawArrays(GL.GL_LINES, 0, n)
                GL.glLineWidth(1.0)
                GL.glBindVertexArray(0)

                cr, cg, cb = overlay.snap_marker_color
                colors = np.tile(np.array([cr, cg, cb], dtype=np.float32), (n, 1))
                data = np.ascontiguousarray(
                    np.concatenate([pos, colors], axis=1).astype(np.float32)
                )

                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._overlay_marker_vbo)
                GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_DYNAMIC_DRAW)
                GL.glBindVertexArray(self._overlay_marker_vao)
                GL.glLineWidth(2.0)
                GL.glDrawArrays(GL.GL_LINES, 0, n)
                GL.glLineWidth(1.0)
                GL.glBindVertexArray(0)
        finally:
            GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glUseProgram(0)

    def _draw_world_segments(self, segs, color, width, view, projection) -> None:
        """Draw (2N,3) world-space GL_LINES in a flat color, on top (depth off).
        Reuses the overlay line VBO."""
        if segs.shape[0] == 0:
            return
        GL.glUseProgram(self._line_program)
        locs = self._line_locs
        _set_mat4(locs["u_view"], view)
        _set_mat4(locs["u_projection"], projection)
        n = int(segs.shape[0])
        colors = np.tile(np.array(color, dtype=np.float32), (n, 1))
        data = np.ascontiguousarray(np.concatenate([segs.astype(np.float32), colors], axis=1))
        GL.glDisable(GL.GL_DEPTH_TEST)
        try:
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._overlay_line_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_DYNAMIC_DRAW)
            GL.glBindVertexArray(self._overlay_line_vao)
            GL.glLineWidth(width)
            GL.glDrawArrays(GL.GL_LINES, 0, n)
            GL.glLineWidth(1.0)
            GL.glBindVertexArray(0)
        finally:
            GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glUseProgram(0)

    def _draw_screen_space_lines(self, segs, color, width) -> None:
        """Draw (2N,3) NDC GL_LINES with identity view/projection (NDC positions
        render directly); depth test off, line width restored to 1.0 afterward.

        Shared by _draw_box_rect and _draw_screen_markers so both go through one
        VBO-upload-and-draw path on the line shader."""
        n = int(segs.shape[0])
        if n == 0:
            return
        identity = np.eye(4, dtype=np.float32)
        colors = np.tile(np.array(color, dtype=np.float32), (n, 1))
        data = np.ascontiguousarray(np.concatenate([segs, colors], axis=1).astype(np.float32))
        GL.glUseProgram(self._line_program)
        _set_mat4(self._line_locs["u_view"], identity)
        _set_mat4(self._line_locs["u_projection"], identity)
        GL.glDisable(GL.GL_DEPTH_TEST)
        try:
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._overlay_line_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_DYNAMIC_DRAW)
            GL.glBindVertexArray(self._overlay_line_vao)
            GL.glLineWidth(width)
            GL.glDrawArrays(GL.GL_LINES, 0, n)
            GL.glLineWidth(1.0)
            GL.glBindVertexArray(0)
        finally:
            GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glUseProgram(0)

    def _draw_box_rect(self, box_rect, color, dashed=False) -> None:
        """Draw the screen-space box-select outline using identity view/projection
        (NDC positions render directly); depth test off."""
        segs = _box_rect_ndc_segments(box_rect, self._viewport_w, self._viewport_h, dashed=dashed)
        self._draw_screen_space_lines(segs, color, 1.5)

    def _draw_world_polylines(self, polylines, view, projection) -> None:
        """Draw each (segments, color, width) as world-space line segments."""
        for segs, color, width in polylines:
            arr = np.asarray(segs, dtype=np.float32).reshape(-1, 3)
            if arr.shape[0] >= 2:
                self._draw_world_segments(arr, color, float(width), view, projection)

    def _draw_screen_markers(self, camera, markers, width, height) -> None:
        """Project each (world_pos, size_px, color) and draw an outlined square
        in screen space (identity matrices), like the box-select rectangle."""
        if not markers:
            return
        for world_pos, size_px, color in markers:
            proj = camera.world_to_screen(world_pos, width, height)
            if proj is None:
                continue
            sx, sy, _depth = proj
            quad = _screen_marker_ndc_quad(sx, sy, size_px, width, height)
            # 4 edges as (2*N,3) GL_LINES at z=0.
            loop = np.zeros((8, 3), dtype=np.float32)
            for i in range(4):
                loop[2 * i, 0:2] = quad[i]
                loop[2 * i + 1, 0:2] = quad[(i + 1) % 4]
            self._draw_screen_space_lines(loop, color, 1.5)

    def _draw_selection(self, scene, selection, view, projection, world_transform=None) -> None:
        if selection is None:
            return
        need_transform = not is_identity_transform(world_transform)
        polys = _selection_face_polygons(scene, selection)
        if polys:
            if need_transform:
                polys = [apply_mat(p, world_transform) for p in polys]
            self.draw_face_fill_overlays(polygons=polys, color=_SELECTION_FILL_COLOR)
        segs = _selection_edge_segments(scene, selection)
        if need_transform and segs.shape[0] > 0:
            segs = apply_mat(segs, world_transform)
        self._draw_world_segments(segs, _SELECTION_EDGE_COLOR, 3.0, view, projection)

    def _init_ghost_fill_buffers(self) -> None:
        """Create the (empty) VAO/VBO for the ghost-fill overlay pass.

        Layout: position-only (vec3) at attribute 0. Buffer is re-uploaded
        on every draw_face_fill_overlays call.
        """
        self._ghost_fill_vao = int(GL.glGenVertexArrays(1))
        self._ghost_fill_vbo = int(GL.glGenBuffers(1))
        GL.glBindVertexArray(self._ghost_fill_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._ghost_fill_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 0, None, GL.GL_DYNAMIC_DRAW)
        stride = 3 * ctypes.sizeof(ctypes.c_float)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(0)
        GL.glBindVertexArray(0)

    def _init_environment_buffers(self) -> None:
        """Create the VAO/VBO for the fullscreen environment quad.

        Layout: position-only (vec2) at attribute 0, in NDC. Uploaded once, in
        clip space, so it needs no matrices and no resize handling.
        """
        quad = np.array(
            [
                [-1.0, -1.0],
                [+1.0, -1.0],
                [+1.0, +1.0],
                [-1.0, -1.0],
                [+1.0, +1.0],
                [-1.0, +1.0],
            ],
            dtype=np.float32,
        )
        self._environment_vao = int(GL.glGenVertexArrays(1))
        self._environment_vbo = int(GL.glGenBuffers(1))
        GL.glBindVertexArray(self._environment_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._environment_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, quad.nbytes, quad, GL.GL_STATIC_DRAW)
        stride = 2 * ctypes.sizeof(ctypes.c_float)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(0)
        GL.glBindVertexArray(0)

    def draw_face_fill_overlays(
        self,
        polygons: list[np.ndarray],
        color: tuple[float, float, float, float] = (0.4, 0.7, 1.0, 0.15),
        world_transform=None,
    ) -> None:
        """Draw alpha-blended filled polygons on top of the scene.

        Each polygon is an (N, 3) float32 ndarray (a closed loop in world
        coords). Earcut-triangulates each by projecting onto its dominant
        axis-aligned plane (XY / XZ / YZ — picked from the polygon's geometric
        normal). Depth-test enabled (overlays behind opaque geometry are
        occluded), depth-write disabled (successive overlay passes don't
        z-fight against each other), standard alpha blend.

        Empty polygon list is a no-op (and avoids touching GL state if the
        renderer wasn't initialized — useful for unit tests).

        world_transform: optional 4x4 float64 to convert local-space polygon
        coords to world space before drawing.  Identity (or None) is a no-op.
        """
        if not polygons:
            return
        if not self._initialized:
            return  # tests may call before initialize_gl; no-op
        if self._current_view_matrix is None or self._current_projection_matrix is None:
            return

        import mapbox_earcut

        # Apply world transform when non-identity (entered-instance context).
        if not is_identity_transform(world_transform):
            polygons = [apply_mat(p, world_transform) for p in polygons]

        triangle_vertices: list[float] = []
        for loop in polygons:
            if loop.shape[0] < 3:
                continue
            e1 = loop[1] - loop[0]
            e2 = loop[-1] - loop[0]
            n = np.cross(e1, e2)
            ax, ay, az = abs(float(n[0])), abs(float(n[1])), abs(float(n[2]))
            if az >= ax and az >= ay:
                xy = loop[:, :2].astype(np.float32)
            elif ax >= ay:
                xy = np.stack([loop[:, 1], loop[:, 2]], axis=1).astype(np.float32)
            else:
                xy = np.stack([loop[:, 0], loop[:, 2]], axis=1).astype(np.float32)
            ring_ends = np.array([len(loop)], dtype=np.uint32)
            tri_indices = mapbox_earcut.triangulate_float32(xy, ring_ends)
            tri_indices = np.asarray(tri_indices, dtype=np.int32).reshape(-1, 3)
            for tri in tri_indices:
                for vi in tri:
                    triangle_vertices.extend(
                        [float(loop[vi, 0]), float(loop[vi, 1]), float(loop[vi, 2])]
                    )

        if not triangle_vertices:
            return

        verts = np.array(triangle_vertices, dtype=np.float32)

        GL.glBindVertexArray(self._ghost_fill_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._ghost_fill_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, verts.nbytes, verts, GL.GL_DYNAMIC_DRAW)

        GL.glUseProgram(self._ghost_fill_program)
        GL.glUniformMatrix4fv(
            self._ghost_fill_locs["u_view"], 1, GL.GL_TRUE, self._current_view_matrix
        )
        GL.glUniformMatrix4fv(
            self._ghost_fill_locs["u_projection"], 1, GL.GL_TRUE, self._current_projection_matrix
        )
        GL.glUniform4f(self._ghost_fill_locs["u_color"], *color)

        # GL state: alpha-blended, depth-test on (LEQUAL so coplanar overlays
        # don't z-fight against source-face geometry already in the depth buffer),
        # depth-write off (successive overlay passes don't z-fight each other),
        # no cull (overlay polygons may be viewed from either side during orbit).
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glDepthFunc(GL.GL_LEQUAL)  # fixes hover-highlight on coplanar source face

        GL.glDrawArrays(GL.GL_TRIANGLES, 0, verts.shape[0] // 3)

        GL.glDepthFunc(GL.GL_LESS)  # restore default
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDisable(GL.GL_BLEND)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)


# --- Uniform helpers --------------------------------------------------------
#
# These take a uniform `loc` (pre-cached via _cache_uniform_locations) instead
# of looking it up per-call. A `loc` of -1 means the uniform isn't present in
# the linked program (e.g., optimized out) — silently skipped.


def _set_mat4(loc: int, m: np.ndarray) -> None:
    if loc < 0:
        return
    # GLSL is column-major; numpy is row-major. Transpose flag handles it.
    GL.glUniformMatrix4fv(loc, 1, GL.GL_TRUE, np.asarray(m, dtype=np.float32))


def _set_vec3(loc: int, v: np.ndarray | tuple[float, float, float]) -> None:
    if loc < 0:
        return
    arr = np.asarray(v, dtype=np.float32)
    GL.glUniform3f(loc, float(arr[0]), float(arr[1]), float(arr[2]))


def _set_float(loc: int, x: float) -> None:
    if loc < 0:
        return
    GL.glUniform1f(loc, float(x))


def _set_int(loc: int, x: int) -> None:
    if loc < 0:
        return
    GL.glUniform1i(loc, int(x))
