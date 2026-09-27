"""The editable polygonal scene — thin Python wrapper over the C++ HalfEdgeMesh.

Pure-Python topology is gone. Every public method delegates into the C++
HalfEdgeMesh held in self._mesh, with mapbox-earcut still owning face
triangulation on the Python side.

Idempotent mutators (`add_vertex`; `add_edge` and `add_face_from_loop`
arrive in this same milestone) so tools never have to check existence
before inserting. A single `dirty` flag tracks "has the renderer
seen the current state yet"; the renderer calls `mark_clean()` after
re-uploading buffers.
"""

from __future__ import annotations

from collections import namedtuple
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import mapbox_earcut
import numpy as np

from bermake._core import HalfEdgeMesh, ray_intersect_mesh

if TYPE_CHECKING:
    from bermake._core import RayMeshHit
from bermake.scene.edge import Edge
from bermake.scene.face import Face
from bermake.scene.uv_transfer import transfer_uvs_across_split, uv_at_point_in_face
from bermake.scene.vertex import Vertex

SplitResult = namedtuple("SplitResult", "vertex edge_a edge_b face_a face_b")

# Material id 0 == MaterialLibrary.DEFAULT_ID: the "unpainted / standard look"
# sentinel. Kept as a literal to avoid a scene -> model import.
_DEFAULT_MATERIAL_ID = 0


class Side(Enum):
    """Which side of a face a material applies to.

    FRONT is the side the face normal points out of. Lives here rather than
    beside Material because `model` imports `scene`, not the reverse.
    """

    FRONT = 0
    BACK = 1


@dataclass(frozen=True, slots=True)
class TexturePlacement:
    """A face's override of its material's default texture projection.

    `scale` is uniform: non-uniform sizing lives on the material, whose
    `texture_size` already carries independent width and height in model units.
    Applied as scale, then rotation, then offset, matching
    viewport/uv_projection.apply_placement. That order is contractual.
    """

    offset_u: float = 0.0
    offset_v: float = 0.0
    scale: float = 1.0
    rotation: float = 0.0


DEFAULT_PLACEMENT = TexturePlacement()


def _newell_normal(positions_3d: np.ndarray) -> np.ndarray:
    """Newell normal of an (N, 3) closed loop, unnormalized, as float64 (3,).

    Sums the twice-signed-area contribution of every loop edge rather than
    reading one corner, so the result is the area-weighted normal of the whole
    polygon. Two things follow, and both are why this exists:

    - A loop whose first three vertices are collinear — the exact shape an edge
      split leaves behind — still yields a normal, because no single corner is
      privileged. A `p1-p0 x p2-p0` estimate returns a zero vector there.
    - A loop whose REFLEX corner lands at index 1 yields the right SIGN.
      `cross(p1 - p0, p2 - p0)` is twice the signed area of the triangle
      (p0, p1, p2), which is the ear test at p1 — so it is a reflex vertex
      sitting at index 1, not at the loop's start, that points it the opposite
      way to the polygon's own normal. The sum cannot go wrong that way: the
      convex corners outweigh the reflex ones by exactly the polygon's area.

    Direction matches `np.cross(p1 - p0, p2 - p0)` for every loop on which that
    estimate is valid: right-handed, so a CCW loop viewed from +Z gives +Z.

    The length is proportional to the polygon's area, so it is only zero for a
    genuinely zero-area face. Callers wanting a unit normal normalize and must
    handle that case; `_project_loop_to_2d_for_earcut` instead wants the raw
    components, to pick a dominant axis and its sign.

    Assumes a planar loop. For a non-planar one this returns the normal of its
    projection, which is the standard and usually wanted behaviour.
    """
    p = np.asarray(positions_3d, dtype=np.float64)
    q = np.roll(p, -1, axis=0)
    return np.array(
        [
            float(np.sum((p[:, 1] - q[:, 1]) * (p[:, 2] + q[:, 2]))),
            float(np.sum((p[:, 2] - q[:, 2]) * (p[:, 0] + q[:, 0]))),
            float(np.sum((p[:, 0] - q[:, 0]) * (p[:, 1] + q[:, 1]))),
        ],
        dtype=np.float64,
    )


def _project_loop_to_2d_for_earcut(positions_3d: np.ndarray) -> np.ndarray:
    """Project an (N, 3) loop onto its dominant axis-aligned plane.

    Returns an (N, 2) float32 array suitable for mapbox_earcut. The projection
    plane is the one perpendicular to the largest component of the polygon's
    normal (Newell's method over the whole loop, so a reflex first corner
    cannot flip the estimate the way a single edge cross product can). Without
    this, vertical faces collapse to collinear points in XY and earcut returns
    zero triangles.

    The SIGN of that dominant component decides the ORDER of the axis pair, and
    it matters as much as the choice of plane. Earcut emits triangles wound the
    same way as the 2D ring it is handed, and those indices are fed straight to
    the renderer; if the projection mirrors the loop, every triangle of that
    face comes back wound backwards relative to the face's own normal. Only a
    right-handed pair (u, v) — one where u x v points along the normal —
    preserves the loop's orientation:

        +Z -> (x, y)    -Z -> (y, x)
        +X -> (y, z)    -X -> (z, y)
        +Y -> (z, x)    -Y -> (x, z)

    Picking the pair from abs(n) instead mirrors three of those six cases, which
    is what made three of a box's six faces render with their back material
    (fixed in v0.7.1; invisible before M7.5a gave the two sides distinct looks).
    """
    if positions_3d.shape[0] < 3:
        return positions_3d[:, :2].astype(np.float32)
    # Newell's method: sum over every loop edge, robust to concave corners.
    n = _newell_normal(positions_3d)
    nx, ny, nz = float(n[0]), float(n[1]), float(n[2])
    ax, ay, az = abs(nx), abs(ny), abs(nz)
    if az >= ax and az >= ay:
        u, v = (0, 1) if nz >= 0.0 else (1, 0)  # +Z -> (x, y) / -Z -> (y, x)
    elif ax >= ay:
        u, v = (1, 2) if nx >= 0.0 else (2, 1)  # +X -> (y, z) / -X -> (z, y)
    else:
        u, v = (2, 0) if ny >= 0.0 else (0, 2)  # +Y -> (z, x) / -Y -> (x, z)
    return np.stack([positions_3d[:, u], positions_3d[:, v]], axis=1).astype(np.float32)


def _merge_corner_uvs(
    uv1: np.ndarray | None,
    loop1: Sequence[int],
    uv2: np.ndarray | None,
    loop2: Sequence[int],
    merged_loop: Sequence[int],
) -> np.ndarray | None:
    """Gather one side's per-corner UVs for a dissolve's merged loop.

    Both parents must have stored UVs at all -- one missing means the merged
    face gets none, per the agree-or-drop rule. Where a merged-loop vertex
    was a corner of only one parent, that parent's value is used outright.
    Where it was a corner of BOTH (the two endpoints of the dissolved edge),
    the two must agree within floating-point noise or the whole array is
    dropped rather than picking a side.
    """
    if uv1 is None or uv2 is None:
        return None
    map1 = {int(v): (float(u), float(w)) for v, (u, w) in zip(loop1, uv1, strict=True)}
    map2 = {int(v): (float(u), float(w)) for v, (u, w) in zip(loop2, uv2, strict=True)}
    merged: list[tuple[float, float]] = []
    for raw_v in merged_loop:
        v = int(raw_v)
        a = map1.get(v)
        b = map2.get(v)
        if a is not None and b is not None:
            if not np.allclose(a, b, atol=1e-6):
                return None
            merged.append(a)
        elif a is not None:
            merged.append(a)
        elif b is not None:
            merged.append(b)
        else:
            # A merged-loop vertex came from neither parent's loop, which
            # should not happen: dissolve only ever combines corners the
            # two original faces already had.
            return None
    return np.asarray(merged, dtype=np.float32)


class Scene:
    """Editable polygonal scene with stable integer IDs (C++ HalfEdgeMesh backed)."""

    # Recommended tolerances for faces_are_coplanar (cos(0.5 deg) angle tolerance,
    # 1e-4 world-unit distance tolerance). Kept at the class top since they're
    # part of Scene's public contract, not an implementation detail local to
    # one method.
    _ANGLE_TOL_COS = 0.9999619  # cos(0.5°)
    _DIST_TOL = 1e-4

    def __init__(self) -> None:
        self._mesh = HalfEdgeMesh()
        self._face_materials_front: dict[int, int] = {}
        self._face_materials_back: dict[int, int] = {}
        self._face_placements_front: dict[int, TexturePlacement] = {}
        self._face_placements_back: dict[int, TexturePlacement] = {}
        self._face_uvs_front: dict[int, np.ndarray] = {}
        self._face_uvs_back: dict[int, np.ndarray] = {}
        self._render_dirty = False

    # --- Mutators ---------------------------------------------------------

    def add_vertex(self, position: np.ndarray) -> int:
        """Insert a vertex at `position` (float32 (3,)) and return its ID.

        Idempotent on exact equality (delegated to C++ HalfEdgeMesh).
        """
        if position.dtype != np.float32 or position.shape != (3,):
            position = np.asarray(position, dtype=np.float32).reshape(3)
        return self._mesh.add_vertex(float(position[0]), float(position[1]), float(position[2]))

    def vertex_slab_size(self) -> int:
        """Number of vertex slots ever allocated — live plus tombstoned.

        `add_vertex` only ever appends a slot when it actually creates a vertex
        (a dedup returns an existing id, and `remove_vertex` tombstones rather
        than freeing), so comparing this across an `add_vertex` call is an
        exact O(1) test for "did that call create, or deduplicate?".
        """
        return self._mesh.vertex_slab_size()

    def add_edge(self, v1_id: int, v2_id: int) -> int:
        """Insert an undirected edge between two existing vertices."""
        if v1_id == v2_id:
            raise ValueError(f"self-loop edge requested at vertex {v1_id}")
        if not self._mesh.vertex_is_live(v1_id):
            raise KeyError(f"add_edge: unknown v1_id={v1_id}")
        if not self._mesh.vertex_is_live(v2_id):
            raise KeyError(f"add_edge: unknown v2_id={v2_id}")
        return self._mesh.add_halfedge_pair(v1_id, v2_id)

    def add_face_from_loop(self, ordered_vertex_ids: Sequence[int]) -> int:
        """Insert a face bounded by the given closed vertex loop.

        Triangulates the loop via mapbox-earcut, then passes both the loop and
        the triangulation into the C++ HalfEdgeMesh.  Any loop edges that do
        not yet exist are auto-inserted before the face is created.
        """
        loop = tuple(ordered_vertex_ids)
        if len(loop) < 3:
            raise ValueError(f"face needs at least 3 vertices, got {len(loop)}")
        for vid in loop:
            if not self._mesh.vertex_is_live(vid):
                raise KeyError(f"add_face_from_loop: unknown vertex_id={vid}")

        triangles = self._triangulate_loop(loop)

        # Auto-insert any loop edges that don't already exist (M2 callers do
        # not pre-insert edges before calling add_face_from_loop, but the C++
        # HalfEdgeMesh requires all boundary edges to be present).
        n = len(loop)
        for i in range(n):
            v_from = loop[i]
            v_to = loop[(i + 1) % n]
            self._mesh.add_halfedge_pair(v_from, v_to)

        return self._mesh.add_face_from_loop(list(loop), triangles)

    def _triangulate_loop(self, loop: Sequence[int]) -> list[int]:
        """Earcut a closed vertex loop to a flat list of GLOBAL vertex ids,
        three per triangle, which is the form the kernel takes."""
        xyz = np.empty((len(loop), 3), dtype=np.float32)
        for i, vid in enumerate(loop):
            pos = self._mesh.vertex_position(vid)
            xyz[i] = (pos[0], pos[1], pos[2])
        # Project onto the dominant axis-aligned plane so vertical faces don't
        # collapse to collinear points (which would yield zero triangles).
        projected = _project_loop_to_2d_for_earcut(xyz)
        ring_ends = np.array([len(loop)], dtype=np.uint32)
        local_indices = mapbox_earcut.triangulate_float32(projected, ring_ends)
        local_indices = np.asarray(local_indices, dtype=np.int32).reshape(-1, 3)
        return [int(loop[i]) for tri in local_indices for i in tri]

    def remove_vertex(self, v_id: int) -> None:
        """Remove a vertex. Raises KeyError if not live, ValueError if still referenced."""
        try:
            self._mesh.remove_vertex(v_id)
        except IndexError as e:
            raise KeyError(str(e)) from None
        except ValueError:
            raise

    def remove_edge(self, e_id: int) -> None:
        try:
            self._mesh.remove_edge(e_id)
        except IndexError as e:
            raise KeyError(str(e)) from None
        except ValueError:
            raise

    def remove_face(self, f_id: int) -> None:
        try:
            self._mesh.remove_face(f_id)
        except IndexError as e:
            raise KeyError(str(e)) from None

    def restore_vertex(self, v_id: int, position: np.ndarray) -> None:
        """Restore a previously-removed vertex with its original ID. Used by undo."""
        position = np.asarray(position, dtype=np.float32).reshape(3)
        self._mesh.restore_vertex(v_id, float(position[0]), float(position[1]), float(position[2]))

    def restore_edge(self, e_id: int, v1_id: int, v2_id: int) -> None:
        """Restore a previously-removed edge with its original ID. Used by undo."""
        self._mesh.restore_edge(e_id, v1_id, v2_id)

    def restore_face(self, f_id: int, ordered_vertex_ids: Sequence[int]) -> None:
        """Restore a previously-removed face with its original ID. Used by undo."""
        loop = tuple(ordered_vertex_ids)
        triangles = self._triangulate_loop(loop)
        self._mesh.restore_face(f_id, list(loop), triangles)

    def set_vertex_position(self, v_id: int, position: np.ndarray) -> None:
        """Move an existing live vertex to `position` (float32 (3,)) in place.

        Recomputes cached normals on incident faces (delegated to C++).
        Raises KeyError if the vertex is not live.
        """
        position = np.asarray(position, dtype=np.float32).reshape(3)
        try:
            self._mesh.set_vertex_position(
                v_id, float(position[0]), float(position[1]), float(position[2])
            )
        except IndexError as e:
            raise KeyError(str(e)) from None

    def clear(self) -> None:
        """Reset the scene to empty. Renderer will re-upload empty buffers."""
        self._mesh.clear()
        self._face_materials_front.clear()
        self._face_materials_back.clear()
        self._face_placements_front.clear()
        self._face_placements_back.clear()
        self._face_uvs_front.clear()
        self._face_uvs_back.clear()
        self._render_dirty = True

    # --- Lifecycle (renderer sync) ----------------------------------------

    def mark_clean(self) -> None:
        self._mesh.mark_clean()
        self._render_dirty = False

    # --- M3b picking + face geometry helpers ---------------------------------

    def ray_pick_face(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
    ) -> RayMeshHit | None:
        """Return the closest live face hit, or None.

        Thin wrapper around the C++ bermake._core.ray_intersect_mesh. Caller
        passes a 3-vector origin + 3-vector direction (need not be unit length).
        The returned RayMeshHit exposes .face_id, .t, .point.
        """
        origin_list = [float(origin[0]), float(origin[1]), float(origin[2])]
        direction_list = [float(direction[0]), float(direction[1]), float(direction[2])]
        return ray_intersect_mesh(self._mesh, origin_list, direction_list)

    def face_loop(self, f_id: int) -> list[int]:
        """Ordered boundary vertex IDs of the given live face."""
        if not self._mesh.face_is_live(f_id):
            raise KeyError(f"face_loop: face {f_id} is not live")
        return list(self._mesh.face_loop_vertices(f_id))

    def face_normal(self, f_id: int) -> np.ndarray:
        """Unit geometric normal of the planar face, via Newell's method.

        Summed over the whole boundary loop rather than estimated from its
        first three vertices, so a loop that is merely ORDERED awkwardly is
        handled: three collinear vertices at the start, which is what an edge
        split leaves behind and which used to raise, or a reflex vertex at
        index 1, which used to flip the sign. The direction is unchanged from
        the old first-three estimate wherever that estimate was valid; six
        interactive tools push, offset and paint along it, so the sign is
        contractual (see tests/test_face_normal_newell.py).

        Raises on a loop with fewer than 3 vertices, and on a genuinely
        degenerate face — one of zero area, where the Newell sum really is
        zero-length.

        Assumes the face is planar (a non-planar loop gives the normal of its
        projection, which is the best single normal such a face has).
        """
        if not self._mesh.face_is_live(f_id):
            raise KeyError(f"face_normal: face {f_id} is not live")
        loop = self._mesh.face_loop_vertices(f_id)
        if len(loop) < 3:
            raise ValueError(f"face_normal: face {f_id} has fewer than 3 vertices")
        positions = np.array(
            [self._mesh.vertex_position(vid) for vid in loop],
            dtype=np.float64,
        )
        n = _newell_normal(positions)
        length = float(np.linalg.norm(n))
        if length < 1e-9:
            raise ValueError(f"face_normal: face {f_id} is degenerate (zero area)")
        return (n / length).astype(np.float32)

    def face_center(self, f_id: int) -> np.ndarray:
        """Centroid (mean) of the face's boundary vertex positions."""
        if not self._mesh.face_is_live(f_id):
            raise KeyError(f"face_center: face {f_id} is not live")
        loop = self._mesh.face_loop_vertices(f_id)
        acc = np.zeros(3, dtype=np.float32)
        for vid in loop:
            pos = self._mesh.vertex_position(vid)
            acc += np.asarray(pos, dtype=np.float32)
        return (acc / float(len(loop))).astype(np.float32)

    # ---- M3c additions ----

    def dissolve_edge(self, edge_id: int) -> int | None:
        """Dissolve an edge between two adjacent faces.

        Returns the new (surviving) face id on success, or None if the edge
        is boundary / dead / would create a degenerate result.

        The merged face's sidecars follow an agree-or-drop rule (M7.6a,
        revising M7.5c's D4a): for each of the three kinds of face-keyed
        sidecar (material, placement, stored UVs), on each side, the merged
        face keeps a value only when BOTH parents carried the identical one.
        Absence counts as a value, so a painted face merged with an
        unpainted one comes out unpainted rather than spreading paint onto
        geometry the user never painted. This is what makes a
        split-then-erase round trip lossless when the two halves still
        agree with each other.
        """
        try:
            f1, f2 = self.edge_faces(edge_id)
        except KeyError:
            return None
        if f1 is None or f2 is None:
            return None

        loop1 = self.face_loop(f1)
        loop2 = self.face_loop(f2)
        materials = {
            side: (self.face_material(f1, side), self.face_material(f2, side))
            for side in (Side.FRONT, Side.BACK)
        }
        placements = {
            side: (self.face_placement(f1, side), self.face_placement(f2, side))
            for side in (Side.FRONT, Side.BACK)
        }
        stored_uvs = {
            side: (self.face_uvs(f1, side), self.face_uvs(f2, side))
            for side in (Side.FRONT, Side.BACK)
        }

        result = self._mesh.dissolve_edge(edge_id)
        if result == self._mesh.INVALID_ID:
            return None
        merged = int(result)
        merged_loop = self.face_loop(merged)

        # Below, disagreement is handled by simply not writing anything --
        # never by an explicit clear. That is only safe because `merged` is
        # always a BRAND NEW id: face ids are never reused (`D5`, docs/
        # 2026-09-15-M7.5c-uv-storage-design.md -- an orphan sidecar entry
        # on a dead face can never be misattributed to a later face for the
        # same reason), so a fresh id starts with no entry in any of the six
        # sidecar dicts, and "don't write" already means "unpainted /
        # identity placement / no stored UVs". This is the inverse use of
        # that same fact: D5 relies on it to justify never CLEANING UP a
        # dead face's entries; this relies on it to justify never CLEARING
        # a new face's. If face ids ever became recyclable, both would need
        # revisiting -- this method would need an explicit clear on
        # disagreement to avoid inheriting a stale entry from whatever face
        # last held this id.
        for side in (Side.FRONT, Side.BACK):
            m1, m2 = materials[side]
            if m1 == m2:
                self.set_face_material(merged, m1, side)

            p1, p2 = placements[side]
            if p1 == p2:
                self.set_face_placement(merged, p1, side)

            uv1, uv2 = stored_uvs[side]
            merged_uv = _merge_corner_uvs(uv1, loop1, uv2, loop2, merged_loop)
            if merged_uv is not None:
                self.set_face_uvs(merged, merged_uv, side)

        return merged

    def faces_are_coplanar(self, f1_id: int, f2_id: int) -> bool:
        """True iff the two faces' normals and planes agree within tolerance.

        Applies Scene's recommended tolerances (_ANGLE_TOL_COS, _DIST_TOL) to
        HalfEdgeMesh.faces_are_coplanar, which takes tolerances as required
        arguments with no built-in defaults of its own.
        """
        return bool(
            self._mesh.faces_are_coplanar(
                f1_id,
                f2_id,
                self._ANGLE_TOL_COS,
                self._DIST_TOL,
            )
        )

    def face_edges(self, f_id: int) -> list[int]:
        """Edge IDs around the face's boundary loop, in order.

        For each consecutive vertex pair (v_i, v_{i+1}) in the boundary loop,
        looks up the edge id via the kernel's idempotent add_halfedge_pair,
        which returns the EXISTING edge id for an already-present live pair
        (no mutation). add_halfedge_pair returns the edge id directly — do
        NOT divide by 2.
        """
        if not self._mesh.face_is_live(f_id):
            raise KeyError(f"Face {f_id} is not live")
        loop = list(self._mesh.face_loop_vertices(f_id))
        edges: list[int] = []
        n = len(loop)
        for i in range(n):
            v_a, v_b = loop[i], loop[(i + 1) % n]
            edge_id = self._mesh.add_halfedge_pair(v_a, v_b)  # returns the edge id
            edges.append(int(edge_id))
        return edges

    def edge_faces(self, e_id: int) -> tuple[int | None, int | None]:
        """The pair of face ids on each side of the edge. None if no face on that side."""
        if not self._mesh.edge_is_live(e_id):
            raise KeyError(f"Edge {e_id} is not live")
        he_a = 2 * e_id
        he_b = 2 * e_id + 1
        f_a = self._mesh.halfedge_face(he_a)
        f_b = self._mesh.halfedge_face(he_b)
        invalid_id = self._mesh.INVALID_ID
        return (
            None if f_a == invalid_id else int(f_a),
            None if f_b == invalid_id else int(f_b),
        )

    def edge_is_boundary(self, e_id: int) -> bool:
        """True iff the edge has fewer than two incident faces."""
        f_a, f_b = self.edge_faces(e_id)
        return f_a is None or f_b is None

    def edge_between(self, v1_id: int, v2_id: int) -> int | None:
        """The live edge joining `v1_id` and `v2_id`, or None.

        Non-mutating, unlike add_halfedge_pair, which this replaces at the
        two call sites that were using it as a lookup (#26).
        """
        return self._mesh.edge_between(v1_id, v2_id)

    def edge_is_live(self, e_id: int) -> bool:
        """True while `e_id` refers to a live edge slot."""
        return self._mesh.edge_is_live(e_id)

    # ---- M3d additions ----

    def point_on_edge(self, e_id: int, t: float) -> np.ndarray:
        """World point at parameter t along edge e (t measured v1→v2 of edge())."""
        e = self.edge(e_id)
        pa = self.vertex(e.v1_id).position
        pb = self.vertex(e.v2_id).position
        return (pa + float(t) * (pb - pa)).astype(np.float32)

    def closest_point_on_edge(self, e_id: int, world_point: np.ndarray) -> tuple[np.ndarray, float]:
        """Closest point on edge segment to `world_point`, plus its clamped t∈[0,1]."""
        e = self.edge(e_id)
        pa = self.vertex(e.v1_id).position
        pb = self.vertex(e.v2_id).position
        ab = pb - pa
        denom = float(np.dot(ab, ab))
        if denom < 1e-18:
            return pa.astype(np.float32), 0.0
        t = float(np.dot(np.asarray(world_point, dtype=np.float32) - pa, ab) / denom)
        t = max(0.0, min(1.0, t))
        return (pa + t * ab).astype(np.float32), t

    def split_edge(self, e_id: int, t: float) -> SplitResult | None:
        """Split edge e at parameter t. Returns a SplitResult (face_* None for a
        boundary edge's empty side), or None if the split is invalid.

        The kernel removes both incident faces and rebuilds them with NEW ids,
        so every face-keyed sidecar would be dropped. Capture them first, then
        replay them onto the rebuilt faces. The old faces' entries are left in
        place deliberately: ids are never reused, and SplitEdgeCommand.undo
        restores the ORIGINAL ids, so the entries still sitting there are what
        makes undo restore the paint.
        """
        e_id = int(e_id)
        t = float(t)
        try:
            edge = self.edge(e_id)
        except KeyError:
            return None
        va, vb = edge.v1_id, edge.v2_id
        f_before = self.edge_faces(e_id)
        captured = [None if fid is None else (fid, tuple(self.face_loop(fid))) for fid in f_before]

        res = self._mesh.split_edge(e_id, t)
        if res is None:
            return None
        invalid = self._mesh.INVALID_ID
        result = SplitResult(
            vertex=int(res.vertex),
            edge_a=int(res.edge_a),
            edge_b=int(res.edge_b),
            face_a=None if res.face_a == invalid else int(res.face_a),
            face_b=None if res.face_b == invalid else int(res.face_b),
        )

        for old, new_fid in zip(captured, (result.face_a, result.face_b), strict=True):
            if old is None or new_fid is None:
                continue
            old_fid, old_loop = old
            self._transfer_face_attributes(
                old_fid, new_fid, old_loop, self.face_loop(new_fid), result.vertex, va, vb, t
            )
        return result

    def split_face(self, f_id: int, chain: Sequence[int]) -> tuple[int, int] | None:
        """Divide face `f_id` along `chain`, a path whose two ends lie on its
        boundary loop. Returns the two new face ids, or None if refused.

        The two sub-loops are built here rather than in the kernel because the
        triangulator is in Python and `add_face_from_loop` already takes a
        caller-supplied triangulation. Walking the parent's loop forward in
        BOTH halves is what preserves its winding, and traversing the chain in
        opposite directions is what makes the kernel's two add_face_from_loop
        calls claim opposite half-edges of each chain edge.
        """
        f_id = int(f_id)
        chain = [int(v) for v in chain]
        if len(chain) < 2 or len(set(chain)) != len(chain):
            return None

        try:
            loop = self.face_loop(f_id)
        except KeyError:
            return None

        first, last = chain[0], chain[-1]
        if first not in loop or last not in loop:
            return None
        if any(v in loop for v in chain[1:-1]):
            return None  # an interior chain vertex on the loop is not a chord

        i, j = loop.index(first), loop.index(last)
        n = len(loop)
        forward = [loop[(i + k) % n] for k in range((j - i) % n + 1)]
        backward = [loop[(j + k) % n] for k in range((i - j) % n + 1)]
        # No separate "ends are the same loop vertex" guard is needed here:
        # first == last would make forward/backward length 1, but first ==
        # last is chain[0] == chain[-1], which the duplicate-vertex check
        # above already rejects.

        interior = chain[1:-1]
        loop_a = forward + interior[::-1]
        loop_b = backward + interior
        if len(loop_a) < 3 or len(loop_b) < 3:
            return None

        out = self._mesh.split_face(
            f_id,
            loop_a,
            self._triangulate_loop(loop_a),
            loop_b,
            self._triangulate_loop(loop_b),
        )
        if out[0] == self._mesh.INVALID_ID:
            return None
        new_a, new_b = int(out[0]), int(out[1])
        self._transfer_subloop_face_attributes(f_id, new_a, loop, self.face_loop(new_a))
        self._transfer_subloop_face_attributes(f_id, new_b, loop, self.face_loop(new_b))
        return new_a, new_b

    def copy_face_attributes_from(self, source: Scene, src_fid: int, dst_fid: int) -> None:
        """Copy one face's sidecars from another Scene onto a face of this one.

        For a face rebuilt VERBATIM in a different Scene: same corners, same
        order, same plane, a brand new id. That is what Make Group and Explode
        do, replaying geometry through `add_face_from_loop`, which mints a
        fresh id and carries nothing with it. Without this every lift silently
        un-paints the geometry it moves (#114).

        Materials and placements copy as they are. Stored UVs are parallel to
        the loop, so they copy only when both loops have the same length; a
        mismatch means the face was not in fact rebuilt verbatim, and dropping
        the array returns that face to the projection rather than mapping
        corners that do not correspond.

        Nothing is written for an attribute the source does not have. Writing
        a default would make every lifted face look painted to
        `faces_with_material`, which drives the user-facing "used on N faces"
        count.

        This is the verbatim sibling of `_transfer_face_attributes`, which
        handles the one case where a rebuilt face's loop GREW: a split inserts
        a corner and interpolates a UV for it. A dissolve's merge -- a loop
        SHRINKING two faces into one -- is handled by `dissolve_edge`'s own
        agree-or-drop rule (M7.6a), not by this method.
        """
        src_fid = int(src_fid)
        dst_fid = int(dst_fid)
        loops_match = len(source.face_loop(src_fid)) == len(self.face_loop(dst_fid))

        for side in (Side.FRONT, Side.BACK):
            material = source._materials_for(side).get(src_fid)
            if material is not None:
                self._materials_for(side)[dst_fid] = material

            placement = source._placements_for(side).get(src_fid)
            if placement is not None:
                self._placements_for(side)[dst_fid] = placement

            stored = source._uvs_for(side).get(src_fid)
            if stored is not None and loops_match:
                self._uvs_for(side)[dst_fid] = np.array(stored, dtype=np.float32)

        self._render_dirty = True

    def _transfer_face_attributes(
        self,
        old_fid: int,
        new_fid: int,
        old_loop: Sequence[int],
        new_loop: Sequence[int],
        w: int,
        va: int,
        vb: int,
        t: float,
    ) -> None:
        """Replay one destroyed face's sidecars onto the face that replaced it.

        Materials and placements copy verbatim: neither depends on the loop, and
        a split does not move the face's plane. Stored UVs are parallel to the
        loop, so they gain one entry interpolated at the split parameter.

        A UV array that does not transfer cleanly is dropped rather than
        guessed at, which returns that face to the projection.
        """
        for side in (Side.FRONT, Side.BACK):
            materials = self._materials_for(side)
            if old_fid in materials:
                materials[int(new_fid)] = materials[old_fid]

            placements = self._placements_for(side)
            if old_fid in placements:
                placements[int(new_fid)] = placements[old_fid]

            stored = self._uvs_for(side).get(old_fid)
            if stored is None:
                continue
            moved = transfer_uvs_across_split(
                [(float(u), float(v)) for u, v in stored],
                old_loop,
                new_loop,
                w,
                va,
                vb,
                t,
            )
            if moved is not None:
                self._uvs_for(side)[int(new_fid)] = np.asarray(moved, dtype=np.float32)

    def _transfer_subloop_face_attributes(
        self,
        old_fid: int,
        new_fid: int,
        old_loop: Sequence[int],
        new_loop: Sequence[int],
    ) -> None:
        """Replay one destroyed face's sidecars onto ONE of the two faces a
        split replaced it with.

        Materials and placements copy verbatim, exactly as in the other two
        siblings: neither depends on the loop, and a split does not move the
        face's plane. Stored UVs are parallel to the loop, so every corner of
        `new_loop` needs a value. A corner that was already on `old_loop`
        copies that corner's UV directly. A corner that was not -- a chain
        vertex, present in `new_loop` only because the split put it there --
        carried no UV of its own, and is resolved by barycentric
        interpolation inside the PARENT's own triangulation (see
        `uv_transfer.uv_at_point_in_face`). A corner the parent's
        triangulation cannot place drops that side's whole array rather than
        leaving some corners guessed and others exact.

        This is the subloop sibling of `copy_face_attributes_from` (identical
        loop, different Scene) and `_transfer_face_attributes` (loop grew by
        exactly one corner at a known edge parameter). A split's new loop is
        neither of those shapes: it is an ARC of the old loop plus vertices
        that were never on it, which is what the barycentric step exists for
        and why this is a third function rather than a generalisation of the
        other two.
        """
        old_fid = int(old_fid)
        sidecar_dicts = (
            self._materials_for(Side.FRONT),
            self._materials_for(Side.BACK),
            self._placements_for(Side.FRONT),
            self._placements_for(Side.BACK),
            self._uvs_for(Side.FRONT),
            self._uvs_for(Side.BACK),
        )
        if not any(old_fid in d for d in sidecar_dicts):
            return  # an unpainted face has nothing to replay onto either child

        new_fid = int(new_fid)
        old_loop = list(old_loop)
        new_loop = list(new_loop)
        old_index = {int(v): i for i, v in enumerate(old_loop)}
        local_triangles = [old_index[int(v)] for v in self._triangulate_loop(old_loop)]
        positions = np.array([self.vertex(v).position for v in old_loop], dtype=np.float64)

        for side in (Side.FRONT, Side.BACK):
            material = self._materials_for(side).get(old_fid)
            if material is not None:
                self._materials_for(side)[new_fid] = material

            placement = self._placements_for(side).get(old_fid)
            if placement is not None:
                self._placements_for(side)[new_fid] = placement

            stored = self._uvs_for(side).get(old_fid)
            if stored is None:
                continue

            new_uvs: list[tuple[float, float]] | None = []
            for raw_vid in new_loop:
                loop_vid = int(raw_vid)
                if loop_vid in old_index:
                    uv = stored[old_index[loop_vid]]
                    new_uvs.append((float(uv[0]), float(uv[1])))
                    continue
                point = self.vertex(loop_vid).position.astype(np.float64)
                resolved = uv_at_point_in_face(positions, stored, local_triangles, point)
                if resolved is None:
                    new_uvs = None
                    break
                new_uvs.append(resolved)
            if new_uvs is not None:
                self._uvs_for(side)[new_fid] = np.asarray(new_uvs, dtype=np.float32)

    # --- Queries ----------------------------------------------------------

    @property
    def dirty(self) -> bool:
        return self._mesh.is_dirty() or self._render_dirty

    def vertex(self, v_id: int) -> Vertex:
        if not self._mesh.vertex_is_live(v_id):
            raise KeyError(f"vertex_id {v_id} is not live")
        pos = self._mesh.vertex_position(v_id)
        return Vertex(id=v_id, position=np.array(pos, dtype=np.float32))

    def edge(self, e_id: int) -> Edge:
        if not self._mesh.edge_is_live(e_id):
            raise KeyError(f"edge_id {e_id} is not live")
        verts = self._mesh.edge_vertices(e_id)
        return Edge(id=e_id, v1_id=verts[0], v2_id=verts[1])

    def face(self, f_id: int) -> Face:
        if not self._mesh.face_is_live(f_id):
            raise KeyError(f"face_id {f_id} is not live")
        loop = self._mesh.face_loop_vertices(f_id)
        tris = self._mesh.face_triangles(f_id)
        triangles = np.array(tris, dtype=np.int32).reshape(-1, 3)
        return Face(
            id=f_id,
            loop_vertex_ids=tuple(loop),
            plane_normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            triangles=triangles,
        )

    def vertices_iter(self) -> Iterable[Vertex]:
        v = self._mesh.next_live_vertex(0)
        while v != HalfEdgeMesh.INVALID_ID:
            yield self.vertex(v)
            v = self._mesh.next_live_vertex(v + 1)

    def edges_iter(self) -> Iterable[Edge]:
        e = self._mesh.next_live_edge(0)
        while e != HalfEdgeMesh.INVALID_ID:
            yield self.edge(e)
            e = self._mesh.next_live_edge(e + 1)

    def faces_iter(self) -> Iterable[Face]:
        f = self._mesh.next_live_face(0)
        while f != HalfEdgeMesh.INVALID_ID:
            yield self.face(f)
            f = self._mesh.next_live_face(f + 1)

    def find_vertex_near(self, world_xyz: np.ndarray, tolerance: float) -> int | None:
        """Return the ID of the live vertex closest to `world_xyz` within `tolerance`."""
        best_id: int | None = None
        best_d2 = tolerance * tolerance
        v = self._mesh.next_live_vertex(0)
        while v != HalfEdgeMesh.INVALID_ID:
            pos = self._mesh.vertex_position(v)
            d0 = pos[0] - float(world_xyz[0])
            d1 = pos[1] - float(world_xyz[1])
            d2 = pos[2] - float(world_xyz[2])
            d_sq = d0 * d0 + d1 * d1 + d2 * d2
            if d_sq <= best_d2:
                best_d2 = d_sq
                best_id = v
            v = self._mesh.next_live_vertex(v + 1)
        return best_id

    # --- Per-face material sidecar ----------------------------------------

    def _materials_for(self, side: Side) -> dict[int, int]:
        return self._face_materials_front if side is Side.FRONT else self._face_materials_back

    def set_face_material(self, f_id: int, material_id: int, side: Side = Side.FRONT) -> None:
        """Paint one side of a face. material_id 0 (Default) clears that side."""
        if material_id == _DEFAULT_MATERIAL_ID:
            self.clear_face_material(f_id, side)
            return
        self._materials_for(side)[f_id] = material_id
        self._render_dirty = True

    def clear_face_material(self, f_id: int, side: Side = Side.FRONT) -> None:
        """Remove any material from one side of a face."""
        if self._materials_for(side).pop(f_id, None) is not None:
            self._render_dirty = True

    def face_material(self, f_id: int, side: Side = Side.FRONT) -> int:
        """Material id painted on one side, or 0 (Default) if unpainted."""
        return self._materials_for(side).get(f_id, _DEFAULT_MATERIAL_ID)

    def faces_with_material(self, material_id: int) -> list[tuple[int, Side]]:
        """Every painted (face, side) referencing `material_id`.

        Only painted faces are stored, since painting Default clears the entry,
        so this scans the two sidecars rather than walking the mesh.
        """
        found = [(f, Side.FRONT) for f, m in self._face_materials_front.items() if m == material_id]
        found += [(f, Side.BACK) for f, m in self._face_materials_back.items() if m == material_id]
        return found

    # --- Per-face texture placement sidecar --------------------------------

    def _placements_for(self, side: Side) -> dict[int, TexturePlacement]:
        return self._face_placements_back if side is Side.BACK else self._face_placements_front

    def face_placement(self, f_id: int, side: Side = Side.FRONT) -> TexturePlacement:
        """This face side's placement, or the identity if it was never adjusted."""
        return self._placements_for(side).get(int(f_id), DEFAULT_PLACEMENT)

    def set_face_placement(
        self, f_id: int, placement: TexturePlacement, side: Side = Side.FRONT
    ) -> None:
        """Store an adjustment, or clear the entry if it is the identity.

        The sidecars hold only adjusted faces, mirroring how painting Default
        clears a material entry, which is what keeps faces_with_placement cheap
        and the serialized form small.
        """
        if placement == DEFAULT_PLACEMENT:
            self.clear_face_placement(f_id, side)
            return
        self._placements_for(side)[int(f_id)] = placement
        self._render_dirty = True

    def clear_face_placement(self, f_id: int, side: Side = Side.FRONT) -> None:
        if self._placements_for(side).pop(int(f_id), None) is not None:
            self._render_dirty = True

    def faces_with_placement(self) -> list[tuple[int, Side]]:
        """Every adjusted (face, side) pair. Never walks the mesh."""
        pairs = [(f, Side.FRONT) for f in self._face_placements_front]
        pairs += [(f, Side.BACK) for f in self._face_placements_back]
        return pairs

    # --- Stored per-corner UVs (M7.5c) ------------------------------------
    #
    # Imported geometry arrives with UVs baked per corner; M7.5b's projection
    # cannot represent them. A face either has an array parallel to its
    # boundary loop or it does not, so there is no default to compare against
    # and face_uvs returns None rather than a sentinel.

    def _uvs_for(self, side: Side) -> dict[int, np.ndarray]:
        return self._face_uvs_back if side is Side.BACK else self._face_uvs_front

    def face_uvs(self, f_id: int, side: Side = Side.FRONT) -> np.ndarray | None:
        """(L, 2) float32 UVs parallel to the face's loop, or None."""
        stored = self._uvs_for(side).get(int(f_id))
        return None if stored is None else stored.copy()

    def set_face_uvs(self, f_id: int, uvs, side: Side = Side.FRONT) -> None:
        """Store per-corner UVs. `uvs` must have one entry per loop vertex."""
        arr = np.asarray(uvs, dtype=np.float32).reshape(-1, 2)
        expected = len(self.face_loop(int(f_id)))
        if arr.shape[0] != expected:
            raise ValueError(
                f"set_face_uvs: face {f_id} has {expected} corners, got {arr.shape[0]}"
            )
        self._uvs_for(side)[int(f_id)] = arr
        self._render_dirty = True

    def clear_face_uvs(self, f_id: int, side: Side = Side.FRONT) -> None:
        if self._uvs_for(side).pop(int(f_id), None) is not None:
            self._render_dirty = True

    def faces_with_uvs(self) -> list[tuple[int, Side]]:
        """Every (face, side) carrying stored UVs, in no particular order."""
        pairs = [(f, Side.FRONT) for f in self._face_uvs_front]
        pairs += [(f, Side.BACK) for f in self._face_uvs_back]
        return pairs

    def face_triangle_materials(self, side: Side = Side.FRONT) -> np.ndarray:
        """Per-triangle material id, aligned 1:1 with face_triangle_buffer().

        Walks live faces in next_live_face ascending order (the exact order the
        C++ face_triangle_buffer uses) and repeats each face's material id by
        its triangle count.
        """
        materials = self._materials_for(side)
        mats: list[int] = []
        f = self._mesh.next_live_face(0)
        while f != HalfEdgeMesh.INVALID_ID:
            n_tris = len(self._mesh.face_triangles(f)) // 3
            mats.extend([materials.get(f, _DEFAULT_MATERIAL_ID)] * n_tris)
            f = self._mesh.next_live_face(f + 1)
        return np.asarray(mats, dtype=np.int64)

    def face_triangle_face_ids(self) -> np.ndarray:
        """Per-triangle owning face id, aligned 1:1 with face_triangle_buffer().

        Same next_live_face walk as face_triangle_materials, which is the exact
        order the C++ face_triangle_buffer uses. Texture UVs are built per face,
        so this is how a triangle finds the face whose plane it was projected on.
        """
        ids: list[int] = []
        f = self._mesh.next_live_face(0)
        while f != HalfEdgeMesh.INVALID_ID:
            n_tris = len(self._mesh.face_triangles(f)) // 3
            ids.extend([f] * n_tris)
            f = self._mesh.next_live_face(f + 1)
        return np.asarray(ids, dtype=np.int64)

    def face_triangle_loop_indices(self) -> np.ndarray:
        """Per-CORNER index into the owning face's boundary loop, (3T,) int32.

        Aligned 1:1 with the rows of face_triangle_buffer()'s positions, and so
        three times longer than face_triangle_face_ids, which is per triangle.

        Stored UVs are parallel to a face's loop; the render buffer is per
        triangle corner and repeats loop vertices. This is how a corner finds
        its UV: stored[loop_indices[corner]].
        """
        out: list[int] = []
        f = self._mesh.next_live_face(0)
        while f != HalfEdgeMesh.INVALID_ID:
            loop = self._mesh.face_loop_vertices(f)
            position = {int(v): i for i, v in enumerate(loop)}
            for vid in self._mesh.face_triangles(f):
                out.append(position[int(vid)])
            f = self._mesh.next_live_face(f + 1)
        return np.asarray(out, dtype=np.int32).reshape(-1)

    def face_loop_indices(self, f_id: int) -> np.ndarray:
        """One face's slice of face_triangle_loop_indices, (3t,) int32.

        The overlay that substitutes stored UVs reads the corners of the
        faces that have them, which is typically a handful out of thousands.
        Building the whole-scene array to serve those is what gave a single
        stored face a cost proportional to the definition's total face count
        (#117); this serves one face in time proportional to that face.
        """
        loop = self._mesh.face_loop_vertices(f_id)
        position = {int(v): i for i, v in enumerate(loop)}
        return np.asarray(
            [position[int(v)] for v in self._mesh.face_triangles(f_id)], dtype=np.int32
        )

    # --- Render-buffer projection -----------------------------------------

    def edge_line_buffer(self) -> np.ndarray:
        buf = self._mesh.edge_line_buffer()
        if not buf:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray(buf, dtype=np.float32).reshape(-1, 3)

    def face_triangle_buffer(self) -> tuple[np.ndarray, np.ndarray]:
        positions, normals = self._mesh.face_triangle_buffer()
        if not positions:
            empty = np.zeros((0, 3), dtype=np.float32)
            return empty, empty
        pos = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        nrm = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
        return pos, nrm
