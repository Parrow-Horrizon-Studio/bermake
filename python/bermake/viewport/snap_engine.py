"""Snap & inference engine: the selection policy over the candidate generators.

Gathers candidates for every `SnapKind` in tolerance (the generators live in
`snap_candidates.py`) and picks one. Precedence is D12's explicit table,
`_PRECEDENCE` below, NOT the numeric value of `SnapKind`: the two were
decoupled in M7.6b so kinds could be added to the enum without renumbering
the ordering. Ties inside one kind break on screen distance, then depth.

Stateless by design (D1). Acquisition (hover-dwell) and inference locking are
stateful and live in `inference.py`, which wraps this.
"""

from __future__ import annotations

import numpy as np

from bermake.geometry.ray import (
    closest_point_on_segment_to_ray as _closest_point_on_segment_to_ray,
)
from bermake.geometry.ray import (
    closest_points_two_lines as _closest_points_two_lines,
)
from bermake.geometry.transforms import apply_mat, is_identity_transform, mat_invert
from bermake.viewport.snap_candidates import Candidate as _Candidate
from bermake.viewport.snap_candidates import (
    axis_candidates as _axis_candidates,
)
from bermake.viewport.snap_candidates import (
    directional_candidates as _directional_candidates,
)
from bermake.viewport.snap_candidates import (
    edge_point_candidates as _edge_point_candidates,
)
from bermake.viewport.snap_candidates import (
    endpoint_candidates as _endpoint_candidates,
)
from bermake.viewport.snap_candidates import (
    face_candidate as _face_candidate,
)
from bermake.viewport.snap_candidates import (
    from_point_candidates as _from_point_candidates,
)
from bermake.viewport.snap_candidates import (
    guide_candidates as _guide_candidates,
)
from bermake.viewport.snap_candidates import (
    guide_intersection_candidates as _guide_intersection_candidates,
)
from bermake.viewport.snap_candidates import (
    intersection_candidates as _intersection_candidates,
)
from bermake.viewport.snap_types import SnapKind, SnapResult

# `SnapKind` and `SnapResult` move to `snap_types.py` in M7.6b (so `snap_engine`
# and `snap_candidates` can share the vocabulary without importing each other)
# but are re-exported here unchanged: every existing importer across
# `python/bermake/tools/`, `python/bermake/viewport/` and `tests/` keeps working
# with no edit. `_closest_point_on_segment_to_ray` and `_closest_points_two_lines`
# are no longer called from this module directly (their call sites moved to
# snap_candidates.py with the generators that used them); they are kept
# importable from here because existing code, including
# tests/test_snap_engine.py, imports them from this module by name. Both are
# listed below so lint does not treat them as unused re-exports.
__all__ = [
    "SnapKind",
    "SnapResult",
    "_closest_point_on_segment_to_ray",
    "_closest_points_two_lines",
]


# Snap-marker colors, keyed by kind. Shared by tools (overlay color); the
# renderer is shape-only. AXIS_LOCK has no marker color (the rubber-band shows
# the axis color instead).
MARKER_COLOR_BY_KIND = {
    SnapKind.GRID: (0.70, 0.70, 0.70),
    SnapKind.MIDPOINT: (0.13, 0.77, 0.84),  # cyan
    SnapKind.ENDPOINT: (0.15, 0.75, 0.26),  # green
    SnapKind.ON_EDGE: (0.89, 0.23, 0.18),  # red
    SnapKind.ON_FACE: (0.18, 0.42, 0.88),  # blue
    # Magenta moved to the directional pair below (spec D14): SketchUp reserves
    # magenta for Parallel and Perpendicular, so keeping it on INTERSECTION would
    # collide with those once M7.6b's directional inferences ship. Deliberate
    # change to shipped behaviour, not a bug fix.
    SnapKind.INTERSECTION: (0.10, 0.10, 0.12),  # near-black, SketchUp's convention
    SnapKind.PARALLEL: (0.82, 0.23, 0.82),  # magenta
    SnapKind.PERPENDICULAR: (0.82, 0.23, 0.82),  # magenta
    # Spec 2.2: both guide inferences share one neutral grey, matching the
    # guide's own construction-geometry color (draw_plan/viewport_widget).
    SnapKind.ON_GUIDE: (0.45, 0.45, 0.52),
    SnapKind.GUIDE_POINT: (0.45, 0.45, 0.52),
    # FROM_POINT gets no entry, and every consumer looks this dict up as
    # MARKER_COLOR_BY_KIND.get(snap.kind, <neutral>), so today it just renders
    # neutral like any other unlisted kind. Colouring it by snap.axis (like
    # AXIS_LOCK's rubber-band) would need plumbing snap.axis through all
    # eleven call sites for a Task-4-scale change; deferred, not done.
}

# Precedence, highest first. Decoupled from the enum's integer values.
#
# ON_FACE sits second from last, above only the GRID fallback. Its candidate is
# ALWAYS in tolerance whenever the cursor ray hits a face, because the candidate's
# position is the ray-face intersection and therefore reprojects onto the cursor.
# Ranking it high meant it swallowed every directional inference while drawing on
# a face, which is the ordinary case. The face itself is not lost: `snap()` copies
# `face_id` onto whichever candidate wins.
_PRECEDENCE = [
    SnapKind.ENDPOINT,
    SnapKind.INTERSECTION,
    SnapKind.MIDPOINT,
    SnapKind.GUIDE_POINT,
    SnapKind.ON_EDGE,
    SnapKind.ON_GUIDE,
    SnapKind.PERPENDICULAR,
    SnapKind.PARALLEL,
    SnapKind.FROM_POINT,
    SnapKind.AXIS_LOCK,
    SnapKind.ON_FACE,
    SnapKind.GRID,
]
_PRECEDENCE_RANK = {k: i for i, k in enumerate(_PRECEDENCE)}  # lower = higher precedence


class SnapEngine:
    """SketchUp-style snap & inference engine."""

    PIXEL_TOLERANCE = 8.0  # screen-space proximity for point/edge inferences
    AXIS_DEG_TOLERANCE = 5.0
    GRID_SIZE_WORLD = 1.0

    def snap(
        self,
        cursor_screen,
        viewport_size,
        camera,
        scene,
        anchor=None,
        world_transform=None,
        acquired=None,
        plane_normal=None,
        guides=None,
        guide_points=None,
    ) -> SnapResult:
        """Return the chosen 3D snap for the given cursor.

        cursor_screen: (px, py) pixel coords. viewport_size: (width, height).
        The cursor ray and ground hit are derived internally from the camera.

        world_transform: optional (4,4) matrix mapping local (scene) coords to world.
        None or identity → behaviour is identical to the no-arg call (regression-safe).
        When non-identity, vertex/edge positions are transformed to world before screen
        projection, and the camera ray is transformed to local space for face picking.

        acquired: an optional Acquired (see inference.py) driving PARALLEL,
        PERPENDICULAR and FROM_POINT. None reproduces the pre-M7.6b path exactly.
        plane_normal: the active drawing plane's normal, used to resolve
        PERPENDICULAR inside that plane. Ignored when acquired is not an edge.
        guides: an optional sequence of world-space (origin, direction) pairs
        driving ON_GUIDE, and guide-versus-scene-edge / guide-versus-guide
        INTERSECTION. guide_points: an optional sequence of world-space
        positions driving GUIDE_POINT. Both already in world space; the
        caller (ViewportWidget) is responsible for the local-to-world
        conversion and for withholding hidden guides entirely.
        """
        if cursor_screen is None or camera is None or scene is None:
            return self._none()
        px, py = float(cursor_screen[0]), float(cursor_screen[1])
        width = int(viewport_size[0])
        height = int(viewport_size[1])
        ray_origin, ray_dir = camera.ray_from_screen(px, py, width, height)
        ground_hit = camera.ray_intersect_ground(px, py, width, height)

        # world_transform support: local→world for screen projections; world→local for ray.
        use_wt = not is_identity_transform(world_transform)
        if use_wt:
            wt = np.asarray(world_transform, dtype=np.float64)
            inv = mat_invert(wt)

            def _to_world(local_pos):
                return apply_mat(local_pos, wt)[0]

            # Camera ray in local space (for face picking).
            ray_origin_local = apply_mat(ray_origin, inv)[0]
            ray_dir_local = (inv[:3, :3] @ np.asarray(ray_dir, dtype=np.float64)).astype(np.float32)
        else:

            def _to_world(local_pos):  # type: ignore[misc]
                return local_pos

            ray_origin_local = ray_origin
            ray_dir_local = ray_dir

        cands: list[_Candidate] = []
        cands += _endpoint_candidates(
            px, py, width, height, camera, scene, _to_world, self.PIXEL_TOLERANCE
        )
        cands += _edge_point_candidates(
            px,
            py,
            width,
            height,
            camera,
            scene,
            ray_origin,
            ray_dir,
            _to_world,
            ray_origin_local,
            ray_dir_local,
            pixel_tolerance=self.PIXEL_TOLERANCE,
        )
        face_cand = _face_candidate(ray_origin_local, ray_dir_local, scene)
        if face_cand is not None:
            if use_wt:
                # Surface the snap result in world space.
                face_cand.world_position = apply_mat(face_cand.world_position, wt)[0]
            cands.append(face_cand)
        cands += _guide_candidates(
            px, py, width, height, camera, guides or (), guide_points or (), self.PIXEL_TOLERANCE
        )
        cands += _guide_intersection_candidates(
            px, py, width, height, camera, scene, guides or (), _to_world, self.PIXEL_TOLERANCE
        )
        if anchor is not None:
            a = np.asarray(anchor, dtype=np.float32)
            cands += _axis_candidates(
                px, py, width, height, camera, a, ray_origin, ray_dir, self.PIXEL_TOLERANCE
            )
            cands += _intersection_candidates(
                px, py, width, height, camera, scene, a, self.PIXEL_TOLERANCE
            )
        if acquired is not None:
            cands += _directional_candidates(
                px,
                py,
                width,
                height,
                camera,
                anchor,
                acquired,
                plane_normal,
                self.PIXEL_TOLERANCE,
            )
            cands += _from_point_candidates(
                px, py, width, height, camera, acquired, self.PIXEL_TOLERANCE
            )

        within = [c for c in cands if c.screen_dist <= self.PIXEL_TOLERANCE]
        if within:
            chosen = self._select(within)
            # The face under the cursor rides every result, not only an ON_FACE
            # win. Tools resolve their drawing plane from it, and a snap to a
            # face's own corner or midpoint is still a snap ON that face: keying
            # the plane off the winning KIND put those gestures on a horizontal
            # plane instead of the wall they were drawn against.
            if chosen.face_id is None and face_cand is not None:
                chosen.face_id = face_cand.face_id
            return self._to_result(chosen)

        if use_wt:
            # The grid fallback belongs on the active context's own local
            # ground plane, not literal world Z=0: inside a group translated
            # or rotated in z, those disagree, and building [gx, gy, 0.0] in
            # world space (as below) would land off the context's floor and,
            # under a rotation, corrupt local x/y too once the wrong point is
            # converted back. Intersect the already-computed local ray with
            # local z=0 instead, round in local space, then go back to world.
            dz_local = float(ray_dir_local[2])
            if abs(dz_local) > 1e-9:
                t_local = -float(ray_origin_local[2]) / dz_local
                if t_local > 0.0:
                    hit_local = np.asarray(
                        ray_origin_local, dtype=np.float64
                    ) + t_local * np.asarray(ray_dir_local, dtype=np.float64)
                    lx = round(float(hit_local[0]) / self.GRID_SIZE_WORLD) * self.GRID_SIZE_WORLD
                    ly = round(float(hit_local[1]) / self.GRID_SIZE_WORLD) * self.GRID_SIZE_WORLD
                    local_point = np.array([lx, ly, 0.0], dtype=np.float64)
                    world_point = apply_mat(local_point, wt)[0]
                    return SnapResult(
                        kind=SnapKind.GRID,
                        world_position=world_point,
                        axis=None,
                        vertex_id=None,
                        label="Grid",
                    )
            # Ray parallel to (or behind, for) the local ground plane: fall
            # back to the world-ground-plane behaviour below, same as identity.

        if ground_hit is not None:
            gx = round(float(ground_hit[0]) / self.GRID_SIZE_WORLD) * self.GRID_SIZE_WORLD
            gy = round(float(ground_hit[1]) / self.GRID_SIZE_WORLD) * self.GRID_SIZE_WORLD
            return SnapResult(
                kind=SnapKind.GRID,
                world_position=np.array([gx, gy, 0.0], dtype=np.float32),
                axis=None,
                vertex_id=None,
                label="Grid",
            )
        return self._none()

    # --- selection --------------------------------------------------------

    def _select(self, candidates: list[_Candidate]) -> _Candidate:
        """Highest precedence, then nearest the cursor, then nearest the camera.

        `screen_dist` is the middle key because of #31: with precedence and depth
        alone, two candidates of the SAME kind are separated only by depth, so a
        vertex sitting exactly under the cursor could lose to one several pixels
        away that happened to be nearer the camera.
        """
        return min(candidates, key=lambda c: (_PRECEDENCE_RANK[c.kind], c.screen_dist, c.depth))

    def _to_result(self, c: _Candidate) -> SnapResult:
        return SnapResult(
            kind=c.kind,
            world_position=np.asarray(c.world_position, dtype=np.float32),
            axis=c.axis,
            vertex_id=c.vertex_id,
            label=c.label,
            edge_id=c.edge_id,
            face_id=c.face_id,
            edge_t=c.edge_t,
            direction=c.direction,
        )

    def _none(self) -> SnapResult:
        return SnapResult(
            kind=SnapKind.NONE,
            world_position=np.zeros(3, dtype=np.float32),
            axis=None,
            vertex_id=None,
            label="—",
        )
