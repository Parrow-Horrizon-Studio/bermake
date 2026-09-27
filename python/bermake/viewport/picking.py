"""Selection picking — pure screen-space (project to pixels via world_to_screen,
2D point-to-segment distance). Independent of the drawing-snap precedence:
selection wants the edge or face under the cursor, not a vertex/midpoint snap.
"""

from __future__ import annotations

import math

import numpy as np

from bermake.geometry.transforms import apply_mat, is_identity_transform, mat_invert

PICK_PIXEL_TOLERANCE = 8.0  # screen-space; matches the M3d snap feel


def _point_segment_distance(px, py, ax, ay, bx, by) -> float:
    """2D distance from point (px,py) to segment (ax,ay)-(bx,by)."""
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    if length2 <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / length2
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def ray_into_local(origin, direction, world_transform):
    """Transform a world-space ray (origin point, direction vector) into the
    local frame of `world_transform`. Returns (origin, direction) unchanged when
    world_transform is None or identity (root context)."""
    if is_identity_transform(world_transform):
        return origin, direction
    inv = mat_invert(np.asarray(world_transform, dtype=np.float64))
    o = apply_mat(np.asarray(origin, dtype=np.float64).reshape(1, 3), inv)[0]
    d = (inv[:3, :3] @ np.asarray(direction, dtype=np.float64)).astype(np.float32)
    return o, d


def world_to_local_point(point, world_transform):
    """Convert a single world-space point (3,) into the local frame of
    world_transform. Returns the point unchanged when world_transform is None
    or identity (root context)."""
    from bermake.geometry.transforms import apply_mat, is_identity_transform, mat_invert

    if is_identity_transform(world_transform):
        return np.asarray(point, dtype=np.float32).reshape(3)
    # apply_mat computes in float64 internally (matrix precision); cast back to
    # float32 so callers get a consistent dtype regardless of which branch ran
    # (some callers feed this straight into float32 vertex-position writes).
    local = apply_mat(
        np.asarray(point, dtype=np.float64).reshape(1, 3), mat_invert(world_transform)
    )[0]
    return local.astype(np.float32)


def pick_selectable(
    cursor_screen, viewport_size, camera, scene, world_transform=None, *, select_vertices=False
):
    """Return ("vertex", id) when `select_vertices` and a vertex is within
    PICK_PIXEL_TOLERANCE; else ("edge", id) for the nearest edge within that
    tolerance; else ("face", id) under the cursor ray; else None.

    Thin targets win because they are harder to hit, which is why the order
    runs vertex, edge, face. Vertex picking is OPT-IN and off by default
    (M7.6c D4): extending that rule to vertices unconditionally would change
    what every click near a corner selects, across every tool that picks, on
    geometry that is mostly corners. With `select_vertices` False this
    function executes exactly the code it executed before M7.6c.

    world_transform: optional (4,4) matrix mapping local (scene) coords to world.
    None or identity gives behaviour identical to the no-arg call (regression-safe).
    """
    px, py = float(cursor_screen[0]), float(cursor_screen[1])
    w, h = int(viewport_size[0]), int(viewport_size[1])

    use_wt = not is_identity_transform(world_transform)
    wt = np.asarray(world_transform, dtype=np.float64) if use_wt else None

    def _to_world(local_pos):
        if not use_wt:
            return local_pos
        return apply_mat(local_pos, wt)[0]

    if select_vertices:
        best_vertex: int | None = None
        best_vd = PICK_PIXEL_TOLERANCE
        for v in scene.vertices_iter():
            s = camera.world_to_screen(_to_world(v.position), w, h)
            if s is None:
                continue
            d = math.hypot(px - s[0], py - s[1])
            if d <= best_vd:
                best_vd = d
                best_vertex = v.id
        if best_vertex is not None:
            return ("vertex", int(best_vertex))

    best_edge: int | None = None
    best_d = PICK_PIXEL_TOLERANCE
    for e in scene.edges_iter():
        p1 = _to_world(scene.vertex(e.v1_id).position)
        p2 = _to_world(scene.vertex(e.v2_id).position)
        s1 = camera.world_to_screen(p1, w, h)
        s2 = camera.world_to_screen(p2, w, h)
        if s1 is None or s2 is None:
            continue
        d = _point_segment_distance(px, py, s1[0], s1[1], s2[0], s2[1])
        if d <= best_d:
            best_d = d
            best_edge = e.id
    if best_edge is not None:
        return ("edge", best_edge)

    origin, direction = camera.ray_from_screen(px, py, w, h)
    if use_wt:
        inv = mat_invert(wt)
        origin_local = apply_mat(origin, inv)[0]
        direction_local = (inv[:3, :3] @ np.asarray(direction, dtype=np.float64)).astype(np.float32)
        hit = scene.ray_pick_face(origin_local, direction_local)
    else:
        hit = scene.ray_pick_face(origin, direction)
    if hit is not None:
        return ("face", int(hit.face_id))
    return None


def _point_in_rect(px, py, rect) -> bool:
    x0, y0, x1, y1 = rect
    return x0 <= px <= x1 and y0 <= py <= y1


def _ccw(ax, ay, bx, by, cx, cy) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _segments_cross(ax, ay, bx, by, cx, cy, dx, dy) -> bool:
    """True if segment AB properly straddles segment CD (and vice-versa)."""
    d1 = _ccw(cx, cy, dx, dy, ax, ay)
    d2 = _ccw(cx, cy, dx, dy, bx, by)
    d3 = _ccw(ax, ay, bx, by, cx, cy)
    d4 = _ccw(ax, ay, bx, by, dx, dy)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _point_in_polygon(px, py, pts) -> bool:
    """Ray-casting point-in-polygon for a list of (x, y) screen points."""
    n = len(pts)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _segment_intersects_rect(ax, ay, bx, by, rect) -> bool:
    if _point_in_rect(ax, ay, rect) or _point_in_rect(bx, by, rect):
        return True
    x0, y0, x1, y1 = rect
    sides = (
        (x0, y0, x1, y0),
        (x1, y0, x1, y1),
        (x1, y1, x0, y1),
        (x0, y1, x0, y0),
    )
    for cx, cy, dx, dy in sides:
        if _segments_cross(ax, ay, bx, by, cx, cy, dx, dy):
            return True
    return False


def _normalize_rect(rect):
    x0, y0, x1, y1 = rect
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def entities_in_box(
    rect_px, mode, viewport_size, camera, scene, world_transform=None, *, select_vertices=False
):
    """Return (edge_ids, face_ids, vertex_ids) inside rect_px under the given mode.
    mode="window": only fully-enclosed; mode="crossing": anything touched.

    world_transform: optional (4,4) matrix mapping local (scene) coords to world.
    None or identity → behaviour is identical to the no-arg call (regression-safe).
    """
    rect = _normalize_rect(rect_px)
    w, h = int(viewport_size[0]), int(viewport_size[1])
    edges: set[int] = set()
    faces: set[int] = set()
    vertices: set[int] = set()

    use_wt = not is_identity_transform(world_transform)
    wt = np.asarray(world_transform, dtype=np.float64) if use_wt else None

    def _to_world(local_pos):
        if not use_wt:
            return local_pos
        return apply_mat(local_pos, wt)[0]

    def proj(local_pos):
        return camera.world_to_screen(_to_world(local_pos), w, h)

    for e in scene.edges_iter():
        s1 = proj(scene.vertex(e.v1_id).position)
        s2 = proj(scene.vertex(e.v2_id).position)
        if mode == "window":
            if (
                s1 is not None
                and s2 is not None
                and _point_in_rect(s1[0], s1[1], rect)
                and _point_in_rect(s2[0], s2[1], rect)
            ):
                edges.add(e.id)
        else:  # crossing
            if s1 is None or s2 is None:
                if (s1 is not None and _point_in_rect(s1[0], s1[1], rect)) or (
                    s2 is not None and _point_in_rect(s2[0], s2[1], rect)
                ):
                    edges.add(e.id)
            elif _segment_intersects_rect(s1[0], s1[1], s2[0], s2[1], rect):
                edges.add(e.id)

    for f in scene.faces_iter():
        loop = scene.face_loop(f.id)
        pts = [proj(scene.vertex(v).position) for v in loop]
        if mode == "window":
            if all(p is not None and _point_in_rect(p[0], p[1], rect) for p in pts):
                faces.add(f.id)
        else:  # crossing
            touched = any(p is not None and _point_in_rect(p[0], p[1], rect) for p in pts)
            if not touched:
                n = len(pts)
                for i in range(n):
                    p, q = pts[i], pts[(i + 1) % n]
                    if (
                        p is not None
                        and q is not None
                        and _segment_intersects_rect(p[0], p[1], q[0], q[1], rect)
                    ):
                        touched = True
                        break
            if not touched:
                valid = [(p[0], p[1]) for p in pts if p is not None]
                if len(valid) >= 3 and _point_in_polygon(rect[0], rect[1], valid):
                    touched = True
            if touched:
                faces.add(f.id)

    if select_vertices:
        for v in scene.vertices_iter():
            s = proj(v.position)
            if s is not None and _point_in_rect(s[0], s[1], rect):
                vertices.add(int(v.id))

    return edges, faces, vertices
