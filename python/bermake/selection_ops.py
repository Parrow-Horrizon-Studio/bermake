"""Pure selection derivations: the neighbour queries behind smart-select, the
connected-component flood behind triple-click, and the set operations behind
Edit > Invert and the right-click Select submenu.

Every function takes ids plus a Scene (or, where tags are involved, a Model)
and returns ids. Nothing here imports Qt, mutates a Selection, or knows that
menus exist, so each function is directly callable in a test with no widget
and no QApplication.

That is deliberate rather than stylistic. M7.6b measured that tests routed
through the widget masked defects which direct calls on the pure generators
caught, and the same milestone shipped one constant in three copies because
two callers each grew their own. Triple-click and `grow` share a walk here
for the same reason: one copy cannot drift.

A dead id is skipped rather than raised on, matching how
`transform_support.selection_vertices` already tolerates a stale id. A
selection can outlive the geometry it names, briefly, between a command and
the `prune_to_live` that follows it.
"""

from __future__ import annotations

from collections import deque


def _live_edges(scene, edge_ids) -> set[int]:
    """The subset of `edge_ids` naming a live edge.

    Final review I6: the module docstring promises "a dead id is skipped
    rather than raised on", and the three neighbour queries kept that promise
    while `grow` and `shrink` merely passed a dead id straight through into
    their own result -- which is not skipping it, and left Grow/Shrink
    re-seating a stale id the next `prune_to_live` would have dropped.
    """
    return {int(e) for e in edge_ids if scene.edge_is_live(int(e))}


def _live_faces(scene, face_ids) -> set[int]:
    """The subset of `face_ids` naming a live face. See `_live_edges`.

    `face_loop` is the liveness probe rather than `face`, which also builds a
    triangulation array the caller throws away.
    """
    out: set[int] = set()
    for f_id in face_ids:
        try:
            scene.face_loop(f_id)
        except KeyError:
            continue
        out.add(int(f_id))
    return out


def _live_vertices(scene, vertex_ids) -> set[int]:
    """The subset of `vertex_ids` naming a live vertex. See `_live_edges`."""
    out: set[int] = set()
    for v_id in vertex_ids:
        try:
            scene.vertex(v_id)
        except KeyError:
            continue
        out.add(int(v_id))
    return out


def bounding_edges(scene, face_ids) -> set[int]:
    """Every edge on the boundary loop of any face in `face_ids`.

    This is the second half of a double-click on a face: the face itself plus
    what bounds it.
    """
    out: set[int] = set()
    for f_id in face_ids:
        try:
            loop_edges = scene.face_edges(f_id)
        except KeyError:
            continue
        out.update(int(e) for e in loop_edges)
    return out


def adjacent_faces(scene, edge_ids) -> set[int]:
    """Every face on either side of any edge in `edge_ids`.

    `Scene.edge_faces` returns a two-tuple whose entries are None where the
    edge is naked on that side, so both are filtered rather than assumed
    present. A wholly naked edge contributes nothing.
    """
    out: set[int] = set()
    for e_id in edge_ids:
        try:
            side_a, side_b = scene.edge_faces(e_id)
        except KeyError:
            continue
        if side_a is not None:
            out.add(int(side_a))
        if side_b is not None:
            out.add(int(side_b))
    return out


def incident_edges(scene, vertex_ids) -> set[int]:
    """Every live edge with one of `vertex_ids` as an endpoint.

    There is no vertex-to-edge index on Scene, so this scans `edges_iter`.
    That is the same scan `pick_selectable` already performs on every pick,
    so it is not a new cost class.
    """
    wanted = {int(v) for v in vertex_ids}
    if not wanted:
        return set()
    return {
        int(e.id) for e in scene.edges_iter() if int(e.v1_id) in wanted or int(e.v2_id) in wanted
    }


def connected_component(scene, seed_vertex_ids) -> tuple[set[int], set[int], set[int]]:
    """Flood from `seed_vertex_ids` across shared vertices.

    Returns `(vertices, edges, faces)`. An edge is included when BOTH its
    endpoints are reached; a face when its WHOLE loop is reached. Both follow
    from the walk rather than being separate rules: an edge with one endpoint
    reached would mean the other endpoint was reachable too, so the stricter
    test is the honest one and it keeps the result symmetric.

    Connectivity is through shared vertices, not shared edges (spec 2.4).
    Two faces meeting at a single corner are one component. That is what makes
    the result independent of which entity kind seeded the walk, which is in
    turn what lets double-click and triple-click compose.

    This is the walk `follow_me_tool._order_path` performs over a selected
    edge set, lifted so there is one copy. That caller keeps its own version
    for now: it needs fork detection and an ORDERED chain, which this does not
    produce.
    """
    adjacency: dict[int, set[int]] = {}
    edges_by_pair: dict[tuple[int, int], int] = {}
    for e in scene.edges_iter():
        v1, v2 = int(e.v1_id), int(e.v2_id)
        adjacency.setdefault(v1, set()).add(v2)
        adjacency.setdefault(v2, set()).add(v1)
        edges_by_pair[(min(v1, v2), max(v1, v2))] = int(e.id)

    seeds = {int(v) for v in seed_vertex_ids if int(v) in adjacency}
    if not seeds:
        return set(), set(), set()

    reached: set[int] = set(seeds)
    queue = deque(seeds)
    while queue:
        v = queue.popleft()
        for n in adjacency[v]:
            if n not in reached:
                reached.add(n)
                queue.append(n)

    edges = {e_id for (v1, v2), e_id in edges_by_pair.items() if v1 in reached and v2 in reached}

    faces: set[int] = set()
    for f in scene.faces_iter():
        loop = scene.face_loop(f.id)
        if loop and all(int(v) in reached for v in loop):
            faces.add(int(f.id))

    return reached, edges, faces


def grow(scene, *, edges, faces, vertices) -> tuple[set[int], set[int], set[int]]:
    """Expand each kind by one step of its own adjacency.

    Faces gain faces sharing an edge; edges gain edges sharing a vertex;
    vertices gain vertices sharing an edge. Kinds do NOT bleed into one
    another (spec D10): growing a face selection yields faces, never the
    loose edges around them. Cross-kind expansion is what double-click
    already does, deliberately and one step at a time.
    """
    edges = _live_edges(scene, edges)
    faces = _live_faces(scene, faces)
    vertices = _live_vertices(scene, vertices)

    grown_faces = set(faces)
    for f_id in faces:
        grown_faces |= adjacent_faces(scene, bounding_edges(scene, {f_id}))

    grown_edges = set(edges)
    if edges:
        endpoints: set[int] = set()
        for e_id in edges:
            try:
                e = scene.edge(e_id)
            except KeyError:
                continue
            endpoints.add(int(e.v1_id))
            endpoints.add(int(e.v2_id))
        grown_edges |= incident_edges(scene, endpoints)

    grown_vertices = set(vertices)
    if vertices:
        for e in scene.edges_iter():
            v1, v2 = int(e.v1_id), int(e.v2_id)
            if v1 in vertices:
                grown_vertices.add(v2)
            if v2 in vertices:
                grown_vertices.add(v1)

    return grown_edges, grown_faces, grown_vertices


def shrink(scene, *, edges, faces, vertices) -> tuple[set[int], set[int], set[int]]:
    """Erode each kind by one step: drop anything on the selection's boundary.

    An entity is on the boundary when at least one of its neighbours, by the
    same per-kind adjacency `grow` uses, is not itself selected. On an
    interior region this is exactly `grow`'s inverse, which is the property
    the tests pin, since neither operation has a SketchUp equivalent to check
    against.
    """
    edges = _live_edges(scene, edges)
    faces = _live_faces(scene, faces)
    vertices = _live_vertices(scene, vertices)

    kept_faces = {
        f_id
        for f_id in faces
        if adjacent_faces(scene, bounding_edges(scene, {f_id})) - {f_id} <= faces
    }

    kept_edges = set()
    if edges:
        # Final review I4: `incident_edges` re-scans every edge in the scene,
        # so calling it once per selected edge made this O(E^2) -- Ctrl+A then
        # Shrink on a 40x40 quad grid (3280 edges) froze the GUI thread for
        # ~10s. One pass builds the same vertex -> incident-edge index the
        # vertex branch below already builds, and the loop becomes two dict
        # lookups per edge.
        edges_at_vertex: dict[int, set[int]] = {}
        endpoints_of: dict[int, tuple[int, int]] = {}
        for e in scene.edges_iter():
            e_id, v1, v2 = int(e.id), int(e.v1_id), int(e.v2_id)
            endpoints_of[e_id] = (v1, v2)
            edges_at_vertex.setdefault(v1, set()).add(e_id)
            edges_at_vertex.setdefault(v2, set()).add(e_id)
        empty: set[int] = set()
        for e_id in edges:
            ends = endpoints_of.get(e_id)
            if ends is None:  # dead id, same skip `scene.edge` used to give
                continue
            v1, v2 = ends
            neighbours = (edges_at_vertex.get(v1, empty) | edges_at_vertex.get(v2, empty)) - {e_id}
            if neighbours <= edges:
                kept_edges.add(e_id)

    kept_vertices = set()
    if vertices:
        neighbours_of: dict[int, set[int]] = {}
        for e in scene.edges_iter():
            v1, v2 = int(e.v1_id), int(e.v2_id)
            neighbours_of.setdefault(v1, set()).add(v2)
            neighbours_of.setdefault(v2, set()).add(v1)
        kept_vertices = {v_id for v_id in vertices if neighbours_of.get(v_id, set()) <= vertices}

    return kept_edges, kept_faces, kept_vertices


_DEFAULT_MATERIAL_ID = 0  # Matches Scene's own Default material id (scene.py).


def same_material(scene, face_ids) -> set[int]:
    """Every live face carrying a material any seed face carries.

    Front and back are separate dictionaries on Scene, and a match on EITHER
    side counts (spec D8): a user asking for "same material" means the paint
    they can see, and they cannot see which side dictionary it came from.

    Material 0 (Default) participates as a seed on a PER-FACE basis: a seed
    face contributes Default only when THAT face itself carries no real
    material on either side. A face painted on one side only contributes
    just its real material, never Default from its own unpainted side --
    that is what stops "All with Same Material" on one painted wall from
    also matching every blank face in the document. A face the caller
    deliberately included in the seed that is wholly unpainted still
    contributes Default, so seeding a painted face together with an
    unpainted one matches both families rather than dropping the unpainted
    seed from its own result. (An earlier version of this rule pooled real
    vs. Default across the whole seed set before deciding, which made a
    mixed seed like that exclude its own unpainted member.)
    """
    from bermake.scene.scene import Side

    # Final review I6: `face_material` is a dict `.get` with a Default
    # fallback and never raises, so the `except KeyError: continue` that used
    # to wrap the two reads below skipped nothing. A dead seed id therefore
    # contributed Default, `real` came out empty, the seed set became {0} and
    # the match loop returned every unpainted face in the document. Liveness
    # is now asked of the mesh up front, so the module's "a dead id is
    # skipped rather than raised on" promise is actually kept here.
    seed_materials: set[int] = set()
    for f_id in _live_faces(scene, face_ids):
        front = int(scene.face_material(f_id, Side.FRONT))
        back = int(scene.face_material(f_id, Side.BACK))
        face_materials = {front, back}
        real = face_materials - {_DEFAULT_MATERIAL_ID}
        seed_materials |= real if real else face_materials
    if not seed_materials:
        return set()

    out: set[int] = set()
    for f in scene.faces_iter():
        front = int(scene.face_material(f.id, Side.FRONT))
        back = int(scene.face_material(f.id, Side.BACK))
        if front in seed_materials or back in seed_materials:
            out.add(int(f.id))
    return out


def same_tag(model, instance_ids) -> set[int]:
    """Every instance in the active context sharing a tag with a seed.

    Tags live on Instance.tag_id and nowhere else (spec D9), so this takes a
    Model rather than a Scene and returns instances rather than geometry.
    """
    children = list(model.active_context.children)
    seeds = {int(i) for i in instance_ids}
    seed_tags = {int(c.tag_id) for c in children if int(c.id) in seeds}
    if not seed_tags:
        return set()
    return {int(c.id) for c in children if int(c.tag_id) in seed_tags}


def invert(model, selection, *, select_vertices) -> tuple[set[int], set[int], set[int], set[int]]:
    """Everything selectable in the active context that is NOT selected.

    The universe is `select_all_ids`, the same one Select All uses (spec D7).
    Inventing a second one would guarantee eventual divergence between the
    two. The visible consequence is that annotations are excluded from both,
    which is today's Select All behaviour rather than a new asymmetry.

    Reads `selection`; never mutates it.
    """
    from bermake.model.model_queries import select_all_ids

    all_edges, all_faces, all_instances = select_all_ids(model)
    verts: set[int] = set()
    if select_vertices:
        all_vertices = {int(v.id) for v in model.active_context.mesh.vertices_iter()}
        verts = all_vertices - {int(v) for v in selection.vertices}
    return (
        all_edges - {int(e) for e in selection.edges},
        all_faces - {int(f) for f in selection.faces},
        all_instances - {int(i) for i in selection.instances},
        verts,
    )
