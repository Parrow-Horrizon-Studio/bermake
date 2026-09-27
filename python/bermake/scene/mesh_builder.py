"""Adapter: walk a generated HalfEdgeMesh into a Scene as ordinary commands
(M7.4 Task 10).

Task 9 added four primitive generators -- `make_box`, `make_cylinder`,
`make_cone`, `make_sphere` (all in `bermake._core`) -- that each build a
standalone `bermake._core.HalfEdgeMesh`. Nothing connected that generated mesh
to a `Scene`: `Scene.__init__` constructs its own private `HalfEdgeMesh`, so
there was no path from a generator's output into editable, undoable
geometry. This module is that path.

It mirrors `build_obj_into_model` (`bermake.io.obj_io`): walk the source's
live elements and re-create them in the destination through ordinary Scene
mutations, so the caller gets back editable geometry with undo, not an
opaque blob. Concretely it walks vertices, then faces (adding each face
loop's not-yet-existing edges first), building one `AddVertexCommand` /
`AddEdgeCommand` / `AddFaceCommand` per element.

Commands execute AS they are built (`cmd.do(scene)`, immediately) -- the
same idiom as `loft_between_loops` in `bermake.tools.sweep_support`, and for
the same reason: every caller registers the finished command list with
`CommandStack.push_executed`, which appends to the undo stack WITHOUT
calling `.do()` again. A caller that instead used `execute()` -- or that
built the commands without calling `.do()` here -- would either double-apply
the geometry or push commands that were never run.

The walk is atomic: if it raises partway through (e.g. a degenerate
transform welding two adjacent loop vertices into a self-loop edge), every
command already applied is undone before the exception propagates, so
`scene` is left exactly as the caller found it.
"""

from __future__ import annotations

import numpy as np

from bermake.commands.command import Command
from bermake.commands.scene_commands import AddEdgeCommand, AddFaceCommand, AddVertexCommand


def build_mesh_into_scene(mesh, scene, transform: np.ndarray | None = None) -> list[Command]:
    """Insert every live vertex and face of `mesh` into `scene`.

    `mesh` is a `bermake._core.HalfEdgeMesh` (what `make_box` / `make_cylinder`
    / `make_cone` / `make_sphere` return), walked via its `next_live_vertex`
    / `next_live_face` / `vertex_position` / `face_loop_vertices` accessors
    and the `INVALID_ID` sentinel as the loop terminator -- the same idiom
    `Scene.vertices_iter` / `Scene.faces_iter` use for the scene's own mesh.

    `transform`, if given, is a 4x4 matrix applied to each source position
    as `p_world = transform @ [x, y, z, 1]` -- column-vector, right-
    multiplied, the SAME convention `sweep_stations` documents in
    `bermake.tools.sweep_support`. A caller (Task 11) can therefore hand this
    function a station transform from that module directly, with no
    row/column trap at the boundary between the two.

    Returns the list of `Command`s, in the order they were built, already
    executed against `scene`. This function does not group them into a
    `CompositeCommand` or push them to a stack -- that is the caller's job,
    exactly as `loft_between_loops`'s callers (Push/Pull, Offset, Follow Me)
    do it themselves.

    Welding: `Scene.add_vertex` deduplicates by exact float32 position
    match. A source mesh's own shared corners therefore weld correctly when
    walked here -- that is required for the result to be a closed manifold
    rather than N disconnected quads. A `transform` that maps two DISTINCT
    source vertices onto the same world position welds them too, silently:
    this adapter does not compare source ids before delegating to
    `Scene.add_vertex`, so the destination ends up with one vertex where the
    source had two. If those two source vertices are adjacent in some face's
    loop, the collapsed loop asks `Scene.add_edge` for a self-loop edge,
    which raises `ValueError`.

    Atomicity: if anything raises partway through the walk -- including
    that self-loop `ValueError` -- every command built so far is undone, in
    reverse order, before the exception propagates. `scene` is left exactly
    as it was found: a caller that catches the exception (e.g. `ValueError`
    from a degenerate, UI-supplied transform) never has to clean up a
    half-built primitive itself.

    A wrinkle the rollback used to hand-guard: `Scene.add_vertex` is
    "idempotent on exact equality", so a welding transform can make two
    DIFFERENT `AddVertexCommand`s resolve onto the SAME underlying scene
    vertex id. `AddVertexCommand` now records whether its own `do()` created
    that vertex or merely resolved onto one already there, and only the
    creator removes it on undo -- so plain reverse-order undo absorbs the
    aliasing, and the id-tracking special case this loop used to carry is
    gone (it would now skip the one command that DOES own the vertex).
    """
    commands: list[Command] = []
    vertex_map: dict[int, int] = {}

    try:
        src_v = mesh.next_live_vertex(0)
        while src_v != mesh.INVALID_ID:
            pos = np.asarray(mesh.vertex_position(src_v), dtype=np.float64)
            if transform is not None:
                pos = (transform @ np.append(pos, 1.0))[:3]
            v_cmd = AddVertexCommand(pos.astype(np.float32))
            v_cmd.do(scene)
            commands.append(v_cmd)
            vertex_map[src_v] = v_cmd.vertex_id
            src_v = mesh.next_live_vertex(src_v + 1)

        src_f = mesh.next_live_face(0)
        while src_f != mesh.INVALID_ID:
            loop = [vertex_map[src_vid] for src_vid in mesh.face_loop_vertices(src_f)]
            n = len(loop)
            for i in range(n):
                a, b = loop[i], loop[(i + 1) % n]
                # Two faces sharing a loop edge must share the ONE edge between
                # them, not each get their own -- Scene.edge_between (Task 2)
                # answers whether the previous face already created it.
                if scene.edge_between(a, b) is None:
                    e_cmd = AddEdgeCommand(a, b)
                    e_cmd.do(scene)
                    commands.append(e_cmd)
            f_cmd = AddFaceCommand(loop)
            f_cmd.do(scene)
            commands.append(f_cmd)
            src_f = mesh.next_live_face(src_f + 1)
    except Exception:
        # Roll back everything already applied, in reverse construction
        # order -- the same order a caller's own undo would use -- so a
        # mid-walk failure never leaves partial geometry behind. Every
        # command in `commands` has already had `do()` called successfully
        # (a command that raised inside `do()` was never appended), so each
        # `undo()` here runs against exactly the state its own `do()` left.
        # Vertex aliasing under a welding transform (see the docstring) needs
        # no special case: each `AddVertexCommand` knows whether it created
        # its vertex, and the ones that merely resolved onto an existing id
        # leave it for its creator to remove.
        for cmd in reversed(commands):
            cmd.undo(scene)
        raise

    return commands
