"""Concrete commands for Scene mutations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from bermake.commands.command import Command


class AddVertexCommand(Command):
    """Add a vertex, or resolve onto the one already sitting at that position.

    `Scene.add_vertex` is idempotent on exact float32 equality, so `do()` may
    hand back a vertex this command did NOT create — one belonging to geometry
    that was already in the scene (a profile vertex welded to the start of a
    Follow Me path is the everyday case). Removing that vertex in `undo()`
    would silently delete the user's pre-existing geometry, so ownership is
    recorded at the first `do()` and every later step honours it.

    Detection: `Scene.add_vertex` only ever APPENDS when it creates (ids are
    slab indices; a dedup returns an id below the slab's current size, and a
    removal tombstones rather than freeing the slot). So the vertex slab
    growing across the call is an exact, O(1) test for "this call created it".
    """

    name = "Add Vertex"

    def __init__(self, position: np.ndarray) -> None:
        self._position = np.asarray(position, dtype=np.float32).reshape(3).copy()
        self._vertex_id: int | None = None
        self._created = False

    @property
    def vertex_id(self) -> int | None:
        """The vertex this command created or resolved onto; None before do()."""
        return self._vertex_id

    def do(self, scene) -> None:
        if self._vertex_id is None:
            # First execution — allocate a new slot, or dedup onto an existing one.
            slab_before = scene.vertex_slab_size()
            self._vertex_id = scene.add_vertex(self._position)
            self._created = scene.vertex_slab_size() != slab_before
        elif self._created:
            # Redo — restore the previously-allocated slot to preserve the ID.
            scene.restore_vertex(self._vertex_id, self._position)
        # else: the first do() resolved onto a vertex owned by someone else.
        # undo() left it alone, and whatever does own it has already been redone
        # by now, so it is still live — redo is a no-op. Calling restore_vertex
        # on a live slot would raise; calling add_vertex again could hand back a
        # different id than the one callers cached.

    def undo(self, scene) -> None:
        assert self._vertex_id is not None, "AddVertexCommand.undo before do"
        if self._created:
            scene.remove_vertex(self._vertex_id)


class AddEdgeCommand(Command):
    """Add an edge, or resolve onto the one already joining the two vertices.

    Same ownership rule as `AddVertexCommand`: `Scene.add_edge` is idempotent,
    so `do()` may return a pre-existing edge, and removing it in `undo()` would
    delete geometry this command never created.

    Detection: `Scene.edge_between` is the non-mutating twin of the lookup
    inside `add_edge` — same canonical vertex-pair key, same liveness check —
    so asking it first tells us exactly what `add_edge` is about to do.
    """

    name = "Add Edge"

    def __init__(self, v1_id: int, v2_id: int) -> None:
        self._v1, self._v2 = v1_id, v2_id
        self._edge_id: int | None = None
        self._created = False

    def do(self, scene) -> None:
        if self._edge_id is None:
            # First execution — allocate a new slot, or dedup onto an existing one.
            existing = scene.edge_between(self._v1, self._v2)
            self._edge_id = scene.add_edge(self._v1, self._v2)
            self._created = existing is None
        elif self._created:
            # Redo — restore the previously-allocated slot to preserve the ID.
            scene.restore_edge(self._edge_id, self._v1, self._v2)
        # else: not ours — see AddVertexCommand.do for why redo is a no-op.

    def undo(self, scene) -> None:
        assert self._edge_id is not None, "AddEdgeCommand.undo before do"
        if self._created:
            scene.remove_edge(self._edge_id)


class AddFaceCommand(Command):
    """Add a face.

    Unlike `add_vertex` / `add_edge`, `Scene.add_face_from_loop` never
    deduplicates — the C++ `add_face_from_loop` allocates `faces_.size()` as
    the new id on every call, so a face command always owns what it made and
    the unconditional `remove_face` in `undo()` is correct.
    """

    name = "Add Face"

    def __init__(self, loop: Sequence[int]) -> None:
        self._loop = tuple(loop)
        self._face_id: int | None = None

    def do(self, scene) -> None:
        if self._face_id is None:
            # First execution — allocate a new slot.
            self._face_id = scene.add_face_from_loop(self._loop)
        else:
            # Redo — restore the previously-allocated slot to preserve the ID.
            scene.restore_face(self._face_id, self._loop)

    def undo(self, scene) -> None:
        assert self._face_id is not None, "AddFaceCommand.undo before do"
        scene.remove_face(self._face_id)


class RemoveFaceCommand(Command):
    name = "Remove Face"

    def __init__(self, face_id: int) -> None:
        self._face_id = face_id
        self._captured_loop: tuple[int, ...] | None = None

    def do(self, scene) -> None:
        self._captured_loop = tuple(scene.face(self._face_id).loop_vertex_ids)
        scene.remove_face(self._face_id)

    def undo(self, scene) -> None:
        assert self._captured_loop is not None, "RemoveFaceCommand.undo before do"
        scene.restore_face(self._face_id, self._captured_loop)


class RemoveEdgeCommand(Command):
    name = "Remove Edge"

    def __init__(self, edge_id: int) -> None:
        self._edge_id = edge_id
        self._captured: tuple[int, int] | None = None

    def do(self, scene) -> None:
        e = scene.edge(self._edge_id)
        self._captured = (e.v1_id, e.v2_id)
        scene.remove_edge(self._edge_id)

    def undo(self, scene) -> None:
        assert self._captured is not None, "RemoveEdgeCommand.undo before do"
        scene.restore_edge(self._edge_id, self._captured[0], self._captured[1])


class RemoveVertexCommand(Command):
    name = "Remove Vertex"

    def __init__(self, vertex_id: int) -> None:
        self._vertex_id = vertex_id
        self._captured_pos: np.ndarray | None = None

    def do(self, scene) -> None:
        self._captured_pos = scene.vertex(self._vertex_id).position.copy()
        scene.remove_vertex(self._vertex_id)

    def undo(self, scene) -> None:
        assert self._captured_pos is not None, "RemoveVertexCommand.undo before do"
        scene.restore_vertex(self._vertex_id, self._captured_pos)


class _AddVertexAtId(Command):
    """Internal: revive a vertex at its original ID. Used by ClearSceneCommand.undo.

    Since HalfEdgeMesh::clear() tombstones rather than shrinking the slabs
    (M7.1 #19), the original slot still exists after a clear, so restore_vertex
    revives it at its exact id — keeping later edge/face references valid.
    """

    def __init__(self, v_id: int, position: np.ndarray) -> None:
        self._v_id = v_id
        self._position = np.asarray(position, dtype=np.float32).reshape(3).copy()

    def do(self, scene) -> None:
        scene.restore_vertex(self._v_id, self._position)

    def undo(self, scene) -> None:
        scene.remove_vertex(self._v_id)


class _AddEdgeAtId(Command):
    """Internal: re-adds an edge at a specific ID."""

    def __init__(self, e_id: int, v1_id: int, v2_id: int) -> None:
        self._e_id = e_id
        self._v1, self._v2 = v1_id, v2_id

    def do(self, scene) -> None:
        scene.restore_edge(self._e_id, self._v1, self._v2)

    def undo(self, scene) -> None:
        scene.remove_edge(self._e_id)


class _AddFaceAtId(Command):
    """Internal: re-adds a face at a specific ID."""

    def __init__(self, f_id: int, loop: Sequence[int]) -> None:
        self._f_id = f_id
        self._loop = tuple(loop)

    def do(self, scene) -> None:
        scene.restore_face(self._f_id, self._loop)

    def undo(self, scene) -> None:
        scene.remove_face(self._f_id)


class ClearSceneCommand(Command):
    """do() captures every live entity and clears; undo() replays Add*AtId children."""

    name = "Clear Scene"

    def __init__(self) -> None:
        self._captured: list[Command] | None = None

    def do(self, scene) -> None:
        captured: list[Command] = []
        for v in scene.vertices_iter():
            captured.append(_AddVertexAtId(v.id, v.position))
        for e in scene.edges_iter():
            captured.append(_AddEdgeAtId(e.id, e.v1_id, e.v2_id))
        for f in scene.faces_iter():
            captured.append(_AddFaceAtId(f.id, f.loop_vertex_ids))
        self._captured = captured
        scene.clear()

    def undo(self, scene) -> None:
        assert self._captured is not None, "ClearSceneCommand.undo before do"
        for cmd in self._captured:
            cmd.do(scene)


class DissolveEdgeCommand(Command):
    """Dissolve an edge between two faces. Reversible.

    do(): on first call, validates the edge, captures the two source faces'
          vertex loops and the shared edge's vertex pair, then dissolves. On
          redo, the shared edge's id has changed (undo recreates it via
          add_face_from_loop), so we re-resolve the current edge id from the
          captured vertex pair. Boundary / dead / unresolvable edges make the
          command a clean no-op (do + undo both return early), keeping the
          undo stack consistent.

          Id-preserving redo (M7.6a fix round 1): `Scene.dissolve_edge` has
          no id-preserving variant of its own, so a redo's dissolve mints a
          FRESH merged-face id every time. Undo, below, always restores the
          two parents to their ORIGINAL ids regardless of how many times
          do() has run, so a stale captured id would desync from the live
          mesh the moment a second dissolve chains onto the first inside one
          CompositeCommand (an ordinary "split a face twice, then
          drag-erase both seams" gesture) and a redo happens in between:
          the second command's undo would call remove_face on an id the
          first command's undo/restore cycle never touches, which is
          already dead. On any run after the first successful one, the
          freshly minted face is immediately removed and the merged face is
          restored to `self._merged_face_id` from the first run instead --
          the same contract `SplitEdgeCommand` and `SplitFaceCommand` give
          their own created ids. Restoring at that original id also revives
          whatever sidecars (material/placement/UVs) the first run's merge
          computed, still sitting under that id, with no recomputation.
    undo(): removes the merged face, then restores the dissolved edge and BOTH
            source faces to their ORIGINAL ids via restore_edge / restore_face
            (id-preserving — NOT add_face_from_loop). This is required so that,
            inside a CompositeCommand, sibling AddFaceCommands whose new-side
            faces this dissolve consumed can still undo by their cached ids.
    """

    name = "Dissolve Edge"

    def __init__(self, edge_id: int) -> None:
        self._edge_id = edge_id
        self._shared_verts: tuple[int, int] | None = None
        self._captured_f1: tuple[int, ...] | None = None
        self._captured_f2: tuple[int, ...] | None = None
        self._f1_id: int | None = None
        self._f2_id: int | None = None
        self._merged_face_id: int | None = None
        self._captured_merged_loop: tuple[int, ...] | None = None
        self._was_noop: bool = False

    def do(self, scene) -> None:
        if self._captured_f1 is None:
            # First execution — validate + capture descriptors for undo/redo.
            try:
                faces = scene.edge_faces(self._edge_id)
            except KeyError:
                self._was_noop = True  # dead edge
                return
            if faces[0] is None or faces[1] is None:
                self._was_noop = True  # boundary edge (only one incident face)
                return
            f1, f2 = faces
            self._f1_id = f1
            self._f2_id = f2
            self._captured_f1 = tuple(scene.face(f1).loop_vertex_ids)
            self._captured_f2 = tuple(scene.face(f2).loop_vertex_ids)
            # Capture the dissolved edge's true endpoints. Vertex ids are stable
            # across undo (undo recreates the edge/faces but reuses vertices), so
            # these drive the redo re-resolution below. This is guaranteed to be
            # the real edge's endpoints — no diagonal-shared-vertex ambiguity.
            e = scene.edge(self._edge_id)
            self._shared_verts = (e.v1_id, e.v2_id)
            edge_to_dissolve = self._edge_id
        else:
            # Redo — the original edge id is stale (undo recreated the edge with
            # a fresh id). Re-resolve the current live edge id from the captured
            # endpoint pair via the non-mutating edge_between lookup.
            assert self._shared_verts is not None, "DissolveEdgeCommand.do (redo) before first do"
            va, vb = self._shared_verts
            found = scene.edge_between(va, vb)
            if found is None:
                # The pair is no longer joined (a sibling command in the same
                # composite removed it). Nothing to dissolve; stay a clean
                # no-op so undo remains consistent.
                self._was_noop = True
                return
            edge_to_dissolve = found

        result = scene.dissolve_edge(edge_to_dissolve)
        if result is None:
            self._was_noop = True
            return
        self._was_noop = False

        if self._captured_merged_loop is None:
            # First successful dissolve: this id and loop become the
            # contract every later run (redo) preserves.
            self._merged_face_id = result
            self._captured_merged_loop = tuple(scene.face(result).loop_vertex_ids)
        else:
            # Redo: `result` is a freshly minted id (see the class
            # docstring). Discard it immediately and restore the merged
            # face at its first-run id instead, so the id this command
            # exposes never changes across a redo.
            scene.remove_face(result)
            scene.restore_face(self._merged_face_id, self._captured_merged_loop)

    def undo(self, scene) -> None:
        if self._was_noop:
            return
        assert self._merged_face_id is not None, "DissolveEdgeCommand.undo before do"
        assert self._captured_f1 is not None and self._captured_f2 is not None, (
            "DissolveEdgeCommand.undo before do"
        )
        assert self._shared_verts is not None, "DissolveEdgeCommand.undo before do"
        assert self._f1_id is not None and self._f2_id is not None, (
            "DissolveEdgeCommand.undo before do"
        )

        # Remove the merged face, then restore the dissolved edge and BOTH source
        # faces to their ORIGINAL ids. Id-preserving restore (not add_face_from_loop)
        # is required so that, inside a CompositeCommand, sibling AddFaceCommands
        # whose faces this dissolve consumed can still undo by their cached ids.
        # restore_edge must run first — restore_face needs the shared edge present.
        scene.remove_face(self._merged_face_id)
        scene.restore_edge(self._edge_id, self._shared_verts[0], self._shared_verts[1])
        scene.restore_face(self._f1_id, self._captured_f1)
        scene.restore_face(self._f2_id, self._captured_f2)


class SplitEdgeCommand(Command):
    """Split an edge at parameter t, inserting a vertex. Reversible.

    do(): first call performs the split via scene.split_edge and captures BOTH
          the originals (edge endpoints, incident face ids+loops) and the created
          ids (vertex w, two edges, up to two faces) plus the rebuilt loops.
          Redo restores every created entity to its FIRST-RUN id (id-preserving),
          so a sibling command in the same gesture composite that cached the new
          vertex id stays valid across undo/redo (the M3c atomic-undo concern).
    undo(): removes the created faces/edges/vertex, then restores the original
            edge and faces to their ORIGINAL ids (restore_edge before restore_face).
    Invalid/degenerate splits make the command a clean no-op.
    """

    name = "Split Edge"

    def __init__(self, edge_id: int, t: float) -> None:
        self._edge_id = edge_id
        self._t = float(t)
        self._was_noop = False
        self._done_once = False
        # originals
        self._orig_verts: tuple[int, int] | None = None  # (va, vb) = edge endpoints
        self._orig_faces: list[tuple[int, tuple[int, ...]]] = []  # (id, loop) captured
        self._w_pos: np.ndarray | None = None
        # created (first-run ids, reused on redo)
        self.new_vertex_id: int | None = None
        self._e1: int | None = None
        self._e2: int | None = None
        self._new_faces: list[tuple[int, tuple[int, ...]]] = []  # (id, loop-with-w)

    def do(self, scene) -> None:
        if not self._done_once:
            self._first_do(scene)
        else:
            self._redo(scene)

    def _first_do(self, scene) -> None:
        try:
            e = scene.edge(self._edge_id)
        except KeyError:
            self._was_noop = True
            self._done_once = True
            return
        va, vb = e.v1_id, e.v2_id
        fa, fb = scene.edge_faces(self._edge_id)
        captured_faces: list[tuple[int, tuple[int, ...]]] = []
        for fid in (fa, fb):
            if fid is not None:
                captured_faces.append((fid, tuple(scene.face(fid).loop_vertex_ids)))

        res = scene.split_edge(self._edge_id, self._t)
        if res is None:
            self._was_noop = True
            self._done_once = True
            return

        self._orig_verts = (va, vb)
        self._orig_faces = captured_faces
        self._w_pos = scene.vertex(res.vertex).position.copy()
        self.new_vertex_id = res.vertex
        self._e1, self._e2 = res.edge_a, res.edge_b
        self._new_faces = []
        for fid in (res.face_a, res.face_b):
            if fid is not None:
                self._new_faces.append((fid, tuple(scene.face(fid).loop_vertex_ids)))
        self._done_once = True
        self._was_noop = False

    def _redo(self, scene) -> None:
        if self._was_noop:
            return
        assert self._orig_verts is not None and self._w_pos is not None, (
            "SplitEdgeCommand._redo before _first_do"
        )
        assert self.new_vertex_id is not None and self._e1 is not None and self._e2 is not None, (
            "SplitEdgeCommand._redo before _first_do"
        )
        va, vb = self._orig_verts
        w = self.new_vertex_id
        scene.restore_vertex(w, self._w_pos)
        for fid, _loop in self._orig_faces:
            scene.remove_face(fid)
        scene.remove_edge(self._edge_id)
        scene.restore_edge(self._e1, va, w)
        scene.restore_edge(self._e2, w, vb)
        for fid, loop in self._new_faces:
            scene.restore_face(fid, loop)

    def undo(self, scene) -> None:
        if self._was_noop:
            return
        assert self._orig_verts is not None, "SplitEdgeCommand.undo before do"
        assert self.new_vertex_id is not None and self._e1 is not None and self._e2 is not None, (
            "SplitEdgeCommand.undo before do"
        )
        for fid, _loop in self._new_faces:
            scene.remove_face(fid)
        scene.remove_edge(self._e1)
        scene.remove_edge(self._e2)
        scene.remove_vertex(self.new_vertex_id)
        va, vb = self._orig_verts
        scene.restore_edge(self._edge_id, va, vb)
        for fid, loop in self._orig_faces:
            scene.restore_face(fid, loop)


class SplitFaceCommand(Command):
    """Split a face along a chain of vertices. Reversible.

    do(): first call performs the split via scene.split_face and captures
          BOTH the original face's id and loop and the created ids (two new
          faces) plus their rebuilt loops. Redo restores the created faces to
          their FIRST-RUN ids (id-preserving), so a sibling command in the
          same gesture composite that cached one of them stays valid across
          undo/redo (the M3c atomic-undo concern, see SplitEdgeCommand).
    undo(): removes the created faces, then restores the original face to its
            ORIGINAL id. split_face creates no vertices and no edges, so
            undo touches only faces.
    A refused split (see Scene.split_face) makes the command a clean no-op.
    """

    name = "Split Face"

    def __init__(self, face_id: int, chain: Sequence[int]) -> None:
        self._face_id = face_id
        self._chain = [int(v) for v in chain]
        self._was_noop = False
        self._done_once = False
        # original
        self._orig_loop: tuple[int, ...] | None = None
        # created (first-run ids, reused on redo)
        self.new_face_ids: tuple[int, int] | None = None
        self._new_loops: list[tuple[int, tuple[int, ...]]] = []

    def do(self, scene) -> None:
        if not self._done_once:
            self._first_do(scene)
        else:
            self._redo(scene)

    def _first_do(self, scene) -> None:
        try:
            loop = scene.face_loop(self._face_id)
        except KeyError:
            self._was_noop = True
            self._done_once = True
            return

        result = scene.split_face(self._face_id, self._chain)
        if result is None:
            self._was_noop = True
            self._done_once = True
            return

        self._orig_loop = tuple(loop)
        self.new_face_ids = result
        self._new_loops = [(fid, tuple(scene.face(fid).loop_vertex_ids)) for fid in result]
        self._done_once = True
        self._was_noop = False

    def _redo(self, scene) -> None:
        if self._was_noop:
            return
        assert self._orig_loop is not None, "SplitFaceCommand._redo before _first_do"
        assert self.new_face_ids is not None, "SplitFaceCommand._redo before _first_do"
        scene.remove_face(self._face_id)
        for fid, loop in self._new_loops:
            scene.restore_face(fid, loop)

    def undo(self, scene) -> None:
        if self._was_noop:
            return
        assert self._orig_loop is not None, "SplitFaceCommand.undo before do"
        assert self.new_face_ids is not None, "SplitFaceCommand.undo before do"
        for fid in self.new_face_ids:
            scene.remove_face(fid)
        scene.restore_face(self._face_id, self._orig_loop)


class TransformVerticesCommand(Command):
    """Move a set of vertices to new positions; undo restores the old ones.

    `moves` maps vertex_id -> (old_xyz, new_xyz). No-op entries (old == new)
    are dropped at construction. Topology is unchanged, so do() and undo() are
    both just absolute position writes — id-preserving and re-entrant.
    """

    name = "Transform"

    def __init__(self, moves: dict) -> None:
        self._moves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for vid, (old, new) in moves.items():
            o = np.asarray(old, dtype=np.float32).reshape(3).copy()
            n = np.asarray(new, dtype=np.float32).reshape(3).copy()
            if not np.array_equal(o, n):
                self._moves[int(vid)] = (o, n)

    def is_empty(self) -> bool:
        return not self._moves

    def do(self, scene) -> None:
        for vid, (_old, new) in self._moves.items():
            scene.set_vertex_position(vid, new)

    def undo(self, scene) -> None:
        for vid, (old, _new) in self._moves.items():
            scene.set_vertex_position(vid, old)
