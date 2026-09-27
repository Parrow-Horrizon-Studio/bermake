"""ImportObjCommand (M6b): undoable wrapper around build_obj_into_model."""

from __future__ import annotations

from bermake.commands.command import Command
from bermake.io.obj_io import build_obj_into_model


class ImportObjCommand(Command):
    """Import an ObjDocument into the model, undoably. do() builds and records
    what it created; undo() removes it. Materials added to the library are NOT
    undone (library adds are not undoable anywhere in Bermake; a re-import reuses
    them via the dedupe in build_obj_into_model)."""

    name = "Import OBJ"

    def __init__(self, doc, target_context, texture_bytes=None, decoder=None) -> None:
        self._doc = doc
        self._target = target_context
        self._texture_bytes = texture_bytes
        self._decoder = decoder
        self._result = None
        self.summary = None

    def do(self, model) -> None:
        self._result = build_obj_into_model(
            self._doc, model, self._target, self._texture_bytes, self._decoder
        )
        self.summary = self._result.summary

    def undo(self, model) -> None:
        result = self._result
        if result is None:
            return
        # group case: detach the created instances from their parent
        for inst in result.created_instances:
            if inst in self._target.children:
                self._target.children.remove(inst)
        # merge case: remove faces, then edges, then vertices (dependency order)
        vids, eids, fids = result.created_geometry
        mesh = self._target.mesh
        for fid in fids:
            try:
                mesh.remove_face(fid)
            except (KeyError, ValueError):
                pass
        for eid in eids:
            try:
                mesh.remove_edge(eid)
            except (KeyError, ValueError):
                pass
        for vid in vids:
            try:
                mesh.remove_vertex(vid)
            except (KeyError, ValueError):
                pass
        self._result = None
