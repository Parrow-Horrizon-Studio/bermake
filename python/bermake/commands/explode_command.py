# python/bermake/commands/explode_command.py
from __future__ import annotations

import numpy as np

from bermake.commands.command import Command
from bermake.geometry.transforms import apply_mat


class ExplodeInstanceCommand(Command):
    name = "Explode"

    def __init__(self, parent_definition, instance) -> None:
        self._parent = parent_definition
        self._inst = instance
        self._baked = None  # list of (new_vid_in_parent) for undo removal
        self._child_records = None  # reparented child instances (for undo)

    def do(self, model) -> None:
        parent_scene = self._parent.mesh
        defn = self._inst.definition
        t = self._inst.transform

        # Bake geometry: copy verts/edges/faces with transformed positions.
        idmap = {}
        new_vids = []
        for v in defn.mesh.vertices_iter():
            world_pos = apply_mat(v.position.reshape(1, 3), t)[0]
            nv = parent_scene.add_vertex(world_pos)
            idmap[v.id] = nv
            new_vids.append(nv)
        new_eids = []
        for e in defn.mesh.edges_iter():
            ne = parent_scene.add_edge(idmap[e.v1_id], idmap[e.v2_id])
            new_eids.append(ne)
        new_faces = []
        for f in defn.mesh.faces_iter():
            nf = parent_scene.add_face_from_loop([idmap[v] for v in f.loop_vertex_ids])
            # The inverse of the lift MakeGroupCommand performs, with the same
            # gap: a fresh face id carries no material, placement or stored
            # UVs (#114). The instance transform moves positions only, so the
            # copy is verbatim; texture coordinates are not in world space.
            parent_scene.copy_face_attributes_from(defn.mesh, f.id, nf)
            new_faces.append(nf)
        self._baked = (new_vids, new_eids, new_faces)

        # Reparent the instance's children into the parent, composing transforms.
        self._child_records = list(defn.children)
        for child in defn.children:
            child.transform = t @ child.transform
            self._parent.children.append(child)
        # The definition no longer owns them — otherwise any OTHER instance of
        # this definition would still render/pick the same child objects (#56).
        defn.children.clear()

        # Remove the exploded instance.
        if self._inst in self._parent.children:
            self._parent.children.remove(self._inst)

    def undo(self, model) -> None:
        parent_scene = self._parent.mesh
        new_vids, new_eids, new_faces = self._baked
        for nf in new_faces:
            parent_scene.remove_face(nf)
        for ne in new_eids:
            try:
                parent_scene.remove_edge(ne)
            except ValueError:
                pass
        for nv in new_vids:
            try:
                parent_scene.remove_vertex(nv)
            except ValueError:
                pass
        # Un-reparent children + undo their transform compose.
        t = self._inst.transform
        tinv = np.linalg.inv(np.asarray(t, np.float64))
        for child in self._child_records:
            if child in self._parent.children:
                self._parent.children.remove(child)
            child.transform = tinv @ child.transform
        # do() cleared the definition's children; restore them in original order.
        self._inst.definition.children.extend(self._child_records)
        self._parent.children.append(self._inst)
