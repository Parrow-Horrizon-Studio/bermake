"""CreateWallCommand (M7a): undoably build a baked solid-box wall group."""

from __future__ import annotations

import numpy as np

from bermake.commands.command import Command
from bermake.geometry.wall import wall_box


class CreateWallCommand(Command):
    """Build one centered wall box into `target_context` as a `"Wall"` group.

    start/end are in the target_context's LOCAL frame (the tool converts from
    world). Undo detaches the single created instance; the definition is not
    globally registered, so its subtree becomes unreachable (matches
    ImportGltfCommand). Redo re-runs do()."""

    name = "Create Wall"

    def __init__(self, start, end, thickness, height, target_context) -> None:
        self._start = start
        self._end = end
        self._thickness = thickness
        self._height = height
        self._target = target_context
        self._instance = None

    def do(self, model) -> None:
        vertices, faces = wall_box(self._start, self._end, self._thickness, self._height)
        if not vertices:
            self._instance = None
            return
        defn = model.new_definition("Wall", is_group=True)
        local = {}
        for i, (x, y, z) in enumerate(vertices):
            local[i] = defn.mesh.add_vertex(np.array([x, y, z], dtype=np.float32))
        for loop in faces:
            defn.mesh.add_face_from_loop([local[i] for i in loop])
        inst = model.new_instance(defn)
        self._target.children.append(inst)
        self._instance = inst

    def undo(self, model) -> None:
        if self._instance is None:
            return
        if self._instance in self._target.children:
            self._target.children.remove(self._instance)
        model.revalidate_active_path()
        self._instance = None
