"""PlaceOpeningCommand (M7b): place a framed door/window Component on a wall."""

from __future__ import annotations

import numpy as np

from bermake.commands.command import Command
from bermake.geometry.opening import opening_frame


def _sig(kind, width, height, depth):
    return (kind, round(float(width), 6), round(float(height), 6), round(float(depth), 6))


class PlaceOpeningCommand(Command):
    """Place a door/window as an Instance of a shared Component Definition.

    Identical (kind, width, height, depth) reuse one Definition via
    model.opening_definitions. Undo detaches the single created instance
    (leaving the Definition + registry entry for reuse); redo re-runs do()."""

    name = "Place Opening"

    def __init__(self, kind, width, height, depth, transform, target_context) -> None:
        self._kind = kind
        self._width = width
        self._height = height
        self._depth = depth
        self._transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
        self._target = target_context
        self._instance = None

    def _definition_for(self, model):
        sig = _sig(self._kind, self._width, self._height, self._depth)
        defn = model.opening_definitions.get(sig)
        if defn is not None:
            return defn
        vertices, faces = opening_frame(self._kind, self._width, self._height, self._depth)
        if not vertices:
            return None
        defn = model.new_definition(self._kind.capitalize(), is_group=False)
        ids = {}
        for i, (x, y, z) in enumerate(vertices):
            ids[i] = defn.mesh.add_vertex(np.array([x, y, z], dtype=np.float32))
        for loop in faces:
            defn.mesh.add_face_from_loop([ids[i] for i in loop])
        model.opening_definitions[sig] = defn
        return defn

    def do(self, model) -> None:
        defn = self._definition_for(model)
        if defn is None:
            self._instance = None
            return
        if self._instance is None:
            self._instance = model.new_instance(defn, self._transform)
        elif self._instance not in defn.instances:
            defn.instances.append(self._instance)  # redo: re-register the same object
        self._target.children.append(self._instance)

    def undo(self, model) -> None:
        if self._instance is None:
            return
        if self._instance in self._target.children:
            self._target.children.remove(self._instance)
        if self._instance in self._instance.definition.instances:
            self._instance.definition.instances.remove(self._instance)
        model.revalidate_active_path()
