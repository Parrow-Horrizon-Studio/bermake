"""CreateRoofCommand (M7c): bake a parametric roof as a "Roof" group."""

from __future__ import annotations

import numpy as np

from bermake.commands.command import Command
from bermake.geometry.roof import roof_solid


class CreateRoofCommand(Command):
    """Bake a Gable/Hip/Shed roof solid into a new "Roof" group in the target
    context, instanced with `transform`. Undo detaches the instance (from both
    children and defn.instances); redo re-attaches the SAME Definition/Instance
    (mirrors CreateInstanceCommand — no fresh Definition, no leak)."""

    name = "Create Roof"

    def __init__(self, kind, width, depth, angle, transform, target_context) -> None:
        self._kind = kind
        self._width = width
        self._depth = depth
        self._angle = angle
        self._transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
        self._target = target_context
        self._definition = None
        self._instance = None

    def do(self, model) -> None:
        if self._definition is None:
            vertices, faces = roof_solid(self._kind, self._width, self._depth, self._angle)
            if not vertices:
                self._instance = None
                return
            defn = model.new_definition("Roof", is_group=True)
            ids = [defn.mesh.add_vertex(np.array(v, dtype=np.float32)) for v in vertices]
            for loop in faces:
                defn.mesh.add_face_from_loop([ids[i] for i in loop])
            self._definition = defn
            self._instance = model.new_instance(defn, self._transform)
        elif self._instance not in self._definition.instances:
            self._definition.instances.append(self._instance)  # redo: re-register
        if self._instance is not None:
            self._target.children.append(self._instance)

    def undo(self, model) -> None:
        if self._instance is None:
            return
        if self._instance in self._target.children:
            self._target.children.remove(self._instance)
        if self._instance in self._instance.definition.instances:
            self._instance.definition.instances.remove(self._instance)
        model.revalidate_active_path()
