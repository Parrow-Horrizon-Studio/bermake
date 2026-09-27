from __future__ import annotations

import numpy as np

from bermake.commands.command import Command


class TransformInstanceCommand(Command):
    name = "Transform"

    def __init__(self, instance, new_transform) -> None:
        self._inst = instance
        self._old = np.asarray(instance.transform, np.float64).copy()
        self._new = np.asarray(new_transform, np.float64).reshape(4, 4).copy()

    def do(self, model) -> None:
        self._inst.transform = self._new.copy()

    def undo(self, model) -> None:
        self._inst.transform = self._old.copy()


class CreateInstanceCommand(Command):
    name = "Create Instance"

    def __init__(self, parent_definition, definition, transform) -> None:
        self._parent = parent_definition
        self._definition = definition
        self._transform = np.asarray(transform, np.float64).reshape(4, 4).copy()
        self.created_instance = None

    def do(self, model) -> None:
        if self.created_instance is None:
            self.created_instance = model.new_instance(self._definition, self._transform)
        else:
            # redo: re-register the same instance object + back-ref
            if self.created_instance not in self._definition.instances:
                self._definition.instances.append(self.created_instance)
        if self.created_instance not in self._parent.children:
            self._parent.children.append(self.created_instance)

    def undo(self, model) -> None:
        if self.created_instance in self._parent.children:
            self._parent.children.remove(self.created_instance)
        if self.created_instance in self._definition.instances:
            self._definition.instances.remove(self.created_instance)
