from __future__ import annotations

from bermake.commands.command import Command


class MakeUniqueCommand(Command):
    name = "Make Unique"

    def __init__(self, instance) -> None:
        self._inst = instance
        self._old_def = instance.definition
        self._clone = None

    def do(self, model) -> None:
        if len(self._old_def.instances) <= 1:
            return  # already unique — no-op
        if self._clone is None:
            self._clone = model.clone_definition(self._old_def)
        self._old_def.instances.remove(self._inst)
        self._inst.definition = self._clone
        if self._inst not in self._clone.instances:
            self._clone.instances.append(self._inst)

    def undo(self, model) -> None:
        if self._clone is None:
            return
        if self._inst in self._clone.instances:
            self._clone.instances.remove(self._inst)
        self._inst.definition = self._old_def
        if self._inst not in self._old_def.instances:
            self._old_def.instances.append(self._inst)


class DeleteInstanceCommand(Command):
    name = "Delete"

    def __init__(self, parent_definition, instance) -> None:
        self._parent = parent_definition
        self._inst = instance

    def do(self, model) -> None:
        if self._inst in self._parent.children:
            self._parent.children.remove(self._inst)
        if self._inst in self._inst.definition.instances:
            self._inst.definition.instances.remove(self._inst)

    def undo(self, model) -> None:
        if self._inst not in self._parent.children:
            self._parent.children.append(self._inst)
        if self._inst not in self._inst.definition.instances:
            self._inst.definition.instances.append(self._inst)
