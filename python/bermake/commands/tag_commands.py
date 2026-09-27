"""Tag commands: TagInstancesCommand (M5c), SetTagColorCommand (M7.5a Task 11)."""

from __future__ import annotations

from bermake.commands.command import Command

_UNTAGGED_ID = 0  # == TagLibrary.UNTAGGED_ID


class TagInstancesCommand(Command):
    """Assign a tag to a set of instances; undo restores each instance's prior tag.

    Captures each instance's previous tag at do() time (id-preserving undo).
    Group commands take the model as their target, so do/undo take `model`.
    """

    name = "Assign Tag"

    def __init__(self, instances, new_tag_id: int) -> None:
        self._instances = list(instances)
        self._new = int(new_tag_id)
        self._old: dict[int, int] = {}

    def do(self, model) -> None:
        for inst in self._instances:
            self._old[inst.id] = inst.tag_id
            inst.tag_id = self._new

    def undo(self, model) -> None:
        for inst in self._instances:
            inst.tag_id = self._old.get(inst.id, _UNTAGGED_ID)


class SetTagColorCommand(Command):
    """Set one tag's colour; undo restores the previous one.

    Colour is document state (M7.5a Task 11), unlike `visible` -- so, like
    TagInstancesCommand, it goes through the command stack rather than
    TagsPage calling TagLibrary.set_color directly. Captures the previous
    colour at do() time (id-preserving undo), the same shape
    EditMaterialCommand uses for a single field.
    """

    name = "Set Tag Color"

    def __init__(self, library, tag_id: int, new_color: tuple[float, float, float]) -> None:
        self._lib = library
        self._tid = int(tag_id)
        self._new = tuple(float(c) for c in new_color)
        self._old: tuple[float, float, float] | None = None

    def do(self, model) -> None:
        self._old = self._lib.get(self._tid).color
        self._lib.set_color(self._tid, self._new)

    def undo(self, model) -> None:
        self._lib.set_color(self._tid, self._old)
