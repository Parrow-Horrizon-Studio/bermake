"""Visibility commands (M7.3): HideInstancesCommand."""

from __future__ import annotations

from bermake.commands.command import Command


class HideInstancesCommand(Command):
    """Hide or unhide a set of instances; undo restores each prior state.

    Captures each instance's previous `hidden` at do() time (id-preserving
    undo), so undoing an Unhide re-hides only the instances that were hidden
    to begin with. Group commands take the model as their target, so do/undo
    take `model`.
    """

    def __init__(self, instances, hidden: bool) -> None:
        self._instances = list(instances)
        self._new = bool(hidden)
        self._old: dict[int, bool] = {}
        self.name = "Hide" if self._new else "Unhide"

    def do(self, model) -> None:
        for inst in self._instances:
            self._old[inst.id] = inst.hidden
            inst.hidden = self._new

    def undo(self, model) -> None:
        for inst in self._instances:
            inst.hidden = self._old.get(inst.id, False)
