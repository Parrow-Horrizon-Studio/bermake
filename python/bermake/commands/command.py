"""Command framework: Command ABC + CompositeCommand."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class Command(ABC):
    """A reversible operation on the Scene.

    do() executes the operation and may capture state needed by undo().
    undo() reverses the operation exactly. Both must be idempotent on
    re-entry — i.e. `do(); undo(); do(); undo()` leaves the scene in the
    same state as `do(); undo()`.
    """

    name: str = "Command"

    @abstractmethod
    def do(self, scene) -> None: ...

    @abstractmethod
    def undo(self, scene) -> None: ...


@dataclass
class CompositeCommand(Command):
    """A sequence of commands executed/undone as one unit (per-gesture grouping)."""

    name: str
    children: list[Command] = field(default_factory=list)

    def do(self, scene) -> None:
        for c in self.children:
            c.do(scene)

    def undo(self, scene) -> None:
        for c in reversed(self.children):
            c.undo(scene)
