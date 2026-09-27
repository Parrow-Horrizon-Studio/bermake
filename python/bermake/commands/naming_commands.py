"""Naming commands (M7.3): RenameInstanceCommand."""

from __future__ import annotations

from bermake.commands.command import Command


class RenameInstanceCommand(Command):
    """Set one instance's name; undo restores the previous one.

    The previous name may legitimately be "" -- which means "fall back to
    definition.name" -- so undo restores it verbatim rather than treating
    empty as "unset".
    """

    name = "Rename"

    def __init__(self, instance, new_name: str) -> None:
        self._instance = instance
        self._new = str(new_name).strip()
        self._old: str = ""

    def do(self, model) -> None:
        self._old = self._instance.name
        self._instance.name = self._new

    def undo(self, model) -> None:
        self._instance.name = self._old
