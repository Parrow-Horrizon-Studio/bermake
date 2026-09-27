"""CommandStack: undo + redo with execute / push_executed semantics."""

from __future__ import annotations

from bermake.commands.command import Command


class CommandStack:
    """Owns the undo + redo stacks. Owned by MainWindow."""

    def __init__(self) -> None:
        self._undo: list[tuple] = []
        self._redo: list[tuple] = []
        self._on_after_undo: list = []  # list[Callable[[], None]]
        self._on_after_redo: list = []  # list[Callable[[], None]]
        self._on_change: list = []  # list[Callable[[], None]]

    def add_undo_listener(self, fn) -> None:
        """Register a zero-arg callable to invoke after each successful undo."""
        self._on_after_undo.append(fn)

    def add_redo_listener(self, fn) -> None:
        """Register a zero-arg callable to invoke after each successful redo."""
        self._on_after_redo.append(fn)

    def add_change_listener(self, fn) -> None:
        """Register a zero-arg callable fired after every stack mutation."""
        self._on_change.append(fn)

    def _fire_change(self) -> None:
        for fn in self._on_change:
            fn()

    def clear(self) -> None:
        """Empty both stacks (used when switching documents). Fires no listeners."""
        self._undo.clear()
        self._redo.clear()

    def execute(self, cmd: Command, target) -> None:
        """Run cmd.do(target), push (cmd, target) to undo stack, clear redo stack."""
        cmd.do(target)
        self._undo.append((cmd, target))
        self._redo.clear()
        self._fire_change()

    def push_executed(self, cmd: Command, target) -> None:
        """Append a command whose do() was already called incrementally.

        Used by tools that build a CompositeCommand mutating the scene as
        the gesture progresses so the snap engine sees in-progress state.
        At gesture completion the tool calls push_executed(composite, scene) to
        register it for undo without re-executing.
        """
        self._undo.append((cmd, target))
        self._redo.clear()
        self._fire_change()

    def undo(self) -> bool:
        if not self._undo:
            return False
        cmd, target = self._undo.pop()
        cmd.undo(target)
        self._redo.append((cmd, target))
        for fn in self._on_after_undo:
            fn()
        self._fire_change()
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        cmd, target = self._redo.pop()
        cmd.do(target)
        self._undo.append((cmd, target))
        for fn in self._on_after_redo:
            fn()
        self._fire_change()
        return True

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)
