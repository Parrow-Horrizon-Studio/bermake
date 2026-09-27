"""Scene-management commands (M7e): undoable create/delete/rename/reorder/update
of SavedViews. All take the Model as their target and mutate model.views.

Recall (going TO a Scene) is deliberately NOT a command — it is a view change,
like an orbit, and bypasses the undo stack (matching SketchUp). Only management
operations are undoable. Captures happen on the FIRST do() only, so repeated
undo/redo restores the true original (the M7d Move/EditLabel lesson)."""

from __future__ import annotations

from bermake.commands.command import Command


class CreateViewCommand(Command):
    """Add a SavedView; undo removes it; redo re-attaches the same object."""

    name = "Create Scene"

    def __init__(self, view) -> None:
        self._view = view

    def do(self, model) -> None:
        model.views.add(self._view)

    def undo(self, model) -> None:
        model.views.remove(self._view.id)


class DeleteViewCommand(Command):
    """Remove a SavedView; undo restores it at its original index."""

    name = "Delete Scene"

    def __init__(self, view_id: int) -> None:
        self._id = int(view_id)
        self._view = None
        self._index = -1

    def do(self, model) -> None:
        if self._view is None:
            self._view = model.views.get(self._id)
            self._index = model.views.index_of(self._id)
        model.views.remove(self._id)

    def undo(self, model) -> None:
        if self._view is not None and self._index >= 0:
            model.views.insert(self._index, self._view)


class RenameViewCommand(Command):
    """Rename a SavedView; undo restores the original name (captured once)."""

    name = "Rename Scene"

    def __init__(self, view_id: int, new_name: str) -> None:
        self._id = int(view_id)
        self._new = str(new_name)
        self._old = None

    def do(self, model) -> None:
        if self._old is None:
            current = model.views.get(self._id)
            self._old = current.name if current is not None else ""
        model.views.rename(self._id, self._new)

    def undo(self, model) -> None:
        model.views.rename(self._id, self._old)


class ReorderViewCommand(Command):
    """Move a SavedView one place up/down; undo reverses it (only if it moved)."""

    name = "Reorder Scene"

    def __init__(self, view_id: int, direction: int) -> None:
        self._id = int(view_id)
        self._dir = int(direction)
        self._moved = False

    def do(self, model) -> None:
        self._moved = model.views.move(self._id, self._dir)

    def undo(self, model) -> None:
        if self._moved:
            model.views.move(self._id, -self._dir)


class UpdateViewCommand(Command):
    """Overwrite a SavedView's snapshot; undo restores the prior one (captured
    once)."""

    name = "Update Scene"

    def __init__(self, view_id: int, new_view) -> None:
        self._id = int(view_id)
        self._new = new_view
        self._old = None

    def do(self, model) -> None:
        if self._old is None:
            self._old = model.views.get(self._id)
        model.views.replace_view(self._id, self._new)

    def undo(self, model) -> None:
        if self._old is not None:
            model.views.replace_view(self._id, self._old)
