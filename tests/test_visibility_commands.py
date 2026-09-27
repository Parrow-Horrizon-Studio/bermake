"""Hide and rename are undoable (M7.3 Task 3)."""

from __future__ import annotations

import numpy as np
from bermake.commands.command_stack import CommandStack
from bermake.commands.naming_commands import RenameInstanceCommand
from bermake.commands.visibility_commands import HideInstancesCommand


def _square(model):
    scene = model.active_context.mesh
    v = [
        scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32)),
    ]
    scene.add_face_from_loop(v)


def test_hide_then_undo_restores_visibility(model_factory, group_factory):
    model = model_factory()
    _square(model)
    instance = group_factory(model)
    stack = CommandStack()

    stack.execute(HideInstancesCommand([instance], True), model)
    assert instance.hidden is True

    stack.undo()
    assert instance.hidden is False


def test_unhide_restores_only_what_was_hidden(model_factory, group_factory):
    # Prior state is captured per instance, so undoing an Unhide must not
    # blanket-hide an instance that was already visible before the command.
    model = model_factory()
    _square(model)
    already_visible = group_factory(model)
    _square(model)
    was_hidden = group_factory(model)
    was_hidden.hidden = True
    stack = CommandStack()

    stack.execute(HideInstancesCommand([already_visible, was_hidden], False), model)
    assert already_visible.hidden is False
    assert was_hidden.hidden is False

    stack.undo()
    assert already_visible.hidden is False
    assert was_hidden.hidden is True


def test_hide_redo_reapplies(model_factory, group_factory):
    model = model_factory()
    _square(model)
    instance = group_factory(model)
    stack = CommandStack()
    stack.execute(HideInstancesCommand([instance], True), model)
    stack.undo()

    stack.redo()

    assert instance.hidden is True


def test_the_hide_command_is_named_for_its_direction(model_factory, group_factory):
    model = model_factory()
    _square(model)
    instance = group_factory(model)
    assert HideInstancesCommand([instance], True).name == "Hide"
    assert HideInstancesCommand([instance], False).name == "Unhide"


def test_rename_then_undo_restores_the_previous_name(model_factory, group_factory):
    model = model_factory()
    _square(model)
    instance = group_factory(model)
    instance.name = "Old"
    stack = CommandStack()

    stack.execute(RenameInstanceCommand(instance, "New"), model)
    assert instance.name == "New"

    stack.undo()
    assert instance.name == "Old"


def test_rename_undo_restores_an_empty_name(model_factory, group_factory):
    # "" is meaningful -- it means "fall back to definition.name" -- so undo
    # must restore it rather than leaving the typed name in place.
    model = model_factory()
    _square(model)
    instance = group_factory(model)
    stack = CommandStack()

    stack.execute(RenameInstanceCommand(instance, "Named"), model)
    stack.undo()

    assert instance.name == ""


def test_rename_strips_surrounding_whitespace(model_factory, group_factory):
    model = model_factory()
    _square(model)
    instance = group_factory(model)
    stack = CommandStack()

    stack.execute(RenameInstanceCommand(instance, "  Padded  "), model)

    assert instance.name == "Padded"
