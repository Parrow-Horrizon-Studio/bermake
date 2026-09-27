"""#60: ScaleTool's local-to-world lift, and defensive append guards.

Brief drift found while implementing (see task-12-brief.md):

- `main_window._tool_manager._tools[...]` doesn't exist. Task 5 re-keyed the
  registry to `_tools_by_id`; the public path is `activate_by_id`. We use
  `main_window._activate("scale")` (which also rebuilds the tool context --
  closer to what actually happens when a user presses S) and then read
  `main_window._tool_manager.active`.
- `main_window._tool_context()` builds a `ToolContext` but does not arm a
  tool (it exists, but arming through it would duplicate what `_activate`
  already does); `_activate` is used throughout instead.
- `test_double_redo_does_not_duplicate_a_group`, as specified in the brief,
  does not reproduce: `CommandStack.redo()` pops one item off its own redo
  stack and is a no-op once that stack is exhausted (see
  `command_stack.py:69-78`), so a second `main_window._command_stack.redo()`
  call never reaches `MakeGroupCommand._redo` a second time in the first
  place -- guarded or not. Further, `_redo` cannot actually be re-entered a
  second time in a row without an intervening `undo()`: its first lines
  unconditionally call `parent_scene.remove_face(f)` on geometry that a
  prior `_redo()` call already removed, which raises `KeyError` from the
  native mesh before the append line is ever reached (verified empirically
  -- see the report). So this specific guard has no reachable duplicate-
  append scenario at all; it is pure belt-and-braces. It is still added
  per the brief's defensive-sweep instruction, but is not given a dedicated
  test here since none can discriminate guarded from unguarded behaviour
  without first hitting an unrelated crash.

  The other two named append sites (`CreateInstanceCommand.do` and
  `DeleteInstanceCommand.undo`) do NOT have this problem -- they never touch
  mesh geometry, so calling `do()`/`undo()` twice in a row (bypassing
  `CommandStack`'s own single-pop protocol, by holding the raw command
  object -- exactly the re-entrancy `Command`'s own docstring in
  `command.py` contracts for) genuinely duplicates the list append pre-fix
  and is idempotent post-fix. Those get direct, discriminating tests below.

- `DeleteInstanceCommand` does not live in `instance_commands.py` as the
  brief states -- it is in `instance_lifecycle_commands.py`, alongside
  `MakeUniqueCommand`. The append-guard fix and its test target that file.
"""

from __future__ import annotations

import numpy as np
import pytest
from bermake.commands.instance_commands import CreateInstanceCommand
from bermake.commands.instance_lifecycle_commands import DeleteInstanceCommand
from bermake.model.model import Model


def test_lift_returns_points_unchanged_at_the_root(main_window):
    main_window._activate("scale")
    tool = main_window._tool_manager.active
    pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    assert np.allclose(tool._lift_local_to_world(pts), pts)


def test_lift_applies_the_active_instance_transform(main_window, group_factory):
    # Inside a group whose instance sits at x=10, a local point at the origin
    # must lift to x=10 in world coordinates.
    scene = main_window._model.active_context.mesh
    v = [
        scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32)),
    ]
    scene.add_face_from_loop(v)
    inst = group_factory(main_window._model)
    inst.transform[:3, 3] = [10.0, 0.0, 0.0]
    main_window._model.enter(inst)

    main_window._activate("scale")
    tool = main_window._tool_manager.active
    lifted = tool._lift_local_to_world(np.array([[0.0, 0.0, 0.0]], dtype=np.float64))
    assert lifted[0][0] == pytest.approx(10.0)


def test_lift_is_none_safe_without_a_model(main_window):
    """Defensive guard: `_world_transform()` returns None when `_model` is
    unset, and `_lift_local_to_world` must not raise on that -- it must
    return the input points unchanged, exactly like the root case."""
    main_window._activate("scale")
    tool = main_window._tool_manager.active
    tool._model = None
    pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    assert np.allclose(tool._lift_local_to_world(pts), pts)


def test_create_instance_do_guards_against_duplicate_append():
    """CreateInstanceCommand.do() must not duplicate the parent-children append.

    `Command`'s own docstring (command.py) contracts do()/undo() to be
    idempotent on re-entry. Calling do() twice in a row with no intervening
    undo() -- possible for any caller holding the raw command object,
    bypassing CommandStack's single-pop redo() -- must not insert the
    instance into parent.children twice.
    """
    m = Model()
    d = m.new_definition("Chair", is_group=False)
    cmd = CreateInstanceCommand(m.root, d, np.eye(4))
    cmd.do(m)
    cmd.do(m)  # duplicate call, no undo() in between
    assert m.root.children.count(cmd.created_instance) == 1


def test_delete_instance_undo_guards_against_duplicate_append():
    """DeleteInstanceCommand.undo() must not duplicate the parent-children append."""
    m = Model()
    d = m.new_definition("Chair", is_group=False)
    inst = m.new_instance(d)
    m.root.children.append(inst)
    cmd = DeleteInstanceCommand(m.root, inst)
    cmd.do(m)
    cmd.undo(m)
    cmd.undo(m)  # duplicate call, no do() in between
    assert m.root.children.count(inst) == 1
