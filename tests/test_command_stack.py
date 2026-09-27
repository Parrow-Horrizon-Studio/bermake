"""Tests for the command framework — Command ABC, CompositeCommand, CommandStack."""

from __future__ import annotations

import numpy as np


class _RecordingCommand:
    """Test helper: records call order without needing a real Scene."""

    def __init__(self, label: str, log: list[str]) -> None:
        self._label = label
        self._log = log

    def do(self, scene) -> None:  # noqa: ANN001
        self._log.append(f"do:{self._label}")

    def undo(self, scene) -> None:  # noqa: ANN001
        self._log.append(f"undo:{self._label}")


def test_composite_do_runs_children_in_order():
    from bermake.commands import CompositeCommand

    log: list[str] = []
    composite = CompositeCommand(
        name="Test",
        children=[_RecordingCommand("a", log), _RecordingCommand("b", log), _RecordingCommand("c", log)],
    )
    composite.do(None)
    assert log == ["do:a", "do:b", "do:c"]


def test_composite_undo_runs_children_in_reverse_order():
    from bermake.commands import CompositeCommand

    log: list[str] = []
    composite = CompositeCommand(
        name="Test",
        children=[_RecordingCommand("a", log), _RecordingCommand("b", log), _RecordingCommand("c", log)],
    )
    composite.undo(None)
    assert log == ["undo:c", "undo:b", "undo:a"]


def test_command_stack_starts_empty():
    from bermake.commands import CommandStack

    s = CommandStack()
    assert not s.can_undo
    assert not s.can_redo


def test_execute_runs_do_and_pushes_to_undo_stack():
    from bermake.commands import CommandStack, CompositeCommand

    log: list[str] = []
    cmd = CompositeCommand(name="C", children=[_RecordingCommand("x", log)])
    s = CommandStack()
    s.execute(cmd, None)
    assert log == ["do:x"]
    assert s.can_undo
    assert not s.can_redo


def test_push_executed_appends_without_calling_do():
    from bermake.commands import CommandStack, CompositeCommand

    log: list[str] = []
    cmd = CompositeCommand(name="C", children=[_RecordingCommand("x", log)])
    s = CommandStack()
    s.push_executed(cmd, None)
    assert log == []  # do was NOT called
    assert s.can_undo


def test_undo_calls_command_undo_and_moves_to_redo():
    from bermake.commands import CommandStack, CompositeCommand

    log: list[str] = []
    cmd = CompositeCommand(name="C", children=[_RecordingCommand("x", log)])
    s = CommandStack()
    s.execute(cmd, None)
    log.clear()
    assert s.undo() is True
    assert log == ["undo:x"]
    assert not s.can_undo
    assert s.can_redo


def test_redo_runs_do_again():
    from bermake.commands import CommandStack, CompositeCommand

    log: list[str] = []
    cmd = CompositeCommand(name="C", children=[_RecordingCommand("x", log)])
    s = CommandStack()
    s.execute(cmd, None)
    s.undo()
    log.clear()
    assert s.redo() is True
    assert log == ["do:x"]


def test_new_execute_clears_redo_stack():
    from bermake.commands import CommandStack, CompositeCommand

    log: list[str] = []
    s = CommandStack()
    cmd_a = CompositeCommand(name="A", children=[_RecordingCommand("a", log)])
    cmd_b = CompositeCommand(name="B", children=[_RecordingCommand("b", log)])
    s.execute(cmd_a, None)
    s.undo()
    assert s.can_redo
    s.execute(cmd_b, None)
    assert not s.can_redo  # new execute cleared redo


def test_undo_on_empty_returns_false():
    from bermake.commands import CommandStack

    s = CommandStack()
    assert s.undo() is False


def test_redo_on_empty_returns_false():
    from bermake.commands import CommandStack

    s = CommandStack()
    assert s.redo() is False


# ---------------------------------------------------------------------------
# Listener API (M4b) — add_undo_listener / add_redo_listener
# ---------------------------------------------------------------------------


def test_undo_listener_fires_once_after_successful_undo():
    from bermake.commands import CommandStack, CompositeCommand

    log: list[str] = []
    cmd = CompositeCommand(name="C", children=[_RecordingCommand("x", log)])
    s = CommandStack()

    undo_fired: list[int] = []
    redo_fired: list[int] = []
    s.add_undo_listener(lambda: undo_fired.append(1))
    s.add_redo_listener(lambda: redo_fired.append(1))

    s.execute(cmd, None)
    assert s.undo() is True

    assert undo_fired == [1], "undo listener must fire exactly once after a successful undo"
    assert redo_fired == [], "redo listener must NOT fire when undo is called"


def test_redo_listener_fires_once_after_successful_redo():
    from bermake.commands import CommandStack, CompositeCommand

    log: list[str] = []
    cmd = CompositeCommand(name="C", children=[_RecordingCommand("x", log)])
    s = CommandStack()

    undo_fired: list[int] = []
    redo_fired: list[int] = []
    s.add_undo_listener(lambda: undo_fired.append(1))
    s.add_redo_listener(lambda: redo_fired.append(1))

    s.execute(cmd, None)
    s.undo()
    undo_fired.clear()  # reset to isolate the redo assertion

    assert s.redo() is True

    assert redo_fired == [1], "redo listener must fire exactly once after a successful redo"
    assert undo_fired == [], "undo listener must NOT fire when redo is called"


def test_undo_listener_does_not_fire_on_empty_stack():
    from bermake.commands import CommandStack

    s = CommandStack()

    undo_fired: list[int] = []
    s.add_undo_listener(lambda: undo_fired.append(1))

    result = s.undo()

    assert result is False
    assert undo_fired == [], "undo listener must not fire when undo stack is empty"


def test_redo_listener_does_not_fire_on_empty_stack():
    from bermake.commands import CommandStack

    s = CommandStack()

    redo_fired: list[int] = []
    s.add_redo_listener(lambda: redo_fired.append(1))

    result = s.redo()

    assert result is False
    assert redo_fired == [], "redo listener must not fire when redo stack is empty"


# ---------------------------------------------------------------------------
# M4e Task 5 — per-command target threading
# ---------------------------------------------------------------------------


class _Recorder:
    name = "Rec"

    def __init__(self, log, tag):
        self._log, self._tag = log, tag

    def do(self, target):
        self._log.append(("do", self._tag, target))

    def undo(self, target):
        self._log.append(("undo", self._tag, target))


def test_stack_threads_per_command_target():
    from bermake.commands import CommandStack

    log = []
    s = CommandStack()
    s.execute(_Recorder(log, "a"), "SCENE_A")
    s.execute(_Recorder(log, "b"), "SCENE_B")
    assert s.undo() is True          # undoes b against SCENE_B
    assert s.undo() is True          # undoes a against SCENE_A
    assert s.undo() is False
    assert ("undo", "b", "SCENE_B") in log
    assert ("undo", "a", "SCENE_A") in log


def test_push_executed_remembers_target():
    from bermake.commands import CommandStack

    log = []
    s = CommandStack()
    s.push_executed(_Recorder(log, "x"), "TARGET_X")
    assert s.undo() is True
    assert ("undo", "x", "TARGET_X") in log


def test_listener_fire_counts_across_full_undo_redo_cycle():
    """Execute one command, undo (listener fires), redo (listener fires),
    then undo back to empty and call undo again — no additional undo-listener fire."""
    from bermake.commands import CommandStack, CompositeCommand

    log: list[str] = []
    cmd = CompositeCommand(name="C", children=[_RecordingCommand("x", log)])
    s = CommandStack()

    undo_fired: list[int] = []
    redo_fired: list[int] = []
    s.add_undo_listener(lambda: undo_fired.append(1))
    s.add_redo_listener(lambda: redo_fired.append(1))

    s.execute(cmd, None)

    # First undo — listener should fire once
    assert s.undo() is True
    assert undo_fired == [1]
    assert redo_fired == []

    # Redo — redo listener fires, undo listener stays silent
    assert s.redo() is True
    assert redo_fired == [1]
    assert undo_fired == [1]  # unchanged since redo

    # Second undo — undo listener fires again (total 2)
    assert s.undo() is True
    assert undo_fired == [1, 1]

    # Stack is now empty — third undo returns False, no additional fire
    assert s.undo() is False
    assert undo_fired == [1, 1], "listener must not fire a third time on an empty stack"
    assert redo_fired == [1], "redo listener must still be at exactly 1"


# ---------------------------------------------------------------------------
# M6a Task 7 — change-listener + clear
# ---------------------------------------------------------------------------


class _NoOpCmd:
    def do(self, target):  # noqa: ANN001
        pass

    def undo(self, target):  # noqa: ANN001
        pass


def test_change_listener_fires_on_every_mutation():
    from bermake.commands.command_stack import CommandStack

    stack = CommandStack()
    calls = []
    stack.add_change_listener(lambda: calls.append(1))

    stack.execute(_NoOpCmd(), object())
    assert len(calls) == 1
    stack.push_executed(_NoOpCmd(), object())
    assert len(calls) == 2
    stack.undo()
    assert len(calls) == 3
    stack.redo()
    assert len(calls) == 4


def test_clear_empties_both_stacks():
    from bermake.commands.command_stack import CommandStack

    stack = CommandStack()
    stack.execute(_NoOpCmd(), object())
    stack.undo()
    assert stack.can_redo
    stack.clear()
    assert not stack.can_undo
    assert not stack.can_redo


def test_clear_does_not_fire_the_change_listener():
    """clear() is documented as firing no listeners (used when switching
    documents, where the caller resets UI state through a different path).
    Guard that against a future change silently wiring clear() into
    _fire_change() and double-notifying listeners on document swap."""
    from bermake.commands.command_stack import CommandStack

    stack = CommandStack()
    stack.execute(_NoOpCmd(), object())
    calls = []
    stack.add_change_listener(lambda: calls.append(1))

    stack.clear()

    assert calls == [], "clear() must not fire the change listener"
