"""An unfinished Line chain is committed before the window acts on the document (#144).

The chain's edges are in the model from the first click, but they reach the
undo stack only when the chain finishes. Anything that leaves the tool or reads
the undo stack or the dirty flag must finish the chain first, or the edges sit
in the model with no undo entry and no unsaved-changes prompt.
"""

from __future__ import annotations

import numpy as np
from bermake.commands.scene_commands import AddVertexCommand


def _grid_snap(world):
    from bermake.viewport.snap_engine import SnapKind, SnapResult

    return SnapResult(
        kind=SnapKind.GRID,
        world_position=np.array(world, dtype=np.float32),
        axis=None,
        vertex_id=None,
        label="Grid",
    )


def _draw_open_chain(window) -> None:
    """Arm Line and click three points without finishing the chain."""
    window._activate("line")
    tool = window._tool_manager.active
    for point in ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0)):
        tool.on_mouse_press(None, _grid_snap(point))
    assert tool.has_active_gesture is True
    assert _edge_count(window) == 2


def _edge_count(window) -> int:
    return len(list(window._model.active_scene.edges_iter()))


def test_switching_tools_mid_chain_keeps_the_edges_as_one_undo_step(main_window):
    _draw_open_chain(main_window)
    assert main_window._command_stack.can_undo is False
    assert main_window._doc_controller.dirty is False

    main_window._activate("select")

    assert _edge_count(main_window) == 2
    assert main_window._doc_controller.dirty is True
    assert main_window._command_stack.can_undo
    main_window._on_undo()
    assert _edge_count(main_window) == 0


def test_new_mid_chain_asks_to_save_the_chain(main_window):
    asked: list[bool] = []

    def prompt():
        asked.append(True)
        return "cancel"

    main_window._prompt_discard = prompt
    _draw_open_chain(main_window)

    main_window._on_file_new()

    assert asked == [True]
    assert _edge_count(main_window) == 2


def test_undo_mid_chain_removes_the_chain_and_keeps_earlier_work(main_window):
    main_window._command_stack.execute(
        AddVertexCommand(np.array([9.0, 9.0, 9.0])), main_window._model.active_scene
    )
    _draw_open_chain(main_window)

    main_window._on_undo()

    assert _edge_count(main_window) == 0
    assert len(list(main_window._model.active_scene.vertices_iter())) == 1
    assert main_window._command_stack.can_redo


def test_save_mid_chain_commits_then_saves_clean(main_window, tmp_path):
    _draw_open_chain(main_window)

    assert main_window._save_to(tmp_path / "Chain.berm") is True

    assert main_window._command_stack.can_undo
    assert main_window._doc_controller.dirty is False
    assert _edge_count(main_window) == 2


def test_revert_mid_chain_on_a_clean_saved_document_reverts(main_window, tmp_path):
    target = tmp_path / "Saved.berm"
    assert main_window._save_to(target) is True
    asked: list[str] = []
    main_window._prompt_revert = lambda name: asked.append(name) or True
    _draw_open_chain(main_window)

    main_window._on_file_revert()

    assert asked == ["Saved.berm"]
    assert _edge_count(main_window) == 0
