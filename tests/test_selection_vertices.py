"""Selection gains a fifth kind, and the four consumers that must move with it."""

from __future__ import annotations

import numpy as np


def test_selection_starts_with_no_vertices():
    from bermake.selection import Selection

    assert Selection().vertices == set()


def test_replace_sets_vertices_and_bumps_the_version():
    from bermake.selection import Selection

    sel = Selection()
    before = sel.version
    sel.replace(vertices={1, 2})
    assert sel.vertices == {1, 2}
    assert sel.version > before


def test_replace_without_vertices_clears_them():
    """replace() is a full reset of every kind. A vertices-shaped hole in it
    would leave stale vertices behind after an ordinary click."""
    from bermake.selection import Selection

    sel = Selection()
    sel.replace(vertices={1, 2})
    assert sel.vertices == {1, 2}, "the first replace must actually set vertices"
    sel.replace(edges={7})
    assert sel.vertices == set()


def test_add_and_remove_vertices():
    from bermake.selection import Selection

    sel = Selection()
    sel.add(vertices={1, 2})
    sel.remove(vertices={1})
    assert sel.vertices == {2}


def test_toggle_vertex():
    from bermake.selection import Selection

    sel = Selection()
    sel.toggle_vertex(5)
    assert sel.vertices == {5}
    sel.toggle_vertex(5)
    assert sel.vertices == set()


def test_clear_drops_vertices():
    from bermake.selection import Selection

    sel = Selection()
    sel.replace(vertices={1})
    sel.clear()
    assert sel.vertices == set()


def test_is_empty_accounts_for_a_vertex_only_selection():
    """A vertex-only selection is NOT empty. Move consults is_empty() before
    capturing a drag, so a False here means a selected vertex cannot move."""
    from bermake.selection import Selection

    sel = Selection()
    sel.replace(vertices={1})
    assert not sel.is_empty()


def test_counts_appends_vertices_last():
    from bermake.selection import Selection

    sel = Selection()
    sel.replace(edges={1}, faces={2, 3}, vertices={4, 5, 6})
    assert sel.counts() == (1, 2, 0, 0, 3)


def test_status_text_names_vertices():
    from bermake.selection import Selection
    from bermake.ui.selection_controller import selection_status_text

    sel = Selection()
    sel.replace(vertices={4, 5})
    assert selection_status_text(sel) == "2 vertices selected"


def test_status_text_singular_vertex():
    from bermake.selection import Selection
    from bermake.ui.selection_controller import selection_status_text

    sel = Selection()
    sel.replace(vertices={4})
    assert selection_status_text(sel) == "1 vertex selected"


def _scene_with_one_edge():
    from bermake.scene import Scene

    scene = Scene()
    a = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    e = scene.add_edge(a, b)
    return scene, a, b, e


def test_selection_vertices_includes_a_bare_vertex_selection():
    """transform_support.selection_vertices is the chokepoint Move, Rotate and
    Scale all read. This one union is what makes a bare vertex draggable."""
    from bermake.selection import Selection
    from bermake.tools.transform_support import selection_vertices

    scene, a, _b, _e = _scene_with_one_edge()
    sel = Selection()
    sel.replace(vertices={a})
    assert selection_vertices(scene, sel) == [a]


def test_selection_vertices_unions_a_vertex_with_an_edge_without_duplicating():
    from bermake.selection import Selection
    from bermake.tools.transform_support import selection_vertices

    scene, a, b, e = _scene_with_one_edge()
    sel = Selection()
    sel.replace(edges={e}, vertices={a})
    assert selection_vertices(scene, sel) == sorted([a, b])


def test_selection_vertices_skips_a_dead_vertex_id():
    from bermake.selection import Selection
    from bermake.tools.transform_support import selection_vertices

    scene, a, _b, _e = _scene_with_one_edge()
    sel = Selection()
    sel.replace(vertices={a, 9999})
    assert selection_vertices(scene, sel) == [a]


def test_delete_does_nothing_to_a_vertex_only_selection(qtbot):
    """D6: deleting a vertex would have to cascade into its edges and faces.
    A no-op is the honest behaviour, not a silent cascade."""
    import numpy as np

    from bermake.ui.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    scene = w._model.active_scene
    a = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    e = scene.add_edge(a, b)
    w._selection.replace(vertices={a})
    depth_before = len(w._command_stack._undo)
    w._on_delete_selection()
    assert scene.edge_is_live(e)
    assert len(w._command_stack._undo) == depth_before


def test_prune_to_live_keeps_a_vertex_whose_edge_died(model_factory):
    """Mutant-3 coverage for prune_to_live's vertices term. In this mesh
    remove_edge does not cascade into its endpoints, so a bare vertex
    legitimately outlives its edge. Omitting the vertices= kwarg from
    prune_to_live's replace() call would fall back to the empty default and
    unconditionally clear every selected vertex, live or not; this test
    catches that by asserting the still-live vertex survives."""
    from bermake.commands.command_stack import CommandStack
    from bermake.commands.scene_commands import RemoveEdgeCommand
    from bermake.selection import Selection
    from bermake.ui.selection_controller import prune_to_live

    model = model_factory()
    scene = model.active_context.mesh
    a = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    e = scene.add_edge(a, b)

    sel = Selection()
    sel.replace(vertices={a})

    stack = CommandStack()
    stack.execute(RemoveEdgeCommand(e), scene)

    prune_to_live(model, sel)

    assert not scene.edge_is_live(e)
    assert sel.vertices == {a}, "a bare vertex outlives its removed edge in this mesh"


def test_prune_to_live_drops_a_dangling_vertex_id(model_factory):
    """The other half of prune_to_live's vertices term: a synthetic id that
    was never live (the #46 shape -- an undo/redo can leave a dangling id
    behind) must be dropped, while a real, still-live vertex survives.
    Mirrors test_selection_vertices_skips_a_dead_vertex_id's synthetic-id
    pattern rather than relying on cascading deletion, since nothing in this
    mesh cascades a vertex's death from its edge."""
    from bermake.selection import Selection
    from bermake.ui.selection_controller import prune_to_live

    model = model_factory()
    scene = model.active_context.mesh
    a = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))

    sel = Selection()
    sel.replace(vertices={a, 9999})

    prune_to_live(model, sel)

    assert sel.vertices == {a}
