"""M7.6c smart-select: the double-click and triple-click matrix.

These drive SelectTool's hooks directly with a hand-built ToolContext, which
is how tests/test_select_tool.py already works.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QMouseEvent


def _camera(w, h):
    from bermake.viewport.camera import Camera

    cam = Camera()
    cam.aspect = float(w) / float(h)
    return cam


def _quad_pair_scene():
    """Two coplanar quads sharing one edge, at z=0."""
    from bermake.scene import Scene

    scene = Scene()
    a = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    c = scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    d = scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    e = scene.add_vertex(np.array([2.0, 0.0, 0.0], dtype=np.float32))
    f = scene.add_vertex(np.array([2.0, 1.0, 0.0], dtype=np.float32))
    left = scene.add_face_from_loop((a, b, c, d))
    right = scene.add_face_from_loop((b, e, f, c))
    return scene, {"a": a, "b": b, "c": c, "left": left, "right": right}


def _tool(scene, w=800, h=600):
    from bermake.selection import Selection
    from bermake.tools.select_tool import SelectTool
    from bermake.tools.tool import ToolContext

    sel = Selection()
    cam = _camera(w, h)
    tool = SelectTool()
    tool.activate(
        ToolContext(
            scene=scene,
            camera=cam,
            widget_size_provider=lambda: (w, h),
            selection=sel,
        )
    )
    return tool, sel, cam


def _event_at(cam, world_point, modifiers=Qt.KeyboardModifier.NoModifier, w=800, h=600):
    sx, sy, _ = cam.world_to_screen(np.asarray(world_point, dtype=np.float32), w, h)
    return QMouseEvent(
        QMouseEvent.Type.MouseButtonDblClick,
        QPointF(float(sx), float(sy)),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        modifiers,
    )


def test_double_click_a_face_selects_it_and_its_bounding_edges():
    scene, ids = _quad_pair_scene()
    tool, sel, cam = _tool(scene)
    # Centre of the left quad, away from every edge.
    tool.on_mouse_double_click(_event_at(cam, (0.5, 0.5, 0.0)), None)
    assert sel.faces == {ids["left"]}
    assert len(sel.edges) == 4


def test_double_click_an_edge_selects_it_and_its_adjacent_faces():
    scene, ids = _quad_pair_scene()
    tool, sel, cam = _tool(scene)
    shared = scene.edge_between(ids["b"], ids["c"])
    # Midpoint of the shared edge.
    tool.on_mouse_double_click(_event_at(cam, (1.0, 0.5, 0.0)), None)
    assert sel.edges == {shared}
    assert sel.faces == {ids["left"], ids["right"]}


def test_double_click_a_face_does_not_select_the_neighbouring_face():
    scene, ids = _quad_pair_scene()
    tool, sel, cam = _tool(scene)
    tool.on_mouse_double_click(_event_at(cam, (0.5, 0.5, 0.0)), None)
    assert ids["right"] not in sel.faces


def test_triple_click_selects_the_whole_connected_component():
    scene, ids = _quad_pair_scene()
    tool, sel, cam = _tool(scene)
    tool.on_mouse_triple_click(_event_at(cam, (0.5, 0.5, 0.0)), None)
    assert sel.faces == {ids["left"], ids["right"]}
    assert len(sel.edges) == 7


def test_triple_click_on_empty_space_leaves_the_selection_alone():
    scene, ids = _quad_pair_scene()
    tool, sel, cam = _tool(scene)
    sel.replace(faces={ids["left"]})
    tool.on_mouse_triple_click(_event_at(cam, (50.0, 50.0, 0.0)), None)
    assert sel.faces == {ids["left"]}


def test_shift_double_click_adds_rather_than_replaces():
    scene, ids = _quad_pair_scene()
    tool, sel, cam = _tool(scene)
    sel.replace(faces={ids["right"]})
    ev = _event_at(cam, (0.5, 0.5, 0.0), modifiers=Qt.KeyboardModifier.ShiftModifier)
    tool.on_mouse_double_click(ev, None)
    assert sel.faces == {ids["left"], ids["right"]}


def test_double_click_on_empty_space_leaves_the_selection_alone():
    scene, ids = _quad_pair_scene()
    tool, sel, cam = _tool(scene)
    sel.replace(faces={ids["left"]})
    tool.on_mouse_double_click(_event_at(cam, (50.0, 50.0, 0.0)), None)
    assert sel.faces == {ids["left"]}


def _tool_with_vertices(scene, w=800, h=600):
    from bermake.selection import Selection
    from bermake.tools.select_tool import SelectTool
    from bermake.tools.tool import ToolContext

    sel = Selection()
    cam = _camera(w, h)
    tool = SelectTool()
    tool.activate(
        ToolContext(
            scene=scene,
            camera=cam,
            widget_size_provider=lambda: (w, h),
            selection=sel,
            select_vertices_provider=lambda: True,
        )
    )
    return tool, sel, cam


def test_double_click_a_vertex_selects_it_and_its_incident_edges():
    scene, ids = _quad_pair_scene()
    tool, sel, cam = _tool_with_vertices(scene)
    tool.on_mouse_double_click(_event_at(cam, (1.0, 0.0, 0.0)), None)
    assert sel.vertices == {ids["b"]}
    # b touches a-b, b-c and b-e.
    assert len(sel.edges) == 3
    assert sel.faces == set()


def test_triple_click_keeps_vertices_only_when_the_mode_is_on():
    """Spec D5: a user who never enabled the mode cannot acquire a vertex
    selection by accident, and so cannot be surprised by a Move that drags
    vertices they cannot see selected."""
    scene, _ids = _quad_pair_scene()
    off_tool, off_sel, cam = _tool(scene)
    off_tool.on_mouse_triple_click(_event_at(cam, (0.5, 0.5, 0.0)), None)
    assert off_sel.vertices == set()

    on_tool, on_sel, cam2 = _tool_with_vertices(scene)
    on_tool.on_mouse_triple_click(_event_at(cam2, (0.5, 0.5, 0.0)), None)
    assert len(on_sel.vertices) == 6


def test_with_the_mode_off_a_corner_double_click_still_smart_selects_the_edge():
    scene, _ids = _quad_pair_scene()
    tool, sel, cam = _tool(scene)
    tool.on_mouse_double_click(_event_at(cam, (0.0, 0.0, 0.0)), None)
    assert sel.vertices == set()
    assert len(sel.edges) == 1


# --- Final review M9: an empty flood must not wipe the selection -----------


def _quad_pair_with_a_loose_vertex():
    """The quad pair plus one live vertex with no incident edge.

    Task 6 confirmed a bare vertex legitimately outlives its removed edge in
    this mesh, so this is a reachable state rather than a contrived one.
    """
    scene, ids = _quad_pair_scene()
    ids["loose"] = scene.add_vertex(np.array([0.5, 3.0, 0.0], dtype=np.float32))
    return scene, ids


def test_triple_click_an_isolated_vertex_leaves_the_selection_alone():
    """`connected_component` returns three empty sets for a live vertex with
    no incident edge -- the seed is not in the adjacency map at all. Applying
    that wiped whatever was selected. A flood that found nothing leaves the
    selection alone, exactly as a double-click that missed does."""
    scene, ids = _quad_pair_with_a_loose_vertex()
    tool, sel, cam = _tool_with_vertices(scene)
    sel.replace(faces={ids["left"]})
    tool.on_mouse_triple_click(_event_at(cam, (0.5, 3.0, 0.0)), None)
    assert sel.faces == {ids["left"]}
    assert sel.vertices == set()


def test_triple_click_an_isolated_vertex_still_floods_a_real_component():
    """Guards the early return against being too eager: the loose vertex in
    the scene must not stop an ordinary triple-click from working."""
    scene, ids = _quad_pair_with_a_loose_vertex()
    tool, sel, cam = _tool_with_vertices(scene)
    tool.on_mouse_triple_click(_event_at(cam, (0.5, 0.5, 0.0)), None)
    assert sel.faces == {ids["left"], ids["right"]}
    assert ids["loose"] not in sel.vertices
