"""Gesture tests for the Select tool (click / Shift / empty-clear / hover)."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent


def _cam(w, h):
    from bermake.viewport.camera import Camera
    c = Camera()
    c.aspect = float(w) / float(h)
    return c


def _press(x, y, mods=Qt.KeyboardModifier.NoModifier):
    return QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(x, y),
                       Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, mods)


def _release(x, y, mods=Qt.KeyboardModifier.NoModifier):
    return QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(x, y),
                       Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton, mods)


def _scene_with_quad():
    from bermake.scene import Scene
    s = Scene()
    a = s.add_vertex(np.array([-1, -1, 0], dtype=np.float32))
    b = s.add_vertex(np.array([1, -1, 0], dtype=np.float32))
    c = s.add_vertex(np.array([1, 1, 0], dtype=np.float32))
    d = s.add_vertex(np.array([-1, 1, 0], dtype=np.float32))
    fid = s.add_face_from_loop((a, b, c, d))
    e_ab = s.add_edge(a, b)
    return s, fid, e_ab


def _make_tool(scene, sel, w=800, h=600):
    from bermake.tools import ToolContext
    from bermake.tools.select_tool import SelectTool
    cam = _cam(w, h)
    tool = SelectTool()
    tool.activate(ToolContext(scene=scene, camera=cam,
                              widget_size_provider=lambda: (w, h), selection=sel))
    return tool, cam


def _click(tool, cam, world, w=800, h=600, mods=Qt.KeyboardModifier.NoModifier):
    sx, sy, _ = cam.world_to_screen(np.asarray(world, dtype=np.float32), w, h)
    tool.on_mouse_press(_press(sx, sy, mods), None)
    tool.on_mouse_release(_release(sx, sy, mods), None)


def test_click_selects_edge_under_cursor(qtbot):
    from bermake.selection import Selection
    scene, fid, e_ab = _scene_with_quad()
    sel = Selection()
    tool, cam = _make_tool(scene, sel)
    _click(tool, cam, [0.0, -1.0, 0.0])
    assert sel.edges == {e_ab}
    assert sel.faces == set()


def test_click_face_interior_selects_face(qtbot):
    from bermake.selection import Selection
    scene, fid, e_ab = _scene_with_quad()
    sel = Selection()
    tool, cam = _make_tool(scene, sel)
    _click(tool, cam, [0.0, 0.0, 0.0])
    assert sel.faces == {fid}


def test_plain_click_replaces(qtbot):
    from bermake.selection import Selection
    scene, fid, e_ab = _scene_with_quad()
    sel = Selection()
    sel.replace(faces=[fid])
    tool, cam = _make_tool(scene, sel)
    _click(tool, cam, [0.0, -1.0, 0.0])
    assert sel.edges == {e_ab}
    assert sel.faces == set()


def test_shift_click_toggles(qtbot):
    from bermake.selection import Selection
    scene, fid, e_ab = _scene_with_quad()
    sel = Selection()
    tool, cam = _make_tool(scene, sel)
    shift = Qt.KeyboardModifier.ShiftModifier
    _click(tool, cam, [0.0, -1.0, 0.0], mods=shift)
    _click(tool, cam, [0.0, 0.0, 0.0], mods=shift)
    assert sel.edges == {e_ab} and sel.faces == {fid}
    _click(tool, cam, [0.0, -1.0, 0.0], mods=shift)
    assert sel.edges == set() and sel.faces == {fid}


def test_empty_click_clears(qtbot):
    from bermake.selection import Selection
    scene, fid, e_ab = _scene_with_quad()
    sel = Selection()
    sel.replace(faces=[fid])
    tool, cam = _make_tool(scene, sel)
    tool.on_mouse_press(_press(3.0, 3.0), None)
    tool.on_mouse_release(_release(3.0, 3.0), None)
    assert sel.is_empty()


def test_esc_clears_selection(qtbot):
    from bermake.selection import Selection
    scene, fid, e_ab = _scene_with_quad()
    sel = Selection()
    sel.replace(faces=[fid])
    tool, cam = _make_tool(scene, sel)
    ev = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
    tool.on_key_press(ev)
    assert sel.is_empty()


def _move(x, y):
    return QMouseEvent(QEvent.Type.MouseMove, QPointF(x, y),
                       Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier)


def _box_drag(tool, cam, p_start, p_end, w=800, h=600, mods=Qt.KeyboardModifier.NoModifier):
    sx0, sy0, _ = cam.world_to_screen(np.asarray(p_start, dtype=np.float32), w, h)
    sx1, sy1, _ = cam.world_to_screen(np.asarray(p_end, dtype=np.float32), w, h)
    tool.on_mouse_press(_press(sx0, sy0, mods), None)
    tool.on_mouse_move(_move(sx1, sy1), None)
    tool.on_mouse_release(_release(sx1, sy1, mods), None)
    return (sx0, sy0), (sx1, sy1)


def test_box_left_to_right_is_window_encloses_only(qtbot):
    from bermake.selection import Selection
    scene, fid, e_ab = _scene_with_quad()
    sel = Selection()
    tool, cam = _make_tool(scene, sel)
    # [-3,0,0] projects to upper-left of quad; [4,-2,0] to lower-right,
    # giving a left->right (window) box that fully encloses the quad.
    _box_drag(tool, cam, [-3.0, 0.0, 0.0], [4.0, -2.0, 0.0])
    assert fid in sel.faces


def test_box_right_to_left_is_crossing(qtbot):
    from bermake.selection import Selection
    scene, fid, e_ab = _scene_with_quad()
    sel = Selection()
    tool, cam = _make_tool(scene, sel)
    _box_drag(tool, cam, [0.0, 0.0, 0.0], [-3.0, 0.5, 0.0])
    assert len(sel.edges) >= 1


def test_box_overlay_sets_box_rect_during_drag(qtbot):
    from bermake.selection import Selection
    scene, fid, e_ab = _scene_with_quad()
    sel = Selection()
    tool, cam = _make_tool(scene, sel)
    sx0, sy0, _ = cam.world_to_screen(np.array([-2, -2, 0], dtype=np.float32), 800, 600)
    sx1, sy1, _ = cam.world_to_screen(np.array([2, 2, 0], dtype=np.float32), 800, 600)
    tool.on_mouse_press(_press(sx0, sy0), None)
    tool.on_mouse_move(_move(sx1, sy1), None)
    ov = tool.overlay()
    assert ov.box_rect is not None
    tool.on_mouse_release(_release(sx1, sy1), None)
    assert tool.overlay().box_rect is None


def test_shift_box_adds_to_existing_selection(qtbot):
    from bermake.selection import Selection
    scene, fid, e_ab = _scene_with_quad()
    sel = Selection()
    sel.replace(faces=[fid])  # pre-existing selection
    tool, cam = _make_tool(scene, sel)
    # Shift + a window box that encloses the quad → ADD (face stays selected).
    _box_drag(tool, cam, [-3.0, 0.0, 0.0], [4.0, -2.0, 0.0],
              mods=Qt.KeyboardModifier.ShiftModifier)
    assert fid in sel.faces


# ---------------------------------------------------------------------------
# Final review I2: View > Select Vertices must reach the click and hover picks
# ---------------------------------------------------------------------------


def _make_vertex_tool(scene, sel, w=800, h=600):
    """A SelectTool with View > Select Vertices on."""
    from bermake.tools import ToolContext
    from bermake.tools.select_tool import SelectTool

    cam = _cam(w, h)
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
    return tool, cam


def _hover(tool, cam, world, w=800, h=600):
    sx, sy, _ = cam.world_to_screen(np.asarray(world, dtype=np.float32), w, h)
    ev = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(sx, sy),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    tool.on_mouse_move(ev, None)


def test_click_on_a_corner_selects_the_vertex_when_the_mode_is_on(qtbot):
    """The mode was reachable by drag (box-select) and by double-click, but
    not by the most obvious gesture of all: a plain click still selected the
    edge because the release-path pick_selectable call omitted the flag."""
    from bermake.selection import Selection

    scene, _fid, _e_ab = _scene_with_quad()
    sel = Selection()
    tool, cam = _make_vertex_tool(scene, sel)
    _click(tool, cam, [-1.0, -1.0, 0.0])
    assert len(sel.vertices) == 1
    assert sel.edges == set()
    assert sel.faces == set()


def test_a_vertex_hit_never_lands_in_the_face_set(qtbot):
    """The latent trap behind I2: the release path read
    `elif hit[0] == "edge": ... else: replace(faces=[hit[1]])`, so a
    ("vertex", id) hit fell into the face branch. Ids are allocated per kind
    here, so that id names a real and unrelated face often enough to matter."""
    from bermake.selection import Selection

    scene, fid, _e_ab = _scene_with_quad()
    sel = Selection()
    tool, cam = _make_vertex_tool(scene, sel)
    _click(tool, cam, [-1.0, -1.0, 0.0])
    assert fid not in sel.faces
    assert sel.faces == set()


def test_click_on_a_corner_still_selects_the_edge_when_the_mode_is_off(qtbot):
    """Regression guard on the default. Vertex picking is opt-in, so an
    untouched bermake must click exactly as it did before M7.6c."""
    from bermake.selection import Selection

    scene, _fid, _e_ab = _scene_with_quad()
    sel = Selection()
    tool, cam = _make_tool(scene, sel)
    _click(tool, cam, [-1.0, -1.0, 0.0])
    assert sel.vertices == set()
    # Two edges meet at this corner; which one wins is a tie the picker
    # settles, and the point of the test is that an EDGE still wins at all.
    assert len(sel.edges) == 1
    assert sel.faces == set()


def test_shift_click_toggles_a_vertex(qtbot):
    """Shift composes for vertices the way it already does for edges and
    faces: on, then off."""
    from bermake.selection import Selection

    scene, _fid, _e_ab = _scene_with_quad()
    sel = Selection()
    tool, cam = _make_vertex_tool(scene, sel)
    _click(tool, cam, [-1.0, -1.0, 0.0], mods=Qt.KeyboardModifier.ShiftModifier)
    assert len(sel.vertices) == 1
    v_id = next(iter(sel.vertices))
    _click(tool, cam, [-1.0, -1.0, 0.0], mods=Qt.KeyboardModifier.ShiftModifier)
    assert v_id not in sel.vertices


def test_shift_click_a_vertex_keeps_an_existing_edge_selection(qtbot):
    from bermake.selection import Selection

    scene, _fid, e_ab = _scene_with_quad()
    sel = Selection()
    sel.replace(edges=[e_ab])
    tool, cam = _make_vertex_tool(scene, sel)
    _click(tool, cam, [-1.0, 1.0, 0.0], mods=Qt.KeyboardModifier.ShiftModifier)
    assert sel.edges == {e_ab}
    assert len(sel.vertices) == 1


def test_hover_reports_a_vertex_when_the_mode_is_on(qtbot):
    """Pre-highlight is the promise the click then keeps, so hover has to
    pick the same entity the click will."""
    from bermake.selection import Selection

    scene, _fid, _e_ab = _scene_with_quad()
    tool, cam = _make_vertex_tool(scene, Selection())
    _hover(tool, cam, [-1.0, -1.0, 0.0])
    assert tool._hovered is not None
    assert tool._hovered[0] == "vertex"


def test_hover_on_a_corner_still_reports_the_edge_when_the_mode_is_off(qtbot):
    from bermake.selection import Selection

    scene, _fid, _e_ab = _scene_with_quad()
    tool, cam = _make_tool(scene, Selection())
    _hover(tool, cam, [-1.0, -1.0, 0.0])
    assert tool._hovered is not None
    assert tool._hovered[0] == "edge"


def test_a_hovered_vertex_draws_no_rubber_band_and_no_face_fill(qtbot):
    """The overlay's `else` was the face branch, so a hovered vertex id would
    have been handed to `face_loop`. Ids are per kind, so on this scene the
    vertex id 0 and the face id 0 both exist and the hover highlight would
    have lit an unrelated polygon."""
    from bermake.selection import Selection

    scene, _fid, _e_ab = _scene_with_quad()
    tool, cam = _make_vertex_tool(scene, Selection())
    _hover(tool, cam, [-1.0, -1.0, 0.0])
    overlay = tool.overlay()
    assert overlay.face_fill_polygons == []
    assert overlay.rubber_band_segments.shape[0] == 0
