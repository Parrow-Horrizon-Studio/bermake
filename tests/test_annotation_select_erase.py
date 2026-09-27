"""SelectTool + annotations (M7d Task 10): click / Shift-toggle / ordering /
active-context scoping / enter-exit clearing.

Annotations are hit-tested BEFORE geometry (they draw on top of the scene), so
an annotation pass runs ahead of the existing instance/edge/face pick in
select_tool.py's on_mouse_release. Mirrors the press/release + QMouseEvent
conventions already used in test_select_tool.py and test_select_tool_objects.py
rather than the bare on_mouse_press-only calls sketched in the task brief,
because the real SelectTool only commits a selection on release.

M7d Task 11 (erase half) appends below: EraserTool.on_mouse_press is the real
click-commit path for annotations -- unlike edges (mutated eagerly on press,
pushed to the undo stack only on release), an annotation hit is deleted via a
single `command_stack.execute(DeleteAnnotationsCommand(...), model)` call
inside on_mouse_press itself, so it is both removed AND undoable the instant
press fires, with no release needed. Confirmed against the real erase_tool.py
rather than the `_ctx(model, sel)` / `_Event(...)` helpers sketched in the task
brief, which don't exist anywhere in this codebase.
"""

from __future__ import annotations

import numpy as np
from bermake.commands.command_stack import CommandStack
from bermake.model.annotation import Dimension
from bermake.model.model import Model
from bermake.selection import Selection
from bermake.tools.erase_tool import EraserTool
from bermake.tools.select_tool import SelectTool
from bermake.tools.tool import ToolContext
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent


class _FlatCamera:
    """Same fixed screen-space projection used by test_annotation_picking.py,
    so the dimension line's known projected pixels can be reused verbatim."""

    def world_to_screen(self, world_xyz, width, height):
        x, y, z = float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2])
        if z < 0.0:
            return None
        return (100.0 + x * 10.0, 200.0 - y * 10.0, 1.0 + z)

    def ray_from_screen(self, cx, cy, w, h):
        return np.array([0.0, 0.0, 50.0]), np.array([0.0, 0.0, -1.0])


def _press(x, y, mods=Qt.KeyboardModifier.NoModifier):
    return QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(x, y),
                       Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, mods)


def _release(x, y, mods=Qt.KeyboardModifier.NoModifier):
    return QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(x, y),
                       Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton, mods)


def _dbl_click(x, y, mods=Qt.KeyboardModifier.NoModifier):
    return QMouseEvent(QMouseEvent.Type.MouseButtonDblClick, QPointF(x, y),
                       Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, mods)


def _click_at(tool, x, y, mods=Qt.KeyboardModifier.NoModifier):
    tool.on_mouse_press(_press(x, y, mods), None)
    tool.on_mouse_release(_release(x, y, mods), None)


def _model_with_dimension():
    model = Model()
    model.active_context.annotations.append(
        Dimension(5, (0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, -2.0, 0.0))
    )
    return model


def _make_tool(model, sel, w=640, h=480, show_guides=True):
    cam = _FlatCamera()
    tool = SelectTool()
    ctx = ToolContext(
        scene=model.active_scene,
        camera=cam,
        widget_size_provider=lambda: (w, h),
        selection=sel,
        model=model,
        show_guides_provider=lambda: show_guides,
    )
    tool.activate(ctx)
    return tool, cam


# ---------------------------------------------------------------------------
# Click selects an annotation
# ---------------------------------------------------------------------------

def test_clicking_a_dimension_selects_it(qtbot):
    model = _model_with_dimension()
    sel = Selection()
    tool, _cam = _make_tool(model, sel)

    _click_at(tool, 120.0, 220.0)

    assert sel.annotations == {5}


def test_clicking_empty_space_clears_the_annotation_selection(qtbot):
    model = _model_with_dimension()
    sel = Selection()
    tool, _cam = _make_tool(model, sel)

    _click_at(tool, 120.0, 220.0)
    assert sel.annotations == {5}
    _click_at(tool, 600.0, 40.0)

    assert sel.annotations == set()


# ---------------------------------------------------------------------------
# Shift-click toggles the annotation without clearing edges/faces
# ---------------------------------------------------------------------------

def test_shift_click_toggles_annotation_and_preserves_edge_and_face_selection(qtbot):
    model = _model_with_dimension()
    scene = model.active_scene
    # Quad placed far from the dimension's screen footprint (~100-140, 220) so
    # its projected pixels can never collide with the annotation pick.
    a = scene.add_vertex(np.array([10.0, 10.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([11.0, 10.0, 0.0], dtype=np.float32))
    c = scene.add_vertex(np.array([11.0, 11.0, 0.0], dtype=np.float32))
    d = scene.add_vertex(np.array([10.0, 11.0, 0.0], dtype=np.float32))
    fid = scene.add_face_from_loop((a, b, c, d))
    e_ab = scene.add_edge(a, b)

    sel = Selection()
    sel.replace(edges=[e_ab], faces=[fid])
    tool, _cam = _make_tool(model, sel)
    shift = Qt.KeyboardModifier.ShiftModifier

    _click_at(tool, 120.0, 220.0, mods=shift)
    assert sel.annotations == {5}
    assert sel.edges == {e_ab}
    assert sel.faces == {fid}

    # Shift-click the same annotation again -> toggles it back off, edges/faces
    # still untouched.
    _click_at(tool, 120.0, 220.0, mods=shift)
    assert sel.annotations == set()
    assert sel.edges == {e_ab}
    assert sel.faces == {fid}


# ---------------------------------------------------------------------------
# Ordering: annotation pass beats the geometry pass
# ---------------------------------------------------------------------------

def test_annotation_pick_beats_geometry_pick_when_they_overlap(qtbot):
    model = _model_with_dimension()
    scene = model.active_scene
    # This edge's endpoints project to EXACTLY the same screen segment as the
    # dimension line ((100,220)-(140,220) under _FlatCamera) -- a genuine
    # overlap, not just a nearby coincidence.
    a = scene.add_vertex(np.array([0.0, -2.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([4.0, -2.0, 0.0], dtype=np.float32))
    e_id = scene.add_edge(a, b)

    sel = Selection()
    tool, _cam = _make_tool(model, sel)

    _click_at(tool, 120.0, 220.0)

    assert sel.annotations == {5}, "the annotation pass must win when it overlaps geometry"
    assert sel.edges == set(), "geometry pick must not also fire once the annotation hit"
    assert e_id not in sel.edges


# ---------------------------------------------------------------------------
# Active-context scoping
# ---------------------------------------------------------------------------

def test_active_context_scoping_changes_which_annotation_is_pickable(qtbot):
    model = Model()
    root_ann = Dimension(
        model.new_annotation_id(), (0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, -2.0, 0.0)
    )
    model.root.annotations.append(root_ann)

    group_def = model.new_definition("Group", is_group=True)
    nested_ann = Dimension(
        model.new_annotation_id(), (0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, -2.0, 0.0)
    )
    group_def.annotations.append(nested_ann)
    inst = model.new_instance(group_def)
    model.root.children.append(inst)

    sel = Selection()
    tool, _cam = _make_tool(model, sel)

    _click_at(tool, 120.0, 220.0)
    assert sel.annotations == {root_ann.id}

    sel.clear()
    model.enter(inst)  # active_context is a live property -- no re-activate needed
    _click_at(tool, 120.0, 220.0)
    assert sel.annotations == {nested_ann.id}
    assert nested_ann.id != root_ann.id


def test_annotation_in_non_active_context_is_not_selectable(qtbot):
    model = Model()
    group_def = model.new_definition("Group", is_group=True)
    nested_ann = Dimension(
        model.new_annotation_id(), (0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, -2.0, 0.0)
    )
    group_def.annotations.append(nested_ann)
    inst = model.new_instance(group_def)
    model.root.children.append(inst)

    sel = Selection()
    tool, _cam = _make_tool(model, sel)  # still at root; nested_ann is NOT in root.annotations

    _click_at(tool, 120.0, 220.0)

    assert sel.annotations == set()


# ---------------------------------------------------------------------------
# Entering/exiting a group clears stale annotation selection
# ---------------------------------------------------------------------------

def test_entering_group_clears_stale_annotation_selection(qtbot, monkeypatch):
    model = _model_with_dimension()
    g = model.new_definition("G", is_group=True)
    inst = model.new_instance(g)
    model.root.children.append(inst)

    sel = Selection()
    tool, _cam = _make_tool(model, sel)
    _click_at(tool, 120.0, 220.0)
    assert sel.annotations == {5}

    monkeypatch.setattr(model, "pick_instance", lambda o, d: inst)
    tool.on_mouse_double_click(_dbl_click(100.0, 100.0), None)

    assert model.active_context is g
    assert sel.annotations == set(), "entering a group should clear a stale annotation selection"


def test_exiting_group_clears_stale_annotation_selection(qtbot):
    model = _model_with_dimension()
    g = model.new_definition("G", is_group=True)
    inst = model.new_instance(g)
    model.root.children.append(inst)
    model.enter(inst)

    sel = Selection()
    sel.replace(annotations=[999])
    tool, _cam = _make_tool(model, sel)

    ev = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
    tool.on_key_press(ev)

    assert model.active_context is model.root
    assert sel.annotations == set(), "exiting a group should clear a stale annotation selection"


# ---------------------------------------------------------------------------
# Regression: existing edge/face/instance selection behaviour is unchanged
# ---------------------------------------------------------------------------

def test_geometry_click_unaffected_when_model_has_no_annotations(qtbot):
    """A Model is present (annotation pick is wired up and runs) but has zero
    annotations anywhere -- the geometry pick must fire exactly as before."""
    model = Model()
    scene = model.active_scene
    a = scene.add_vertex(np.array([-1.0, -1.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([1.0, -1.0, 0.0], dtype=np.float32))
    c = scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    d = scene.add_vertex(np.array([-1.0, 1.0, 0.0], dtype=np.float32))
    scene.add_face_from_loop((a, b, c, d))
    e_ab = scene.add_edge(a, b)

    sel = Selection()
    tool, _cam = _make_tool(model, sel)

    # world (0,-1,0) -> screen (100, 210) under _FlatCamera -- lands on edge a-b.
    _click_at(tool, 100.0, 210.0)

    assert sel.edges == {e_ab}
    assert sel.faces == set()
    assert sel.annotations == set()


# ---------------------------------------------------------------------------
# M7d Task 11: EraserTool erases a clicked annotation via DeleteAnnotationsCommand
# ---------------------------------------------------------------------------

def _make_eraser(model, sel, stack, w=640, h=480, show_guides=True):
    cam = _FlatCamera()
    tool = EraserTool()
    ctx = ToolContext(
        scene=model.active_scene,
        command_stack=stack,
        camera=cam,
        widget_size_provider=lambda: (w, h),
        selection=sel,
        model=model,
        show_guides_provider=lambda: show_guides,
    )
    tool.activate(ctx)
    return tool, cam


def test_eraser_click_removes_an_annotation(qtbot):
    model = _model_with_dimension()
    sel = Selection()
    stack = CommandStack()
    tool, _cam = _make_eraser(model, sel, stack)

    # The real click-commit path: EraserTool deletes an annotation eagerly
    # inside on_mouse_press (like its edge cascade), not on release.
    tool.on_mouse_press(_press(120.0, 220.0), None)

    assert model.active_context.annotations == []


def test_eraser_annotation_erase_is_undoable(qtbot):
    """The brief only asserts deletion -- this proves the deletion actually
    went through DeleteAnnotationsCommand on the real command stack, not a
    direct list mutation that would be unrecoverable."""
    model = _model_with_dimension()
    sel = Selection()
    stack = CommandStack()
    tool, _cam = _make_eraser(model, sel, stack)

    tool.on_mouse_press(_press(120.0, 220.0), None)
    assert model.active_context.annotations == []

    assert stack.undo()
    assert len(model.active_context.annotations) == 1


# ---------------------------------------------------------------------------
# Fix wave (Minor finding): ordering parity with SelectTool -- the annotation
# pass must beat the geometry pass for the Eraser too, mirroring
# test_annotation_pick_beats_geometry_pick_when_they_overlap above.
# ---------------------------------------------------------------------------

def test_eraser_click_on_overlap_erases_annotation_not_geometry(qtbot):
    """A click that hits BOTH an annotation and underlying geometry must erase
    the annotation and leave the geometry untouched -- the same overlap setup
    as test_annotation_pick_beats_geometry_pick_when_they_overlap, driven
    through the Eraser instead of the Select tool."""
    model = _model_with_dimension()
    scene = model.active_scene
    # This edge's endpoints project to EXACTLY the same screen segment as the
    # dimension line ((100,220)-(140,220) under _FlatCamera) -- a genuine
    # overlap, not just a nearby coincidence.
    a = scene.add_vertex(np.array([0.0, -2.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([4.0, -2.0, 0.0], dtype=np.float32))
    e_id = scene.add_edge(a, b)

    sel = Selection()
    stack = CommandStack()
    tool, _cam = _make_eraser(model, sel, stack)

    tool.on_mouse_press(_press(120.0, 220.0), None)

    assert model.active_context.annotations == [], "annotation pass must win on overlap"
    assert e_id in {e.id for e in scene.edges_iter()}, "geometry must survive the annotation hit"


def test_eraser_annotation_erase_is_active_context_scoped(qtbot):
    """Mirrors test_annotation_in_non_active_context_is_not_selectable: a
    click at an annotation's screen position must not erase it when that
    annotation lives in a context other than the active one."""
    model = Model()
    group_def = model.new_definition("Group", is_group=True)
    nested_ann = Dimension(
        model.new_annotation_id(), (0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, -2.0, 0.0)
    )
    group_def.annotations.append(nested_ann)
    inst = model.new_instance(group_def)
    model.root.children.append(inst)

    sel = Selection()
    stack = CommandStack()
    tool, _cam = _make_eraser(model, sel, stack)  # still at root

    tool.on_mouse_press(_press(120.0, 220.0), None)

    assert nested_ann in group_def.annotations, (
        "must not erase an annotation from an inactive context"
    )
    assert not stack.can_undo, "no command should have been pushed for a scoped-out miss"


# ---------------------------------------------------------------------------
# M7d Task 11: Delete key removes selected annotations via DeleteAnnotationsCommand
# ---------------------------------------------------------------------------

def test_delete_removes_selected_annotations_and_undo_restores():
    from bermake.commands.annotation_commands import DeleteAnnotationsCommand

    model = _model_with_dimension()
    ctx = model.active_context
    stack = CommandStack()
    stack.execute(DeleteAnnotationsCommand([5], ctx), model)
    assert ctx.annotations == []
    stack.undo()
    assert len(ctx.annotations) == 1


# ---------------------------------------------------------------------------
# M7.6b Task 7 fix round 1: a hidden guide (View > Guides off) must not be
# pickable either -- Select must not hover/select it, Erase must not delete
# it. `_FlatCamera.world_to_screen` projects (x, y, z) -> (100 + x*10,
# 200 - y*10, ...), so a GuidePoint at world (0, 0, 0) lands at pixel
# (100, 200), the same convention `_model_with_dimension`'s (120.0, 220.0)
# click already relies on for its own p1 at world (0, 0, 0).
# ---------------------------------------------------------------------------


def _model_with_guide_point():
    from bermake.model.annotation import GuidePoint

    model = Model()
    model.active_context.annotations.append(GuidePoint(7, (0.0, 0.0, 0.0)))
    return model


def test_select_cannot_click_a_hidden_guide_point(qtbot):
    model = _model_with_guide_point()
    sel = Selection()
    tool, _cam = _make_tool(model, sel, show_guides=False)

    _click_at(tool, 100.0, 200.0)

    assert sel.annotations == set()


def test_select_can_click_the_same_guide_point_when_visible(qtbot):
    """Control for the test above: the same click, same fixture, guides
    visible -- proves the miss above is the show_guides filter, not a
    fixture/geometry mistake."""
    model = _model_with_guide_point()
    sel = Selection()
    tool, _cam = _make_tool(model, sel, show_guides=True)

    _click_at(tool, 100.0, 200.0)

    assert sel.annotations == {7}


def test_eraser_cannot_erase_a_hidden_guide_point(qtbot):
    model = _model_with_guide_point()
    sel = Selection()
    stack = CommandStack()
    tool, _cam = _make_eraser(model, sel, stack, show_guides=False)

    tool.on_mouse_press(_press(100.0, 200.0), None)

    assert len(model.active_context.annotations) == 1
    assert not stack.can_undo


def test_eraser_can_erase_the_same_guide_point_when_visible(qtbot):
    model = _model_with_guide_point()
    sel = Selection()
    stack = CommandStack()
    tool, _cam = _make_eraser(model, sel, stack, show_guides=True)

    tool.on_mouse_press(_press(100.0, 200.0), None)

    assert model.active_context.annotations == []
