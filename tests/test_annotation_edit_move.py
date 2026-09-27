"""M7d Task 12: edit label text (double-click) + Move tool translates
annotations.

The first two tests are the brief's command-stack-only checks (they already
pass once Task 5's EditLabelTextCommand/MoveAnnotationsCommand exist -- kept
here as the documented baseline). Everything below wires the real gesture
entry points: SelectTool.on_mouse_double_click (grounded against the real
_pick_annotation / prompt_text pattern already used by TextTool) and
MoveTool's press/move/release + apply_typed_value (grounded against the real
composition already used for instance moves via CompositeCommand).
"""

from __future__ import annotations

import types

import numpy as np
import pytest
from bermake.commands.annotation_commands import EditLabelTextCommand, MoveAnnotationsCommand
from bermake.commands.command_stack import CommandStack
from bermake.model.annotation import Dimension, Label
from bermake.model.model import Model
from bermake.selection import Selection
from bermake.tools.move_tool import MoveTool
from bermake.tools.select_tool import SelectTool
from bermake.tools.tool import ToolContext
from bermake.viewport.snap_engine import SnapKind
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent


def test_edit_label_text_through_the_command_stack():
    model = Model()
    ctx = model.active_context
    ctx.annotations.append(Label(1, (0, 0, 0), (1, 1, 0), "before"))
    stack = CommandStack()
    stack.execute(EditLabelTextCommand(1, "after", ctx), model)
    assert ctx.annotations[0].text == "after"
    stack.undo()
    assert ctx.annotations[0].text == "before"


def test_move_selected_annotations_through_the_command_stack():
    model = Model()
    ctx = model.active_context
    ctx.annotations.append(Dimension(1, (0, 0, 0), (4, 0, 0), (0, -1, 0)))
    stack = CommandStack()
    stack.execute(MoveAnnotationsCommand([1], (0.0, -0.5, 0.0), ctx), model)
    assert ctx.annotations[0].offset == (0.0, -1.5, 0.0)
    stack.undo()
    assert ctx.annotations[0].offset == (0.0, -1.0, 0.0)


# ---------------------------------------------------------------------------
# SelectTool: double-click a label to edit it (Step 3.1)
# ---------------------------------------------------------------------------

class _FlatCamera:
    """Same fixed screen-space projection used by test_annotation_picking.py
    and test_annotation_select_erase.py, so known-good hit pixels carry over."""

    def world_to_screen(self, world_xyz, width, height):
        x, y, z = float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2])
        if z < 0.0:
            return None
        return (100.0 + x * 10.0, 200.0 - y * 10.0, 1.0 + z)

    def ray_from_screen(self, cx, cy, w, h):
        return np.array([0.0, 0.0, 50.0]), np.array([0.0, 0.0, -1.0])


def _dbl_click(x, y):
    return QMouseEvent(QMouseEvent.Type.MouseButtonDblClick, QPointF(x, y),
                       Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier)


def _select_tool(model, sel, stack):
    tool = SelectTool()
    ctx = ToolContext(
        scene=model.active_scene,
        command_stack=stack,
        camera=_FlatCamera(),
        widget_size_provider=lambda: (640, 480),
        selection=sel,
        model=model,
    )
    tool.activate(ctx)
    return tool


def test_double_click_label_prompts_prefilled_and_commits_edit_label_text_command(qtbot):
    model = Model()
    model.active_context.annotations.append(
        Label(9, (0.0, 0.0, 0.0), (5.0, 3.0, 0.0), "note")
    )
    stack = CommandStack()
    tool = _select_tool(model, Selection(), stack)
    seen_defaults = []

    def _stub(default=""):
        seen_defaults.append(default)
        return "renamed"

    tool.prompt_text = _stub
    # text sits at the projected text_pos -> (152, 168) hits label 9
    # (same pixel proven in test_annotation_picking.py).
    tool.on_mouse_double_click(_dbl_click(152.0, 168.0), None)

    assert seen_defaults == ["note"], "prompt must be pre-filled with the current text"
    assert model.active_context.annotations[0].text == "renamed"
    assert stack.can_undo
    stack.undo()
    assert model.active_context.annotations[0].text == "note"


def test_double_click_label_cancelled_prompt_makes_no_change(qtbot):
    model = Model()
    model.active_context.annotations.append(
        Label(9, (0.0, 0.0, 0.0), (5.0, 3.0, 0.0), "note")
    )
    stack = CommandStack()
    tool = _select_tool(model, Selection(), stack)
    tool.prompt_text = lambda default="": None

    tool.on_mouse_double_click(_dbl_click(152.0, 168.0), None)

    assert model.active_context.annotations[0].text == "note"
    assert not stack.can_undo


def test_double_click_label_blank_prompt_makes_no_change(qtbot):
    model = Model()
    model.active_context.annotations.append(
        Label(9, (0.0, 0.0, 0.0), (5.0, 3.0, 0.0), "note")
    )
    stack = CommandStack()
    tool = _select_tool(model, Selection(), stack)
    tool.prompt_text = lambda default="": "   "

    tool.on_mouse_double_click(_dbl_click(152.0, 168.0), None)

    assert model.active_context.annotations[0].text == "note"
    assert not stack.can_undo


def test_double_click_dimension_does_nothing(qtbot):
    model = Model()
    model.active_context.annotations.append(
        Dimension(5, (0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, -2.0, 0.0))
    )
    stack = CommandStack()
    tool = _select_tool(model, Selection(), stack)
    calls = []
    tool.prompt_text = lambda default="": (calls.append(default), "should-not-be-used")[1]

    # dimension line hits at (120, 220) -- same pixel as test_annotation_picking.py.
    tool.on_mouse_double_click(_dbl_click(120.0, 220.0), None)

    assert calls == [], "a dimension has no stored text -- prompt_text must never be called"
    assert not stack.can_undo
    assert model.active_context.annotations[0].kind == "dimension"


def test_double_click_label_in_non_active_context_is_not_editable(qtbot):
    model = Model()
    group_def = model.new_definition("Group", is_group=True)
    nested_label = Label(
        model.new_annotation_id(), (0.0, 0.0, 0.0), (5.0, 3.0, 0.0), "nested"
    )
    group_def.annotations.append(nested_label)
    inst = model.new_instance(group_def)
    model.root.children.append(inst)

    stack = CommandStack()
    tool = _select_tool(model, Selection(), stack)  # still at root
    tool.prompt_text = lambda default="": "hijacked"

    tool.on_mouse_double_click(_dbl_click(152.0, 168.0), None)

    assert nested_label.text == "nested"
    assert not stack.can_undo


# ---------------------------------------------------------------------------
# MoveTool: translate selected annotations, composed into ONE undo entry
# (Step 3.2) -- the previous task's Critical was a mixed-selection delete
# that pushed two undo entries so a single Ctrl+Z restored only half; these
# prove a Move gesture never repeats that.
# ---------------------------------------------------------------------------

def _press(x=0.0, y=0.0):
    return QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(x, y),
                       Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier)


def _release(x=0.0, y=0.0):
    return QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(x, y),
                       Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
                       Qt.KeyboardModifier.NoModifier)


def _snap(pos, kind=SnapKind.ENDPOINT):
    return types.SimpleNamespace(
        kind=kind, world_position=np.asarray(pos, np.float32),
        axis=None, vertex_id=None, edge_id=None, edge_t=None,
    )


def _square(scene):
    a = scene.add_vertex(np.array([0, 0, 0], np.float32))
    b = scene.add_vertex(np.array([2, 0, 0], np.float32))
    c = scene.add_vertex(np.array([2, 2, 0], np.float32))
    d = scene.add_vertex(np.array([0, 2, 0], np.float32))
    f = scene.add_face_from_loop([a, b, c, d])
    return a, b, c, d, f


def _move_ctx(model, stack, sel):
    return ToolContext(scene=model.active_scene, command_stack=stack, camera=None,
                       widget_size_provider=lambda: (800, 600), selection=sel, model=model)


def test_move_with_dimension_annotation_selected_shifts_its_offset(qtbot):
    model = Model()
    dim = Dimension(model.new_annotation_id(), (0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, -2.0, 0.0))
    model.active_context.annotations.append(dim)

    sel = Selection()
    sel.replace(annotations=[dim.id])
    stack = CommandStack()
    tool = MoveTool()
    tool.activate(_move_ctx(model, stack, sel))

    tool.on_mouse_press(_press(), _snap([0, 0, 0]))
    tool.on_mouse_move(_press(), _snap([0, -1, 0]))
    tool.on_mouse_release(_release(), _snap([0, -1, 0]))

    assert dim.offset == pytest.approx((0.0, -3.0, 0.0))
    assert stack.can_undo
    stack.undo()
    assert dim.offset == pytest.approx((0.0, -2.0, 0.0))


def test_move_with_geometry_and_annotation_selected_is_one_undo_entry_and_restores_both(qtbot):
    model = Model()
    scene = model.active_scene
    a, b, c, d, f = _square(scene)
    ann = Label(model.new_annotation_id(), (0.0, 0.0, 0.0), (1.0, 1.0, 0.0), "note")
    model.active_context.annotations.append(ann)

    sel = Selection()
    sel.replace(faces=[f], annotations=[ann.id])
    stack = CommandStack()
    tool = MoveTool()
    tool.activate(_move_ctx(model, stack, sel))

    tool.on_mouse_press(_press(), _snap([0, 0, 0]))
    tool.on_mouse_move(_press(), _snap([0, 0, 3]))
    tool.on_mouse_release(_release(), _snap([0, 0, 3]))

    for vid, base in ((a, [0, 0, 0]), (b, [2, 0, 0]), (c, [2, 2, 0]), (d, [0, 2, 0])):
        assert np.allclose(scene.vertex(vid).position, np.array(base) + np.array([0, 0, 3]))
    assert ann.text_pos == pytest.approx((1.0, 1.0, 3.0))
    assert ann.anchor == pytest.approx((0.0, 0.0, 0.0)), "anchor stays put; only text_pos shifts"

    assert stack.can_undo
    stack.undo()
    assert not stack.can_undo, (
        "a single undo must consume the ENTIRE mixed-selection Move gesture -- "
        "two separate undo entries would restore only half"
    )
    for vid, base in ((a, [0, 0, 0]), (b, [2, 0, 0]), (c, [2, 2, 0]), (d, [0, 2, 0])):
        assert np.allclose(scene.vertex(vid).position, base)
    assert ann.text_pos == pytest.approx((1.0, 1.0, 0.0))


def test_move_with_instance_and_annotation_selected_is_one_undo_entry_and_restores_both(qtbot):
    model = Model()
    child_def = model.new_definition("Child", is_group=True)
    inst = model.new_instance(child_def, np.eye(4))
    model.root.children.append(inst)
    ann = Label(model.new_annotation_id(), (0.0, 0.0, 0.0), (1.0, 1.0, 0.0), "note")
    model.active_context.annotations.append(ann)

    sel = Selection()
    sel.replace(instances=[inst.id], annotations=[ann.id])
    stack = CommandStack()
    tool = MoveTool()
    tool.activate(_move_ctx(model, stack, sel))

    tool.on_mouse_press(_press(), _snap([0, 0, 0]))
    tool.on_mouse_move(_press(), _snap([0, 0, 3]))
    tool.on_mouse_release(_release(), _snap([0, 0, 3]))

    assert np.allclose(inst.transform[:3, 3], [0, 0, 3])
    assert ann.text_pos == pytest.approx((1.0, 1.0, 3.0))

    assert stack.can_undo
    stack.undo()
    assert not stack.can_undo, (
        "a single undo must consume the ENTIRE mixed-selection Move gesture -- "
        "two separate undo entries would restore only half"
    )
    assert np.allclose(inst.transform[:3, 3], [0, 0, 0])
    assert ann.text_pos == pytest.approx((1.0, 1.0, 0.0))


def _entered_rotated_scaled_group(model):
    """Enter a group rotated 90deg CCW about Z, non-uniformly scaled
    (sx=2, sy=0.5, sz=1), translated (10,20,0) -- identical construction to
    test_text_tool.py's _entered_rotated_scaled_group, so the hand-verified
    WORLD-delta (0,4,0) -> LOCAL-delta (2,0,0) derivation carries over: a
    world-vs-local mixup in the Move tool's annotation path would fail this."""
    grp = model.new_definition("G", is_group=True)
    ang = np.pi / 2.0
    c, s = np.cos(ang), np.sin(ang)
    rot = np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])
    scale = np.diag([2.0, 0.5, 1.0])
    tf = np.eye(4)
    tf[:3, :3] = rot @ scale
    tf[:3, 3] = [10.0, 20.0, 0.0]
    inst = model.new_instance(grp, tf)
    model.root.children.append(inst)
    model.enter(inst)
    return inst


def test_move_annotation_only_inside_entered_transformed_group_uses_local_delta(qtbot):
    model = Model()
    _entered_rotated_scaled_group(model)
    assert not np.allclose(model.active_world_transform, np.eye(4))

    ann = Label(model.new_annotation_id(), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), "note")
    model.active_context.annotations.append(ann)

    sel = Selection()
    sel.replace(annotations=[ann.id])
    stack = CommandStack()
    tool = MoveTool()
    tool.activate(_move_ctx(model, stack, sel))

    # WORLD delta (0,4,0) -> LOCAL delta (2,0,0) under this group's transform
    # (hand-verified: see _entered_rotated_scaled_group docstring).
    tool.on_mouse_press(_press(), _snap([10.0, 20.0, 0.0]))
    tool.on_mouse_move(_press(), _snap([10.0, 24.0, 0.0]))
    tool.on_mouse_release(_release(), _snap([10.0, 24.0, 0.0]))

    assert stack.can_undo
    assert ann.text_pos == pytest.approx((3.0, 0.0, 0.0), abs=1e-5)
    assert ann.anchor == pytest.approx((0.0, 0.0, 0.0), abs=1e-5)

    stack.undo()
    assert not stack.can_undo
    assert ann.text_pos == pytest.approx((1.0, 0.0, 0.0), abs=1e-5)


def test_move_annotation_only_typed_value_applies_local_delta(qtbot):
    """apply_typed_value (VCB path) is the OTHER point the tool commits a
    translation delta -- must compose the annotation move the same way."""
    model = Model()
    from bermake.units import Units
    dim = Dimension(model.new_annotation_id(), (0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, -2.0, 0.0))
    model.active_context.annotations.append(dim)

    sel = Selection()
    sel.replace(annotations=[dim.id])
    stack = CommandStack()
    tool = MoveTool()
    tool.activate(_move_ctx(model, stack, sel))

    tool.on_mouse_press(_press(), _snap([0, 0, 0]))
    tool.on_mouse_move(_press(), _snap([0, -1, 0]))  # direction -Y

    assert tool.apply_typed_value("3", Units()) is True
    assert dim.offset == pytest.approx((0.0, -5.0, 0.0))
    assert stack.can_undo
    stack.undo()
    assert dim.offset == pytest.approx((0.0, -2.0, 0.0))


def test_move_noop_on_purely_empty_selection_still_commits_nothing(qtbot):
    """Regression guard for the on_mouse_press relaxation: an annotation-only
    selection must now start a gesture, but a totally empty one still must
    not (unchanged from the pre-Task-12 behaviour)."""
    model = Model()
    sel = Selection()
    stack = CommandStack()
    tool = MoveTool()
    tool.activate(_move_ctx(model, stack, sel))

    tool.on_mouse_press(_press(), _snap([0, 0, 0]))
    tool.on_mouse_release(_release(), _snap([0, 0, 3]))

    assert not stack.can_undo
