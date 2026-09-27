from __future__ import annotations

import numpy as np
from bermake.commands.command_stack import CommandStack
from bermake.model.model import Model
from bermake.tools.roof_tool import RoofTool
from bermake.tools.tool import ToolContext
from bermake.viewport.snap_engine import SnapKind
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent


class _Snap:
    def __init__(self, x, y, z=0.0, kind=SnapKind.ON_FACE):
        self.kind = kind
        self.world_position = np.array([x, y, z], dtype=np.float64)


def _ctx(model, stack):
    return ToolContext(
        scene=model.active_scene, command_stack=stack, model=model,
        camera=None, widget_size_provider=lambda: (100, 100), units_provider=lambda: None,
    )


def _place(tool, ax, ay, bx, by):
    tool.on_mouse_press(None, _Snap(ax, ay))
    tool.on_mouse_move(None, _Snap(bx, by))
    tool.on_mouse_press(None, _Snap(bx, by))


def test_footprint_drag_places_one_roof():
    model = Model()
    stack = CommandStack()
    tool = RoofTool()
    tool.activate(_ctx(model, stack))
    _place(tool, 0.0, 0.0, 4.0, 6.0)
    assert len(model.active_context.children) == 1
    defn = model.active_context.children[-1].definition
    assert defn.is_group is True and defn.name == "Roof"


def test_ridge_runs_along_longer_edge_by_default():
    # footprint 4 (x) x 8 (y): ridge should run along Y (the longer edge), so the
    # roof's apex line spans ~8 in Y and the cross-section spans ~4 in X.
    model = Model()
    stack = CommandStack()
    tool = RoofTool()
    tool.kind = "gable"
    tool.activate(_ctx(model, stack))
    _place(tool, 0.0, 0.0, 4.0, 8.0)
    inst = model.active_context.children[-1]
    m = inst.transform
    verts = [(m @ np.append(np.array(v), 1.0))[:3]
             for v in _defn_local_verts(inst.definition)]
    a = np.array(verts)
    # X extent ~4 (across ridge), Y extent ~8 (along ridge)
    assert np.isclose(a[:, 0].max() - a[:, 0].min(), 4.0, atol=1e-6)
    assert np.isclose(a[:, 1].max() - a[:, 1].min(), 8.0, atol=1e-6)


def test_no_footprint_no_placement_on_degenerate():
    model = Model()
    stack = CommandStack()
    tool = RoofTool()
    tool.activate(_ctx(model, stack))
    _place(tool, 1.0, 1.0, 1.0, 1.0)   # zero-area footprint
    assert len(model.active_context.children) == 0


def _defn_local_verts(defn):
    return [v.position for v in defn.mesh.vertices_iter()]


def _apex_axis(model):
    inst = model.active_context.children[-1]
    m = inst.transform
    verts = np.array([(m @ np.append(v.position, 1.0))[:3]
                      for v in inst.definition.mesh.vertices_iter()])
    apex = verts[np.isclose(verts[:, 2], verts[:, 2].max())]
    span_x = apex[:, 0].max() - apex[:, 0].min()
    span_y = apex[:, 1].max() - apex[:, 1].min()
    return "y" if span_y > span_x else "x"


def test_up_arrow_flips_ridge_axis():
    # 4 (x) x 8 (y) gable: default ridge runs along the longer edge (Y);
    # one Up press mid-gesture rotates it 90 deg so the ridge runs along X.
    m1 = Model()
    t1 = RoofTool()
    t1.activate(_ctx(m1, CommandStack()))
    _place(t1, 0.0, 0.0, 4.0, 8.0)
    assert _apex_axis(m1) == "y"

    m2 = Model()
    t2 = RoofTool()
    t2.activate(_ctx(m2, CommandStack()))
    t2.on_mouse_press(None, _Snap(0.0, 0.0))
    t2.on_key_press(
        QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Up, Qt.KeyboardModifier.NoModifier)
    )
    t2.on_mouse_move(None, _Snap(4.0, 8.0))
    t2.on_mouse_press(None, _Snap(4.0, 8.0))
    assert _apex_axis(m2) == "x"


def test_flip_quarters_resets_after_each_completed_gesture():
    # Flip the ridge with Up during the first gesture and commit it, then run
    # a SECOND gesture on the same footprint shape with no further flip key.
    # If _flip_quarters were sticky across gestures (the M7c carry-over bug),
    # the second roof would inherit the first's flip; it must instead default
    # back to ridge-along-the-longer-edge.
    model = Model()
    stack = CommandStack()
    tool = RoofTool()
    tool.activate(_ctx(model, stack))

    tool.on_mouse_press(None, _Snap(0.0, 0.0))
    tool.on_key_press(
        QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Up, Qt.KeyboardModifier.NoModifier)
    )
    tool.on_mouse_move(None, _Snap(4.0, 8.0))
    tool.on_mouse_press(None, _Snap(4.0, 8.0))
    assert _apex_axis(model) == "x"  # flipped, as in test_up_arrow_flips_ridge_axis

    _place(tool, 0.0, 0.0, 4.0, 8.0)  # second gesture, no flip key pressed
    assert len(model.active_context.children) == 2
    assert _apex_axis(model) == "y"  # back to default (longer-edge) orientation


def test_placement_inside_entered_group_lands_at_world_footprint():
    # Placing a roof while entered into a translated+rotated group must still
    # land at the DRAWN WORLD footprint (transform_local = inv(active_world) @ m_world
    # => rendered at m_world). This guards the M7b-class overlay/placement frame bug.
    model = Model()
    grp = model.new_definition("G", is_group=True)
    ang = np.radians(35.0)
    c, s = np.cos(ang), np.sin(ang)
    tf = np.array(
        [[c, -s, 0.0, 10.0], [s, c, 0.0, 20.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    inst = model.new_instance(grp, tf)
    model.active_context.children.append(inst)
    model.active_path = [inst]
    tool = RoofTool()
    tool.activate(_ctx(model, CommandStack()))
    _place(tool, 0.0, 0.0, 4.0, 8.0)   # world footprint corners (0,0)-(4,8)
    roof = model.active_context.children[-1]
    world = model.active_world_transform @ roof.transform
    verts = np.array([(world @ np.append(v.position, 1.0))[:3]
                      for v in roof.definition.mesh.vertices_iter()])
    assert np.isclose(verts[:, 0].min(), 0.0, atol=1e-6)
    assert np.isclose(verts[:, 0].max(), 4.0, atol=1e-6)
    assert np.isclose(verts[:, 1].min(), 0.0, atol=1e-6)
    assert np.isclose(verts[:, 1].max(), 8.0, atol=1e-6)
