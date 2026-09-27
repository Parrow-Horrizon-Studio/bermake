"""Tests for SelectTool object-pick, double-click enter/exit, Esc-exit, and hover instance."""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent

from bermake.model.model import Model
from bermake.selection import Selection
from bermake.tools.select_tool import SelectTool
from bermake.tools.tool import ToolContext
from bermake.viewport.camera import Camera


def _cam(w=800, h=600):
    c = Camera()
    c.aspect = float(w) / float(h)
    return c


def _dbl_click(x, y, mods=Qt.KeyboardModifier.NoModifier):
    return QMouseEvent(
        QMouseEvent.Type.MouseButtonDblClick,
        QPointF(x, y),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        mods,
    )


def _press(x, y, mods=Qt.KeyboardModifier.NoModifier):
    return QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(x, y),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        mods,
    )


def _release(x, y, mods=Qt.KeyboardModifier.NoModifier):
    return QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        QPointF(x, y),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        mods,
    )


def _make_model_with_instance():
    """Return (model, instance, group_def) with one instance at the root."""
    m = Model()
    g = m.new_definition("G", is_group=True)
    inst = m.new_instance(g)
    m.root.children.append(inst)
    return m, inst, g


def _make_tool(model, sel, w=800, h=600):
    cam = _cam(w, h)
    tool = SelectTool()
    ctx = ToolContext(
        scene=model.active_scene,
        camera=cam,
        widget_size_provider=lambda: (w, h),
        selection=sel,
        model=model,
    )
    tool.activate(ctx)
    return tool, cam


# ---------------------------------------------------------------------------
# Double-click enters the instance
# ---------------------------------------------------------------------------

def test_double_click_enters_instance(qtbot, monkeypatch):
    m, inst, g = _make_model_with_instance()
    sel = Selection()
    # Monkeypatch pick_instance so any ray hits our instance
    monkeypatch.setattr(m, "pick_instance", lambda o, d: inst)
    tool, cam = _make_tool(m, sel)

    tool.on_mouse_double_click(_dbl_click(100, 100), None)

    assert m.active_context is g, "double-click should enter the instance's definition"


def test_double_click_prefers_entering_instance_over_geometry_under_it(qtbot, monkeypatch):
    """M7.6c: the instance branch must win even when real geometry sits at
    the same screen location as the enterable instance. Smart-select on raw
    geometry is only reachable once you are inside a group, never as a way
    to skip entering one (spec's Risks section calls this out by name).

    `sel.faces` alone cannot prove this: entering a group runs
    `_enter_or_exit_cleanup`, which unconditionally clears the selection, so
    even a wrongly-ordered smart-select-then-enter leaves the selection
    empty at the end, same as a correctly-ordered enter that never touched
    it. The spy on `_apply_smart_selection` (the interface seam this task
    and Task 7 both build on) observes whether the geometry branch ran at
    all, which the final Selection state cannot distinguish.
    """
    m, inst, g = _make_model_with_instance()
    scene = m.active_scene
    a = scene.add_vertex(np.array([-1.0, -1.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([1.0, -1.0, 0.0], dtype=np.float32))
    c = scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    d = scene.add_vertex(np.array([-1.0, 1.0, 0.0], dtype=np.float32))
    scene.add_face_from_loop((a, b, c, d))
    sel = Selection()
    monkeypatch.setattr(m, "pick_instance", lambda o, d: inst)
    tool, cam = _make_tool(m, sel)
    smart_select_calls = []
    monkeypatch.setattr(
        tool,
        "_apply_smart_selection",
        lambda event, *, edges, faces: smart_select_calls.append((edges, faces)),
    )
    sx, sy, _ = cam.world_to_screen(np.array([0.0, 0.0, 0.0], dtype=np.float32), 800, 600)

    tool.on_mouse_double_click(_dbl_click(sx, sy), None)

    assert m.active_context is g, "double-click should still enter the instance"
    assert smart_select_calls == [], "the geometry branch must not run when instance-enter fires"


def test_double_click_clears_selection(qtbot, monkeypatch):
    m, inst, g = _make_model_with_instance()
    sel = Selection()
    sel.replace(instances=[inst.id])  # pre-existing selection
    monkeypatch.setattr(m, "pick_instance", lambda o, d: inst)
    tool, sel_out = _make_tool(m, sel), sel

    tool_obj, _cam = _make_tool(m, sel)
    monkeypatch.setattr(m, "pick_instance", lambda o, d: inst)
    tool_obj.on_mouse_double_click(_dbl_click(100, 100), None)

    assert sel.is_empty(), "entering a group clears the selection"


def test_double_click_suppresses_trailing_release(qtbot, monkeypatch):
    m, inst, g = _make_model_with_instance()
    sel = Selection()
    monkeypatch.setattr(m, "pick_instance", lambda o, d: inst)
    tool, cam = _make_tool(m, sel)

    tool.on_mouse_double_click(_dbl_click(100, 100), None)
    # After double-click we entered; the trailing release should be swallowed
    tool.on_mouse_press(_press(100, 100), None)
    tool.on_mouse_release(_release(100, 100), None)
    # Still inside the group — release was suppressed, so active_context is still g
    assert m.active_context is g


# ---------------------------------------------------------------------------
# Single-click selects an instance
# ---------------------------------------------------------------------------

def test_single_click_selects_instance(qtbot, monkeypatch):
    m, inst, g = _make_model_with_instance()
    sel = Selection()
    monkeypatch.setattr(m, "pick_instance", lambda o, d: inst)
    tool, cam = _make_tool(m, sel)

    tool.on_mouse_press(_press(100, 100), None)
    tool.on_mouse_release(_release(100, 100), None)

    assert inst.id in sel.instances, "single click should select the instance"
    assert m.active_context is m.root, "single click should NOT enter the instance"


def test_shift_click_toggles_instance(qtbot, monkeypatch):
    m, inst, g = _make_model_with_instance()
    sel = Selection()
    monkeypatch.setattr(m, "pick_instance", lambda o, d: inst)
    tool, cam = _make_tool(m, sel)
    shift = Qt.KeyboardModifier.ShiftModifier

    # First shift-click adds
    tool.on_mouse_press(_press(100, 100, shift), None)
    tool.on_mouse_release(_release(100, 100, shift), None)
    assert inst.id in sel.instances

    # Second shift-click removes (toggle)
    tool.on_mouse_press(_press(100, 100, shift), None)
    tool.on_mouse_release(_release(100, 100, shift), None)
    assert inst.id not in sel.instances


# ---------------------------------------------------------------------------
# Esc exits one level inside a group; clears at root
# ---------------------------------------------------------------------------

def test_esc_inside_group_exits_one_level(qtbot):
    m, inst, g = _make_model_with_instance()
    m.enter(inst)  # manually enter
    sel = Selection()
    tool, cam = _make_tool(m, sel)
    assert m.active_context is g

    ev = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
    tool.on_key_press(ev)

    assert m.active_context is m.root, "Esc should exit one level back to root"


def test_esc_at_root_clears_selection(qtbot):
    m, inst, g = _make_model_with_instance()
    sel = Selection()
    sel.replace(instances=[inst.id])
    tool, cam = _make_tool(m, sel)
    assert not m.active_path  # at root

    ev = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
    tool.on_key_press(ev)

    assert sel.is_empty(), "Esc at root should clear selection"
    assert m.active_context is m.root, "Esc at root should not change context"


# ---------------------------------------------------------------------------
# Empty click exits one level inside a group
# ---------------------------------------------------------------------------

def test_empty_click_inside_group_exits(qtbot, monkeypatch):
    m, inst, g = _make_model_with_instance()
    m.enter(inst)  # manually enter
    sel = Selection()
    # No pick_instance hit, no entity hit either
    monkeypatch.setattr(m, "pick_instance", lambda o, d: None)
    tool, cam = _make_tool(m, sel)
    assert m.active_context is g

    # Click somewhere with no geometry (corner)
    tool.on_mouse_press(_press(3.0, 3.0), None)
    tool.on_mouse_release(_release(3.0, 3.0), None)

    assert m.active_context is m.root, "empty click inside group exits one level"


# ---------------------------------------------------------------------------
# request_context_rebuild is called on enter/exit
# ---------------------------------------------------------------------------

def test_double_click_calls_request_rebuild(qtbot, monkeypatch):
    m, inst, g = _make_model_with_instance()
    sel = Selection()
    monkeypatch.setattr(m, "pick_instance", lambda o, d: inst)

    rebuilt = []
    cam = _cam()
    tool = SelectTool()
    ctx = ToolContext(
        scene=m.active_scene,
        camera=cam,
        widget_size_provider=lambda: (800, 600),
        selection=sel,
        model=m,
        request_context_rebuild=lambda: rebuilt.append(1),
    )
    tool.activate(ctx)

    tool.on_mouse_double_click(_dbl_click(100, 100), None)

    assert rebuilt, "double-click enter should call request_context_rebuild"


# ---------------------------------------------------------------------------
# Root regression: no instances → falls through to entity pick (unchanged)
# ---------------------------------------------------------------------------

def test_root_no_instances_entity_pick_unchanged(qtbot):
    """When pick_instance returns None, behavior is identical to pre-M4e."""
    from bermake.scene import Scene

    s = Scene()
    a = s.add_vertex(np.array([-1, -1, 0], dtype=np.float32))
    b = s.add_vertex(np.array([1, -1, 0], dtype=np.float32))
    c = s.add_vertex(np.array([1, 1, 0], dtype=np.float32))
    d = s.add_vertex(np.array([-1, 1, 0], dtype=np.float32))
    fid = s.add_face_from_loop((a, b, c, d))
    e_ab = s.add_edge(a, b)

    m = Model()
    # root has no child instances, so pick_instance always returns None
    sel = Selection()
    cam = _cam()
    tool = SelectTool()
    ctx = ToolContext(
        scene=s,
        camera=cam,
        widget_size_provider=lambda: (800, 600),
        selection=sel,
        model=m,
    )
    tool.activate(ctx)

    # Click on edge
    sx, sy, _ = cam.world_to_screen(np.array([0.0, -1.0, 0.0], dtype=np.float32), 800, 600)
    tool.on_mouse_press(_press(sx, sy), None)
    tool.on_mouse_release(_release(sx, sy), None)

    assert sel.edges == {e_ab}, "entity pick falls through correctly when no instance is hit"
