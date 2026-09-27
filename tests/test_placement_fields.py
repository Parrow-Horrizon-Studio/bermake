"""M7.5b Task 10: editing a face's texture placement numerically."""

from __future__ import annotations

import math

import numpy as np
import pytest
from bermake.scene.scene import Side, TexturePlacement


def _square(scene, x0: float = 0.0):
    v = [
        scene.add_vertex(np.array([x0 + 0.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([x0 + 1.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([x0 + 1.0, 1.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([x0 + 0.0, 1.0, 0.0], dtype=np.float32)),
    ]
    for a, b in zip(v, v[1:] + v[:1], strict=True):
        scene.add_edge(a, b)
    return scene.add_face_from_loop(v)


def test_editing_a_field_goes_through_the_command_stack(main_window):
    win = main_window
    scene = win._model.active_context.mesh
    f = _square(scene)
    panel = win._properties_dock
    panel.set_placement_target(f, Side.FRONT)
    depth = len(win._command_stack._undo)

    panel._apply_placement(TexturePlacement(offset_u=0.5, scale=2.0))

    assert len(win._command_stack._undo) == depth + 1
    assert scene.face_placement(f).offset_u == 0.5
    assert scene.face_placement(f).scale == 2.0


def test_undo_restores_the_previous_placement(main_window):
    win = main_window
    scene = win._model.active_context.mesh
    f = _square(scene)
    scene.set_face_placement(f, TexturePlacement(offset_u=1.0))
    panel = win._properties_dock
    panel.set_placement_target(f, Side.FRONT)

    panel._apply_placement(TexturePlacement(offset_u=9.0))
    win._command_stack.undo()
    assert scene.face_placement(f).offset_u == 1.0


def test_the_panel_edits_the_side_it_was_given(main_window):
    win = main_window
    scene = win._model.active_context.mesh
    f = _square(scene)
    panel = win._properties_dock
    panel.set_placement_target(f, Side.BACK)

    panel._apply_placement(TexturePlacement(scale=3.0))
    assert scene.face_placement(f, Side.BACK).scale == 3.0
    assert scene.face_placement(f, Side.FRONT).scale == 1.0


def test_rotation_is_shown_in_degrees_and_stored_in_radians(main_window):
    win = main_window
    scene = win._model.active_context.mesh
    f = _square(scene)
    panel = win._properties_dock
    panel.set_placement_target(f, Side.FRONT)

    panel._apply_placement_from_widgets(offset_u=0.0, offset_v=0.0, scale=1.0, rotation_deg=90.0)
    assert scene.face_placement(f).rotation == __import__("pytest").approx(math.pi / 2.0)

    panel.set_placement_target(f, Side.FRONT)
    assert panel._rotation_widget_value() == __import__("pytest").approx(90.0)


def test_setting_every_field_back_to_the_identity_clears_the_entry(main_window):
    win = main_window
    scene = win._model.active_context.mesh
    f = _square(scene)
    panel = win._properties_dock
    panel.set_placement_target(f, Side.FRONT)

    panel._apply_placement(TexturePlacement(scale=2.0))
    panel._apply_placement(TexturePlacement())
    assert scene.faces_with_placement() == []


# --- selection wiring (M7.5b Task 10 fix round 1) --------------------------
#
# _refresh_entity_info() is the existing hook (~15 call sites, plus
# _on_after_undo_redo) that MainWindow already uses to keep Entity Info in
# sync with the selection; the placement group now rides along on it rather
# than getting a second, parallel refresh path. These drive the real
# MainWindow selection + refresh machinery (main_window._selection.replace +
# main_window._refresh_selection_status(), the same idiom
# test_entity_info_page.py already uses) instead of calling
# set_placement_target directly, so a broken _refresh_entity_info wiring
# fails here even though the direct-call tests above would not notice it.


def test_selecting_one_face_enables_the_group_and_shows_its_placement(main_window):
    # Catches: set_selected_face never called from _refresh_entity_info (the
    # group would stay disabled and show the identity forever), and catches
    # a version that enables the group but doesn't actually read this face's
    # stored placement (e.g. always shows the identity).
    win = main_window
    scene = win._model.active_context.mesh
    f = _square(scene)
    scene.set_face_placement(f, TexturePlacement(offset_u=0.3))
    panel = win._properties_dock

    win._selection.replace(faces=[f])
    win._refresh_selection_status()

    assert panel._placement_group.isEnabled()
    assert panel._offset_u_spin.value() == pytest.approx(0.3)


def test_selecting_a_second_face_shows_that_faces_placement(main_window):
    # Catches: a wiring that latches onto the first selected face and never
    # updates (stale caching), or one that always shows whichever face has
    # the lowest/highest id rather than the one actually selected.
    win = main_window
    scene = win._model.active_context.mesh
    f1 = _square(scene, x0=0.0)
    f2 = _square(scene, x0=2.0)
    scene.set_face_placement(f1, TexturePlacement(offset_u=0.3))
    scene.set_face_placement(f2, TexturePlacement(offset_u=0.7))
    panel = win._properties_dock

    win._selection.replace(faces=[f1])
    win._refresh_selection_status()
    assert panel._offset_u_spin.value() == pytest.approx(0.3)

    win._selection.replace(faces=[f2])
    win._refresh_selection_status()
    assert panel._offset_u_spin.value() == pytest.approx(0.7)


def test_the_group_disables_for_anything_but_exactly_one_face(main_window):
    # Catches: a version that enables the group whenever selection.faces is
    # non-empty (`bool(sel.faces)` instead of `len(sel.faces) == 1`), which
    # would wrongly enable it for the two-face and mixed cases below, and a
    # version that ignores a mixed edge+face selection entirely.
    win = main_window
    scene = win._model.active_context.mesh
    f1 = _square(scene, x0=0.0)
    f2 = _square(scene, x0=2.0)
    edge_id = next(iter(scene.edges_iter())).id
    panel = win._properties_dock

    win._selection.replace(faces=[f1])
    win._refresh_selection_status()
    assert panel._placement_group.isEnabled()

    win._selection.clear()
    win._refresh_selection_status()
    assert not panel._placement_group.isEnabled()

    win._selection.replace(faces=[f1, f2])
    win._refresh_selection_status()
    assert not panel._placement_group.isEnabled()

    win._selection.replace(faces=[f1], edges=[edge_id])
    win._refresh_selection_status()
    assert not panel._placement_group.isEnabled()


def test_flipping_the_side_control_reads_and_writes_the_other_side(main_window):
    # Catches: a side control that updates the displayed fields but a
    # _apply_placement path that still hardcodes Side.FRONT underneath (the
    # front assertion below would move too), and a control that overwrites
    # the side it did NOT switch to.
    win = main_window
    scene = win._model.active_context.mesh
    f = _square(scene)
    scene.set_face_placement(f, TexturePlacement(offset_u=0.2), Side.FRONT)
    scene.set_face_placement(f, TexturePlacement(offset_u=0.9), Side.BACK)
    panel = win._properties_dock

    win._selection.replace(faces=[f])
    win._refresh_selection_status()
    assert panel._offset_u_spin.value() == pytest.approx(0.2)

    panel._side_combo.setCurrentIndex(1)  # Back
    assert panel._offset_u_spin.value() == pytest.approx(0.9)

    panel._apply_placement(TexturePlacement(offset_u=0.5))
    assert scene.face_placement(f, Side.BACK).offset_u == 0.5
    assert scene.face_placement(f, Side.FRONT).offset_u == 0.2


def test_the_group_updates_after_an_undo_that_changes_a_placement(main_window):
    # Catches: a wiring that only resyncs from inside _apply_placement's own
    # follow-up call (which never runs on an EXTERNAL undo/redo) instead of
    # genuinely riding on _refresh_entity_info / _on_after_undo_redo -- such
    # a version would leave the field stuck at 0.4 after undo.
    win = main_window
    scene = win._model.active_context.mesh
    f = _square(scene)
    panel = win._properties_dock
    win._selection.replace(faces=[f])
    win._refresh_selection_status()

    panel._apply_placement(TexturePlacement(offset_u=0.4))
    assert panel._offset_u_spin.value() == pytest.approx(0.4)

    win._command_stack.undo()

    assert scene.face_placement(f).offset_u == 0.0
    assert panel._offset_u_spin.value() == pytest.approx(0.0)


def test_a_placement_drag_refreshes_the_selected_faces_panel(main_window):
    # M7.5b Task 12 fix round 1: CommandStack._fire_change() (fired by
    # end_placement_drag's push_executed) only reaches _rebuild_outliner and
    # _on_document_changed -- neither touches _refresh_entity_info, the sole
    # place that repopulates these fields -- so without PaintTool calling the
    # ToolContext's on_placement_committed hook, the panel would keep
    # showing the pre-drag value (0.0) forever after a Shift-drag on the
    # very face Properties has selected.
    win = main_window
    scene = win._model.active_context.mesh
    f = _square(scene)
    panel = win._properties_dock
    win._selection.replace(faces=[f])
    win._refresh_selection_status()
    assert panel._offset_u_spin.value() == pytest.approx(0.0)

    tool = win._paint_tool
    tool.begin_placement_drag(f, Side.FRONT)
    tool.update_placement_drag(du=0.6, dv=0.0)
    tool.end_placement_drag()

    assert scene.face_placement(f).offset_u == pytest.approx(0.6)
    assert panel._offset_u_spin.value() == pytest.approx(0.6)


def test_a_placement_drag_that_moves_nothing_does_not_refresh(main_window):
    # A no-op drag pushes no command (tests/test_placement_drag.py), so it
    # must not fire on_placement_committed either -- confirms the hook is
    # gated on end_placement_drag's own push, not called unconditionally.
    win = main_window
    scene = win._model.active_context.mesh
    f = _square(scene)
    panel = win._properties_dock
    win._selection.replace(faces=[f])
    win._refresh_selection_status()

    calls = []
    win._paint_tool._on_placement_committed = lambda: calls.append(True)
    tool = win._paint_tool
    tool.begin_placement_drag(f, Side.FRONT)
    tool.end_placement_drag()

    assert calls == []


def test_setting_a_spin_box_value_directly_drives_the_command(main_window):
    # Drives the real valueChanged path (setValue on a live widget) instead
    # of calling _apply_placement/_apply_placement_from_widgets directly, so
    # a copy-paste bug in _on_placement_field_changed that reads the wrong
    # widget for a field (e.g. mapping the scale spin's value onto rotation)
    # is caught: only `scale` may differ from the identity afterward.
    win = main_window
    scene = win._model.active_context.mesh
    f = _square(scene)
    panel = win._properties_dock
    panel.set_placement_target(f, Side.FRONT)

    panel._scale_spin.setValue(4.0)

    placement = scene.face_placement(f)
    assert placement.scale == pytest.approx(4.0)
    assert placement.offset_u == 0.0
    assert placement.offset_v == 0.0
    assert placement.rotation == 0.0


# --- the viewport repaint (M7.5b final review, item 1) ----------------------
#
# These assert on ViewportWidget.update being CALLED, not on the scene having
# changed. A test of the second kind passes against the broken code these
# exist for: SetFacePlacementCommand always updated scene.face_placement
# correctly, and the only thing missing was anything asking for a repaint --
# so the spec's "primary mechanism" (design line 172) moved the texture and
# the screen went on showing the old one until an unrelated event repainted.
# MainWindow routes the repaint through a slot that looks self._viewport up at
# call time, which is what lets these patch it.


def _count_repaints(win, monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(win._viewport, "update", lambda: calls.append(1))
    return calls


def _selected_square(win):
    scene = win._model.active_context.mesh
    f = _square(scene)
    win._selection.replace(faces=[f])
    win._refresh_selection_status()
    return scene, f


def test_nudging_a_placement_spin_box_repaints_the_viewport(main_window, monkeypatch):
    win = main_window
    scene, f = _selected_square(win)
    calls = _count_repaints(win, monkeypatch)

    win._properties_dock._offset_u_spin.setValue(0.25)

    assert scene.face_placement(f).offset_u == pytest.approx(0.25)
    assert calls, "the numeric placement edit never asked the viewport to repaint"


def test_every_placement_field_repaints_the_viewport(main_window, monkeypatch):
    # One assertion per field, so a wiring attached to a single spin box's
    # signal instead of to the shared commit path is caught.
    win = main_window
    _, _ = _selected_square(win)
    panel = win._properties_dock
    calls = _count_repaints(win, monkeypatch)

    for spin, value in (
        (panel._offset_u_spin, 0.25),
        (panel._offset_v_spin, 0.5),
        (panel._scale_spin, 2.0),
        (panel._rotation_spin, 30.0),
    ):
        before = len(calls)
        spin.setValue(value)
        assert len(calls) > before, f"{spin} committed without a repaint"


def test_flipping_the_side_control_does_not_repaint(main_window, monkeypatch):
    # The discriminating half: a fix that repainted from set_placement_target
    # (or emitted unconditionally) would repaint on every selection change and
    # on a side flip, neither of which changes a single pixel. Both would pass
    # the two tests above.
    win = main_window
    _, _ = _selected_square(win)
    calls = _count_repaints(win, monkeypatch)

    win._properties_dock._side_combo.setCurrentIndex(1)  # Back

    assert calls == [], "flipping Front/Back issues no command and must not repaint"


def test_an_untargeted_placement_edit_does_not_repaint(main_window, monkeypatch):
    # _apply_placement returns early with no face targeted, pushing no command;
    # emitting there would repaint for a scene change that never happened.
    win = main_window
    win._properties_dock.set_placement_target(None)
    calls = _count_repaints(win, monkeypatch)

    win._properties_dock._apply_placement(TexturePlacement(offset_u=0.5))

    assert calls == []


def test_a_committed_placement_drag_repaints_the_viewport(main_window, monkeypatch):
    # Task 12's drag repaints incidentally, because a mouse release repaints
    # anyway. Driven here without any mouse event, so the repaint has to come
    # from the commit path itself -- the asymmetry that made the fallback look
    # healthy while the primary mechanism looked broken.
    win = main_window
    scene, f = _selected_square(win)
    calls = _count_repaints(win, monkeypatch)

    tool = win._paint_tool
    tool.begin_placement_drag(f, Side.FRONT)
    tool.update_placement_drag(du=0.6, dv=0.0)
    tool.end_placement_drag()

    assert scene.face_placement(f).offset_u == pytest.approx(0.6)
    assert calls, "the committed drag never asked the viewport to repaint"


def test_a_drag_that_moves_nothing_does_not_repaint(main_window, monkeypatch):
    win = main_window
    _, f = _selected_square(win)
    calls = _count_repaints(win, monkeypatch)

    tool = win._paint_tool
    tool.begin_placement_drag(f, Side.FRONT)
    tool.end_placement_drag()

    assert calls == []


# --- reset to projection (M7.5c Task 8 fix round 1) -------------------------
#
# ResetFaceUvsCommand's do/undo/redo are covered command-level in
# tests/test_reset_uvs_command.py. What was missing is coverage of the
# MainWindow wiring around it: the button's enabled state (the sole
# stored-UV indicator, spec D12), which side it targets, the no/multi
# selection guard, and -- the M7.5b case again -- whether the click actually
# asks the viewport to repaint. Reuses this file's fixture idioms
# (_selected_square, _count_repaints, win._selection.replace +
# _refresh_selection_status) rather than introducing a second style.

_QUAD_UVS = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def test_clicking_reset_uvs_repaints_the_viewport(main_window, monkeypatch):
    win = main_window
    scene, f = _selected_square(win)
    scene.set_face_uvs(f, _QUAD_UVS, Side.FRONT)
    win._refresh_selection_status()
    calls = _count_repaints(win, monkeypatch)

    win._properties_dock._reset_uvs_btn.click()

    assert scene.face_uvs(f, Side.FRONT) is None
    assert calls, "reset to projection never asked the viewport to repaint"


def test_reset_uvs_enabled_state_tracks_the_target(main_window):
    # The last assertion is the discriminating one: it fails against a
    # version that sets the button's enabled state once (e.g. only inside
    # set_placement_target's initial call) instead of recomputing it fresh
    # on every retarget, including the one _on_after_undo_redo drives.
    win = main_window
    scene, f = _selected_square(win)
    panel = win._properties_dock

    assert not panel._reset_uvs_btn.isEnabled(), "no stored UVs yet"

    scene.set_face_uvs(f, _QUAD_UVS, Side.FRONT)
    win._refresh_selection_status()
    assert panel._reset_uvs_btn.isEnabled(), "stored UVs exist but the button stayed disabled"

    win._on_reset_uvs_requested()
    assert not panel._reset_uvs_btn.isEnabled(), "reset ran but the button stayed enabled"

    win._command_stack.undo()
    assert panel._reset_uvs_btn.isEnabled(), "undo restored the UVs but the button stayed disabled"


def test_reset_uvs_respects_the_side(main_window):
    win = main_window
    scene, f = _selected_square(win)
    scene.set_face_uvs(f, _QUAD_UVS, Side.FRONT)
    win._refresh_selection_status()
    panel = win._properties_dock

    assert panel._reset_uvs_btn.isEnabled(), "Front has stored UVs"

    panel._side_combo.setCurrentIndex(1)  # Back
    assert not panel._reset_uvs_btn.isEnabled(), "Back has no stored UVs of its own"

    scene.set_face_uvs(f, [(0.9, 0.9)] * 4, Side.BACK)
    panel._side_combo.setCurrentIndex(0)  # Front
    panel._side_combo.setCurrentIndex(1)  # Back again, now with stored UVs
    assert panel._reset_uvs_btn.isEnabled()

    win._on_reset_uvs_requested()
    assert scene.face_uvs(f, Side.BACK) is None
    assert scene.face_uvs(f, Side.FRONT) is not None, "resetting Back must not touch Front"


def test_reset_uvs_guard_holds_with_no_or_multiple_selection(main_window):
    win = main_window
    scene = win._model.active_context.mesh
    f1 = _square(scene, x0=0.0)
    f2 = _square(scene, x0=2.0)
    scene.set_face_uvs(f1, _QUAD_UVS, Side.FRONT)
    panel = win._properties_dock

    win._selection.clear()
    win._refresh_selection_status()
    assert not panel._reset_uvs_btn.isEnabled()
    win._on_reset_uvs_requested()  # must be a clean no-op, not an exception
    assert scene.face_uvs(f1, Side.FRONT) is not None

    win._selection.replace(faces=[f1, f2])
    win._refresh_selection_status()
    assert not panel._reset_uvs_btn.isEnabled()
    win._on_reset_uvs_requested()  # must be a clean no-op, not an exception
    assert scene.face_uvs(f1, Side.FRONT) is not None
