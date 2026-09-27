"""The five set operations and the vertex-mode toggle, through MainWindow."""

from __future__ import annotations

import numpy as np


def _window_with_quad_pair(qtbot):
    from bermake.ui.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    scene = w._model.active_scene
    a = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    c = scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    d = scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    e = scene.add_vertex(np.array([2.0, 0.0, 0.0], dtype=np.float32))
    f = scene.add_vertex(np.array([2.0, 1.0, 0.0], dtype=np.float32))
    left = scene.add_face_from_loop((a, b, c, d))
    right = scene.add_face_from_loop((b, e, f, c))
    return w, {"left": left, "right": right, "a": a, "b": b, "c": c}


def test_invert_selects_the_other_face(qtbot):
    w, ids = _window_with_quad_pair(qtbot)
    w._selection.replace(faces={ids["left"]})
    w._on_invert_selection()
    assert ids["right"] in w._selection.faces
    assert ids["left"] not in w._selection.faces


def test_grow_adds_the_neighbouring_face(qtbot):
    w, ids = _window_with_quad_pair(qtbot)
    w._selection.replace(faces={ids["left"]})
    w._on_grow_selection()
    assert w._selection.faces == {ids["left"], ids["right"]}


def test_shrink_removes_a_boundary_face(qtbot):
    w, ids = _window_with_quad_pair(qtbot)
    w._selection.replace(faces={ids["left"]})
    w._on_shrink_selection()
    assert w._selection.faces == set()


def test_select_same_material_finds_the_other_painted_face(qtbot):
    from bermake.scene.scene import Side

    w, ids = _window_with_quad_pair(qtbot)
    scene = w._model.active_scene
    scene.set_face_material(ids["left"], 7, Side.FRONT)
    scene.set_face_material(ids["right"], 7, Side.FRONT)
    w._selection.replace(faces={ids["left"]})
    w._on_select_same_material()
    assert w._selection.faces == {ids["left"], ids["right"]}


def test_the_operations_preserve_the_selection_version_contract(qtbot):
    """Every handler goes through Selection.replace, which bumps `version`.
    The renderer uses that bump to detect changes; a handler that mutated the
    sets directly would leave the viewport stale."""
    w, ids = _window_with_quad_pair(qtbot)
    w._selection.replace(faces={ids["left"]})
    before = w._selection.version
    w._on_grow_selection()
    assert w._selection.version > before


def test_grow_preserves_an_instance_selected_alongside_a_face(qtbot):
    """Mutant 1 (Step 8): drop the instances= pass-through in
    _on_grow_selection and this is the test that catches it. grow() only
    ever touches geometry, so an instance selected alongside a face must
    survive the round trip through Selection.replace unchanged."""
    w, ids = _window_with_quad_pair(qtbot)
    scene = w._model.active_scene
    g = scene.add_vertex(np.array([5.0, 0.0, 0.0], dtype=np.float32))
    h = scene.add_vertex(np.array([6.0, 0.0, 0.0], dtype=np.float32))
    i = scene.add_vertex(np.array([6.0, 1.0, 0.0], dtype=np.float32))
    j = scene.add_vertex(np.array([5.0, 1.0, 0.0], dtype=np.float32))
    lone_face = scene.add_face_from_loop((g, h, i, j))

    w._selection.clear()
    w._selection.toggle_face(lone_face)
    w._on_make_group()
    instance_id = next(iter(w._selection.instances))

    w._selection.replace(faces={ids["left"]}, instances={instance_id})
    w._on_grow_selection()
    assert instance_id in w._selection.instances


def test_grow_on_an_empty_selection_is_a_no_op(qtbot):
    w, _ids = _window_with_quad_pair(qtbot)
    w._selection.clear()
    w._on_grow_selection()
    assert w._selection.is_empty()


def test_toggling_vertex_mode_flips_the_viewport_flag(qtbot):
    w, _ids = _window_with_quad_pair(qtbot)
    assert w._viewport.select_vertices is False
    w._on_toggle_select_vertices(True)
    assert w._viewport.select_vertices is True
    w._on_toggle_select_vertices(False)
    assert w._viewport.select_vertices is False


def test_turning_vertex_mode_off_drops_selected_vertices(qtbot):
    """Same reasoning as View > Guides deselecting hidden guides: a selection
    the user can no longer see, but which Move would still drag, is a trap."""
    w, ids = _window_with_quad_pair(qtbot)
    w._on_toggle_select_vertices(True)
    w._selection.replace(vertices={ids["a"]})
    w._on_toggle_select_vertices(False)
    assert w._selection.vertices == set()


def test_every_new_action_id_is_registered(qtbot):
    w, _ids = _window_with_quad_pair(qtbot)
    for action_id in (
        "edit_invert_selection",
        "select_grow",
        "select_shrink",
        "select_same_material",
        "select_same_tag",
        "view_select_vertices",
    ):
        assert action_id in w._actions, action_id
