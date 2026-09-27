"""The M7.2 command handlers (Task 9).

These run against a real MainWindow, so they need pytest-qt. Mirror the
fixture style already used by tests/test_main_window_scenes.py.
"""

from __future__ import annotations

import numpy as np
from bermake.model.annotation import Dimension, Label
from PySide6.QtWidgets import QInputDialog


def test_select_all_selects_the_active_context(qtbot, main_window_with_square):
    window = main_window_with_square
    window._selection.clear()

    window._on_select_all()

    assert len(window._selection.edges) == 4
    assert len(window._selection.faces) == 1


def test_select_none_clears_the_selection(qtbot, main_window_with_square):
    window = main_window_with_square
    window._on_select_all()
    assert not window._selection.is_empty()

    window._on_select_none()

    assert window._selection.is_empty()


def test_select_all_on_an_empty_model_is_a_no_op(qtbot, main_window):
    main_window._on_select_all()
    assert main_window._selection.is_empty()


def test_zoom_extents_moves_the_camera_to_frame_the_model(qtbot, main_window_with_square):
    window = main_window_with_square
    camera = window._viewport.camera
    before = np.array(camera.position, dtype=float).copy()

    window._on_zoom_extents()

    after = np.array(camera.position, dtype=float)
    assert not np.allclose(before, after)
    # The target lands on the square's centre (0.5, 0.5, 0).
    assert np.allclose(np.array(camera.target, dtype=float), [0.5, 0.5, 0.0], atol=1e-4)


def test_zoom_extents_preserves_the_view_direction(qtbot, main_window_with_square):
    window = main_window_with_square
    camera = window._viewport.camera
    before = np.array(camera.target, dtype=float) - np.array(camera.position, dtype=float)
    before /= np.linalg.norm(before)

    window._on_zoom_extents()

    after = np.array(camera.target, dtype=float) - np.array(camera.position, dtype=float)
    after /= np.linalg.norm(after)
    assert np.allclose(before, after, atol=1e-5)


def test_zoom_extents_on_an_empty_model_leaves_the_camera_alone(qtbot, main_window):
    camera = main_window._viewport.camera
    before = np.array(camera.position, dtype=float).copy()

    main_window._on_zoom_extents()

    assert np.allclose(before, np.array(camera.position, dtype=float))


def test_close_group_exits_one_level(qtbot, main_window_with_group):
    window = main_window_with_group
    window._model.enter(window._model.active_context.children[-1])
    depth = len(window._model.active_path)

    window._on_close_group()

    assert len(window._model.active_path) == depth - 1


def test_close_group_at_the_root_is_a_no_op(qtbot, main_window):
    main_window._on_close_group()
    assert main_window._model.active_path == []


def test_edit_group_enters_the_selected_instance(qtbot, main_window_with_group):
    window = main_window_with_group
    instance = window._model.active_context.children[-1]
    window._selection.clear()
    window._selection.toggle_instance(instance.id)

    window._on_edit_group()

    assert window._model.active_path[-1] is instance


def test_edit_group_with_nothing_selected_is_a_no_op(qtbot, main_window):
    main_window._selection.clear()
    main_window._on_edit_group()
    assert main_window._model.active_path == []


# ---------------------------------------------------------------------------
# Fix wave: _on_paint_selection and _on_edit_label_text (Task 9's own
# deliverables) had zero committed coverage. Added here, following the
# fixture/assertion style already used above -- observable state (material on
# the face, text on the annotation, command-stack depth), not method calls.
# ---------------------------------------------------------------------------


def _two_faces(window):
    """Two triangular faces sharing an edge -- mirrors the _two_face_scene
    helper in tests/test_scene_materials.py, built on the live MainWindow
    scene so _on_paint_selection can be exercised through the real handler."""
    scene = window.scene
    a = [
        scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32)),
    ]
    fa = scene.add_face_from_loop(a)
    b = [a[1], scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32)), a[2]]
    fb = scene.add_face_from_loop(b)
    return scene, fa, fb


def test_paint_selection_paints_every_face_and_one_undo_restores_all(qtbot, main_window):
    window = main_window
    scene, fa, fb = _two_faces(window)
    materials = window._model.materials.materials()
    active_id = materials[1].id
    previous_id = materials[2].id
    # fa starts unpainted (Default, id 0); fb starts painted with a DIFFERENT
    # material. Two distinct starting materials prove the composite undo
    # restores each face's own prior value, not a single shared one.
    scene.set_face_material(fb, previous_id)
    window._active_material_id = active_id
    window._selection.replace(faces=[fa, fb])

    assert len(window._command_stack._undo) == 0
    window._on_paint_selection()

    assert scene.face_material(fa) == active_id
    assert scene.face_material(fb) == active_id
    # Painting N faces must land as ONE undo-stack entry (a CompositeCommand
    # wrapping N PaintFaceCommands), so a single Ctrl+Z undoes the whole
    # paint gesture at once.
    assert len(window._command_stack._undo) == 1

    assert window._command_stack.undo() is True
    assert scene.face_material(fa) == 0
    assert scene.face_material(fb) == previous_id
    assert len(window._command_stack._undo) == 0


def test_paint_selection_with_nothing_selected_is_a_no_op(qtbot, main_window_with_square):
    window = main_window_with_square
    window._selection.clear()
    depth_before = len(window._command_stack._undo)

    window._on_paint_selection()

    assert len(window._command_stack._undo) == depth_before


def _label(window, text):
    ann = Label(window._model.new_annotation_id(), (0.0, 0.0, 0.0), (1.0, 1.0, 0.0), text)
    window._model.active_context.annotations.append(ann)
    return ann


def test_edit_label_text_applies_change_and_undo_restores_original(qtbot, main_window, monkeypatch):
    window = main_window
    ann = _label(window, "Hello")
    window._selection.replace(annotations=[ann.id])
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("Renamed", True)))
    depth_before = len(window._command_stack._undo)

    window._on_edit_label_text()

    assert ann.text == "Renamed"
    assert len(window._command_stack._undo) == depth_before + 1

    assert window._command_stack.undo() is True
    assert ann.text == "Hello"
    assert len(window._command_stack._undo) == depth_before


def test_edit_label_text_cancel_is_a_no_op(qtbot, main_window, monkeypatch):
    window = main_window
    ann = _label(window, "Hello")
    window._selection.replace(annotations=[ann.id])
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("Renamed", False)))
    depth_before = len(window._command_stack._undo)

    window._on_edit_label_text()

    assert ann.text == "Hello"
    assert len(window._command_stack._undo) == depth_before


def test_edit_label_text_on_a_non_label_annotation_is_a_no_op(qtbot, main_window, monkeypatch):
    """Pins the `kind == "label"` guard: a selected Dimension must be left
    untouched and must never even reach the text dialog."""
    window = main_window
    ann = Dimension(
        window._model.new_annotation_id(), (0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, -2.0, 0.0)
    )
    window._model.active_context.annotations.append(ann)
    window._selection.replace(annotations=[ann.id])
    before = (ann.p1, ann.p2, ann.offset)

    def _fail_if_called(*_a, **_k):
        raise AssertionError("QInputDialog.getText must not be reached for a non-label kind")

    monkeypatch.setattr(QInputDialog, "getText", staticmethod(_fail_if_called))
    depth_before = len(window._command_stack._undo)

    window._on_edit_label_text()

    assert (ann.p1, ann.p2, ann.offset) == before
    assert len(window._command_stack._undo) == depth_before
