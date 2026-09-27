"""M7.5a Task 9: the material editor drives commands, not the library."""

from __future__ import annotations

import pytest

from bermake.ui.materials_page import MaterialsPage, _swatch_style


def test_editing_alpha_goes_through_the_command_stack(main_window):
    page = main_window._materials_page
    lib = main_window._model.materials
    m = lib.add_custom("Glass", (0.5, 0.6, 0.7))
    page.set_library(lib)
    page.set_active(m.id)
    depth = len(main_window._command_stack._undo)

    page._apply_edit(alpha=0.4)

    assert lib.get(m.id).alpha == 0.4
    assert len(main_window._command_stack._undo) == depth + 1, "edit must be undoable"
    main_window._command_stack.undo()
    assert lib.get(m.id).alpha == 1.0


def test_adding_a_material_is_undoable(main_window):
    page = main_window._materials_page
    lib = main_window._model.materials
    page.set_library(lib)
    before = [m.id for m in lib.materials()]

    page._add_material("Glass", (0.5, 0.6, 0.7))

    assert len(lib.materials()) == len(before) + 1
    main_window._command_stack.undo()
    assert [m.id for m in lib.materials()] == before


def test_deleting_reassigns_painted_faces_and_undoes_as_one_step(main_window_with_square):
    win = main_window_with_square
    page = win._materials_page
    lib = win._model.materials
    scene = win._model.active_context.mesh
    m = lib.add_custom("Brick", (0.7, 0.3, 0.2))
    face = next(iter(scene.faces_iter())).id
    scene.set_face_material(face, m.id)
    page.set_library(lib)
    page.set_active(m.id)
    depth = len(win._command_stack._undo)

    page._delete_active(confirm=lambda count: True)

    assert scene.face_material(face) == 0
    assert len(win._command_stack._undo) == depth + 1, "one undo step, not two"
    win._command_stack.undo()
    assert scene.face_material(face) == m.id
    assert lib.get(m.id).name == "Brick"


def test_the_delete_confirmation_is_told_the_affected_count(main_window_with_square):
    win = main_window_with_square
    page = win._materials_page
    lib = win._model.materials
    scene = win._model.active_context.mesh
    m = lib.add_custom("Brick", (0.7, 0.3, 0.2))
    face = next(iter(scene.faces_iter())).id
    scene.set_face_material(face, m.id)
    page.set_library(lib)
    page.set_active(m.id)
    seen = []

    page._delete_active(confirm=lambda count: seen.append(count) or False)

    assert seen == [1]
    # cancelling must change nothing
    assert scene.face_material(face) == m.id
    assert lib.get(m.id).name == "Brick"


def test_the_default_material_cannot_be_deleted(main_window):
    page = main_window._materials_page
    lib = main_window._model.materials
    page.set_library(lib)
    page.set_active(0)
    assert page._can_delete_active() is False

    # End-to-end, not just the predicate: a mutant that dropped the guard from
    # _delete_active() itself would sail past the assertion above.
    before = [m.id for m in lib.materials()]
    depth = len(main_window._command_stack._undo)
    calls = []

    page._delete_active(confirm=lambda count: calls.append(count) or True)

    assert calls == [], "must not even ask before refusing the Default"
    assert [m.id for m in lib.materials()] == before
    assert len(main_window._command_stack._undo) == depth, "nothing may be pushed"


def test_undo_through_the_menu_path_refreshes_the_swatch_grid(main_window):
    """Ctrl+Z routes through MainWindow._on_undo, not CommandStack.undo() --
    and the Materials page must come back in sync with the rewound library."""
    win = main_window
    page = win._materials_page
    lib = win._model.materials
    page.set_library(lib)
    before = len(page._buttons)

    page._add_material("Glass", (0.5, 0.6, 0.7))
    assert len(page._buttons) == before + 1

    win._on_undo()

    assert len(lib.materials()) == before
    assert len(page._buttons) == before, "swatch grid still shows the undone material"


def test_undo_through_the_menu_path_refreshes_the_editor_fields(main_window):
    win = main_window
    lib = win._model.materials
    m = lib.add_custom("Glass", (0.5, 0.6, 0.7))
    page = win._materials_page
    page.set_library(lib)
    page.set_active(m.id)

    page._apply_edit(name="Frosted")
    assert page._name_edit.text() == "Frosted"

    win._on_undo()

    assert lib.get(m.id).name == "Glass"
    assert page._name_edit.text() == "Glass", "editor still shows the pre-undo name"


def test_picking_the_same_colour_pushes_no_undo_entry(main_window, monkeypatch):
    from PySide6.QtGui import QColor

    from bermake.ui import materials_page as materials_page_module

    win = main_window
    lib = win._model.materials
    m = lib.add_custom("Glass", (0.5, 0.6, 0.7))
    page = win._materials_page
    page.set_library(lib)
    page.set_active(m.id)
    depth = len(win._command_stack._undo)

    r, g, b = (round(c * 255) for c in lib.get(m.id).base_color)
    monkeypatch.setattr(
        materials_page_module.QColorDialog,
        "getColor",
        staticmethod(lambda *a, **k: QColor(r, g, b)),
    )

    page._on_edit_color()

    assert len(win._command_stack._undo) == depth, "an unchanged colour is not an edit"


@pytest.mark.parametrize("field,value", [("metallic", 0.9), ("roughness", 0.1)])
def test_every_pbr_field_round_trips_through_the_editor(main_window, field, value):
    page = main_window._materials_page
    lib = main_window._model.materials
    m = lib.add_custom("Steel", (0.6, 0.6, 0.6))
    page.set_library(lib)
    page.set_active(m.id)
    page._apply_edit(**{field: value})
    assert getattr(lib.get(m.id), field) == value


def _stylesheet_alpha(stylesheet: str) -> int:
    """Pull the alpha channel out of a `rgba(r,g,b,a)` stylesheet fragment."""
    start = stylesheet.index("rgba(") + len("rgba(")
    end = stylesheet.index(")", start)
    parts = [p.strip() for p in stylesheet[start:end].split(",")]
    return int(parts[3])


def test_translucent_swatch_carries_a_lower_alpha_than_opaque():
    """A solid-swatch implementation (background-color: rgb(...), or
    rgba(...) with a hardcoded 255) renders an alpha-0.3 material exactly
    like an opaque one -- a lie the user would paint with unknowingly. This
    kills that: it demands the stylesheet's own alpha channel scale with the
    material's alpha, not just that *some* string changed."""
    opaque_style = _swatch_style((0.5, 0.5, 0.5), 1.0, False)
    translucent_style = _swatch_style((0.5, 0.5, 0.5), 0.3, False)

    opaque_alpha = _stylesheet_alpha(opaque_style)
    translucent_alpha = _stylesheet_alpha(translucent_style)

    assert opaque_alpha == 255
    assert translucent_alpha < opaque_alpha
    # Loosely tracks alpha * 255, not just "some smaller number".
    assert abs(translucent_alpha - round(0.3 * 255)) <= 2


def test_swatch_grid_reflects_material_alpha(qtbot):
    from bermake.model.material import MaterialLibrary

    lib = MaterialLibrary()
    mat = lib.add_custom("Glass", (0.4, 0.6, 0.9))
    lib.edit(mat.id, alpha=0.25)
    page = MaterialsPage(lib)
    qtbot.addWidget(page)

    btn = page._buttons[mat.id]
    assert _stylesheet_alpha(btn.styleSheet()) < 255
