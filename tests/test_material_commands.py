"""M7.5a Task 8: material lifecycle under undo."""

from __future__ import annotations

import numpy as np
import pytest
from bermake.commands.material_commands import (
    AddMaterialCommand,
    DeleteMaterialCommand,
    EditMaterialCommand,
    PaintFaceCommand,
)
from bermake.model.material import MaterialLibrary
from bermake.model.model import Model
from bermake.scene.scene import Scene, Side


def _square(scene, z=0.0):
    v = [
        scene.add_vertex(np.array([0.0, 0.0, z], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 0.0, z], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 1.0, z], dtype=np.float32)),
        scene.add_vertex(np.array([0.0, 1.0, z], dtype=np.float32)),
    ]
    for a, b in zip(v, v[1:] + v[:1], strict=True):
        scene.add_edge(a, b)
    return scene.add_face_from_loop(v)


def _model_with_square(z=0.0):
    """A Model whose root definition holds one square face."""
    model = Model()
    return model, _square(model.root.mesh, z)


def _grouped(model, name):
    """Add a child group holding one square, and return (definition, instance)."""
    definition = model.new_definition(name, is_group=True)
    face = _square(definition.mesh)
    instance = model.new_instance(definition)
    model.root.children.append(instance)
    return definition, instance, face


def test_paint_targets_the_named_side_and_undoes_only_that_side():
    s = Scene()
    f = _square(s)
    s.set_face_material(f, 4, Side.FRONT)
    cmd = PaintFaceCommand(f, 7, Side.BACK)
    cmd.do(s)
    assert s.face_material(f, Side.BACK) == 7
    assert s.face_material(f, Side.FRONT) == 4  # discriminating
    cmd.undo(s)
    assert s.face_material(f, Side.BACK) == 0
    assert s.face_material(f, Side.FRONT) == 4


def test_paint_defaults_to_the_front():
    s = Scene()
    f = _square(s)
    PaintFaceCommand(f, 7).do(s)
    assert s.face_material(f, Side.FRONT) == 7
    assert s.face_material(f, Side.BACK) == 0


def test_add_then_undo_leaves_the_library_as_it_was():
    lib = MaterialLibrary()
    before = [m.id for m in lib.materials()]
    cmd = AddMaterialCommand(lib, "Glass", (0.5, 0.6, 0.7))
    cmd.do(Scene())
    assert lib.get(cmd.material_id).name == "Glass"
    cmd.undo(Scene())
    assert [m.id for m in lib.materials()] == before


def test_edit_undo_restores_every_field_not_just_the_changed_one():
    lib = MaterialLibrary()
    m = lib.add_custom("Glass", (0.5, 0.6, 0.7))
    lib.edit(m.id, alpha=0.8, metallic=0.3)
    cmd = EditMaterialCommand(lib, m.id, base_color=(1.0, 0.0, 0.0), alpha=0.2)
    cmd.do(Scene())
    assert lib.get(m.id).base_color == (1.0, 0.0, 0.0)
    cmd.undo(Scene())
    restored = lib.get(m.id)
    assert restored.base_color == (0.5, 0.6, 0.7)
    assert restored.alpha == 0.8  # the untouched-by-this-edit field
    assert restored.metallic == 0.3


def test_edit_can_rename():
    lib = MaterialLibrary()
    m = lib.add_custom("Glass", (0.5, 0.6, 0.7))
    EditMaterialCommand(lib, m.id, name="Frosted").do(Scene())
    assert lib.get(m.id).name == "Frosted"


def test_delete_counts_affected_sides_before_it_runs():
    model, a = _model_with_square(0.0)
    s = model.root.mesh
    b = _square(s, 1.0)
    lib = model.materials
    m = lib.add_custom("Brick", (0.7, 0.3, 0.2))
    s.set_face_material(a, m.id, Side.FRONT)
    s.set_face_material(b, m.id, Side.BACK)
    cmd = DeleteMaterialCommand(lib, m.id, model)
    assert cmd.affected_count == 2  # readable BEFORE do()
    assert lib.get(m.id).name == "Brick"


def test_delete_repaints_affected_sides_to_default():
    model, f = _model_with_square()
    s = model.root.mesh
    lib = model.materials
    m = lib.add_custom("Brick", (0.7, 0.3, 0.2))
    s.set_face_material(f, m.id, Side.BACK)
    DeleteMaterialCommand(lib, m.id, model).do(model)
    assert s.face_material(f, Side.BACK) == 0
    assert m.id not in [x.id for x in lib.materials()]


def test_delete_undo_restores_the_material_and_the_paint():
    model, f = _model_with_square()
    s = model.root.mesh
    lib = model.materials
    m = lib.add_custom("Brick", (0.7, 0.3, 0.2))
    s.set_face_material(f, m.id, Side.BACK)
    cmd = DeleteMaterialCommand(lib, m.id, model)
    cmd.do(model)
    cmd.undo(model)
    assert lib.get(m.id).name == "Brick"
    assert s.face_material(f, Side.BACK) == m.id  # both halves


def test_delete_undo_restores_display_order():
    model = Model()
    lib = model.materials
    lib.add_custom("A", (1.0, 0.0, 0.0))
    target = lib.add_custom("B", (0.0, 1.0, 0.0))
    lib.add_custom("C", (0.0, 0.0, 1.0))
    before = [m.id for m in lib.materials()]
    cmd = DeleteMaterialCommand(lib, target.id, model)
    cmd.do(model)
    cmd.undo(model)
    # appending on undo would put B last and still "restore" it
    assert [m.id for m in lib.materials()] == before


def test_delete_refuses_the_default():
    with pytest.raises(ValueError):
        DeleteMaterialCommand(MaterialLibrary(), 0, Model())


def test_delete_spans_every_definition_not_just_the_entered_one():
    """The discriminating test for the model-wide fix.

    MaterialLibrary is model-wide but each Definition owns its own Scene. A
    scene-scoped delete only repaints the *entered* definition, leaving the
    other definition's face tagged with an id the library no longer holds --
    which renders as Default (MaterialLibrary.get() falls back) yet survives
    save/load as a dangling reference.
    """
    model = Model()
    lib = model.materials
    m = lib.add_custom("Brick", (0.7, 0.3, 0.2))
    def_a, inst_a, face_a = _grouped(model, "A")
    def_b, _inst_b, face_b = _grouped(model, "B")
    def_a.mesh.set_face_material(face_a, m.id)
    def_b.mesh.set_face_material(face_b, m.id)
    model.enter(inst_a)  # editing inside A only

    cmd = DeleteMaterialCommand(lib, m.id, model)
    assert cmd.affected_count == 2, "must count both definitions, not just the entered one"

    cmd.do(model)
    assert def_a.mesh.face_material(face_a) == 0
    assert def_b.mesh.face_material(face_b) == 0, "the other definition is stranded"

    cmd.undo(model)
    assert def_a.mesh.face_material(face_a) == m.id
    assert def_b.mesh.face_material(face_b) == m.id


def test_delete_counts_a_shared_definition_once_per_definition_not_per_instance():
    """traverse() yields a definition once per instance; the scan must dedupe."""
    model = Model()
    lib = model.materials
    m = lib.add_custom("Brick", (0.7, 0.3, 0.2))
    shared, _inst, face = _grouped(model, "Shared")
    model.root.children.append(model.new_instance(shared))  # second instance
    shared.mesh.set_face_material(face, m.id)

    cmd = DeleteMaterialCommand(lib, m.id, model)
    assert cmd.affected_count == 1
