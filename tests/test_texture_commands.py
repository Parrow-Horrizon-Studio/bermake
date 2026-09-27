"""M7.5b Task 8: the texture lifecycle under undo."""

from __future__ import annotations

import numpy as np
from bermake.commands.material_commands import (
    AddTextureCommand,
    DeleteTextureCommand,
    SetFacePlacementCommand,
    SetMaterialTextureCommand,
)
from bermake.model.material import MaterialLibrary
from bermake.model.texture import TextureLibrary
from bermake.scene.scene import DEFAULT_PLACEMENT, Scene, Side, TexturePlacement

_PNG = b"\x89PNG\r\n\x1a\nbytes"


def _square(scene):
    v = [
        scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32)),
    ]
    for a, b in zip(v, v[1:] + v[:1], strict=True):
        scene.add_edge(a, b)
    return scene.add_face_from_loop(v)


def test_adding_a_texture_is_undoable():
    lib = TextureLibrary()
    scene = Scene()
    cmd = AddTextureCommand(lib, "brick.png", _PNG, "png", 4, 4, False)
    cmd.do(scene)
    assert cmd.texture_id is not None
    assert lib.get(cmd.texture_id).data == _PNG
    cmd.undo(scene)
    assert lib.get(cmd.texture_id) is None


def test_redo_restores_the_same_id():
    # A redo that allocated a fresh id would leave every material that pointed
    # at the old one dangling.
    lib = TextureLibrary()
    scene = Scene()
    cmd = AddTextureCommand(lib, "brick.png", _PNG, "png", 4, 4, False)
    cmd.do(scene)
    first = cmd.texture_id
    cmd.undo(scene)
    cmd.do(scene)
    assert cmd.texture_id == first
    assert lib.get(first) is not None


def test_setting_a_material_texture_undoes_to_untextured():
    textures, materials, scene = TextureLibrary(), MaterialLibrary(), Scene()
    tex = textures.add("b.png", _PNG, "png", 4, 4, False)
    mat = materials.add_custom("Brick", (1.0, 1.0, 1.0))

    cmd = SetMaterialTextureCommand(materials, mat.id, tex.id, texture_size=(2.0, 2.0))
    cmd.do(scene)
    assert materials.get(mat.id).texture_id == tex.id
    assert materials.get(mat.id).texture_size == (2.0, 2.0)

    cmd.undo(scene)
    assert materials.get(mat.id).texture_id is None
    assert materials.get(mat.id).texture_size == (1.0, 1.0)


def test_deleting_a_texture_counts_its_materials_before_it_runs():
    textures, materials, scene = TextureLibrary(), MaterialLibrary(), Scene()
    tex = textures.add("b.png", _PNG, "png", 4, 4, False)
    other = textures.add("w.png", _PNG, "png", 4, 4, False)
    a = materials.add_custom("A", (1.0, 1.0, 1.0))
    b = materials.add_custom("B", (1.0, 1.0, 1.0))
    c = materials.add_custom("C", (1.0, 1.0, 1.0))
    materials.edit(a.id, texture_id=tex.id)
    materials.edit(b.id, texture_id=tex.id)
    materials.edit(c.id, texture_id=other.id)

    cmd = DeleteTextureCommand(textures, materials, tex.id)
    assert cmd.affected_material_count == 2       # readable BEFORE do()
    assert textures.get(tex.id) is not None       # and nothing has happened yet


def test_deleting_clears_every_referencing_material_and_undo_restores_them():
    # The M7.5a lesson, one level up. A delete that cleared only some materials
    # leaves the rest pointing at a texture that no longer exists, and that
    # dangling state survives save and load.
    textures, materials, scene = TextureLibrary(), MaterialLibrary(), Scene()
    tex = textures.add("b.png", _PNG, "png", 4, 4, False)
    a = materials.add_custom("A", (1.0, 1.0, 1.0))
    b = materials.add_custom("B", (1.0, 1.0, 1.0))
    materials.edit(a.id, texture_id=tex.id)
    materials.edit(b.id, texture_id=tex.id)

    cmd = DeleteTextureCommand(textures, materials, tex.id)
    cmd.do(scene)
    assert textures.get(tex.id) is None
    assert materials.get(a.id).texture_id is None
    assert materials.get(b.id).texture_id is None

    cmd.undo(scene)
    assert textures.get(tex.id) is not None
    assert materials.get(a.id).texture_id == tex.id
    assert materials.get(b.id).texture_id == tex.id


def test_deleting_restores_the_texture_at_its_original_index():
    # Three textures, delete the MIDDLE one. With two, or the last, a restore
    # that appends is indistinguishable from a correct one.
    textures, materials, scene = TextureLibrary(), MaterialLibrary(), Scene()
    a = textures.add("a.png", _PNG, "png", 4, 4, False)
    b = textures.add("b.png", _PNG, "png", 4, 4, False)
    c = textures.add("c.png", _PNG, "png", 4, 4, False)
    before = [t.id for t in textures.textures()]

    cmd = DeleteTextureCommand(textures, materials, b.id)
    cmd.do(scene)
    assert [t.id for t in textures.textures()] == [a.id, c.id]
    cmd.undo(scene)
    assert [t.id for t in textures.textures()] == before


def test_placement_targets_the_named_side_and_undoes_only_that_side():
    scene = Scene()
    f = _square(scene)
    scene.set_face_placement(f, TexturePlacement(scale=4.0), Side.FRONT)

    cmd = SetFacePlacementCommand(f, TexturePlacement(scale=2.0), Side.BACK)
    cmd.do(scene)
    assert scene.face_placement(f, Side.BACK).scale == 2.0
    assert scene.face_placement(f, Side.FRONT).scale == 4.0

    cmd.undo(scene)
    assert scene.face_placement(f, Side.BACK) == DEFAULT_PLACEMENT
    assert scene.face_placement(f, Side.FRONT).scale == 4.0


def test_placement_undo_restores_a_previous_placement_not_just_the_identity():
    # A command that always undoes to DEFAULT_PLACEMENT passes any test whose
    # face started unadjusted. This one starts adjusted.
    scene = Scene()
    f = _square(scene)
    scene.set_face_placement(f, TexturePlacement(offset_u=1.0))

    cmd = SetFacePlacementCommand(f, TexturePlacement(offset_u=9.0))
    cmd.do(scene)
    cmd.undo(scene)
    assert scene.face_placement(f).offset_u == 1.0
