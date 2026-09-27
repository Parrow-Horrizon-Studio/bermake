"""M7.5a Task 1: PBR fields on Material, edit/remove on the library."""

from __future__ import annotations

import pytest
from bermake.model.material import Material, MaterialLibrary


def test_material_defaults_are_opaque_dielectric():
    m = Material(7, "Glass", (0.5, 0.6, 0.7))
    assert m.alpha == 1.0
    assert m.metallic == 0.0
    assert m.roughness == 0.5
    assert m.is_translucent is False


def test_is_translucent_is_strictly_below_one():
    assert Material(1, "a", (0, 0, 0), alpha=0.999).is_translucent is True
    assert Material(1, "a", (0, 0, 0), alpha=1.0).is_translucent is False


def test_edit_returns_a_new_record_and_replaces_it_in_the_library():
    lib = MaterialLibrary()
    original = lib.add_custom("Glass", (0.5, 0.6, 0.7))
    edited = lib.edit(original.id, alpha=0.4, roughness=0.1)
    assert edited.alpha == 0.4
    assert edited.roughness == 0.1
    # untouched fields survive
    assert edited.base_color == (0.5, 0.6, 0.7)
    assert edited.name == "Glass"
    # and the library now hands out the edited record
    assert lib.get(original.id) is edited


def test_edit_can_rename():
    lib = MaterialLibrary()
    m = lib.add_custom("Glass", (0.5, 0.6, 0.7))
    assert lib.edit(m.id, name="Frosted").name == "Frosted"


def test_edit_rejects_an_unknown_field():
    lib = MaterialLibrary()
    m = lib.add_custom("Glass", (0.5, 0.6, 0.7))
    with pytest.raises(TypeError):
        lib.edit(m.id, shininess=4.0)


def test_remove_drops_the_material_and_keeps_display_order_of_the_rest():
    lib = MaterialLibrary()
    a = lib.add_custom("A", (1.0, 0.0, 0.0))
    b = lib.add_custom("B", (0.0, 1.0, 0.0))
    before = [m.id for m in lib.materials()]
    lib.remove(a.id)
    after = [m.id for m in lib.materials()]
    assert a.id not in after
    assert b.id in after
    # every surviving id keeps its relative order
    assert after == [i for i in before if i != a.id]


def test_remove_refuses_the_default():
    lib = MaterialLibrary()
    with pytest.raises(ValueError):
        lib.remove(MaterialLibrary.DEFAULT_ID)


def test_restore_puts_a_material_back_at_its_original_index():
    lib = MaterialLibrary()
    lib.add_custom("A", (1.0, 0.0, 0.0))
    target = lib.add_custom("B", (0.0, 1.0, 0.0))
    lib.add_custom("C", (0.0, 0.0, 1.0))
    order_before = [m.id for m in lib.materials()]
    idx = lib.index_of(target.id)
    removed = lib.remove(target.id)
    lib.restore(removed, idx)
    assert [m.id for m in lib.materials()] == order_before


def test_records_round_trip_the_pbr_fields():
    lib = MaterialLibrary()
    lib.add_custom("Glass", (0.5, 0.6, 0.7))
    lib.edit(lib.materials()[-1].id, alpha=0.3, metallic=0.8, roughness=0.15)
    rebuilt = MaterialLibrary.from_records(lib.to_records(), lib.next_id)
    m = rebuilt.materials()[-1]
    assert (m.alpha, m.metallic, m.roughness) == (0.3, 0.8, 0.15)
    assert m.base_color == (0.5, 0.6, 0.7)


def test_from_records_accepts_a_schema_4_color_key():
    # Schema <= 4 wrote "color" and no PBR fields. Task 11 relies on this.
    rebuilt = MaterialLibrary.from_records(
        [{"id": 0, "name": "Default", "color": [0.65, 0.65, 0.70]}], 1
    )
    m = rebuilt.get(0)
    assert m.base_color == (0.65, 0.65, 0.70)
    assert (m.alpha, m.metallic, m.roughness) == (1.0, 0.0, 0.5)
