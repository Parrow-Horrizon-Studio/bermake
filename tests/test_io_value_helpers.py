from bermake.document import DocumentSettings
from bermake.model.material import MaterialLibrary
from bermake.model.tag import TagLibrary
from bermake.units import UnitSystem, Units, units_from_dict, units_to_dict


def test_material_library_roundtrip_preserves_customs_and_next_id():
    lib = MaterialLibrary()
    custom = lib.add_custom("My Teal", (0.1, 0.6, 0.6))
    records, nid = lib.to_records(), lib.next_id
    rebuilt = MaterialLibrary.from_records(records, nid)
    assert [(m.id, m.name, m.base_color) for m in rebuilt.materials()] == \
           [(m.id, m.name, m.base_color) for m in lib.materials()]
    assert rebuilt.next_id == nid
    assert rebuilt.get(custom.id).name == "My Teal"
    # Default sentinel still resolves after rebuild.
    assert rebuilt.get(MaterialLibrary.DEFAULT_ID).name == "Default"


def test_tag_library_roundtrip_preserves_visibility_and_next_id():
    lib = TagLibrary()
    walls = lib.add("Walls")
    furn = lib.add("Furniture")
    lib.set_visible(furn.id, False)
    rebuilt = TagLibrary.from_records(lib.to_records(), lib.next_id)
    assert [(t.id, t.name, t.visible) for t in rebuilt.tags()] == \
           [(t.id, t.name, t.visible) for t in lib.tags()]
    assert rebuilt.next_id == lib.next_id
    assert rebuilt.is_visible(walls.id) is True
    assert rebuilt.is_visible(furn.id) is False
    # Untagged sentinel survives.
    assert rebuilt.get(TagLibrary.UNTAGGED_ID).name == "Untagged"


def test_units_roundtrip_imperial():
    u = Units(system=UnitSystem.IMPERIAL, metric_unit="cm",
              metric_precision=2, imperial_denominator=32)
    back = units_from_dict(units_to_dict(u))
    assert back == u


def test_document_settings_set_units():
    doc = DocumentSettings()
    u = Units(system=UnitSystem.IMPERIAL, imperial_denominator=8)
    doc.set_units(u)
    assert doc.units == u
