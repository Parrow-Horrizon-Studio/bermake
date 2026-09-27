"""M7.5b Task 1: the texture record and its library."""

from __future__ import annotations

from bermake.model.material import MaterialLibrary
from bermake.model.texture import Texture, TextureLibrary

_PNG = b"\x89PNG\r\n\x1a\n-pretend-this-is-an-image-"


def _lib():
    lib = TextureLibrary()
    brick = lib.add("brick.png", _PNG, "png", 512, 512, False)
    glass = lib.add("glass.png", _PNG, "png", 256, 256, True)
    return lib, brick, glass


def test_add_assigns_increasing_ids_and_keeps_the_bytes():
    lib, brick, glass = _lib()
    assert brick.id != glass.id
    assert lib.get(brick.id).data == _PNG
    assert lib.get(brick.id).width == 512
    assert lib.get(glass.id).has_transparency is True


def test_get_returns_none_for_an_unknown_id():
    # Unlike MaterialLibrary.get, which falls back to Default. There is no
    # default texture: an id naming nothing means untextured.
    lib, _, _ = _lib()
    assert lib.get(9999) is None


def test_textures_are_returned_in_insertion_order():
    lib, brick, glass = _lib()
    assert [t.id for t in lib.textures()] == [brick.id, glass.id]


def test_edit_returns_the_new_record_and_leaves_the_others_alone():
    lib, brick, glass = _lib()
    renamed = lib.edit(brick.id, name="brickwork.png")
    assert renamed.name == "brickwork.png"
    assert renamed.data == _PNG
    assert lib.get(glass.id).name == "glass.png"


def test_remove_then_restore_puts_it_back_at_its_original_index():
    # Three textures, remove the MIDDLE one. With two, or by removing the last,
    # a restore that appends is indistinguishable from a correct one.
    lib, brick, glass = _lib()
    wood = lib.add("wood.png", _PNG, "png", 64, 64, False)
    before = [t.id for t in lib.textures()]

    index = lib.index_of(glass.id)
    record = lib.remove(glass.id)
    assert [t.id for t in lib.textures()] == [brick.id, wood.id]

    lib.restore(record, index)
    assert [t.id for t in lib.textures()] == before


def test_records_omit_the_bytes():
    # The blobs become zip entries in Task 7. A record carrying `data` would
    # base64 megabytes into document.json.
    lib, brick, _ = _lib()
    rec = next(r for r in lib.to_records() if r["id"] == brick.id)
    assert "data" not in rec
    assert rec["width"] == 512
    assert rec["has_transparency"] is False


def test_records_round_trip_with_the_blobs_supplied_separately():
    lib, brick, glass = _lib()
    records = lib.to_records()
    blobs = {t.id: t.data for t in lib.textures()}

    restored = TextureLibrary.from_records(records, blobs)
    assert [t.id for t in restored.textures()] == [brick.id, glass.id]
    assert restored.get(glass.id).has_transparency is True
    assert restored.get(brick.id).data == _PNG


def test_a_record_whose_blob_is_missing_loads_with_empty_data():
    # Spec 1.8: a document referencing a missing container entry must load,
    # not refuse. That material renders untextured rather than failing the open.
    lib, brick, _ = _lib()
    restored = TextureLibrary.from_records(lib.to_records(), {})
    assert restored.get(brick.id).data == b""


def test_restore_keeps_next_id_ahead_of_the_restored_record():
    # Otherwise an undo of a delete, followed by a new add, reuses a live id.
    lib, _, glass = _lib()
    record = lib.remove(glass.id)
    lib.restore(record, 0)
    assert lib.next_id > glass.id


def test_the_model_layer_imports_no_qt():
    import subprocess
    import sys

    code = (
        "import bermake.model.texture, sys; "
        "print(any(m.startswith('PySide6') for m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_material_defaults_to_untextured():
    lib = MaterialLibrary()
    m = lib.add_custom("Plain", (0.5, 0.5, 0.5))
    assert m.texture_id is None
    assert m.texture_size == (1.0, 1.0)


def test_material_can_carry_a_texture_and_a_real_world_size():
    lib = MaterialLibrary()
    m = lib.add_custom("Brick", (1.0, 1.0, 1.0))
    edited = lib.edit(m.id, texture_id=7, texture_size=(2.0, 1.5))
    assert edited.texture_id == 7
    assert edited.texture_size == (2.0, 1.5)


def test_a_schema_5_material_record_without_texture_keys_still_loads():
    records = [{"id": 3, "name": "Old", "color": [0.2, 0.3, 0.4]}]
    lib = MaterialLibrary.from_records(records)
    assert lib.get(3).texture_id is None
    assert lib.get(3).texture_size == (1.0, 1.0)


def test_a_model_has_a_texture_library():
    from bermake.model.model import Model

    assert isinstance(Model().textures, TextureLibrary)
