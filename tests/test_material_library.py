from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from bermake.model.material import Material, MaterialLibrary


def test_default_material_is_first_with_default_id():
    lib = MaterialLibrary()
    mats = lib.materials()
    assert MaterialLibrary.DEFAULT_ID == 0
    assert mats[0].id == 0
    assert mats[0].name == "Default"


def test_builtin_palette_seeded_with_contiguous_monotonic_ids():
    lib = MaterialLibrary()
    ids = [m.id for m in lib.materials()]
    assert len(ids) >= 9                      # Default + >= 8 builtins
    assert ids == list(range(len(ids)))       # 0..N contiguous & ascending


def test_get_returns_material_by_id():
    lib = MaterialLibrary()
    brick = next(m for m in lib.materials() if m.name == "Brick Red")
    assert lib.get(brick.id) is brick


def test_get_unknown_id_falls_back_to_default():
    lib = MaterialLibrary()
    assert lib.get(9999).id == MaterialLibrary.DEFAULT_ID


def test_add_custom_appends_with_fresh_id_and_keeps_default_first():
    lib = MaterialLibrary()
    before = len(lib.materials())
    mat = lib.add_custom("#A1B2C3", (0.63, 0.70, 0.76))
    assert mat.id == before                   # next id == old count
    assert lib.get(mat.id) is mat
    assert lib.materials()[-1] is mat
    assert lib.materials()[0].id == MaterialLibrary.DEFAULT_ID


def test_material_is_frozen():
    m = Material(1, "X", (0.1, 0.2, 0.3))
    with pytest.raises(FrozenInstanceError):
        m.base_color = (0.0, 0.0, 0.0)  # type: ignore[misc]


def test_from_records_with_empty_list_falls_back_to_sentinel_default():
    """Empty records is a real (if unusual) input: no materials survive into
    _materials/_order, but `get()` must still return a usable Default rather
    than KeyError/None — the sentinel Default seeded by `cls()` before the
    records loop overwrote the dicts is the fallback that makes this work."""
    lib = MaterialLibrary.from_records([], next_id=1)
    assert lib.materials() == []
    default = lib.get(MaterialLibrary.DEFAULT_ID)
    assert default is not None
    assert default.name == "Default"
