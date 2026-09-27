from __future__ import annotations

from bermake.model.tag import Tag, TagLibrary


def test_untagged_is_first_and_id_zero():
    lib = TagLibrary()
    tags = lib.tags()
    assert TagLibrary.UNTAGGED_ID == 0
    assert tags[0].id == 0
    assert tags[0].name == "Untagged"
    assert tags[0].visible is True


def test_add_mints_fresh_monotonic_ids():
    lib = TagLibrary()
    a = lib.add("Walls")
    b = lib.add("Furniture")
    assert a.id == 1
    assert b.id == 2
    assert [t.id for t in lib.tags()] == [0, 1, 2]


def test_get_unknown_falls_back_to_untagged():
    lib = TagLibrary()
    assert lib.get(999).id == TagLibrary.UNTAGGED_ID


def test_from_records_with_empty_list_falls_back_to_sentinel_untagged():
    """Empty records is a real (if unusual) input: no tags survive into
    _tags/_order, but `get()` must still return a usable Untagged rather
    than KeyError/None — the sentinel Untagged seeded by `cls()` before the
    records loop overwrote the dicts is the fallback that makes this work."""
    lib = TagLibrary.from_records([], next_id=1)
    assert lib.tags() == []
    untagged = lib.get(TagLibrary.UNTAGGED_ID)
    assert untagged is not None
    assert untagged.name == "Untagged"


def test_set_visible_toggles_user_tag():
    lib = TagLibrary()
    w = lib.add("Walls")
    lib.set_visible(w.id, False)
    assert lib.is_visible(w.id) is False
    lib.set_visible(w.id, True)
    assert lib.is_visible(w.id) is True


def test_untagged_cannot_be_hidden():
    lib = TagLibrary()
    lib.set_visible(TagLibrary.UNTAGGED_ID, False)
    assert lib.is_visible(TagLibrary.UNTAGGED_ID) is True


def test_is_visible_unknown_id_is_true():
    lib = TagLibrary()
    assert lib.is_visible(999) is True


def test_tag_is_mutable():
    t = Tag(1, "X", True)
    t.visible = False
    assert t.visible is False


def test_rename_changes_name():
    lib = TagLibrary()
    w = lib.add("Walls")
    lib.rename(w.id, "Exterior Walls")
    assert lib.get(w.id).name == "Exterior Walls"


def test_rename_untagged_is_noop():
    lib = TagLibrary()
    lib.rename(TagLibrary.UNTAGGED_ID, "Nope")
    assert lib.get(TagLibrary.UNTAGGED_ID).name == "Untagged"


def test_rename_empty_is_noop():
    lib = TagLibrary()
    w = lib.add("Walls")
    lib.rename(w.id, "")
    assert lib.get(w.id).name == "Walls"
