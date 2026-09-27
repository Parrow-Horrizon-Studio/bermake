from __future__ import annotations

from bermake.model.model import Model
from bermake.model.tag import TagLibrary


def test_new_instance_defaults_to_untagged():
    m = Model()
    d = m.new_definition("D", is_group=True)
    inst = m.new_instance(d)
    assert inst.tag_id == TagLibrary.UNTAGGED_ID


def test_model_has_tag_library():
    m = Model()
    assert isinstance(m.tags, TagLibrary)
    assert m.tags.tags()[0].id == TagLibrary.UNTAGGED_ID


def test_clone_definition_copies_child_tag_ids():
    m = Model()
    outer = m.new_definition("Outer", is_group=True)
    inner = m.new_definition("Inner", is_group=True)
    child = m.new_instance(inner)
    child.tag_id = 5
    outer.children.append(child)
    clone = m.clone_definition(outer)
    assert clone.children[0].tag_id == 5
