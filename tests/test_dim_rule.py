import numpy as np
from bermake.model.model import Model
from bermake.viewport.scene_renderer import definition_is_dimmed


def _child_group(m):
    d = m.new_definition("G", is_group=True)
    inst = m.new_instance(d)
    m.root.children.append(inst)
    return d, inst


def test_nothing_dimmed_at_root():
    m = Model()
    d, _inst = _child_group(m)
    # At root (no active_path), neither the root nor the child group is dimmed.
    assert definition_is_dimmed(m.root, m) is False
    assert definition_is_dimmed(d, m) is False


def test_inside_group_dims_everything_but_active():
    m = Model()
    d, inst = _child_group(m)
    m.enter(inst)
    assert m.active_context is d
    assert definition_is_dimmed(d, m) is False        # active context: full colour
    assert definition_is_dimmed(m.root, m) is True     # the rest: dimmed
