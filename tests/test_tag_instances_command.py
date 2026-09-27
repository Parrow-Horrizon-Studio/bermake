from __future__ import annotations

from bermake.commands.tag_commands import TagInstancesCommand
from bermake.model.model import Model


def _inst(m, tag_id=0):
    d = m.new_definition("D", is_group=True)
    i = m.new_instance(d)
    i.tag_id = tag_id
    return i


def test_do_assigns_and_undo_restores_mixed():
    m = Model()
    a = _inst(m, 0)
    b = _inst(m, 3)
    cmd = TagInstancesCommand([a, b], 5)
    cmd.do(m)
    assert a.tag_id == 5 and b.tag_id == 5
    cmd.undo(m)
    assert a.tag_id == 0 and b.tag_id == 3


def test_redo_reapplies():
    m = Model()
    a = _inst(m, 1)
    cmd = TagInstancesCommand([a], 7)
    cmd.do(m)
    cmd.undo(m)
    cmd.do(m)
    assert a.tag_id == 7
