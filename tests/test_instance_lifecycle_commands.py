import numpy as np
from bermake.model.model import Model
from bermake.commands.instance_lifecycle_commands import (
    DeleteInstanceCommand, MakeUniqueCommand,
)


def test_make_unique_detaches_shared_definition():
    m = Model()
    d = m.new_definition("Chair", is_group=False)
    d.mesh.add_vertex(np.array([0, 0, 0], np.float32))
    a = m.new_instance(d); b = m.new_instance(d)
    m.root.children += [a, b]

    cmd = MakeUniqueCommand(b)
    cmd.do(m)
    assert b.definition is not d         # b now has its own clone
    assert a.definition is d
    assert len(list(b.definition.mesh.vertices_iter())) == 1  # geometry copied
    cmd.undo(m)
    assert b.definition is d


def test_delete_instance_removes_and_restores():
    m = Model()
    d = m.new_definition("G", is_group=True)
    inst = m.new_instance(d)
    m.root.children.append(inst)
    cmd = DeleteInstanceCommand(m.root, inst)
    cmd.do(m)
    assert inst not in m.root.children
    cmd.undo(m)
    assert inst in m.root.children


def test_make_unique_redo_reuses_same_clone():
    """Redo must reuse the same clone object, not allocate a second one."""
    m = Model()
    d = m.new_definition("Box", is_group=False)
    d.mesh.add_vertex(np.array([1, 2, 3], np.float32))
    a = m.new_instance(d); b = m.new_instance(d)
    m.root.children += [a, b]

    cmd = MakeUniqueCommand(b)
    cmd.do(m)                        # do
    clone_after_first_do = b.definition
    assert clone_after_first_do is not d

    cmd.undo(m)                      # undo
    assert b.definition is d

    cmd.do(m)                        # redo
    assert b.definition is clone_after_first_do  # same clone object, not a second one
