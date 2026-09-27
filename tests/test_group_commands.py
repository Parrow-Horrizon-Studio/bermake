import numpy as np
from bermake.model.model import Model
from bermake.commands.group_commands import MakeGroupCommand


def _triangle(scene):
    a = scene.add_vertex(np.array([0, 0, 0], np.float32))
    b = scene.add_vertex(np.array([1, 0, 0], np.float32))
    c = scene.add_vertex(np.array([0, 1, 0], np.float32))
    f = scene.add_face_from_loop([a, b, c])
    return [a, b, c], f


def test_make_group_moves_geometry_into_new_definition():
    m = Model()
    verts, face = _triangle(m.root.mesh)
    cmd = MakeGroupCommand(m.root, verts, [], [face])
    cmd.do(m)

    # Parent mesh is now empty; a child instance exists.
    assert list(m.root.mesh.faces_iter()) == []
    assert len(m.root.children) == 1
    inst = m.root.children[0]
    assert inst is cmd.created_instance
    # The new definition holds the triangle.
    assert len(list(inst.definition.mesh.faces_iter())) == 1
    assert inst.definition.is_group is True


def test_make_group_undo_restores_parent_geometry():
    m = Model()
    verts, face = _triangle(m.root.mesh)
    before_face_ids = {f.id for f in m.root.mesh.faces_iter()}
    cmd = MakeGroupCommand(m.root, verts, [], [face])
    cmd.do(m)
    cmd.undo(m)
    assert {f.id for f in m.root.mesh.faces_iter()} == before_face_ids
    assert m.root.children == []


def test_make_group_redo_is_id_stable():
    m = Model()
    verts, face = _triangle(m.root.mesh)
    cmd = MakeGroupCommand(m.root, verts, [], [face])
    cmd.do(m)
    inst_id = cmd.created_instance.id
    def_obj = cmd.created_instance.definition
    cmd.undo(m)
    cmd.do(m)  # redo
    assert cmd.created_instance.id == inst_id          # same instance, stable id
    assert cmd.created_instance.definition is def_obj  # no second definition leaked
    assert len(m.root.children) == 1
    assert list(m.root.mesh.faces_iter()) == []        # geometry lifted again
    cmd.undo(m)                                          # undo after redo restores again
    assert len(list(m.root.mesh.faces_iter())) == 1
