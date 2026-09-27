# tests/test_explode_command.py
import numpy as np
from bermake.model.model import Model
from bermake.commands.explode_command import ExplodeInstanceCommand


def test_explode_bakes_geometry_into_parent_at_transformed_positions():
    m = Model()
    d = m.new_definition("G", is_group=True)
    d.mesh.add_vertex(np.array([0, 0, 0], np.float32))
    d.mesh.add_vertex(np.array([1, 0, 0], np.float32))
    t = np.eye(4); t[:3, 3] = [10, 0, 0]
    inst = m.new_instance(d, t)
    m.root.children.append(inst)

    cmd = ExplodeInstanceCommand(m.root, inst)
    cmd.do(m)

    assert inst not in m.root.children
    xs = sorted(float(v.position[0]) for v in m.root.mesh.vertices_iter())
    assert np.allclose(xs, [10.0, 11.0])  # baked by the +10 translation


def test_explode_undo_restores_instance_and_clears_parent():
    m = Model()
    d = m.new_definition("G", is_group=True)
    d.mesh.add_vertex(np.array([0, 0, 0], np.float32))
    inst = m.new_instance(d, np.eye(4))
    m.root.children.append(inst)
    cmd = ExplodeInstanceCommand(m.root, inst)
    cmd.do(m)
    cmd.undo(m)
    assert inst in m.root.children
    assert list(m.root.mesh.vertices_iter()) == []


def test_explode_redo_round_trips():
    """do -> undo -> do (redo) -> undo: parent mesh should be correctly baked on each do."""
    m = Model()
    d = m.new_definition("G", is_group=True)
    d.mesh.add_vertex(np.array([0, 0, 0], np.float32))
    d.mesh.add_vertex(np.array([1, 0, 0], np.float32))
    t = np.eye(4); t[:3, 3] = [10, 0, 0]
    inst = m.new_instance(d, t)
    m.root.children.append(inst)

    cmd = ExplodeInstanceCommand(m.root, inst)

    # First do
    cmd.do(m)
    assert inst not in m.root.children
    xs = sorted(float(v.position[0]) for v in m.root.mesh.vertices_iter())
    assert np.allclose(xs, [10.0, 11.0])

    # Undo
    cmd.undo(m)
    assert inst in m.root.children
    assert list(m.root.mesh.vertices_iter()) == []

    # Redo (second do)
    cmd.do(m)
    assert inst not in m.root.children
    xs_redo = sorted(float(v.position[0]) for v in m.root.mesh.vertices_iter())
    assert np.allclose(xs_redo, [10.0, 11.0])

    # Undo again
    cmd.undo(m)
    assert inst in m.root.children
    assert list(m.root.mesh.vertices_iter()) == []


def test_explode_reparents_child_instance():
    """G has a nested child c (+1x). G's instance is placed at +10x.
    After do: c is in parent.children with composed x~11.
    After undo: c is back to x~1, NOT in parent.children, G's instance is back.
    """
    m = Model()

    # Build definition G with one nested child c
    d_child = m.new_definition("C", is_group=False)
    t_child = np.eye(4); t_child[:3, 3] = [1, 0, 0]
    c = m.new_instance(d_child, t_child)

    d_g = m.new_definition("G", is_group=True)
    d_g.children.append(c)

    # Place G's instance at +10x
    t_g = np.eye(4); t_g[:3, 3] = [10, 0, 0]
    inst_g = m.new_instance(d_g, t_g)
    m.root.children.append(inst_g)

    cmd = ExplodeInstanceCommand(m.root, inst_g)
    cmd.do(m)

    # c should now be in parent (root) children
    assert c in m.root.children
    # c's transform x component should be ~11 (10 + 1 composed)
    assert np.isclose(float(c.transform[0, 3]), 11.0, atol=1e-5)
    # inst_g should be gone from parent
    assert inst_g not in m.root.children

    cmd.undo(m)

    # c should no longer be in root.children
    assert c not in m.root.children
    # c's transform x component should be back to ~1
    assert np.isclose(float(c.transform[0, 3]), 1.0, atol=1e-5)
    # inst_g should be back
    assert inst_g in m.root.children


def test_explode_undo_removes_baked_edges_and_faces():
    m = Model()
    d = m.new_definition("G", is_group=True)
    a = d.mesh.add_vertex(np.array([0, 0, 0], np.float32))
    b = d.mesh.add_vertex(np.array([1, 0, 0], np.float32))
    c = d.mesh.add_vertex(np.array([0, 1, 0], np.float32))
    d.mesh.add_face_from_loop([a, b, c])  # triangle: 3 verts, 3 edges, 1 face
    t = np.eye(4); t[:3, 3] = [10, 0, 0]
    inst = m.new_instance(d, t)
    m.root.children.append(inst)

    cmd = ExplodeInstanceCommand(m.root, inst)
    cmd.do(m)
    # baked into parent:
    assert len(list(m.root.mesh.faces_iter())) == 1
    assert len(list(m.root.mesh.edges_iter())) == 3
    cmd.undo(m)
    # undo must FULLY clear the baked geometry — this fails before the fix:
    assert list(m.root.mesh.faces_iter()) == []
    assert list(m.root.mesh.edges_iter()) == []
    assert list(m.root.mesh.vertices_iter()) == []
    assert inst in m.root.children
