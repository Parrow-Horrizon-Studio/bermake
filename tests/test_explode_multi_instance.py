# tests/test_explode_multi_instance.py
"""Regression guard: exploding one instance of a SHARED (multi-instance)
definition must not corrupt the sibling instances.

Fix: _on_explode (main_window.py) first issues MakeUniqueCommand to detach the
definition, then ExplodeInstanceCommand — as one CompositeCommand.
This test replicates that exact composite at the model/command level (no Qt).
"""
from __future__ import annotations

import numpy as np

from bermake.commands import CommandStack, CompositeCommand
from bermake.commands.explode_command import ExplodeInstanceCommand
from bermake.commands.instance_lifecycle_commands import MakeUniqueCommand
from bermake.model.model import Model


def _build_scene():
    """Model with definition d (triangle) and TWO instances a, b of d.
    b has a +10x translation.
    Returns (model, d, a, b).
    """
    m = Model()
    d = m.new_definition("Shared", is_group=True)
    v0 = d.mesh.add_vertex(np.array([0.0, 0.0, 0.0], np.float32))
    v1 = d.mesh.add_vertex(np.array([1.0, 0.0, 0.0], np.float32))
    v2 = d.mesh.add_vertex(np.array([0.0, 1.0, 0.0], np.float32))
    d.mesh.add_face_from_loop([v0, v1, v2])

    a = m.new_instance(d, np.eye(4, dtype=np.float64))
    m.root.children.append(a)

    t_b = np.eye(4, dtype=np.float64)
    t_b[:3, 3] = [10.0, 0.0, 0.0]
    b = m.new_instance(d, t_b)
    m.root.children.append(b)

    return m, d, a, b


def test_sibling_untouched_after_explode():
    """Exploding b must leave a + its definition d intact."""
    m, d, a, b = _build_scene()
    assert len(d.instances) == 2

    stack = CommandStack()
    composite = CompositeCommand(name="Explode", children=[
        MakeUniqueCommand(b),
        ExplodeInstanceCommand(m.root, b),
    ])
    stack.execute(composite, m)

    # a is still a root child
    assert a in m.root.children
    # a still points at the original definition d
    assert a.definition is d
    # b is gone from root children
    assert b not in m.root.children

    # The triangle is baked into root.mesh at the +10x position
    xs = sorted(float(v.position[0]) for v in m.root.mesh.vertices_iter())
    assert len(xs) == 3
    assert np.allclose(sorted(xs), [10.0, 10.0, 11.0], atol=1e-5)

    # The original definition d still has its untouched triangle
    d_verts = sorted(float(v.position[0]) for v in d.mesh.vertices_iter())
    assert np.allclose(sorted(d_verts), [0.0, 0.0, 1.0], atol=1e-5)


def test_undo_restores_two_instances():
    """After undo the model must return to 2 instances of d with empty root mesh."""
    m, d, a, b = _build_scene()

    stack = CommandStack()
    composite = CompositeCommand(name="Explode", children=[
        MakeUniqueCommand(b),
        ExplodeInstanceCommand(m.root, b),
    ])
    stack.execute(composite, m)
    stack.undo()

    # Both instances back in root
    assert a in m.root.children
    assert b in m.root.children
    # b's definition is restored to the shared d
    assert b.definition is d
    assert len(d.instances) == 2
    # root mesh is clear
    assert list(m.root.mesh.vertices_iter()) == []
    assert list(m.root.mesh.faces_iter()) == []


# --- #56: exploding an instance must not leave its NESTED child instances
# double-owned by the source definition (distinct from the sibling-instance
# corruption covered above, which is guarded by MakeUniqueCommand). ---


def test_explode_removes_children_from_source_definition():
    model = Model()
    # A component definition that itself contains one child instance.
    inner = model.new_definition("Inner", is_group=False)
    comp = model.new_definition("Comp", is_group=False)
    comp.children.append(model.new_instance(inner, np.eye(4, dtype=np.float64)))

    # Place the SAME component twice in the root.
    a = model.new_instance(comp, np.eye(4, dtype=np.float64))
    b = model.new_instance(comp, np.eye(4, dtype=np.float64))
    model.root.children.extend([a, b])
    assert len(comp.children) == 1

    stack = CommandStack()
    stack.execute(ExplodeInstanceCommand(model.root, a), model)

    # The child moved to the root; the definition must no longer own it,
    # otherwise instance `b` still renders it too (double ownership).
    assert len(comp.children) == 0
    assert len(model.root.children) == 2      # b + the reparented child


def test_explode_undo_restores_children_to_the_definition():
    model = Model()
    inner = model.new_definition("Inner", is_group=False)
    comp = model.new_definition("Comp", is_group=False)
    child = model.new_instance(inner, np.eye(4, dtype=np.float64))
    comp.children.append(child)
    a = model.new_instance(comp, np.eye(4, dtype=np.float64))
    model.root.children.append(a)

    stack = CommandStack()
    stack.execute(ExplodeInstanceCommand(model.root, a), model)
    stack.undo()

    assert comp.children == [child]
    assert a in model.root.children


def test_explode_redo_reclears_definition_children():
    """Redo re-runs do(), which re-captures defn.children — it must re-clear
    the definition too, not just re-append to the parent."""
    model = Model()
    inner = model.new_definition("Inner", is_group=False)
    comp = model.new_definition("Comp", is_group=False)
    child = model.new_instance(inner, np.eye(4, dtype=np.float64))
    comp.children.append(child)
    a = model.new_instance(comp, np.eye(4, dtype=np.float64))
    b = model.new_instance(comp, np.eye(4, dtype=np.float64))
    model.root.children.extend([a, b])

    stack = CommandStack()
    stack.execute(ExplodeInstanceCommand(model.root, a), model)
    stack.undo()
    stack.redo()

    assert len(comp.children) == 0
    assert child in model.root.children
