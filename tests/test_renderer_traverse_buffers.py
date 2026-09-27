"""Headless tests for the per-definition buffer data path added in M4e Task 11.

These tests exercise the data path that feeds the renderer (Model.traverse())
without requiring a GL context. GL draw calls cannot be unit-tested headlessly;
the pytest-qt tests in test_viewport.py cover construction.
"""

from __future__ import annotations

import numpy as np

from bermake.model.model import Model


def test_model_traverse_provides_drawable_pairs():
    """Renderer receives (definition, world_transform) pairs with correct transforms."""
    m = Model()
    m.root.mesh.add_vertex(np.array([0, 0, 0], np.float32))
    g = m.new_definition("G", is_group=True)
    g.mesh.add_vertex(np.array([0, 0, 0], np.float32))
    t = np.eye(4); t[:3, 3] = [4, 0, 0]
    inst = m.new_instance(g, t)
    m.root.children.append(inst)
    pairs = list(m.traverse())
    assert len(pairs) == 2
    # Root definition is first with identity transform.
    root_def, root_world = pairs[0]
    assert root_def is m.root
    assert np.allclose(root_world, np.eye(4))
    # Child definition carries the instance transform.
    child_def, child_world = pairs[1]
    assert child_def is g
    # The renderer will draw g's buffer with this model matrix:
    assert np.allclose(child_world[:3, 3], [4, 0, 0])


def test_traverse_root_only_single_pair():
    """A model with no children yields exactly one (root, identity) pair."""
    m = Model()
    pairs = list(m.traverse())
    assert len(pairs) == 1
    d, t = pairs[0]
    assert d is m.root
    assert np.allclose(t, np.eye(4))


def test_traverse_nested_transforms_accumulate():
    """Nested instances accumulate transforms depth-first (parent @ child)."""
    m = Model()
    # parent group at [1, 0, 0]
    parent_def = m.new_definition("Parent", is_group=True)
    t_parent = np.eye(4); t_parent[:3, 3] = [1, 0, 0]
    parent_inst = m.new_instance(parent_def, t_parent)
    m.root.children.append(parent_inst)

    # child group at [0, 2, 0] relative to parent → world [1, 2, 0]
    child_def = m.new_definition("Child", is_group=True)
    t_child = np.eye(4); t_child[:3, 3] = [0, 2, 0]
    child_inst = m.new_instance(child_def, t_child)
    parent_def.children.append(child_inst)

    pairs = list(m.traverse())
    # root, parent, child
    assert len(pairs) == 3
    defs = [d for d, _ in pairs]
    assert defs == [m.root, parent_def, child_def]
    # Child world translation: parent [1,0,0] + child [0,2,0] = [1,2,0]
    child_world = pairs[2][1]
    assert np.allclose(child_world[:3, 3], [1, 2, 0])


def test_scene_renderer_has_def_buffers_cache():
    """SceneRenderer exposes _def_buffers dict (per-definition GL buffer cache)."""
    from bermake.viewport.scene_renderer import SceneRenderer

    renderer = SceneRenderer()
    assert hasattr(renderer, "_def_buffers")
    assert isinstance(renderer._def_buffers, dict)
    assert len(renderer._def_buffers) == 0  # empty before any GL init


def test_scene_renderer_render_accepts_model_param():
    """render() signature uses 'model' not 'scene'."""
    import inspect
    from bermake.viewport.scene_renderer import SceneRenderer

    sig = inspect.signature(SceneRenderer.render)
    assert "model" in sig.parameters
    assert "scene" not in sig.parameters
