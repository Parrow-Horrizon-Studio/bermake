"""Read-only model queries backing Select All and Zoom Extents (M7.2 Task 6)."""

from __future__ import annotations

import numpy as np
from bermake.model.model_queries import model_bounds, select_all_ids


def _unit_square(scene):
    """Four vertices, four edges, one face on the z=0 plane, 0..1 in x and y."""
    v = [
        scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32)),
    ]
    scene.add_face_from_loop(v)
    return v


def test_select_all_collects_the_active_context(model_factory):
    model = model_factory()
    _unit_square(model.active_context.mesh)

    edges, faces, instances = select_all_ids(model)

    assert len(edges) == 4
    assert len(faces) == 1
    assert instances == set()


def test_select_all_on_an_empty_model_returns_empty_sets(model_factory):
    edges, faces, instances = select_all_ids(model_factory())
    assert (edges, faces, instances) == (set(), set(), set())


def test_select_all_returns_instances_at_the_active_level(model_factory, group_factory):
    model = model_factory()
    _unit_square(model.active_context.mesh)
    instance = group_factory(model)  # wraps the loose geometry into one group

    _edges, _faces, instances = select_all_ids(model)

    assert instances == {instance.id}


def test_select_all_excludes_hidden_tag_instances(model_factory, group_factory):
    model = model_factory()
    _unit_square(model.active_context.mesh)
    instance = group_factory(model)
    hidden_tag = model.tags.add("Hidden")
    instance.tag_id = hidden_tag.id
    model.tags.set_visible(hidden_tag.id, False)

    _edges, _faces, instances = select_all_ids(model)

    assert instances == set()


def test_select_all_does_not_reach_outside_the_active_context(model_factory, group_factory):
    model = model_factory()
    _unit_square(model.active_context.mesh)
    group_factory(model)  # lifts the unit square into a child definition

    edges, faces, _instances = select_all_ids(model)

    # The root context's own geometry was lifted out; select_all_ids must not
    # reach into the child definition's mesh to "find" it again.
    assert edges == set()
    assert faces == set()


def test_select_all_scoped_to_a_non_root_active_context(model_factory, group_factory):
    """A regression that read model.root instead of model.active_context would
    pass every other test in this file (they all leave active_context at
    root). Enter a group and prove Select All follows the active context: it
    must return exactly that group's own geometry, not the root's."""
    model = model_factory()
    _unit_square(model.active_context.mesh)  # becomes the group's own geometry
    instance = group_factory(model)  # lifts the unit square into a new group
    group_def = instance.definition

    # Distinct, differently-shaped geometry left behind at the root level
    # (a triangle: 3 edges, vs the group's unit square: 4 edges -- so the two
    # id sets can never coincide by accident, even under id reuse).
    root_verts = [
        model.root.mesh.add_vertex(np.array([10.0, 10.0, 0.0], dtype=np.float32)),
        model.root.mesh.add_vertex(np.array([11.0, 10.0, 0.0], dtype=np.float32)),
        model.root.mesh.add_vertex(np.array([11.0, 11.0, 0.0], dtype=np.float32)),
    ]
    model.root.mesh.add_face_from_loop(root_verts)

    root_edges = {e.id for e in model.root.mesh.edges_iter()}
    root_faces = {f.id for f in model.root.mesh.faces_iter()}
    group_edges = {e.id for e in group_def.mesh.edges_iter()}
    group_faces = {f.id for f in group_def.mesh.faces_iter()}

    model.enter(instance)
    edges, faces, _instances = select_all_ids(model)

    assert edges == group_edges
    assert faces == group_faces
    assert edges != root_edges
    assert faces != root_faces


def test_bounds_of_a_unit_square(model_factory):
    model = model_factory()
    _unit_square(model.active_context.mesh)

    bounds = model_bounds(model)

    assert bounds is not None
    lo, hi = bounds
    assert np.allclose(lo, [0.0, 0.0, 0.0])
    assert np.allclose(hi, [1.0, 1.0, 0.0])


def test_bounds_of_an_empty_model_is_none(model_factory):
    assert model_bounds(model_factory()) is None


def test_bounds_transforms_geometry_inside_a_transformed_instance(model_factory):
    from bermake.geometry.transforms import mat_translate

    model = model_factory()
    child = model.new_definition("Child", is_group=True)
    _unit_square(child.mesh)
    inst = model.new_instance(child, transform=mat_translate((5.0, 5.0, 5.0)))
    model.root.children.append(inst)

    bounds = model_bounds(model)

    assert bounds is not None
    lo, hi = bounds
    assert np.allclose(lo, [5.0, 5.0, 5.0])
    assert np.allclose(hi, [6.0, 6.0, 5.0])


def test_bounds_excludes_geometry_under_a_hidden_tag(model_factory):
    """model_bounds walks traverse_visible(), which prunes hidden-tag
    subtrees. A regression that swapped in traverse() (all definitions,
    tag visibility ignored) would leave every other bounds test unchanged --
    none of them hide a tag -- so this must exercise that path directly."""
    from bermake.geometry.transforms import mat_translate

    model = model_factory()

    visible_def = model.new_definition("Visible", is_group=True)
    _unit_square(visible_def.mesh)
    visible_inst = model.new_instance(visible_def)
    model.root.children.append(visible_inst)

    hidden_def = model.new_definition("Hidden", is_group=True)
    _unit_square(hidden_def.mesh)
    hidden_inst = model.new_instance(hidden_def, transform=mat_translate((100.0, 100.0, 100.0)))
    model.root.children.append(hidden_inst)

    hidden_tag = model.tags.add("Hidden")
    hidden_inst.tag_id = hidden_tag.id
    model.tags.set_visible(hidden_tag.id, False)

    bounds = model_bounds(model)

    assert bounds is not None
    lo, hi = bounds
    # The visible unit square's box only -- if the hidden instance at
    # (100, 100, 100) leaked in, hi would be far past [1, 1, 0].
    assert np.allclose(lo, [0.0, 0.0, 0.0])
    assert np.allclose(hi, [1.0, 1.0, 0.0])
