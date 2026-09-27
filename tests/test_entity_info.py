"""Entity Info summaries, selection bounds, and area formatting (M7.3 Task 6)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from bermake.model.entity_info import entity_summary
from bermake.model.model_queries import selection_bounds
from bermake.selection import Selection
from bermake.units import Units, UnitSystem, format_area


def _square(model, size=1.0):
    scene = model.active_context.mesh
    v = [
        scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([size, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([size, size, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([0.0, size, 0.0], dtype=np.float32)),
    ]
    return scene.add_face_from_loop(v)


def _right_triangle(model, leg=2.0):
    """A right triangle whose own vertex hull is NOT its local AABB.

    Its local bounding box is the full `leg x leg` square, but the triangle
    itself only occupies half of that box (there is no vertex at the
    box's fourth corner). That makes it a discriminating fixture for
    rotated-extent tests: an implementation that (incorrectly) transforms
    the local AABB's corners drags the box's untouched fourth corner along
    for the ride and reports a larger rotated extent than the triangle's
    actual vertices produce.
    """
    scene = model.active_context.mesh
    v = [
        scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([leg, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([0.0, leg, 0.0], dtype=np.float32)),
    ]
    return scene.add_face_from_loop(v)


def _tilted_square(model):
    """A unit square lying in a plane that is not aligned to any coordinate
    plane, built from a 3-4-5 triangle so the coordinates -- and the exact
    area -- are exact rationals rather than trig approximations.

    Starts as a unit square in the XZ-plane (y=0) and rotates it about the
    X-axis by the 3-4-5 angle (sin=3/5, cos=4/5). Rotation is an isometry,
    so the tilted square's true area is still exactly 1 -- but its normal
    (0, -4/5, -3/5) has no zero component along Y or Z, so it is genuinely
    tilted rather than flat in a coordinate plane.
    """
    scene = model.active_context.mesh
    v = [
        scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, -0.6, 0.8], dtype=np.float32)),
        scene.add_vertex(np.array([0.0, -0.6, 0.8], dtype=np.float32)),
    ]
    return scene.add_face_from_loop(v)


def test_nothing_selected(model_factory):
    summary = entity_summary(model_factory(), Selection())
    assert summary.kind == "Nothing"
    assert summary.count == 0


def test_one_face_reports_its_area_and_material(model_factory):
    model = model_factory()
    face_id = _square(model, size=2.0)
    selection = Selection()
    selection.replace(faces=[face_id])

    summary = entity_summary(model, selection)

    assert summary.kind == "Face"
    assert summary.count == 1
    assert summary.area == pytest.approx(4.0)
    assert summary.material_id == model.active_context.mesh.face_material(face_id)
    assert summary.length is None


def test_tilted_face_reports_its_true_area(model_factory):
    # _polygon_area (Newell's method) is correct for a polygon in any plane,
    # but the only other area test lies flat in the XY plane, so a sign/axis
    # bug in picking which plane to project onto would slip through. This
    # face is a unit square tilted out of every coordinate plane; rotation
    # is an isometry, so its true area is still exactly 1 despite the tilt.
    model = model_factory()
    face_id = _tilted_square(model)
    selection = Selection()
    selection.replace(faces=[face_id])

    summary = entity_summary(model, selection)

    assert summary.area == pytest.approx(1.0)


def test_one_edge_reports_its_length(model_factory):
    model = model_factory()
    _square(model, size=3.0)
    scene = model.active_context.mesh
    edge_id = next(iter(scene.edges_iter())).id
    selection = Selection()
    selection.replace(edges=[edge_id])

    summary = entity_summary(model, selection)

    assert summary.kind == "Edge"
    assert summary.length == pytest.approx(3.0)
    assert summary.area is None


def test_one_group_reports_name_definition_and_counts(model_factory, group_factory):
    model = model_factory()
    _square(model)
    instance = group_factory(model)
    instance.definition.name = "Wall"
    instance.name = "North"
    selection = Selection()
    selection.replace(instances=[instance.id])

    summary = entity_summary(model, selection)

    assert summary.kind == "Group"
    assert summary.name == "North"
    assert summary.definition_name == "Wall"
    assert summary.edge_count == 4
    assert summary.face_count == 1
    assert summary.child_count == 0
    assert summary.hidden is False
    assert summary.tag_id == instance.tag_id


def test_mixed_kinds_report_mixed(model_factory, group_factory):
    model = model_factory()
    face_id = _square(model)
    instance = group_factory(model)
    _square(model)
    selection = Selection()
    selection.replace(faces=[face_id], instances=[instance.id])

    summary = entity_summary(model, selection)

    assert summary.kind == "Mixed"
    # A cross-type selection (instances + faces here) is a DIFFERENT "Mixed"
    # from a same-type-but-heterogeneous instance selection (see the test
    # below) -- Entity Info's Tag/Hidden fields must stay disabled for this
    # one, and instances_only is what tells the two apart.
    assert summary.instances_only is False


def test_mixed_group_and_component_reports_instances_only(model_factory, group_factory):
    # A Group and a Component together are the SAME selection shape
    # (instances only) but heterogeneous definitions -- _instance_kind
    # collapses that to "Mixed" too, just like the cross-type case above.
    # instances_only is what lets Entity Info tell them apart: Tag and
    # Hidden apply to any instance selection (spec 1.4), homogeneous or not.
    model = model_factory()
    _square(model)
    group = group_factory(model)
    component_def = model.new_definition("Chair", is_group=False)
    component = model.new_instance(component_def)
    model.active_context.children.append(component)
    selection = Selection()
    selection.replace(instances=[group.id, component.id])

    summary = entity_summary(model, selection)

    assert summary.kind == "Mixed"
    assert summary.instances_only is True


def test_hidden_is_none_when_the_selection_disagrees(model_factory, group_factory):
    model = model_factory()
    _square(model)
    first = group_factory(model)
    _square(model)
    second = group_factory(model)
    second.hidden = True
    selection = Selection()
    selection.replace(instances=[first.id, second.id])

    summary = entity_summary(model, selection)

    assert summary.count == 2
    assert summary.hidden is None
    # Name is single-instance only: a shared name across a multi-selection has
    # no meaning, so the field must report nothing rather than pick one.
    assert summary.name is None


def test_tag_id_is_none_when_the_selection_disagrees(model_factory, group_factory):
    model = model_factory()
    _square(model)
    first = group_factory(model)
    _square(model)
    second = group_factory(model)
    second.tag_id = model.tags.add("Walls").id
    selection = Selection()
    selection.replace(instances=[first.id, second.id])

    assert entity_summary(model, selection).tag_id is None


def test_selection_bounds_is_none_without_instances(model_factory):
    assert selection_bounds(model_factory(), Selection()) is None


def test_selection_bounds_measures_transformed_extents(model_factory, group_factory):
    # A plain square is a bad fixture here: its own vertices ARE its local
    # AABB's corners, so a spec-violating "transform the local AABB's 8
    # corners" shortcut produces the identical result to the correct
    # "transform every actual vertex" approach for a square. A right
    # triangle discriminates: its local AABB is the full 2x2 square, but the
    # triangle only occupies half of it (see _right_triangle). Rotating the
    # box's untouched fourth corner along with the real vertices inflates
    # the AABB-shortcut's Y-extent to 2*sqrt(2); the true vertex-derived
    # Y-extent is only sqrt(2). (X happens to agree between the two
    # approaches for this particular triangle -- Y is what actually catches
    # the bug.)
    model = model_factory()
    _right_triangle(model, leg=2.0)
    instance = group_factory(model)
    angle = math.radians(45.0)
    rotation = np.eye(4, dtype=np.float64)
    rotation[0, 0] = math.cos(angle)
    rotation[0, 1] = -math.sin(angle)
    rotation[1, 0] = math.sin(angle)
    rotation[1, 1] = math.cos(angle)
    instance.transform = rotation
    selection = Selection()
    selection.replace(instances=[instance.id])

    lo, hi = selection_bounds(model, selection)

    assert float(hi[0] - lo[0]) == pytest.approx(2.0 * math.sqrt(2.0), abs=1e-4)
    assert float(hi[1] - lo[1]) == pytest.approx(math.sqrt(2.0), abs=1e-4)


def test_summary_size_comes_from_selection_bounds(model_factory, group_factory):
    model = model_factory()
    _square(model, size=2.0)
    instance = group_factory(model)
    selection = Selection()
    selection.replace(instances=[instance.id])

    summary = entity_summary(model, selection)

    assert summary.size == pytest.approx((2.0, 2.0, 0.0), abs=1e-5)


def test_format_area_metric_is_always_square_metres():
    # mm2 / cm2 produce unreadable magnitudes for architectural faces, so the
    # metric unit setting deliberately does not apply here.
    assert format_area(4.0, Units(system=UnitSystem.METRIC, metric_unit="mm")) == "4.00 m²"


def test_format_area_imperial_is_square_feet():
    assert format_area(1.0, Units(system=UnitSystem.IMPERIAL)) == "10.76 ft²"
