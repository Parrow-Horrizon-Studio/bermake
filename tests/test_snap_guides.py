"""Unit tests for guide inference (M7.6b Task 8): ON_GUIDE, GUIDE_POINT, and
guide-crossing INTERSECTION, plus the show_guides gate that keeps a hidden
guide from snapping at all.

Follows tests/test_snap_engine.py's helper style: _camera_at_default,
_screen_of, imports inside the test bodies.
"""

from __future__ import annotations

import math

import numpy as np


def _camera_at_default():
    from bermake.viewport.camera import Camera

    cam = Camera()
    cam.aspect = 1280.0 / 800.0
    return cam


def _screen_of(cam, world):
    """Pixel coords that project onto `world` in a 1280x800 viewport."""
    sx, sy, _ = cam.world_to_screen(np.asarray(world, dtype=np.float32), 1280, 800)
    return (sx, sy)


def test_cursor_on_guide_line_yields_on_guide():
    from bermake.scene import Scene
    from bermake.viewport.snap_engine import SnapEngine, SnapKind

    eng = SnapEngine()
    scene = Scene()
    cam = _camera_at_default()
    origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cursor = _screen_of(cam, [0.0, 0.0, 5.0])  # a point on the guide, off the origin
    res = eng.snap(
        cursor, (1280, 800), cam, scene, guides=[(origin, direction)], guide_points=[]
    )
    assert res.kind == SnapKind.ON_GUIDE
    np.testing.assert_allclose(res.world_position, [0.0, 0.0, 5.0], atol=1e-3)


def test_cursor_on_guide_point_yields_guide_point():
    from bermake.scene import Scene
    from bermake.viewport.snap_engine import SnapEngine, SnapKind

    eng = SnapEngine()
    scene = Scene()
    cam = _camera_at_default()
    point = [3.0, 1.0, 2.0]
    cursor = _screen_of(cam, point)
    res = eng.snap(cursor, (1280, 800), cam, scene, guides=[], guide_points=[point])
    assert res.kind == SnapKind.GUIDE_POINT
    np.testing.assert_allclose(res.world_position, point, atol=1e-3)


def test_guide_point_beats_on_edge_but_loses_to_endpoint():
    from bermake.scene import Scene
    from bermake.viewport.snap_engine import SnapEngine, SnapKind

    cam = _camera_at_default()

    # A guide point sitting ON a live edge, away from its midpoint and ends,
    # so ON_EDGE and GUIDE_POINT both land in tolerance at the same pixel.
    eng = SnapEngine()
    scene = Scene()
    v0 = scene.add_vertex(np.array([0.0, 0.0, 2.0], dtype=np.float32))
    v1 = scene.add_vertex(np.array([4.0, 0.0, 2.0], dtype=np.float32))
    scene.add_edge(v0, v1)
    guide_point = [1.0, 0.0, 2.0]  # on the segment, not the midpoint (2,0,2)
    cursor = _screen_of(cam, guide_point)
    res = eng.snap(cursor, (1280, 800), cam, scene, guides=[], guide_points=[guide_point])
    assert res.kind == SnapKind.GUIDE_POINT

    # The same point, now also a real vertex: ENDPOINT must win.
    eng2 = SnapEngine()
    scene2 = Scene()
    vid = scene2.add_vertex(np.array(guide_point, dtype=np.float32))
    res2 = eng2.snap(cursor, (1280, 800), cam, scene2, guides=[], guide_points=[guide_point])
    assert res2.kind == SnapKind.ENDPOINT
    assert res2.vertex_id == vid


def test_on_edge_beats_on_guide_when_only_those_two_compete():
    """D12's real-geometry-outranks-construction-geometry rule, isolated.

    An edge you can actually build on beats a guide drawn only to help you
    aim: ON_EDGE outranks ON_GUIDE. Task 8's other precedence tests establish
    this only as a side effect of a stronger kind (INTERSECTION) beating
    both at once, which never exercises the ON_EDGE-versus-ON_GUIDE relation
    itself -- a mutant that ranked ON_GUIDE above ON_EDGE would pass every
    other test in this file. This test puts only those two candidates on the
    board.

    The scene edge and the guide are placed skew to each other (no real 3D
    crossing, so no INTERSECTION candidate), but each is positioned to cross
    the SAME camera ray at a different depth: any point on that ray
    reprojects to the exact same pixel, so both an ON_EDGE and an ON_GUIDE
    candidate land in tolerance at the cursor with nothing else competing --
    the edge is built asymmetrically around its crossing point so that point
    is neither the segment's midpoint nor either endpoint, ruling out
    MIDPOINT and ENDPOINT too.
    """
    from bermake.scene import Scene
    from bermake.viewport.snap_engine import SnapEngine, SnapKind

    cam = _camera_at_default()
    cursor = _screen_of(cam, [4.0, 0.0, 0.0])
    ray_origin, ray_dir = cam.ray_from_screen(cursor[0], cursor[1], 1280, 800)
    ray_origin = np.asarray(ray_origin, dtype=np.float64)
    ray_dir = np.asarray(ray_dir, dtype=np.float64)

    guide_origin = ray_origin + 10.0 * ray_dir
    guide_direction = np.array([0.0, 1.0, 0.0])

    edge_crossing = ray_origin + 15.0 * ray_dir
    edge_direction = np.array([0.0, 0.0, 1.0])
    # Asymmetric around the crossing point: not the midpoint, not an end.
    p1 = edge_crossing - edge_direction * 0.5
    p2 = edge_crossing + edge_direction * 3.5

    eng = SnapEngine()
    scene = Scene()
    va = scene.add_vertex(p1.astype(np.float32))
    vb = scene.add_vertex(p2.astype(np.float32))
    eid = scene.add_edge(va, vb)

    res = eng.snap(
        cursor,
        (1280, 800),
        cam,
        scene,
        guides=[(guide_origin, guide_direction)],
        guide_points=[],
    )
    assert res.kind == SnapKind.ON_EDGE
    assert res.edge_id == eid


def test_midpoint_beats_guide_point_when_both_compete():
    """D12: inserting GUIDE_POINT between MIDPOINT and ON_EDGE created a new
    adjacent pair on MIDPOINT's other side that nothing previously pinned.
    Real geometry (an edge's own midpoint) still outranks a construction
    point placed at the same spot.

    Placing the guide point exactly at the edge's geometric midpoint also
    puts an ON_EDGE candidate at that identical world point (the midpoint
    IS a point on the segment, so the ray through it is the segment's own
    closest approach too). That does not weaken this test: ON_EDGE already
    ranks below both MIDPOINT and GUIDE_POINT and stays there regardless of
    which of those two wins, so its presence cannot mask a MIDPOINT/
    GUIDE_POINT reordering the way INTERSECTION masked ON_EDGE/ON_GUIDE in
    the fix-round-1 tests.
    """
    from bermake.scene import Scene
    from bermake.viewport.snap_engine import SnapEngine, SnapKind

    eng = SnapEngine()
    scene = Scene()
    cam = _camera_at_default()
    v0 = scene.add_vertex(np.array([0.0, 0.0, 2.0], dtype=np.float32))
    v1 = scene.add_vertex(np.array([4.0, 0.0, 2.0], dtype=np.float32))
    scene.add_edge(v0, v1)
    midpoint = [2.0, 0.0, 2.0]
    cursor = _screen_of(cam, midpoint)
    res = eng.snap(cursor, (1280, 800), cam, scene, guides=[], guide_points=[midpoint])
    assert res.kind == SnapKind.MIDPOINT


def test_on_guide_beats_perpendicular_when_only_those_two_compete():
    """D12: inserting ON_GUIDE between ON_EDGE and PERPENDICULAR created a
    second new adjacent pair, on ON_GUIDE's other side, that nothing
    previously pinned. Guides outrank the directional inferences: a
    construction line the user placed still beats a direction merely
    inferred from an acquired edge.

    The guide is placed exactly along the same infinite line the
    PERPENDICULAR inference resolves to (through the anchor, perpendicular
    to the acquired edge within the drawing plane), so both land on the
    cursor with nothing else in tolerance to interfere: the anchor also
    sits on the world Y axis here, so an AXIS_LOCK candidate tags along too,
    but it ranks below both ON_GUIDE and PERPENDICULAR and stays there
    regardless of their relative order, so it cannot mask this pair either.
    """
    from bermake.scene import Scene
    from bermake.viewport.inference import Acquired, AcquiredKind
    from bermake.viewport.snap_engine import SnapEngine, SnapKind

    eng = SnapEngine()
    scene = Scene()
    cam = _camera_at_default()
    anchor = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    acquired = Acquired(
        kind=AcquiredKind.EDGE,
        position=np.array([3.0, 0.0, 0.0], dtype=np.float64),
        direction=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        entity_id=0,
    )
    # Perpendicular to (1,0,0) in the Z-normal plane runs along (0,1,0),
    # through the anchor -- the same infinite line as this guide.
    guide_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    guide_direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    probe = [0.0, 3.0, 0.0]
    cursor = _screen_of(cam, probe)

    res = eng.snap(
        cursor,
        (1280, 800),
        cam,
        scene,
        anchor=anchor,
        acquired=acquired,
        plane_normal=np.array([0.0, 0.0, 1.0]),
        guides=[(guide_origin, guide_direction)],
        guide_points=[],
    )
    assert res.kind == SnapKind.ON_GUIDE


def test_guide_crossing_scene_edge_yields_intersection_not_on_guide():
    from bermake.scene import Scene
    from bermake.viewport.snap_engine import SnapEngine, SnapKind

    cam = _camera_at_default()
    origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)  # the guide is the X axis

    # A segment that genuinely crosses the guide within its own bounds.
    eng = SnapEngine()
    scene = Scene()
    va = scene.add_vertex(np.array([2.0, -1.0, 0.0], dtype=np.float32))
    vb = scene.add_vertex(np.array([2.0, 1.0, 0.0], dtype=np.float32))
    eid = scene.add_edge(va, vb)
    crossing = [2.0, 0.0, 0.0]
    cursor = _screen_of(cam, crossing)
    res = eng.snap(
        cursor, (1280, 800), cam, scene, guides=[(origin, direction)], guide_points=[]
    )
    assert res.kind == SnapKind.INTERSECTION
    assert res.edge_id == eid
    assert abs(res.edge_t - 0.5) < 1e-3
    np.testing.assert_allclose(res.world_position, crossing, atol=1e-3)

    # A segment whose INFINITE line crosses the guide at the same screen
    # point, but only outside its own [0, 1] span: the edge's own bound must
    # still be enforced for a guide crossing, so this must NOT be reported
    # as an intersection -- the cursor is on the (unobstructed) guide only.
    eng2 = SnapEngine()
    scene2 = Scene()
    va2 = scene2.add_vertex(np.array([2.0, 3.0, 0.0], dtype=np.float32))
    vb2 = scene2.add_vertex(np.array([2.0, 5.0, 0.0], dtype=np.float32))
    scene2.add_edge(va2, vb2)
    res2 = eng2.snap(
        cursor, (1280, 800), cam, scene2, guides=[(origin, direction)], guide_points=[]
    )
    assert res2.kind == SnapKind.ON_GUIDE


def test_two_guides_crossing_yields_intersection():
    from bermake.scene import Scene
    from bermake.viewport.snap_engine import SnapEngine, SnapKind

    eng = SnapEngine()
    scene = Scene()
    cam = _camera_at_default()
    guide_a = (np.array([0.0, 0.0, 0.0], dtype=np.float64), np.array([1.0, 0.0, 0.0], dtype=np.float64))
    guide_b = (np.array([2.0, -5.0, 0.0], dtype=np.float64), np.array([0.0, 1.0, 0.0], dtype=np.float64))
    crossing = [2.0, 0.0, 0.0]
    cursor = _screen_of(cam, crossing)
    res = eng.snap(cursor, (1280, 800), cam, scene, guides=[guide_a, guide_b], guide_points=[])
    assert res.kind == SnapKind.INTERSECTION
    np.testing.assert_allclose(res.world_position, crossing, atol=1e-3)


def test_hidden_guides_never_reach_the_snap_engine(qtbot):
    """Task 7 made a hidden guide unpickable; a hidden guide that still snaps
    is the same trap with nothing on screen to explain the cursor jump."""
    from bermake.model.annotation import Guide, GuidePoint
    from bermake.model.model import Model
    from bermake.viewport.viewport_widget import ViewportWidget

    model = Model()
    model.active_context.annotations.append(Guide(1, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
    model.active_context.annotations.append(GuidePoint(2, (3.0, 0.0, 0.0)))

    widget = ViewportWidget(model=model)
    qtbot.addWidget(widget)

    widget.show_guides = False
    lines, points = widget._gather_guides()
    assert lines == []
    assert points == []

    widget.show_guides = True
    lines, points = widget._gather_guides()
    assert len(lines) == 1
    assert len(points) == 1

    # And driven all the way through the engine: with guides hidden, no
    # ON_GUIDE/GUIDE_POINT candidate is produced even when the cursor sits
    # right on top of one.
    from bermake.scene import Scene
    from bermake.viewport.snap_engine import SnapEngine, SnapKind

    cam = _camera_at_default()
    cursor = _screen_of(cam, [3.0, 0.0, 0.0])
    widget.show_guides = False
    hidden_lines, hidden_points = widget._gather_guides()
    res = SnapEngine().snap(
        cursor, (1280, 800), cam, Scene(), guides=hidden_lines, guide_points=hidden_points
    )
    assert res.kind not in (SnapKind.ON_GUIDE, SnapKind.GUIDE_POINT)


def test_gather_guides_transforms_correctly_in_a_rotated_translated_context(qtbot):
    """The Task 7 cautionary tale: mixed local/world math cancels exactly at
    the model root, so a rotated + translated active context is the only
    setup that can catch a wrong conversion. Directions transform by the
    linear block only; points (the origin) transform by the full matrix."""
    from bermake.geometry.transforms import mat_rotate, mat_translate
    from bermake.model.annotation import Guide
    from bermake.model.model import Model
    from bermake.viewport.viewport_widget import ViewportWidget

    model = Model()
    grp_def = model.new_definition("Grp", is_group=True)
    grp_def.annotations.append(Guide(1, (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)))

    # 90 degree rotation about Z, then a translation of +5 on X.
    wt = mat_translate([5.0, 0.0, 0.0]) @ mat_rotate([0, 0, 0], [0, 0, 1], math.radians(90))
    grp_inst = model.new_instance(grp_def, wt)
    model.root.children.append(grp_inst)
    model.enter(grp_inst)

    widget = ViewportWidget(model=model)
    qtbot.addWidget(widget)

    lines, points = widget._gather_guides()
    assert len(lines) == 1
    origin, direction = lines[0]
    # Local origin (1,0,0) rotates to (0,1,0), then translates by (5,0,0).
    np.testing.assert_allclose(origin, [5.0, 1.0, 0.0], atol=1e-5)
    # Local direction (1,0,0) rotates to (0,1,0); no translation applied.
    np.testing.assert_allclose(direction, [0.0, 1.0, 0.0], atol=1e-5)


def test_gather_guides_direction_uses_the_vector_convention_under_nonuniform_scale(qtbot):
    """A pure rotation is orthogonal (R == inv(R).T), so the previous test
    cannot tell the correct convention for transforming a direction (the
    plain linear block) from the wrong one (the inverse-transpose, correct
    only for surface NORMALS) -- both give the same answer whenever the
    context's linear part is a rotation alone. `model.py`'s
    `pick_face_local` uses the inverse-transpose for normals in code
    adjacent to this exact distinction, and Task 7 of this milestone had to
    be fixed for getting it wrong here, so this needs its own test with a
    non-uniform scale, the only case that separates the two.

    Expected value derived BY HAND, independent of `_gather_guides`:

    Context linear block L = R @ S, R a 90 degree rotation about Z
    ((x,y,z) -> (-y,x,z)) and S = diag(2,1,1):

        R = [[0,-1,0],[1,0,0],[0,0,1]]      S = diag(2,1,1)
        L = R @ S = [[0,-1,0],[2,0,0],[0,0,1]]

    `Guide.__post_init__` normalises (1,1,0) to d = (1/sqrt2, 1/sqrt2, 0).

    Correct (vector) convention -- plain L applied to d:
        L @ d = (0*d0 + -1*d1, 2*d0 + 0*d1, 0) = (-1/sqrt2, 2/sqrt2, 0)
              = (-0.70710678, 1.41421356, 0.0)

    Wrong (normal) convention -- inverse-transpose of L applied to d:
        L^-1 = S^-1 @ R^-1 = diag(0.5,1,1) @ R^T = [[0,0.5,0],[-1,0,0],[0,0,1]]
        (L^-1)^T = [[0,-1,0],[0.5,0,0],[0,0,1]]
        (L^-1)^T @ d = (-1/sqrt2, 0.5/sqrt2, 0) = (-0.70710678, 0.35355339, 0.0)

    The two are not scalar multiples of each other (the y-component ratio is
    4, the x-component ratio is 1), so this is a genuine direction
    difference, not just a magnitude difference an allclose on a normalised
    vector could miss.
    """
    from bermake.geometry.transforms import mat_rotate, mat_scale, mat_translate
    from bermake.model.annotation import Guide
    from bermake.model.model import Model
    from bermake.viewport.viewport_widget import ViewportWidget

    model = Model()
    grp_def = model.new_definition("Grp", is_group=True)
    grp_def.annotations.append(Guide(1, (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)))

    wt = (
        mat_translate([5.0, 0.0, 0.0])
        @ mat_rotate([0, 0, 0], [0, 0, 1], math.radians(90))
        @ mat_scale([0, 0, 0], [2.0, 1.0, 1.0])
    )
    grp_inst = model.new_instance(grp_def, wt)
    model.root.children.append(grp_inst)
    model.enter(grp_inst)

    widget = ViewportWidget(model=model)
    qtbot.addWidget(widget)

    lines, _points = widget._gather_guides()
    assert len(lines) == 1
    _origin, direction = lines[0]
    np.testing.assert_allclose(direction, [-0.70710678, 1.41421356, 0.0], atol=1e-5)


def _rotated_scaled_context(model):
    """Enter a group rotated 90 degrees about Z and scaled non-uniformly.

    Rotation alone cannot tell a direction convention from a normal
    convention apart, because a rotation matrix is orthogonal and its
    inverse transpose is itself. The non-uniform scale is what separates
    them, and it is also what M7.4 #92 was about. The scale is applied
    BEFORE the rotation (right-most factor) so it acts on the group's own
    local axes.
    """
    from bermake.geometry.transforms import mat_rotate, mat_scale, mat_translate

    wt = (
        mat_translate([5.0, 0.0, 0.0])
        @ mat_rotate([0, 0, 0], [0, 0, 1], math.radians(90))
        @ mat_scale([0, 0, 0], [1.0, 4.0, 1.0])
    )
    grp_def = model.new_definition("Grp", is_group=True)
    grp_inst = model.new_instance(grp_def, wt)
    model.root.children.append(grp_inst)
    model.enter(grp_inst)
    return grp_def, wt


def test_edge_direction_is_world_space_in_a_rotated_scaled_context(qtbot):
    """`_edge_direction` feeds `Acquired.direction`, which is consumed in world
    space beside a world anchor, but it reads the ACTIVE CONTEXT's local mesh.

    An edge running along local +X in a context rotated 90 degrees about Z
    runs along world +Y. Returning the local vector made Parallel,
    Perpendicular and the Down-arrow edge lock all point the wrong way
    inside any rotated context.
    """
    from bermake.model.model import Model
    from bermake.viewport.snap_engine import SnapKind, SnapResult
    from bermake.viewport.viewport_widget import ViewportWidget

    model = Model()
    grp_def, _wt = _rotated_scaled_context(model)
    scene = grp_def.mesh
    v0 = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    v1 = scene.add_vertex(np.array([2.0, 0.0, 0.0], dtype=np.float32))
    eid = scene.add_edge(v0, v1)

    widget = ViewportWidget(model=model)
    qtbot.addWidget(widget)

    snap = SnapResult(
        kind=SnapKind.ON_EDGE,
        world_position=np.zeros(3, dtype=np.float32),
        axis=None,
        vertex_id=None,
        label="On Edge",
        edge_id=eid,
    )
    direction = widget._edge_direction(snap)
    np.testing.assert_allclose(direction, [0.0, 1.0, 0.0], atol=1e-9)


def test_the_gesture_plane_normal_uses_the_inverse_transpose(qtbot):
    """A normal is not an ordinary vector (M7.4 #92).

    In a context scaled by 4 along local Y and then rotated 90 degrees about
    Z, a face whose local normal is (0, 1, 1) has world normal proportional
    to (-1/4, 0, 1) -- the inverse transpose of the linear block. Pushing it
    through the plain linear block would give (-4, 0, 1) instead, which is
    not perpendicular to the face any more, and Perpendicular resolves
    inside that plane.
    """
    from bermake.model.model import Model
    from bermake.viewport.viewport_widget import ViewportWidget

    model = Model()
    _grp_def, _wt = _rotated_scaled_context(model)

    widget = ViewportWidget(model=model)
    qtbot.addWidget(widget)

    got = widget._normal_to_world(np.array([0.0, 1.0, 1.0], dtype=np.float64))
    expected = np.array([-0.25, 0.0, 1.0])
    expected = expected / np.linalg.norm(expected)
    np.testing.assert_allclose(got, expected, atol=1e-9)

    # And the plain-linear-block answer, which a rotation alone could not
    # have distinguished from it, is genuinely a different direction.
    wrong = np.array([-4.0, 0.0, 1.0])
    wrong = wrong / np.linalg.norm(wrong)
    assert float(np.linalg.norm(got - wrong)) > 0.5
