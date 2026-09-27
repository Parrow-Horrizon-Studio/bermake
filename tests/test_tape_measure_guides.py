"""M7.6b Task 7: Tape Measure creates construction guides.

Reuses tests/test_tape_measure_tool.py's fixture and event-synthesis helpers
(`_press`, `_snap`, `_ctx`-style shape) rather than inventing new ones, plus a
`_move`/`_release` pair for the drag half of the new gestures.
"""

from __future__ import annotations

import types

import numpy as np
from bermake.commands import CommandStack
from bermake.model import Model
from bermake.tools.tape_measure_tool import TapeMeasureTool
from bermake.tools.tool import ToolContext
from bermake.units import Units
from bermake.viewport.snap_engine import SnapKind
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

U = Units()


def _press(modifiers=Qt.KeyboardModifier.NoModifier):
    return QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(0, 0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        modifiers,
    )


def _move():
    return QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(0, 0),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )


def _release(modifiers=Qt.KeyboardModifier.NoModifier):
    return QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        QPointF(0, 0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        modifiers,
    )


def _snap(p, kind=SnapKind.ENDPOINT, vertex_id=None, edge_id=None, face_id=None, edge_t=None):
    return types.SimpleNamespace(
        kind=kind,
        world_position=np.asarray(p, np.float32),
        axis=None,
        vertex_id=vertex_id,
        edge_id=edge_id,
        edge_t=edge_t,
        face_id=face_id,
    )


def _ctx(model, command_stack):
    return ToolContext(
        scene=model.active_scene,
        command_stack=command_stack,
        camera=None,
        widget_size_provider=lambda: (800, 600),
        units_provider=lambda: U,
        model=model,
    )


def _guides(model):
    return [a for a in model.active_context.annotations if a.kind == "guide"]


def _guide_points(model):
    return [a for a in model.active_context.annotations if a.kind == "guide_point"]


# ---------------------------------------------------------------------------
# Behaviour 5 FIRST: the offset-rule spike. Spec 2.4 (reasoned, not measured)
# says a guide's offset is perpendicular to the clicked edge WITHIN THE PLANE
# OF THE FACE that edge bounds. Verify with an actual vertical wall before
# trusting that rule for anything else in this file.
# ---------------------------------------------------------------------------


def _vertical_wall_model():
    """A model whose root scene holds one vertical wall: a quad face in the
    x=0 plane (so its normal is +/-X), matching
    tests/test_drawing_plane_resolution.py's own `_wall_scene` layout.

    vertex order: 0=(0,0,0) 1=(0,6,0) 2=(0,6,4) 3=(0,0,4). Edge (3, 0) is the
    wall's LEFT VERTICAL edge (y=0 fixed, z varies) -- deliberately chosen
    because it stress-tests the rule: a wrong "always fall back to the
    horizontal ground plane" implementation would try to find a direction
    perpendicular to a vertical edge that also lies in a horizontal plane,
    which does not exist (the cross product degenerates to zero), so that
    bug would fail loudly (no guide at all) rather than quietly producing a
    horizontal one.
    """
    model = Model()
    scene = model.active_scene
    vids = [
        scene.add_vertex(np.array(p, dtype=np.float32))
        for p in [(0, 0, 0), (0, 6, 0), (0, 6, 4), (0, 0, 4)]
    ]
    edge_ids = {}
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        edge_ids[(a, b)] = scene.add_edge(vids[a], vids[b])
    face_id = scene.add_face_from_loop(vids)
    return model, scene, vids, edge_ids, face_id


def test_guide_offset_is_coplanar_with_the_clicked_edges_face(qtbot):
    model, scene, vids, edge_ids, face_id = _vertical_wall_model()
    stack = CommandStack()
    t = TapeMeasureTool()
    t.activate(_ctx(model, stack))

    vertical_edge_id = edge_ids[(3, 0)]
    edge = scene.edge(vertical_edge_id)
    p1 = scene.vertex(edge.v1_id).position
    p2 = scene.vertex(edge.v2_id).position
    edge_dir = (np.asarray(p2, np.float64) - np.asarray(p1, np.float64))
    edge_dir = edge_dir / np.linalg.norm(edge_dir)
    midpoint = (np.asarray(p1, np.float64) + np.asarray(p2, np.float64)) / 2.0

    press_snap = _snap(
        midpoint, kind=SnapKind.ON_EDGE, edge_id=vertical_edge_id, face_id=face_id, edge_t=0.5
    )
    t.on_mouse_press(_press(), press_snap)
    assert t._drag_active, "a click on an edge must open a drag gesture"

    # Drag purely along +Y -- NOT the wall's own edge direction (+/-Z) -- so
    # a coplanar (x == 0) result is not a coincidence of the drag direction
    # chosen.
    drag_to = midpoint + np.array([0.0, 1.5, 0.0])
    t.on_mouse_move(_move(), _snap(drag_to, kind=SnapKind.ON_FACE, face_id=face_id))
    t.on_mouse_release(_release(), _snap(drag_to, kind=SnapKind.ON_FACE, face_id=face_id))

    guides = _guides(model)
    assert len(guides) == 1, (
        "expected exactly one Guide from the edge-drag; the offset rule may "
        "have degenerated instead of resolving the wall's own plane"
    )
    guide = guides[0]

    # Coplanar with the wall (x == 0 for both the origin and the direction),
    # NOT dropped onto the horizontal ground plane.
    assert abs(guide.origin[0]) < 1e-4, f"guide origin left the wall's plane: {guide.origin}"
    assert abs(guide.direction[0]) < 1e-4, (
        f"guide direction left the wall's plane: {guide.direction}"
    )
    # And genuinely not horizontal: the clicked edge was vertical, so a
    # guide parallel to it must keep a large Z component.
    assert abs(guide.direction[2]) > 0.9, f"guide direction reads as horizontal: {guide.direction}"

    # The guide is parallel (or anti-parallel) to the clicked edge.
    assert abs(abs(float(np.dot(guide.direction, edge_dir))) - 1.0) < 1e-4

    # It undoes cleanly, as one step.
    assert stack.can_undo
    assert stack.undo()
    assert _guides(model) == []


# ---------------------------------------------------------------------------
# Behaviour 1: click an edge, drag, release -> one Guide, parallel to the edge.
# ---------------------------------------------------------------------------


def _free_edge_model():
    """A model whose root scene holds a single free edge (no face at all),
    along +X, so the offset rule falls back to the gesture's own drawing
    plane (horizontal, per resolve_drawing_plane's own fallback rule)."""
    model = Model()
    scene = model.active_scene
    v0 = scene.add_vertex(np.array([0.0, 0.0, 0.0], np.float32))
    v1 = scene.add_vertex(np.array([4.0, 0.0, 0.0], np.float32))
    edge_id = scene.add_edge(v0, v1)
    return model, scene, edge_id


def test_edge_drag_and_release_creates_one_parallel_guide(qtbot):
    model, scene, edge_id = _free_edge_model()
    stack = CommandStack()
    t = TapeMeasureTool()
    t.activate(_ctx(model, stack))

    press_snap = _snap([2.0, 0.0, 0.0], kind=SnapKind.ON_EDGE, edge_id=edge_id, edge_t=0.5)
    t.on_mouse_press(_press(), press_snap)
    t.on_mouse_move(_move(), _snap([2.0, 2.0, 0.0], kind=SnapKind.ON_FACE))
    t.on_mouse_release(_release(), _snap([2.0, 2.0, 0.0], kind=SnapKind.ON_FACE))

    guides = _guides(model)
    assert len(guides) == 1
    guide = guides[0]
    assert abs(abs(float(guide.direction[0])) - 1.0) < 1e-6, guide.direction


# ---------------------------------------------------------------------------
# Behaviour 2: the same gesture with Ctrl held creates nothing, and the
# measure-only readout keeps working.
# ---------------------------------------------------------------------------


def test_ctrl_held_suppresses_guide_creation(qtbot):
    model, scene, edge_id = _free_edge_model()
    stack = CommandStack()
    t = TapeMeasureTool()
    t.activate(_ctx(model, stack))

    press_snap = _snap([2.0, 0.0, 0.0], kind=SnapKind.ON_EDGE, edge_id=edge_id, edge_t=0.5)
    t.on_mouse_press(_press(Qt.KeyboardModifier.ControlModifier), press_snap)
    t.on_mouse_move(_move(), _snap([2.0, 2.0, 0.0], kind=SnapKind.ON_FACE))
    t.on_mouse_release(
        _release(Qt.KeyboardModifier.ControlModifier), _snap([2.0, 2.0, 0.0], kind=SnapKind.ON_FACE)
    )

    assert _guides(model) == []
    assert _guide_points(model) == []
    # measure-only readout intact: one point placed, cursor tracked, distance shown.
    assert t.status_text is not None
    assert "Distance" in t.status_text


# ---------------------------------------------------------------------------
# Behaviour 3: creating a guide pushes exactly one command.
# ---------------------------------------------------------------------------


def test_guide_creation_is_one_undo_step(qtbot):
    model, scene, edge_id = _free_edge_model()
    stack = CommandStack()
    t = TapeMeasureTool()
    t.activate(_ctx(model, stack))

    press_snap = _snap([2.0, 0.0, 0.0], kind=SnapKind.ON_EDGE, edge_id=edge_id, edge_t=0.5)
    t.on_mouse_press(_press(), press_snap)
    t.on_mouse_move(_move(), _snap([2.0, 2.0, 0.0], kind=SnapKind.ON_FACE))
    t.on_mouse_release(_release(), _snap([2.0, 2.0, 0.0], kind=SnapKind.ON_FACE))

    assert len(_guides(model)) == 1
    assert stack.can_undo
    assert stack.undo()
    assert _guides(model) == []
    assert not stack.can_undo


# ---------------------------------------------------------------------------
# Behaviour 4: click a vertex, drag along an axis, type a distance -> one
# GuidePoint at that distance.
# ---------------------------------------------------------------------------


def test_vertex_drag_and_typed_value_creates_one_guide_point(qtbot):
    model = Model()
    scene = model.active_scene
    v0 = scene.add_vertex(np.array([0.0, 0.0, 0.0], np.float32))
    stack = CommandStack()
    t = TapeMeasureTool()
    t.activate(_ctx(model, stack))

    press_snap = _snap([0.0, 0.0, 0.0], kind=SnapKind.ENDPOINT, vertex_id=v0)
    t.on_mouse_press(_press(), press_snap)
    t.on_mouse_move(_move(), _snap([3.0, 0.0, 0.0], kind=SnapKind.AXIS_LOCK))

    assert t.apply_typed_value("5", U)

    points = _guide_points(model)
    assert len(points) == 1
    np.testing.assert_allclose(points[0].position, (5.0, 0.0, 0.0), atol=1e-5)
    # Typed-value commit ends the gesture like every other one-shot tool.
    assert not t._drag_active


def test_vertex_drag_typed_value_rejects_zero_and_negative(qtbot):
    model = Model()
    scene = model.active_scene
    v0 = scene.add_vertex(np.array([0.0, 0.0, 0.0], np.float32))
    stack = CommandStack()
    t = TapeMeasureTool()
    t.activate(_ctx(model, stack))

    t.on_mouse_press(_press(), _snap([0.0, 0.0, 0.0], kind=SnapKind.ENDPOINT, vertex_id=v0))
    t.on_mouse_move(_move(), _snap([3.0, 0.0, 0.0], kind=SnapKind.AXIS_LOCK))

    assert not t.apply_typed_value("0", U)
    assert not t.apply_typed_value("-2", U)
    assert _guide_points(model) == []


# ---------------------------------------------------------------------------
# Fix round 1: the offset maths must hold up inside a rotated + translated
# group, not just at the model root (identity transform), where local and
# world vectors happen to look identical and a local/world mix-up cancels
# out silently.
# ---------------------------------------------------------------------------


def _rotated_wall_model():
    """A wall (vertical quad, local x=0 plane, same layout as
    `_vertical_wall_model`) living inside a GROUP instance transformed by a
    90-degree rotation about Z plus a translation -- so `model.active_context`
    is the group's own definition, `model.active_world_transform` is that
    rotation+translation, and every local quantity genuinely differs from its
    world counterpart (unlike the root-context tests above, where the two
    coincide and a local/world mix-up is invisible).

    Returns (model, scene, edge_ids, face_id, to_world_point, to_world_vec)
    where the last two are the SAME rotation+translation the SnapEngine would
    have applied, for building expected-world-space snaps and assertions.
    """
    model = Model()
    inner = model.new_definition("Wall", is_group=True)
    scene = inner.mesh
    vids = [
        scene.add_vertex(np.array(p, dtype=np.float32))
        for p in [(0, 0, 0), (0, 6, 0), (0, 6, 4), (0, 0, 4)]
    ]
    edge_ids = {}
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        edge_ids[(a, b)] = scene.add_edge(vids[a], vids[b])
    face_id = scene.add_face_from_loop(vids)

    # +90 degrees about Z: (x, y, z) -> (-y, x, z), then translate.
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.array([10.0, 20.0, 0.0])
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = translation

    inst = model.new_instance(inner, transform)
    model.root.children.append(inst)
    model.enter(inst)
    assert model.active_context is inner

    def to_world_point(local_pt):
        return rot @ np.asarray(local_pt, dtype=np.float64) + translation

    def to_world_vec(local_vec):
        return rot @ np.asarray(local_vec, dtype=np.float64)

    return model, scene, edge_ids, face_id, to_world_point, to_world_vec


def test_guide_from_edge_in_rotated_translated_group_is_correct_in_world_space(qtbot):
    """Fix round 1 regression: the old code mixed a LOCAL perpendicular/edge
    vector into WORLD-space press/drag points. At the model root (identity
    transform) local and world vectors are numerically identical, so every
    other test in this file could not catch it. Inside a rotated group the
    two genuinely differ, and the bug showed up two ways: a guide's stored
    direction came out perpendicular to the edge it was supposed to be
    parallel to, and a guide's world origin landed off the source face's own
    world plane.

    This test creates two guides in the SAME rotated+translated group -- one
    from a horizontal edge (exposes the direction bug: a Z-axis rotation
    happens to leave a pure-Z perpendicular vector unchanged, so the
    horizontal edge's OWN direction is the quantity that actually differs
    between local and world here) and one from the vertical edge (exposes
    the origin bug: its perpendicular is a pure-X/Y vector, which a Z-axis
    rotation does change) -- and checks both required properties, in WORLD
    space, on each.
    """
    model, scene, edge_ids, face_id, to_world_point, to_world_vec = _rotated_wall_model()
    stack = CommandStack()
    t = TapeMeasureTool()
    t.activate(_ctx(model, stack))

    # --- horizontal edge: (0, 1) -> local direction (0, 1, 0) -------------
    horiz_edge_id = edge_ids[(0, 1)]
    horiz_local_dir = np.array([0.0, 1.0, 0.0])
    horiz_press_local = np.array([0.0, 3.0, 0.0])  # midpoint of (0,0,0)-(0,6,0)
    horiz_press_world = to_world_point(horiz_press_local)
    horiz_drag_world = horiz_press_world + np.array([0.0, 0.0, 1.5])  # along world Z

    t.on_mouse_press(
        _press(),
        _snap(horiz_press_world, kind=SnapKind.ON_EDGE, edge_id=horiz_edge_id, face_id=face_id),
    )
    t.on_mouse_move(_move(), _snap(horiz_drag_world, kind=SnapKind.ON_FACE, face_id=face_id))
    t.on_mouse_release(_release(), _snap(horiz_drag_world, kind=SnapKind.ON_FACE, face_id=face_id))

    guides = _guides(model)
    assert len(guides) == 1, "expected the horizontal-edge drag to create one Guide"
    horiz_guide = guides[0]

    horiz_guide_dir_world = to_world_vec(horiz_guide.direction)
    horiz_guide_dir_world /= np.linalg.norm(horiz_guide_dir_world)
    horiz_edge_dir_world = to_world_vec(horiz_local_dir)
    horiz_edge_dir_world /= np.linalg.norm(horiz_edge_dir_world)
    assert abs(abs(float(np.dot(horiz_guide_dir_world, horiz_edge_dir_world))) - 1.0) < 1e-6, (
        f"guide direction {horiz_guide.direction} (world {horiz_guide_dir_world}) is not "
        f"parallel to the clicked edge (world {horiz_edge_dir_world})"
    )

    # --- vertical edge: (3, 0) -> local direction (0, 0, -1) --------------
    vert_edge_id = edge_ids[(3, 0)]
    vert_press_local = np.array([0.0, 0.0, 2.0])  # midpoint of (0,0,4)-(0,0,0)
    vert_press_world = to_world_point(vert_press_local)
    # A world-space drag with components along BOTH the correct perpendicular
    # (world X here) and the stale-local-treated-as-world one (world Y) --
    # chosen so a buggy implementation still creates a guide (a non-zero
    # projection either way) but at a visibly different, off-plane origin.
    vert_drag_world = vert_press_world + np.array([2.0, 1.0, 0.0])

    t.on_mouse_press(
        _press(),
        _snap(vert_press_world, kind=SnapKind.ON_EDGE, edge_id=vert_edge_id, face_id=face_id),
    )
    t.on_mouse_move(_move(), _snap(vert_drag_world, kind=SnapKind.ON_FACE, face_id=face_id))
    t.on_mouse_release(_release(), _snap(vert_drag_world, kind=SnapKind.ON_FACE, face_id=face_id))

    guides = _guides(model)
    assert len(guides) == 2, "expected the vertical-edge drag to create a second Guide"
    vert_guide = guides[1]

    # The wall's world plane is Y == 20 (the local x=0 plane rotated 90
    # degrees about Z, then translated): every point on the wall has local
    # x == 0, and this transform maps local x to world (y - 20), so world Y
    # is pinned at the translation's Y component regardless of local y/z.
    vert_guide_origin_world = to_world_point(vert_guide.origin)
    assert abs(float(vert_guide_origin_world[1]) - 20.0) < 1e-6, (
        f"guide origin {vert_guide.origin} (world {vert_guide_origin_world}) left the "
        "wall's own world plane (Y == 20)"
    )
