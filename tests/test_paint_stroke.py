"""M7.5a Task 10: drag-to-paint and which side gets painted."""

from __future__ import annotations

import numpy as np
from bermake.scene.scene import Side


def _faces(scene):
    return [f.id for f in scene.faces_iter()]


def test_a_stroke_across_three_faces_is_one_undo_step(main_window):
    win = main_window
    scene = win._model.active_context.mesh
    # three coplanar squares in a row
    for i in range(3):
        v = [
            scene.add_vertex(np.array([i, 0.0, 0.0], dtype=np.float32)),
            scene.add_vertex(np.array([i + 1.0, 0.0, 0.0], dtype=np.float32)),
            scene.add_vertex(np.array([i + 1.0, 1.0, 0.0], dtype=np.float32)),
            scene.add_vertex(np.array([i, 1.0, 0.0], dtype=np.float32)),
        ]
        for a, b in zip(v, v[1:] + v[:1], strict=True):
            scene.add_edge(a, b)
        scene.add_face_from_loop(v)
    ids = _faces(scene)
    win._activate("paint")
    tool = win._tool_manager.active
    mat = win._model.materials.materials()[1]
    depth = len(win._command_stack._undo)

    tool._begin_stroke(mat.id)
    for f in ids:
        tool._paint_during_stroke(f, Side.FRONT, mat.id)
    tool._end_stroke()

    assert all(scene.face_material(f) == mat.id for f in ids)
    # three faces, ONE step. A per-face push would give three.
    assert len(win._command_stack._undo) == depth + 1
    win._command_stack.undo()
    assert all(scene.face_material(f) == 0 for f in ids)


def test_re_crossing_a_face_does_not_add_a_second_command(main_window_with_square):
    win = main_window_with_square
    scene = win._model.active_context.mesh
    f = _faces(scene)[0]
    win._activate("paint")
    tool = win._tool_manager.active
    mat = win._model.materials.materials()[1]
    depth = len(win._command_stack._undo)

    tool._begin_stroke(mat.id)
    for _ in range(5):
        tool._paint_during_stroke(f, Side.FRONT, mat.id)
    # Discriminates a real dedup guard from a "rebuilt every move"
    # implementation: without the guard, five re-crossings of the same
    # (face, side) collect five identical PaintFaceCommands here, and
    # len(undo_stack) == 1 / face_material == 0 after undo would STILL hold
    # (push_executed pushes once regardless of child count, and undoing five
    # paired do/undo commands is a correct no-op) -- neither assertion below
    # would catch it. Only the child count does.
    assert len(tool._stroke_commands) == 1, "re-crossing must not collect a second command"
    tool._end_stroke()

    assert len(win._command_stack._undo) == depth + 1
    composite = win._command_stack._undo[-1][0]
    assert len(composite.children) == 1, "the pushed composite must have exactly one child"
    win._command_stack.undo()
    assert scene.face_material(f) == 0, "one undo must clear it, not five"


def test_deactivate_mid_stroke_rolls_back_every_painted_face_and_pushes_nothing(main_window):
    """PaintTool.deactivate() (mirroring EraserTool.deactivate) must undo any
    in-progress stroke when a tool switch interrupts a drag, so a wrong
    rollback never leaves the scene mutated with nothing on the undo stack
    for the user to recover with."""
    win = main_window
    scene = win._model.active_context.mesh
    for i in range(2):
        v = [
            scene.add_vertex(np.array([i, 0.0, 0.0], dtype=np.float32)),
            scene.add_vertex(np.array([i + 1.0, 0.0, 0.0], dtype=np.float32)),
            scene.add_vertex(np.array([i + 1.0, 1.0, 0.0], dtype=np.float32)),
            scene.add_vertex(np.array([i, 1.0, 0.0], dtype=np.float32)),
        ]
        for a, b in zip(v, v[1:] + v[:1], strict=True):
            scene.add_edge(a, b)
        scene.add_face_from_loop(v)
    ids = _faces(scene)
    assert len(ids) == 2
    win._activate("paint")
    tool = win._tool_manager.active
    mat = win._model.materials.materials()[1]
    depth = len(win._command_stack._undo)

    tool._begin_stroke(mat.id)
    for f in ids:
        tool._paint_during_stroke(f, Side.FRONT, mat.id)
    assert all(scene.face_material(f) == mat.id for f in ids)  # live, pre-release

    tool.deactivate()  # e.g. the user switches tools mid-drag

    assert all(scene.face_material(f) == 0 for f in ids), (
        "mid-drag rollback must revert every painted face"
    )
    assert len(win._command_stack._undo) == depth, (
        "an aborted stroke must leave nothing on the undo stack"
    )


def test_has_active_gesture_reflects_stroke_state(main_window_with_square):
    """Mirrors EraserTool: has_active_gesture must be True only for a live
    stroke, so ViewportWidget.contextMenuEvent suppresses a right-click as a
    mid-gesture cancel instead of popping the context menu -- False both
    before a stroke starts and after it ends, whether by release or by a
    mid-drag deactivate()."""
    win = main_window_with_square
    win._activate("paint")
    tool = win._tool_manager.active
    mat = win._model.materials.materials()[1]

    assert tool.has_active_gesture is False

    tool._begin_stroke(mat.id)
    assert tool.has_active_gesture is True

    tool._end_stroke()
    assert tool.has_active_gesture is False

    tool._begin_stroke(mat.id)
    assert tool.has_active_gesture is True

    tool.deactivate()
    assert tool.has_active_gesture is False


def test_a_stroke_that_paints_nothing_pushes_nothing(main_window_with_square):
    win = main_window_with_square
    win._activate("paint")
    tool = win._tool_manager.active
    depth = len(win._command_stack._undo)
    tool._begin_stroke(1)
    tool._end_stroke()
    assert len(win._command_stack._undo) == depth


def test_the_stroke_paints_the_side_it_is_given(main_window_with_square):
    win = main_window_with_square
    scene = win._model.active_context.mesh
    f = _faces(scene)[0]
    win._activate("paint")
    tool = win._tool_manager.active
    mat = win._model.materials.materials()[1]

    tool._begin_stroke(mat.id)
    tool._paint_during_stroke(f, Side.BACK, mat.id)
    tool._end_stroke()

    assert scene.face_material(f, Side.BACK) == mat.id
    assert scene.face_material(f, Side.FRONT) == 0  # discriminating


def test_clicking_from_either_side_paints_that_side_and_only_that_side():
    """The spec's §3 trap, closed: front and back asserted as distinct slots,
    from cameras on both sides of one face.

    The final review found `PaintTool._resolve_side` could be replaced wholesale
    with `return Side.FRONT` and every test in test_paint_tool.py and
    test_paint_stroke.py still passed (14/14). The stroke tests call
    `_paint_during_stroke(f, side, mid)` directly, so `_resolve_side` was never
    on the path, and the one test that does call it asserts FRONT, which the
    mutant returns anyway.

    This drives the real entry point, `on_mouse_press`, with a real Camera, a
    real Scene and the real picker, once from +Z and once from -Z. Anything
    that ignores the ray (a constant side, or a side read from the normal
    alone) paints the same slot twice and leaves the other at Default.
    """
    from bermake.model.material import MaterialLibrary
    from bermake.scene.scene import Scene
    from bermake.tools.paint_tool import PaintTool
    from bermake.tools.tool import ToolContext
    from bermake.viewport.camera import Camera
    from PySide6.QtCore import QPointF, Qt

    class _Event:
        def position(self):
            return QPointF(50.0, 50.0)

        def modifiers(self):
            return Qt.KeyboardModifier.NoModifier

    class _Stack:
        def __init__(self):
            self.pushed = []

        def push_executed(self, cmd, target):
            self.pushed.append(cmd)

    scene = Scene()
    corners = [
        scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32)),
    ]
    f = scene.add_face_from_loop(corners)
    assert tuple(scene.face_normal(f)) == (0.0, 0.0, 1.0)  # +Z: front faces up

    lib = MaterialLibrary()
    red = lib.add_custom("Red", (0.8, 0.1, 0.1))
    blue = lib.add_custom("Blue", (0.1, 0.1, 0.8))
    stack = _Stack()

    def camera_at(z):
        # up is +Y, not +Z: looking straight down the Z axis makes a +Z up
        # vector parallel to the view direction and the view matrix degenerate.
        return Camera(
            position=np.array([0.5, 0.5, z], dtype=np.float32),
            target=np.array([0.5, 0.5, 0.0], dtype=np.float32),
            up=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )

    def click_from(z, material):
        tool = PaintTool()
        tool.activate(
            ToolContext(
                scene=scene,
                command_stack=stack,
                camera=camera_at(z),
                widget_size_provider=lambda: (100, 100),
                model=None,
                active_material_provider=lambda: material,
            )
        )
        tool.on_mouse_press(_Event(), snap=None)
        tool.on_mouse_release(_Event(), snap=None)

    click_from(+3.0, red)  # above the face: the FRONT is what you see
    assert scene.face_material(f, Side.FRONT) == red.id
    assert scene.face_material(f, Side.BACK) == 0, "painting the front must not touch the back"

    click_from(-3.0, blue)  # below the same face: the BACK
    assert scene.face_material(f, Side.BACK) == blue.id
    assert scene.face_material(f, Side.FRONT) == red.id, "the front must survive a back paint"

    assert len(stack.pushed) == 2  # two clicks, two undoable strokes


def test_side_for_ray_reads_the_normal_direction():
    from bermake.tools.paint_tool import side_for_ray

    normal = np.array([0.0, 0.0, 1.0])
    # looking down at a face whose normal points up: we see the FRONT
    assert side_for_ray(normal, np.array([0.0, 0.0, -1.0])) is Side.FRONT
    # looking up from below at the same face: we see the BACK
    assert side_for_ray(normal, np.array([0.0, 0.0, 1.0])) is Side.BACK


def test_side_for_ray_is_not_fooled_by_a_grazing_angle():
    from bermake.tools.paint_tool import side_for_ray

    normal = np.array([0.0, 0.0, 1.0])
    nearly_edge_on = np.array([1.0, 0.0, -0.01])
    assert side_for_ray(normal, nearly_edge_on) is Side.FRONT


def test_resolve_side_uses_the_inverse_transpose_under_non_uniform_scale():
    """side_for_ray alone can't tell an inverse-transpose from a plain linear
    block: both agree on an axis-aligned face under uniform scale (they agree
    whenever the face normal already lies along one of the scale matrix's own
    axes, which an axis-aligned face always does). This test picks a face
    normal NOT aligned to any axis (from a tilted triangle) and a diagonal
    non-uniform scale, a combination where the two transforms genuinely
    disagree on direction -- and, for the ray direction chosen below, disagree
    on which SIDE it is:

      local normal n = (1, 1, 1)/sqrt(3); scale L = diag(2, 1, 1)

      correct (inverse-transpose): (L^-1)^T n = (0.5, 1, 1)/sqrt(3)
      wrong   (plain linear block): L n        = (2, 1, 1)/sqrt(3)

    Dotted with ray direction d = (1, -1, 0):
      correct . d = 0.5 - 1 = -0.5   -> FRONT
      wrong   . d = 2   - 1 =  0.5   -> BACK

    So this test fails if PaintTool._resolve_side is changed to use the plain
    linear block instead of the inverse-transpose.
    """
    from bermake.tools.paint_tool import PaintTool
    from bermake.tools.tool import ToolContext

    class _FakeScene:
        def face_normal(self, fid):
            return np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)

    class _FakeCamera:
        def ray_from_screen(self, x, y, w, h):
            return np.zeros(3), np.array([1.0, -1.0, 0.0])

    class _FakeModel:
        # diag(2, 1, 1): non-uniform, and the face normal above is not
        # aligned to any coordinate axis -- the trap case the brief warns
        # an axis-aligned or uniformly-scaled fixture cannot expose.
        active_world_transform = np.diag([2.0, 1.0, 1.0, 1.0])

    tool = PaintTool()
    tool.activate(
        ToolContext(
            scene=_FakeScene(),
            camera=_FakeCamera(),
            widget_size_provider=lambda: (100, 100),
            model=_FakeModel(),
        )
    )
    # _resolve_side calls event.position() via _cursor(); _FakeCamera ignores
    # the resulting (x, y), so any QPointF stand-in works.
    class _Event:
        def position(self):
            from PySide6.QtCore import QPointF

            return QPointF(0.0, 0.0)

    assert tool._resolve_side(_Event(), 0) is Side.FRONT
