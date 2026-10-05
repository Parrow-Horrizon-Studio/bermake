"""M7.12 Task 3 (#77): `Tool.holds_uncommitted_changes`.

An autosave reads the model while the user works, so it must never run while
a tool holds a temporary change of its own that a cancel would roll back. The
audit behind this file (see the M7.12 task-3 report) read every registered
tool's press, drag and release code. Three tools write to the Model/Scene
before their command reaches the undo stack:

- Line: every click runs its AddVertex/AddEdge commands at once and keeps
  them in a private composite; the composite is pushed only on Enter, a
  double-click or loop closure, and Escape undoes it.
- Eraser: a press-drag stroke executes each removal at once and pushes the
  stroke on release.
- Paint: a stroke paints faces at once and pushes on release, and a Shift
  placement drag writes intermediate placements straight into the scene.

The other 21 tools only draw an overlay until the final click and then
execute or push their command in a single call.

Each tool is driven through its own mouse and key handlers. Two things keep
the expected values honest rather than asserted by fiat. A model fingerprint
taken before and during the gesture must differ for exactly the tools that
claim to hold changes (so True is not vacuous, and False means the model
really was untouched). The undo depth must not move mid-gesture for any tool
(so "uncommitted" really means "not yet on the undo stack").
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
from bermake.scene.scene import Side, TexturePlacement
from bermake.tools.tool import Tool
from bermake.viewport.camera import Camera
from bermake.viewport.snap_engine import SnapKind, SnapResult
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent

_W, _H = 800, 600


# ---- events and snaps ------------------------------------------------------


def _mouse(kind, pos, button, buttons, mods=Qt.KeyboardModifier.NoModifier):
    return QMouseEvent(kind, QPointF(float(pos[0]), float(pos[1])), button, buttons, mods)


def _press(pos=(0.0, 0.0)):
    return _mouse(
        QEvent.Type.MouseButtonPress, pos, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton
    )


def _move(pos=(0.0, 0.0)):
    # A drag: no button changed, left still held (what Erase, Paint, Select read).
    return _mouse(QEvent.Type.MouseMove, pos, Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)


def _release(pos=(0.0, 0.0)):
    return _mouse(
        QEvent.Type.MouseButtonRelease, pos, Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton
    )


def _key(qt_key):
    return QKeyEvent(QEvent.Type.KeyPress, qt_key, Qt.KeyboardModifier.NoModifier)


def _snap(world, kind=SnapKind.GRID, vertex_id=None):
    return SnapResult(
        kind=kind,
        world_position=np.asarray(world, dtype=np.float32),
        axis=None,
        vertex_id=vertex_id,
        label="t",
    )


# ---- the world each script runs in ----------------------------------------


class _Env:
    """A MainWindow's real model, command stack and selection, plus a real
    Camera at a known size and two unit squares (run 0 and run 1 each use
    their own target so a committed run never disturbs the next)."""

    def __init__(self, window) -> None:
        self.window = window
        self.model = window._model
        self.scene = window._model.active_context.mesh
        self.stack = window._command_stack
        self.selection = window._selection
        self.cam = Camera()
        self.cam.aspect = _W / _H
        self.ctx = dataclasses.replace(
            window._tool_context(),
            camera=self.cam,
            widget_size_provider=lambda: (_W, _H),
        )
        self.squares: list[tuple[int, list[int], float]] = []
        for x0 in (0.0, 3.0):
            v = [
                self.scene.add_vertex(np.array([x0, 0.0, 0.0], dtype=np.float32)),
                self.scene.add_vertex(np.array([x0 + 1.0, 0.0, 0.0], dtype=np.float32)),
                self.scene.add_vertex(np.array([x0 + 1.0, 1.0, 0.0], dtype=np.float32)),
                self.scene.add_vertex(np.array([x0, 1.0, 0.0], dtype=np.float32)),
            ]
            self.squares.append((self.scene.add_face_from_loop(v), v, x0))

    def px(self, world) -> tuple[float, float]:
        sx, sy, _depth = self.cam.world_to_screen(np.asarray(world, np.float32), _W, _H)
        return (float(sx), float(sy))

    def org(self, k: int) -> np.ndarray:
        """A free-space base point for drawing tools, distinct per run."""
        return np.array([-4.0 - 4.0 * k, 3.0, 0.0])

    def face_center(self, k: int) -> np.ndarray:
        return np.array([self.squares[k][2] + 0.5, 0.5, 0.0])

    def fingerprint(self):
        """Everything a save would write that a tool can touch."""
        s = self.scene
        faces = tuple(
            (
                f.id,
                s.face_material(f.id),
                s.face_placement(f.id, Side.FRONT),
                s.face_material(f.id, Side.BACK),
            )
            for f in s.faces_iter()
        )
        return (
            tuple((v.id, v.position.tobytes()) for v in s.vertices_iter()),
            tuple(sorted(e.id for e in s.edges_iter())),
            faces,
            len(self.model.active_context.children),
            len(self.model.active_context.annotations),
        )

    def undo_depth(self) -> int:
        return len(self.stack._undo)


# ---- one script per tool ---------------------------------------------------


@dataclass(frozen=True)
class _Script:
    begin: Callable  # (env, tool, k) -> leaves the tool mid-gesture
    commit: Callable  # (env, tool, k) -> finishes the gesture normally
    mid: bool  # does the tool hold uncommitted changes mid-gesture?
    after_escape: bool = False  # ... and after Escape (Eraser, Paint: Escape is inert)
    prepare: Callable | None = None  # (env, k) -> selection or scene setup, before activate
    commits: bool = True  # does `commit` push an undo entry? (Select only changes selection)
    gesture: bool = True  # is the tool in a gesture mid-script? (False: hover-only tools)


def _escape(env, tool, k):
    tool.on_key_press(_key(Qt.Key.Key_Escape))


def _enter(env, tool, k):
    tool.on_key_press(_key(Qt.Key.Key_Return))


def _click_through(*offsets):
    """A tool whose gesture is a run of snapped clicks from org(k): press each
    offset in turn. `begin` presses all but the last, `commit` presses the last."""

    def begin(env, tool, k):
        for off in offsets[:-1]:
            tool.on_mouse_press(_press(), _snap(env.org(k) + off))
        tool.on_mouse_move(_move(), _snap(env.org(k) + offsets[-1]))

    def commit(env, tool, k):
        tool.on_mouse_press(_press(), _snap(env.org(k) + offsets[-1]))

    return begin, commit


def _o(x, y, z=0.0):
    return np.array([x, y, z])


def _scripted(offsets, mid=False):
    begin, commit = _click_through(*offsets)
    return _Script(begin, commit, mid)


# Tools that draw an overlay and nothing else until the closing click.
_RECT = _scripted([_o(0, 0), _o(2, 2)])
_CIRCLE = _scripted([_o(0, 0), _o(1, 0)])


def _select_faces(env, k):
    env.selection.replace(faces=[env.squares[k][0]])


# -- Line: live (see module docstring) --


def _line_begin(env, tool, k):
    tool.on_mouse_press(_press(), _snap(env.org(k)))
    tool.on_mouse_press(_press(), _snap(env.org(k) + _o(1, 0)))
    tool.on_mouse_move(_move(), _snap(env.org(k) + _o(2, 0)))


_LINE = _Script(_line_begin, _enter, mid=True)


# -- Arc: three clicks, built at the third --


def _arc_begin(env, tool, k):
    tool.on_mouse_press(_press(), _snap(env.org(k)))
    tool.on_mouse_press(_press(), _snap(env.org(k) + _o(2, 0)))
    tool.on_mouse_move(_move(), _snap(env.org(k) + _o(1, 1)))


def _arc_commit(env, tool, k):
    tool.on_mouse_press(_press(), _snap(env.org(k) + _o(1, 1)))


_ARC = _Script(_arc_begin, _arc_commit, mid=False)


# -- Primitives: footprint, then a height read off the camera ray --


def _prim_begin(env, tool, k):
    tool.on_mouse_press(_press(), _snap(env.org(k)))
    tool.on_mouse_press(_press(), _snap(env.org(k) + _o(2, 2)))
    tool.on_mouse_move(_move(env.px(env.org(k) + _o(1, 1, 1.5))), _snap(env.org(k)))


def _prim_commit(env, tool, k):
    tool.on_mouse_press(_press(env.px(env.org(k) + _o(1, 1, 1.5))), _snap(env.org(k)))


_PRIMITIVE = _Script(_prim_begin, _prim_commit, mid=False)


# -- Push/Pull and Offset: hover a face, click to arm, move, click to commit --


def _arm_begin(target):
    def begin(env, tool, k):
        tool.on_mouse_move(_move(env.px(env.face_center(k))), None)
        tool.on_mouse_press(_press(env.px(env.face_center(k))), None)
        tool.on_mouse_move(_move(env.px(target(env, k))), None)

    return begin


def _arm_commit(target):
    def commit(env, tool, k):
        tool.on_mouse_press(_press(env.px(target(env, k))), None)

    return commit


def _pp_target(env, k):
    return env.face_center(k) + _o(0, 0, 1.0)


def _offset_target(env, k):
    return np.array([env.squares[k][2] + 0.2, 0.5, 0.0])


_PUSH_PULL = _Script(_arm_begin(_pp_target), _arm_commit(_pp_target), mid=False)
_OFFSET = _Script(_arm_begin(_offset_target), _arm_commit(_offset_target), mid=False)


# -- Follow Me: preselect a path, hover the profile, click --


def _follow_prepare(env, k):
    _face, v, x0 = env.squares[k]
    corner = v[0]
    up = env.scene.add_vertex(np.array([x0, 0.0, 3.0], dtype=np.float32))
    over = env.scene.add_vertex(np.array([x0, 3.0, 3.0], dtype=np.float32))
    e1 = env.scene.add_edge(corner, up)
    e2 = env.scene.add_edge(up, over)
    env.selection.replace(edges=[e1, e2])


def _follow_begin(env, tool, k):
    tool.on_mouse_move(_move(env.px(env.face_center(k))), None)


def _follow_commit(env, tool, k):
    tool.on_mouse_press(_press(env.px(env.face_center(k))), None)


_FOLLOW_ME = _Script(
    _follow_begin, _follow_commit, mid=False, prepare=_follow_prepare, gesture=False
)


# -- Door/Window: a preview on hover, placed on click. It picks faces of child
#    instances only, so each run gets its own wall instance to aim at --


def _opening_prepare(env, k):
    from bermake.geometry.wall import wall_box

    x0 = 3.0 * k
    verts, faces = wall_box((x0, 5.0, 0.0), (x0 + 2.0, 5.0, 0.0), 0.2, 2.4)
    defn = env.model.new_definition("Wall", is_group=True)
    ids = [defn.mesh.add_vertex(np.array(v, dtype=np.float32)) for v in verts]
    for loop in faces:
        defn.mesh.add_face_from_loop([ids[i] for i in loop])
    env.model.active_context.children.append(env.model.new_instance(defn))


def _opening_aim(env, k):
    # The wall's near (-Y) face, which the default camera looks straight at.
    return env.px([3.0 * k + 1.0, 4.9, 1.2])


def _opening_begin(env, tool, k):
    tool.on_mouse_move(_move(_opening_aim(env, k)), None)


def _opening_commit(env, tool, k):
    tool.on_mouse_press(_press(_opening_aim(env, k)), None)


_DOOR_WINDOW = _Script(
    _opening_begin, _opening_commit, mid=False, prepare=_opening_prepare, gesture=False
)


# -- Text: patch the modal prompt --


def _text_commit(env, tool, k):
    tool.prompt_text = lambda default="": "label"
    tool.on_mouse_press(_press(), _snap(env.org(k) + _o(2, 0)))


def _text_begin(env, tool, k):
    tool.on_mouse_press(_press(), _snap(env.org(k)))
    tool.on_mouse_move(_move(), _snap(env.org(k) + _o(2, 0)))


_TEXT = _Script(_text_begin, _text_commit, mid=False)


# -- Tape Measure: a guide-point drag from an existing vertex --


def _tape_begin(env, tool, k):
    vid = env.squares[k][1][0]
    x0 = env.squares[k][2]
    tool.on_mouse_press(_press(), _snap([x0, 0, 0], SnapKind.ENDPOINT, vid))
    tool.on_mouse_move(_move(), _snap([x0, 0, 1.5]))


def _tape_commit(env, tool, k):
    x0 = env.squares[k][2]
    tool.on_mouse_release(_release(), _snap([x0, 0, 1.5]))


_TAPE = _Script(_tape_begin, _tape_commit, mid=False)


# -- Move / Rotate / Scale: overlay preview, one transform command at the end --


def _move_begin(env, tool, k):
    x0 = env.squares[k][2]
    tool.on_mouse_press(_press(), _snap([x0, 0, 0], SnapKind.ENDPOINT))
    tool.on_mouse_move(_move(), _snap([x0, 0, 3], SnapKind.ENDPOINT))


def _move_commit(env, tool, k):
    x0 = env.squares[k][2]
    tool.on_mouse_release(_release(), _snap([x0, 0, 3], SnapKind.ENDPOINT))


_MOVE = _Script(_move_begin, _move_commit, mid=False, prepare=_select_faces)


def _rotate_begin(env, tool, k):
    x0 = env.squares[k][2]
    tool.on_mouse_press(_press(), _snap([x0, 0, 0]))
    tool.on_mouse_press(_press(), _snap([x0 + 1, 0, 0]))
    tool.on_mouse_move(_move(), _snap([x0 + 1, 1, 0]))


def _rotate_commit(env, tool, k):
    x0 = env.squares[k][2]
    tool.on_mouse_press(_press(), _snap([x0 + 1, 1, 0]))


_ROTATE = _Script(_rotate_begin, _rotate_commit, mid=False, prepare=_select_faces)


def _scale_begin(env, tool, k):
    from bermake.tools.transform_support import GripSpec

    x0 = env.squares[k][2]
    grip = GripSpec(
        position=np.array([x0 + 1, 1, 0], np.float32),
        opposite=np.array([x0, 0, 0], np.float32),
        axes=(0, 1),
    )
    # The grip pick and cursor plane need a camera ray on a grip; pin them the
    # way test_scale_tool.py does so the script drives the real press/drag/release.
    tool._pick_grip = lambda ev: grip
    tool._cursor_world = lambda ev: np.array([x0 + 2, 2, 0], np.float32)
    tool.on_mouse_press(_press(), None)
    tool.on_mouse_move(_move(), None)


def _scale_commit(env, tool, k):
    tool.on_mouse_release(_release(), None)


_SCALE = _Script(_scale_begin, _scale_commit, mid=False, prepare=_select_faces)


# -- Select: a box drag, the selection is not model content --


def _select_begin(env, tool, k):
    tool.on_mouse_press(_press((100, 100)), None)
    tool.on_mouse_move(_move((300, 260)), None)


def _select_commit(env, tool, k):
    tool.on_mouse_release(_release((300, 260)), None)


_SELECT = _Script(_select_begin, _select_commit, mid=False, commits=False)


# -- Eraser: live, Escape is inert --


def _erase_begin(env, tool, k):
    x0 = env.squares[k][2]
    tool.on_mouse_press(_press(env.px([x0 + 0.5, 0.0, 0.0])), None)


def _erase_commit(env, tool, k):
    tool.on_mouse_release(_release(), None)


_ERASER = _Script(_erase_begin, _erase_commit, mid=True, after_escape=True)


# -- Paint: live, Escape is inert --


def _paint_prepare(env, k):
    mat = env.model.materials.materials()[1]
    env.window._active_material_id = mat.id


def _paint_begin(env, tool, k):
    tool.on_mouse_press(_press(env.px(env.face_center(k))), None)


def _paint_commit(env, tool, k):
    tool.on_mouse_release(_release(), None)


_PAINT = _Script(_paint_begin, _paint_commit, mid=True, after_escape=True, prepare=_paint_prepare)


# -- Wall and Roof: committed walls stay committed, the tip is only drawn --


def _wall_begin(env, tool, k):
    tool.on_mouse_press(_press(), _snap(env.org(k)))
    tool.on_mouse_move(_move(), _snap(env.org(k) + _o(3, 0)))


def _wall_commit(env, tool, k):
    tool.on_mouse_press(_press(), _snap(env.org(k) + _o(3, 0)))


_WALL = _Script(_wall_begin, _wall_commit, mid=False)

_ROOF = _scripted([_o(0, 0), _o(4, 3)])
_DIMENSION = _scripted([_o(0, 0), _o(2, 0), _o(1, 1)])

_SCRIPTS: dict[str, _Script] = {
    "arc": _ARC,
    "box": _PRIMITIVE,
    "circle": _CIRCLE,
    "cone": _PRIMITIVE,
    "cylinder": _PRIMITIVE,
    "dimension": _DIMENSION,
    "door_window": _DOOR_WINDOW,
    "eraser": _ERASER,
    "follow_me": _FOLLOW_ME,
    "line": _LINE,
    "move": _MOVE,
    "offset": _OFFSET,
    "paint": _PAINT,
    "polygon": _CIRCLE,
    "push_pull": _PUSH_PULL,
    "rectangle": _RECT,
    "roof": _ROOF,
    "rotate": _ROTATE,
    "scale": _SCALE,
    "select": _SELECT,
    "sphere": _PRIMITIVE,
    "tape_measure": _TAPE,
    "text": _TEXT,
    "wall": _WALL,
}


# ---- tests -----------------------------------------------------------------


def test_the_script_table_covers_every_registered_tool(main_window):
    """A 25th tool registered in MainWindow fails here until someone audits it."""
    assert set(_SCRIPTS) == main_window._tool_manager.tool_ids()
    assert len(_SCRIPTS) == 24


def test_the_base_tool_holds_nothing_by_default():
    prop = Tool.__dict__["holds_uncommitted_changes"]
    assert isinstance(prop, property)

    class _Plain(Tool):
        name = "p"
        shortcut = ""
        id = "p"
        has_active_gesture = False
        anchor_or_none = None

        def activate(self, ctx): ...
        def deactivate(self): ...
        def overlay(self): ...

    assert _Plain().holds_uncommitted_changes is False


def _prepare_and_activate(env, tool, script, k):
    if script.prepare is not None:
        script.prepare(env, k)
    tool.activate(env.ctx)


@pytest.mark.parametrize("tool_id", sorted(_SCRIPTS))
def test_tool_reports_uncommitted_changes_only_while_it_holds_them(main_window, tool_id):
    env = _Env(main_window)
    tool = main_window._tool_manager._tools_by_id[tool_id]
    script = _SCRIPTS[tool_id]

    # Run 0: idle, mid-gesture, commit.
    _prepare_and_activate(env, tool, script, 0)
    assert tool.holds_uncommitted_changes is False, "idle"
    before = env.fingerprint()
    depth = env.undo_depth()

    script.begin(env, tool, 0)
    assert tool.has_active_gesture is script.gesture, "the script reached the mid-gesture state"
    assert tool.holds_uncommitted_changes is script.mid, "mid-gesture"
    assert env.undo_depth() == depth, "nothing is on the undo stack mid-gesture"
    if script.mid:
        assert env.fingerprint() != before, "a holding tool has really changed the model"
    else:
        assert env.fingerprint() == before, "a tool that holds nothing left the model alone"

    script.commit(env, tool, 0)
    assert tool.holds_uncommitted_changes is False, "after the gesture commits"
    if script.commits:
        assert env.undo_depth() > depth, "the script really did commit a command"

    # Run 1: idle again, mid-gesture, Escape.
    _prepare_and_activate(env, tool, script, 1)
    assert tool.holds_uncommitted_changes is False, "idle again"
    before = env.fingerprint()

    script.begin(env, tool, 1)
    assert tool.holds_uncommitted_changes is script.mid, "mid-gesture, second run"
    _escape(env, tool, 1)
    assert tool.holds_uncommitted_changes is script.after_escape, "after Escape"
    if not script.after_escape:
        assert env.fingerprint() == before, "Escape left the model exactly as it was"
    else:
        # Escape is inert for this tool; the stroke is still live and releasing finishes it.
        script.commit(env, tool, 1)
        assert tool.holds_uncommitted_changes is False, "after the stroke is released"


@pytest.mark.parametrize("tool_id", ["eraser", "paint"])
def test_switching_tools_mid_stroke_rolls_back_so_nothing_is_held(main_window, tool_id):
    env = _Env(main_window)
    tool = main_window._tool_manager._tools_by_id[tool_id]
    script = _SCRIPTS[tool_id]
    _prepare_and_activate(env, tool, script, 0)
    before = env.fingerprint()
    script.begin(env, tool, 0)
    assert tool.holds_uncommitted_changes is True
    tool.deactivate()
    assert tool.holds_uncommitted_changes is False
    assert env.fingerprint() == before


def test_line_holds_nothing_when_every_click_so_far_reused_existing_vertices(main_window):
    """The start point snapped to an existing vertex adds no command, so the
    composite is empty and an autosave there would save exactly what is in the
    model. The first click on empty space does add a vertex and does hold."""
    env = _Env(main_window)
    tool = main_window._tool_manager._tools_by_id["line"]
    tool.activate(env.ctx)
    vid = env.squares[0][1][0]

    tool.on_mouse_press(_press(), _snap([0, 0, 0], SnapKind.ENDPOINT, vid))
    assert tool.has_active_gesture is True
    assert tool.holds_uncommitted_changes is False

    tool.on_mouse_press(_press(), _snap([-3, 5, 0]))
    assert tool.holds_uncommitted_changes is True


def test_line_holds_from_the_first_click_on_empty_space(main_window):
    env = _Env(main_window)
    tool = main_window._tool_manager._tools_by_id["line"]
    tool.activate(env.ctx)
    tool.on_mouse_press(_press(), _snap([-3, 5, 0]))
    assert tool.holds_uncommitted_changes is True
    _escape(env, tool, 0)
    assert tool.holds_uncommitted_changes is False


def test_paint_placement_drag_holds_only_once_it_has_moved(main_window):
    env = _Env(main_window)
    tool = main_window._tool_manager._tools_by_id["paint"]
    tool.activate(env.ctx)
    face = env.squares[0][0]
    start = env.scene.face_placement(face, Side.FRONT)

    tool.begin_placement_drag(face, Side.FRONT)
    assert tool.has_active_gesture is True
    assert tool.holds_uncommitted_changes is False, "armed but nothing written yet"
    assert env.scene.face_placement(face, Side.FRONT) == start

    tool.update_placement_drag(0.25, 0.0)
    assert env.scene.face_placement(face, Side.FRONT) != start, "the drag writes live"
    assert tool.holds_uncommitted_changes is True

    tool.end_placement_drag()
    assert tool.holds_uncommitted_changes is False
    assert env.scene.face_placement(face, Side.FRONT) != start, "the drag was committed"


def test_paint_placement_drag_rolls_back_on_tool_switch(main_window):
    env = _Env(main_window)
    tool = main_window._tool_manager._tools_by_id["paint"]
    tool.activate(env.ctx)
    face = env.squares[0][0]

    tool.begin_placement_drag(face, Side.FRONT)
    tool.update_placement_drag(0.25, 0.0)
    assert tool.holds_uncommitted_changes is True
    tool.deactivate()
    assert tool.holds_uncommitted_changes is False
    assert env.scene.face_placement(face, Side.FRONT) == TexturePlacement()


def test_a_stroke_that_paints_nothing_holds_nothing(main_window):
    """Painting the face's current material changes no model state, so there is
    nothing to roll back and an autosave is free to run."""
    env = _Env(main_window)
    tool = main_window._tool_manager._tools_by_id["paint"]
    # Active material is the default, which the face already has.
    tool.activate(env.ctx)
    tool.on_mouse_press(_press(env.px(env.face_center(0))), None)
    assert tool.has_active_gesture is True
    assert tool.holds_uncommitted_changes is False
    tool.on_mouse_release(_release(), None)


def test_eraser_that_misses_every_edge_holds_nothing(main_window):
    env = _Env(main_window)
    tool = main_window._tool_manager._tools_by_id["eraser"]
    tool.activate(env.ctx)
    tool.on_mouse_press(_press((5, 5)), None)
    assert tool.has_active_gesture is True
    assert tool.holds_uncommitted_changes is False
    tool.on_mouse_release(_release((5, 5)), None)
