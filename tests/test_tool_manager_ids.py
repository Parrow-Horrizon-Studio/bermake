"""Tools are identified by id; the shortcut is optional (M7.4 Task 5).

Before this task, ToolManager.register keyed its registry by
`tool.shortcut.upper()`. That made the shortcut a tool's identity: two tools
sharing no shortcut (both `""`) silently overwrote each other, and a
shortcut-less tool could never be armed at all -- exactly the situation M7.4
needs for the four primitives and Follow Me (spec D9). These tests pin the
id-based registry and confirm the shortcut path is unchanged for tools that
still have one.
"""

from __future__ import annotations

from bermake.scene import Scene
from bermake.tools.arc_tool import ArcTool
from bermake.tools.circle_tool import CircleTool
from bermake.tools.dimension_tool import DimensionTool
from bermake.tools.erase_tool import EraserTool
from bermake.tools.follow_me_tool import FollowMeTool
from bermake.tools.line_tool import LineTool
from bermake.tools.move_tool import MoveTool
from bermake.tools.offset_tool import OffsetTool
from bermake.tools.opening_tool import DoorWindowTool
from bermake.tools.paint_tool import PaintTool
from bermake.tools.polygon_tool import PolygonTool
from bermake.tools.primitive_tool import BoxTool, ConeTool, CylinderTool, SphereTool
from bermake.tools.push_pull_tool import PushPullTool
from bermake.tools.rectangle_tool import RectangleTool
from bermake.tools.roof_tool import RoofTool
from bermake.tools.rotate_tool import RotateTool
from bermake.tools.scale_tool import ScaleTool
from bermake.tools.select_tool import SelectTool
from bermake.tools.tape_measure_tool import TapeMeasureTool
from bermake.tools.text_tool import TextTool
from bermake.tools.tool import ToolContext
from bermake.tools.tool_manager import ToolManager
from bermake.tools.wall_tool import WallTool
from bermake.ui.actions import ACTIONS, TOOL_GROUP

_ALL_TOOL_CLASSES = (
    ArcTool,
    BoxTool,
    CircleTool,
    ConeTool,
    CylinderTool,
    DimensionTool,
    EraserTool,
    FollowMeTool,
    LineTool,
    MoveTool,
    OffsetTool,
    DoorWindowTool,
    PaintTool,
    PolygonTool,
    PushPullTool,
    RectangleTool,
    RoofTool,
    RotateTool,
    ScaleTool,
    SelectTool,
    SphereTool,
    TapeMeasureTool,
    TextTool,
    WallTool,
)


def _ctx() -> ToolContext:
    return ToolContext(scene=Scene())


class _NoShortcutTool(LineTool):
    """A tool with no keyboard shortcut, which the old registry could not hold."""

    @property
    def id(self) -> str:
        return "no_shortcut"

    @property
    def shortcut(self) -> str:
        return ""


def test_every_shipped_tool_has_a_unique_id_matching_its_action():
    # Ties tool.py's ids directly to actions.py's handler_arg (which _tool()
    # derives from the action id) so the two cannot drift apart -- exactly
    # what the brief warns against.
    expected_ids = {
        spec.handler_arg for spec in ACTIONS if spec.group == TOOL_GROUP
    }
    assert len(expected_ids) == 24

    mgr = ToolManager()
    for cls in _ALL_TOOL_CLASSES:
        mgr.register(cls())

    assert mgr.tool_ids() == expected_ids


def test_activate_by_id_arms_the_right_tool_among_several():
    # A broken implementation that ignores the id and arms "the only
    # registered tool" would pass a single-tool test trivially -- register
    # three and check each id resolves to its own instance.
    mgr = ToolManager(_ctx())
    line, select, circle = LineTool(), SelectTool(), CircleTool()
    mgr.register(line)
    mgr.register(select)
    mgr.register(circle)

    assert mgr.activate_by_id("circle") is True
    assert mgr.active is circle

    assert mgr.activate_by_id("select") is True
    assert mgr.active is select

    assert mgr.activate_by_id("line") is True
    assert mgr.active is line


def test_two_shortcutless_tools_can_both_register_and_are_independently_reachable():
    # The old registry keyed on shortcut.upper(), so two tools with no
    # shortcut would have overwritten each other silently -- the exact
    # collision this task exists to prevent. This is the sharpest version of
    # that regression: two shortcut-less tools, each independently armable.
    mgr = ToolManager(_ctx())

    class _A(_NoShortcutTool):
        @property
        def id(self) -> str:
            return "a"

    class _B(_NoShortcutTool):
        @property
        def id(self) -> str:
            return "b"

    a, b = _A(), _B()
    mgr.register(a)
    mgr.register(b)
    assert mgr.tool_ids() == {"a", "b"}

    assert mgr.activate_by_id("a") is True
    assert mgr.active is a

    assert mgr.activate_by_id("b") is True
    assert mgr.active is b


def test_activate_by_shortcut_still_works_for_tools_that_have_one():
    mgr = ToolManager(_ctx())
    line = LineTool()
    mgr.register(line)
    assert mgr.activate_by_shortcut("L") is True
    assert mgr.active is line


def test_activate_by_shortcut_ignores_the_empty_string():
    # Otherwise a stray empty shortcut would arm an arbitrary shortcut-less
    # tool -- the empty key must never be treated as "no filter".
    mgr = ToolManager(_ctx())
    mgr.register(_NoShortcutTool())
    assert mgr.activate_by_shortcut("") is False
    assert mgr.active is None


def test_activate_by_id_with_an_unknown_id_returns_false():
    mgr = ToolManager(_ctx())
    mgr.register(LineTool())
    assert mgr.activate_by_id("not_a_real_tool") is False
    assert mgr.active is None


def test_registering_two_tools_with_the_same_id_the_second_replaces_the_first():
    # register() is a plain dict-keyed overwrite, same silent-replace
    # semantics the old shortcut-keyed dict always had -- deliberately kept,
    # not accidental, and pinned here so a future change to that behaviour
    # is a conscious decision rather than a regression.
    mgr = ToolManager(_ctx())

    class _First(_NoShortcutTool):
        @property
        def id(self) -> str:
            return "dup"

    class _Second(_NoShortcutTool):
        @property
        def id(self) -> str:
            return "dup"

    first, second = _First(), _Second()
    mgr.register(first)
    mgr.register(second)

    assert mgr.tool_ids() == {"dup"}
    assert mgr.activate_by_id("dup") is True
    assert mgr.active is second
