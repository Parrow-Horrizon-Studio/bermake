"""Tool framework: Tool ABC, ToolOverlay, ToolManager, and concrete tools.

M2 ships LineTool and RectangleTool against this framework. M3's PushPullTool
and M4's full roster plug into the same shapes.

Concrete tool classes are re-exported lazily (PEP 562 module `__getattr__`)
rather than imported eagerly at package-init time. Most tool modules import
PySide6 for event types; importing any of them unconditionally here meant
importing ANY submodule of `bermake.tools` — including a Qt-free one, such as
`bermake.tools.sweep_support` — transitively loaded PySide6, since Python
always runs a package's `__init__.py` before the submodule itself. Lazy
attribute access keeps `from bermake.tools import PushPullTool` working
unchanged while letting Qt-free submodules stay Qt-free on import.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bermake.tools.arc_tool import ArcTool
    from bermake.tools.circle_tool import CircleTool
    from bermake.tools.erase_tool import EraserTool
    from bermake.tools.line_tool import LineTool
    from bermake.tools.move_tool import MoveTool
    from bermake.tools.polygon_tool import PolygonTool
    from bermake.tools.push_pull_tool import PushPullTool
    from bermake.tools.rectangle_tool import RectangleTool
    from bermake.tools.rotate_tool import RotateTool
    from bermake.tools.scale_tool import ScaleTool
    from bermake.tools.select_tool import SelectTool
    from bermake.tools.tape_measure_tool import TapeMeasureTool
    from bermake.tools.tool import Tool, ToolContext, ToolOverlay
    from bermake.tools.tool_manager import ToolManager

__all__ = [
    "ArcTool",
    "CircleTool",
    "EraserTool",
    "LineTool",
    "MoveTool",
    "PolygonTool",
    "PushPullTool",
    "RectangleTool",
    "RotateTool",
    "ScaleTool",
    "SelectTool",
    "TapeMeasureTool",
    "Tool",
    "ToolContext",
    "ToolManager",
    "ToolOverlay",
]

_MODULE_BY_NAME = {
    "ArcTool": "bermake.tools.arc_tool",
    "CircleTool": "bermake.tools.circle_tool",
    "EraserTool": "bermake.tools.erase_tool",
    "LineTool": "bermake.tools.line_tool",
    "MoveTool": "bermake.tools.move_tool",
    "PolygonTool": "bermake.tools.polygon_tool",
    "PushPullTool": "bermake.tools.push_pull_tool",
    "RectangleTool": "bermake.tools.rectangle_tool",
    "RotateTool": "bermake.tools.rotate_tool",
    "ScaleTool": "bermake.tools.scale_tool",
    "SelectTool": "bermake.tools.select_tool",
    "TapeMeasureTool": "bermake.tools.tape_measure_tool",
    "Tool": "bermake.tools.tool",
    "ToolContext": "bermake.tools.tool",
    "ToolOverlay": "bermake.tools.tool",
    "ToolManager": "bermake.tools.tool_manager",
}


def __getattr__(name: str):
    module_name = _MODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(__all__)
