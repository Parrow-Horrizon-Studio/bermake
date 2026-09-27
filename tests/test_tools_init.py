"""`bermake.tools` re-exports concrete tool classes lazily (PEP 562
`__getattr__`) instead of importing them eagerly at package-init time, so
that a Qt-free submodule such as `bermake.tools.sweep_support` does not pull
in PySide6 merely because Python runs the package's `__init__.py` before the
submodule itself (M7.4 Task 3).
"""

from __future__ import annotations

import pytest

import bermake.tools as tools_pkg

# Every name the package advertises, and the module each must come from.
_EXPECTED = {
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
    "ToolManager": "bermake.tools.tool_manager",
    "ToolOverlay": "bermake.tools.tool",
}


def test_all_matches_the_expected_export_set():
    assert set(tools_pkg.__all__) == set(_EXPECTED)


@pytest.mark.parametrize("name", sorted(_EXPECTED))
def test_each_public_name_resolves_to_the_right_class(name: str):
    import importlib

    resolved = getattr(tools_pkg, name)
    expected_module = importlib.import_module(_EXPECTED[name])
    assert resolved is getattr(expected_module, name)


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError):
        tools_pkg.NotARealTool


def test_dir_reports_the_public_names():
    assert sorted(dir(tools_pkg)) == sorted(_EXPECTED)
