from __future__ import annotations

from bermake.tools.wall_tool import WallTool
from bermake.ui.wall_options_bar import WallOptionsBar
from bermake.units import Units


def test_fields_reflect_and_update_tool(qtbot):
    tool = WallTool()
    tool.thickness = 0.1
    tool.height = 2.4
    bar = WallOptionsBar(tool, units_provider=lambda: Units())
    qtbot.addWidget(bar)
    bar.refresh()
    # Editing the thickness field to a metric value updates the tool (meters).
    bar._thickness_edit.setText("200mm")
    bar._on_thickness_committed()
    assert abs(tool.thickness - 0.2) < 1e-6
    bar._height_edit.setText("3m")
    bar._on_height_committed()
    assert abs(tool.height - 3.0) < 1e-6


def test_bad_input_is_ignored(qtbot):
    tool = WallTool()
    bar = WallOptionsBar(tool, units_provider=lambda: Units())
    qtbot.addWidget(bar)
    bar._thickness_edit.setText("not a number")
    bar._on_thickness_committed()
    assert tool.thickness == 0.1        # unchanged
