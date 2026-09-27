from __future__ import annotations

from bermake.tools.roof_tool import RoofTool
from bermake.ui.roof_options_bar import RoofOptionsBar


def test_slope_field_updates_tool(qtbot):
    tool = RoofTool()
    bar = RoofOptionsBar(tool, units_provider=lambda: None)
    qtbot.addWidget(bar)
    bar._slope_edit.setText("45")
    bar._on_slope_committed()
    assert abs(tool.slope - 45.0) < 1e-6


def test_toggle_sets_kind(qtbot):
    tool = RoofTool()
    bar = RoofOptionsBar(tool, units_provider=lambda: None)
    qtbot.addWidget(bar)
    bar.set_kind("hip")
    assert tool.kind == "hip"
    bar.set_kind("shed")
    assert tool.kind == "shed"
    bar.set_kind("gable")
    assert tool.kind == "gable"


def test_bad_slope_ignored(qtbot):
    tool = RoofTool()
    bar = RoofOptionsBar(tool, units_provider=lambda: None)
    qtbot.addWidget(bar)
    bar._slope_edit.setText("bogus")
    bar._on_slope_committed()
    assert tool.slope == 30.0
    bar._slope_edit.setText("90")   # out of (0, 85]
    bar._on_slope_committed()
    assert tool.slope == 30.0
