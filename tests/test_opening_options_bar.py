from __future__ import annotations

from bermake.tools.opening_tool import DoorWindowTool
from bermake.ui.opening_options_bar import OpeningOptionsBar
from bermake.units import Units


def test_fields_update_tool(qtbot):
    tool = DoorWindowTool()
    bar = OpeningOptionsBar(tool, units_provider=lambda: Units())
    qtbot.addWidget(bar)
    bar._width_edit.setText("1000mm")
    bar._on_width_committed()
    assert abs(tool.width - 1.0) < 1e-6
    bar._sill_edit.setText("800mm")
    bar._on_sill_committed()
    assert abs(tool.sill - 0.8) < 1e-6


def test_toggle_sets_kind_and_reloads(qtbot):
    tool = DoorWindowTool()
    bar = OpeningOptionsBar(tool, units_provider=lambda: Units())
    qtbot.addWidget(bar)
    bar.set_kind("window")
    assert tool.kind == "window"
    bar.set_kind("door")
    assert tool.kind == "door"


def test_bad_input_ignored(qtbot):
    tool = DoorWindowTool()
    bar = OpeningOptionsBar(tool, units_provider=lambda: Units())
    qtbot.addWidget(bar)
    bar._height_edit.setText("bogus")
    bar._on_height_committed()
    assert tool.height == 2.1
