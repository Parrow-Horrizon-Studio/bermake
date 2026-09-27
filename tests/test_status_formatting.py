"""Numeric status_text honours the unit system when a units_provider is set."""

from __future__ import annotations

import numpy as np
from bermake.scene.scene import Scene
from bermake.tools.push_pull_tool import PushPullTool
from bermake.tools.tool import ToolContext
from bermake.units import Units, UnitSystem


def _ctx(s, units):
    return ToolContext(scene=s, command_stack=None, camera=None,
                       widget_size_provider=lambda: (800, 600), units_provider=lambda: units)


def test_pushpull_depth_formats_imperial(qtbot):
    s = Scene()
    a = s.add_vertex(np.array([0, 0, 0], np.float32))
    b = s.add_vertex(np.array([1, 0, 0], np.float32))
    c = s.add_vertex(np.array([1, 1, 0], np.float32))
    d = s.add_vertex(np.array([0, 1, 0], np.float32))
    f = s.add_face_from_loop([a, b, c, d])
    t = PushPullTool()
    t.activate(_ctx(s, Units(system=UnitSystem.IMPERIAL)))
    t._arm_face(f)
    t._current_depth = 0.0254 * 12  # 1 foot
    assert "'" in (t.status_text or "")   # shows feet/inches, not "0.305"
