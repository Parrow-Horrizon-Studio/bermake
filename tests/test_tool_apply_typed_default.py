from __future__ import annotations

from bermake.tools.tool import ToolContext


def test_toolcontext_has_units_provider():
    ctx = ToolContext(scene=object())
    assert ctx.units_provider is None


def test_apply_typed_value_default_false():
    from bermake.tools.line_tool import LineTool
    assert LineTool().apply_typed_value("3", None) is False
