"""Unit tests for the Tool framework — ABC, overlay dataclass, manager."""

from __future__ import annotations

import numpy as np

from bermake.scene import Scene
from bermake.tools import Tool, ToolContext, ToolManager, ToolOverlay


class FakeTool(Tool):
    """Minimal Tool subclass for unit-testing ToolManager.

    Defined inline rather than as a shared fixture because the existing
    `tests/` directory isn't a Python package and `pythonpath` isn't
    configured — sharing across test files would require build-system
    changes not worth it for a single helper.
    """

    def __init__(self, name: str = "Fake", shortcut: str = "F") -> None:
        self._name = name
        self._shortcut = shortcut
        self.activated = False
        self.deactivated = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def shortcut(self) -> str:
        return self._shortcut

    @property
    def id(self) -> str:
        return self._name.lower()

    def activate(self, ctx: ToolContext) -> None:
        self.activated = True

    def deactivate(self) -> None:
        self.deactivated = True

    def overlay(self) -> ToolOverlay:
        return ToolOverlay(
            rubber_band_segments=np.zeros((0, 3), dtype=np.float32),
            rubber_band_color=(1.0, 1.0, 1.0),
            snap_marker_position=None,
            snap_marker_color=(1.0, 1.0, 1.0),
        )

    @property
    def has_active_gesture(self) -> bool:
        return False

    @property
    def anchor_or_none(self) -> np.ndarray | None:
        return None


def _ctx() -> ToolContext:
    return ToolContext(scene=Scene())


def test_tool_manager_starts_with_no_active_tool():
    mgr = ToolManager()
    assert mgr.active is None


def test_register_and_activate_by_shortcut():
    mgr = ToolManager(_ctx())
    t = FakeTool(name="Fake", shortcut="F")
    mgr.register(t)
    assert mgr.activate_by_shortcut("F") is True
    assert mgr.active is t
    assert t.activated is True


def test_activating_switches_and_deactivates_previous():
    mgr = ToolManager(_ctx())
    a = FakeTool(name="A", shortcut="A")
    b = FakeTool(name="B", shortcut="B")
    mgr.register(a)
    mgr.register(b)
    mgr.activate_by_shortcut("A")
    mgr.activate_by_shortcut("B")
    assert mgr.active is b
    assert a.deactivated is True


def test_deactivate_current_clears_active_tool():
    mgr = ToolManager(_ctx())
    t = FakeTool(name="T", shortcut="T")
    mgr.register(t)
    mgr.activate_by_shortcut("T")
    mgr.deactivate_current()
    assert mgr.active is None
    assert t.deactivated is True


def test_unknown_shortcut_returns_false():
    mgr = ToolManager()
    assert mgr.activate_by_shortcut("X") is False
    assert mgr.active is None


def test_activate_without_context_raises_runtime_error():
    """Activating a tool when no ToolContext has been set must raise."""
    import pytest as _pytest  # local import — test file doesn't otherwise import pytest

    mgr = ToolManager()  # no ctx
    mgr.register(FakeTool(name="F", shortcut="F"))
    with _pytest.raises(RuntimeError):
        mgr.activate_by_shortcut("F")
    # And no tool should be left "active" after the failure.
    assert mgr.active is None
