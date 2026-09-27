"""Rotate tool: cycling the forced axis back to None must restore auto-tilt (#49).

The primary test drives the REAL on_key_press cycle handler (Key_Up x4: None ->
0 -> 1 -> 2 -> None) rather than calling _effective_normal directly, because the
bug lives at the call site inside on_key_press (it fed the already-forced
self._normal back in as the "inferred" value), not inside _effective_normal
itself.
"""

from __future__ import annotations

import numpy as np
from bermake.tools.rotate_tool import RotateTool, _Stage
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent


def _key_up():
    return QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Up, Qt.KeyboardModifier.NoModifier)


def test_cycling_forced_axis_back_to_none_restores_inferred_normal_via_on_key_press(qtbot):
    tool = RotateTool()
    inferred = np.array([0.0, 1.0, 0.0], np.float32)  # a non-Z face normal

    # Simulate state right after center-placement inference: the raw inferred
    # normal is remembered, and _normal currently reflects it (forced_axis=None).
    tool._inferred_normal = inferred.copy()
    tool._forced_axis = None
    tool._normal = inferred.copy()
    tool._stage = _Stage.HAVE_CENTER  # non-IDLE, so the on_key_press guard applies

    # Cycle Up four times: None -> 0 -> 1 -> 2 -> None.
    for _ in range(4):
        tool.on_key_press(_key_up())

    assert tool._forced_axis is None
    assert np.allclose(tool._normal, inferred), (
        f"auto-tilt not restored: {tool._normal.tolist()} (expected {inferred.tolist()})"
    )


def test_effective_normal_helper_restores_inferred_when_cycled_back_to_none(qtbot):
    """Secondary: the brief's helper-pattern test, exercising _effective_normal directly."""
    tool = RotateTool()
    inferred = np.array([0.0, 1.0, 0.0], np.float32)
    tool._inferred_normal = inferred.copy()
    tool._forced_axis = None
    tool._normal = tool._effective_normal(inferred)
    assert np.allclose(tool._normal, inferred)

    order = [0, 1, 2, None]
    for axis in order:
        tool._forced_axis = axis
        tool._normal = tool._effective_normal(tool._inferred_normal)

    assert np.allclose(tool._normal, inferred), f"auto-tilt not restored: {tool._normal}"
