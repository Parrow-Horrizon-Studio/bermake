"""PrimitiveOptionsBar (M7.4 Task 11): segments/rings settings row shared by
the four primitive tools.

Box has neither field -- its footprint and height come entirely from the
drag gesture, not a typed value -- so its bar is intentionally blank. It
still gets its own key in ToolSettingsPage (see MainWindow) so arming Box
behaves like arming any other primitive tool rather than needing a special
case in `_refresh_tool_options`.

One shared class, not four near-identical ones: M7.4 Tasks 6 and 8 each
wrote a private helper that hand-duplicated shared logic and had it removed
on review. WallOptionsBar / RoofOptionsBar / OpeningOptionsBar are one class
per tool because each binds different, tool-specific fields; the four
primitive bars differ only in *which* of the same two fields they show, so
one parametrized class is the right shape here instead of four copies of it.
"""

from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QLabel, QSpinBox, QWidget

from bermake.tools.primitive_tool import RINGS_FLOOR, SEGMENTS_FLOOR

_SPIN_MAX = 512  # generous ceiling; the generators have no upper limit of their own


class PrimitiveOptionsBar(QWidget):
    """A compact row with Rings (sphere only) + Segments (cylinder/cone/
    sphere) fields bound to a `PrimitiveTool`. MainWindow shows it only
    while the matching tool is active."""

    def __init__(self, tool, *, has_segments: bool, has_rings: bool) -> None:
        super().__init__()
        self._tool = tool
        self._segments_spin: QSpinBox | None = None
        self._rings_spin: QSpinBox | None = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)

        if has_rings:
            layout.addWidget(QLabel("Rings:"))
            self._rings_spin = QSpinBox()
            self._rings_spin.setRange(RINGS_FLOOR, _SPIN_MAX)
            layout.addWidget(self._rings_spin)
            self._rings_spin.valueChanged.connect(self._on_rings_changed)

        if has_segments:
            layout.addWidget(QLabel("Segments:"))
            self._segments_spin = QSpinBox()
            self._segments_spin.setRange(SEGMENTS_FLOOR, _SPIN_MAX)
            layout.addWidget(self._segments_spin)
            self._segments_spin.valueChanged.connect(self._on_segments_changed)

        layout.addStretch(1)
        self.refresh()

    def refresh(self) -> None:
        if self._rings_spin is not None:
            self._rings_spin.blockSignals(True)
            self._rings_spin.setValue(self._tool.rings)
            self._rings_spin.blockSignals(False)
        if self._segments_spin is not None:
            self._segments_spin.blockSignals(True)
            self._segments_spin.setValue(self._tool.segments)
            self._segments_spin.blockSignals(False)

    def _on_segments_changed(self, value: int) -> None:
        self._tool.segments = value

    def _on_rings_changed(self, value: int) -> None:
        self._tool.rings = value
