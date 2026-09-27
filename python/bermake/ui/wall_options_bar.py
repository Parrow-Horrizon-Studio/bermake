"""WallOptionsBar (M7a): thickness/height settings row for the Wall tool."""

from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QWidget

from bermake.units import format_length, parse_length


class WallOptionsBar(QWidget):
    """A compact row with unit-aware Thickness + Height fields bound to a
    WallTool. MainWindow shows it only while the Wall tool is active."""

    def __init__(self, wall_tool, units_provider) -> None:
        super().__init__()
        self._tool = wall_tool
        self._units = units_provider
        self._thickness_edit = QLineEdit()
        self._height_edit = QLineEdit()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.addWidget(QLabel("Wall thickness:"))
        layout.addWidget(self._thickness_edit)
        layout.addWidget(QLabel("height:"))
        layout.addWidget(self._height_edit)
        layout.addStretch(1)
        self._thickness_edit.editingFinished.connect(self._on_thickness_committed)
        self._height_edit.editingFinished.connect(self._on_height_committed)
        self.refresh()

    def refresh(self) -> None:
        u = self._units()
        self._thickness_edit.setText(format_length(self._tool.thickness, u))
        self._height_edit.setText(format_length(self._tool.height, u))

    def _on_thickness_committed(self) -> None:
        v = parse_length(self._thickness_edit.text(), self._units())
        if v is not None and v > 0:
            self._tool.thickness = v
        self.refresh()

    def _on_height_committed(self) -> None:
        v = parse_length(self._height_edit.text(), self._units())
        if v is not None and v > 0:
            self._tool.height = v
        self.refresh()
