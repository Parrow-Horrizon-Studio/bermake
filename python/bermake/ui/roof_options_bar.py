"""RoofOptionsBar (M7c): Gable|Hip|Shed toggle + slope field for the tool."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QRadioButton,
    QWidget,
)

from bermake.geometry.roof import _MAX_SLOPE_DEG
from bermake.units import format_angle, parse_angle

_KINDS = ("gable", "hip", "shed")


class RoofOptionsBar(QWidget):
    """A compact row: Gable|Hip|Shed radio toggle + a slope-degrees field bound
    to a RoofTool. MainWindow shows it only while the tool is active."""

    def __init__(self, tool, units_provider) -> None:
        super().__init__()
        self._tool = tool
        self._units = units_provider
        self._buttons = {}
        self._group = QButtonGroup(self)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        for kind in _KINDS:
            btn = QRadioButton(kind.capitalize())
            self._buttons[kind] = btn
            self._group.addButton(btn)
            layout.addWidget(btn)
            btn.clicked.connect(lambda _checked=False, k=kind: self.set_kind(k))
        layout.addWidget(QLabel("Slope:"))
        self._slope_edit = QLineEdit()
        layout.addWidget(self._slope_edit)
        layout.addStretch(1)

        self._slope_edit.editingFinished.connect(self._on_slope_committed)
        self.refresh()

    def set_kind(self, kind) -> None:
        self._tool.kind = kind
        self._buttons[kind].setChecked(True)
        self.refresh()

    def refresh(self) -> None:
        self._buttons[self._tool.kind].setChecked(True)
        self._slope_edit.setText(format_angle(self._tool.slope))

    def _on_slope_committed(self) -> None:
        value = parse_angle(self._slope_edit.text())
        if value is not None and 0.0 < value <= _MAX_SLOPE_DEG:
            self._tool.slope = value
        self.refresh()
