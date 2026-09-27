"""OpeningOptionsBar (M7b): Door|Window toggle + size fields for the tool."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QRadioButton,
    QWidget,
)

from bermake.units import format_length, parse_length


class OpeningOptionsBar(QWidget):
    """A compact row: Door|Window radio toggle + unit-aware width/height/sill/
    depth fields bound to a DoorWindowTool. MainWindow shows it only while the
    tool is active."""

    def __init__(self, tool, units_provider) -> None:
        super().__init__()
        self._tool = tool
        self._units = units_provider
        self._door_btn = QRadioButton("Door")
        self._window_btn = QRadioButton("Window")
        self._group = QButtonGroup(self)
        self._group.addButton(self._door_btn)
        self._group.addButton(self._window_btn)
        self._width_edit = QLineEdit()
        self._height_edit = QLineEdit()
        self._sill_edit = QLineEdit()
        self._depth_edit = QLineEdit()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.addWidget(self._door_btn)
        layout.addWidget(self._window_btn)
        layout.addWidget(QLabel("W:"))
        layout.addWidget(self._width_edit)
        layout.addWidget(QLabel("H:"))
        layout.addWidget(self._height_edit)
        layout.addWidget(QLabel("Sill:"))
        layout.addWidget(self._sill_edit)
        layout.addWidget(QLabel("Depth:"))
        layout.addWidget(self._depth_edit)
        layout.addStretch(1)

        self._door_btn.setChecked(self._tool.kind == "door")
        self._window_btn.setChecked(self._tool.kind == "window")
        self._door_btn.clicked.connect(lambda: self.set_kind("door"))
        self._window_btn.clicked.connect(lambda: self.set_kind("window"))
        self._width_edit.editingFinished.connect(self._on_width_committed)
        self._height_edit.editingFinished.connect(self._on_height_committed)
        self._sill_edit.editingFinished.connect(self._on_sill_committed)
        self._depth_edit.editingFinished.connect(self._on_depth_committed)
        self.refresh()

    def set_kind(self, kind) -> None:
        self._tool.kind = kind
        self._door_btn.setChecked(kind == "door")
        self._window_btn.setChecked(kind == "window")
        self._sill_edit.setEnabled(kind == "window")
        self.refresh()

    def refresh(self) -> None:
        u = self._units()
        self._width_edit.setText(format_length(self._tool.width, u))
        self._height_edit.setText(format_length(self._tool.height, u))
        self._sill_edit.setText(format_length(self._tool.sill, u))
        self._depth_edit.setText(format_length(self._tool.depth, u))
        self._sill_edit.setEnabled(self._tool.kind == "window")

    def _commit(self, edit, attr) -> None:
        value = parse_length(edit.text(), self._units())
        if value is not None and value > 0:
            setattr(self._tool, attr, value)
        self.refresh()

    def _on_width_committed(self) -> None:
        self._commit(self._width_edit, "width")

    def _on_height_committed(self) -> None:
        self._commit(self._height_edit, "height")

    def _on_sill_committed(self) -> None:
        self._commit(self._sill_edit, "sill")

    def _on_depth_committed(self) -> None:
        self._commit(self._depth_edit, "depth")
