"""Bottom-of-viewport status bar (M4d, split into three fields in M7.3).

Left: the prompt -- tool, snap, breadcrumb, one-line message and selection,
joined by `·`, exactly as before. Middle: a unit-aware cursor-coordinate
readout. Right: the Measurements box, which through v0.4.0 was one more
anonymous run of text in the same label.

The Measurements box stays a DISPLAY driven by MainWindow's existing
type-anywhere key capture. It is deliberately not a QLineEdit and takes no
focus: making it focusable risks swallowing keystrokes the viewport needs.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

# The status bar paints over the VIEWPORT, so these two alphas have to work
# against whatever environment the document carries. At the original 0.5, black
# over a white background resolved to about mid grey behind #dddddd text, close
# to 2.3:1 and unreadable. 0.72 is near 6.6:1 and still reads over the dark
# environment, so one value serves every environment and the status bar stays
# independent of document state (spec D10).
_CHIP_ALPHA = 0.72
_BOX_ALPHA = 0.82

_FIELD_STYLE = (
    f"QLabel {{ background-color: rgba(0, 0, 0, {_CHIP_ALPHA}); color: #dddddd;"
    " padding: 4px 10px; font-family: sans-serif; font-size: 11px; }"
)
_BOX_STYLE = (
    f"QLabel {{ background-color: rgba(0, 0, 0, {_BOX_ALPHA}); color: #ffffff;"
    " border: 1px solid #888; padding: 3px 8px; font-family: monospace;"
    " font-size: 11px; min-width: 70px; }"
)

_MEASUREMENTS_LABEL = "Measurements"


class StatusBar(QWidget):
    """Prompt (left) · coordinates (middle) · Measurements box (right)."""

    def __init__(self) -> None:
        super().__init__()
        self._tool: str = ""
        self._snap: str = ""
        self._selection: str = ""
        self._breadcrumb: str = ""
        self._message: str = ""

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._prompt = QLabel("", self)
        self._prompt.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._prompt.setStyleSheet(_FIELD_STYLE)

        self._coordinates = QLabel("", self)
        self._coordinates.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._coordinates.setStyleSheet(_FIELD_STYLE)

        self._measurements_label = QLabel(_MEASUREMENTS_LABEL, self)
        self._measurements_label.setStyleSheet(_FIELD_STYLE)
        self._measurements = QLabel("", self)
        self._measurements.setStyleSheet(_BOX_STYLE)
        self._measurements.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._measurements.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout.addWidget(self._prompt, stretch=1)
        layout.addWidget(self._coordinates, stretch=0)
        layout.addWidget(self._measurements_label, stretch=0)
        layout.addWidget(self._measurements, stretch=0)
        self.setMinimumHeight(24)

    # --- setters (unchanged signatures) ----------------------------------
    def set_tool(self, name: str) -> None:
        self._tool = name
        self._refresh()

    def set_snap(self, label: str) -> None:
        self._snap = label
        self._refresh()

    def set_status(self, text: str) -> None:
        """The Measurements box content (typed value, or a tool's read-out)."""
        self._measurements.setText(text or "")

    def set_message(self, text: str) -> None:
        """A one-line notice for the prompt area -- not a measurement."""
        self._message = text or ""
        self._refresh()

    def set_selection(self, text: str) -> None:
        self._selection = text or ""
        self._refresh()

    def set_breadcrumb(self, text: str) -> None:
        self._breadcrumb = text or ""
        self._refresh()

    def set_coordinates(self, xyz_text: str) -> None:
        self._coordinates.setText(xyz_text or "")

    # --- readers ---------------------------------------------------------
    def prompt_text(self) -> str:
        return self._prompt.text()

    def coordinates_text(self) -> str:
        return self._coordinates.text()

    def measurements_text(self) -> str:
        return self._measurements.text()

    def measurements_label_text(self) -> str:
        return self._measurements_label.text()

    def measurements_widget(self) -> QLabel:
        return self._measurements

    def _refresh(self) -> None:
        if not self._tool:
            parts = [p for p in (self._breadcrumb, self._message, self._selection) if p]
            self._prompt.setText(" · ".join(parts))
            return
        snap = self._snap if self._snap else "—"
        parts = [f"{self._tool} · {snap}"]
        if self._breadcrumb:
            parts.append(self._breadcrumb)
        if self._message:
            parts.append(self._message)
        if self._selection:
            parts.append(self._selection)
        self._prompt.setText(" · ".join(parts))
