"""The Tool Settings tab (M7.3): the active tool's options, or nothing.

Hosts the three option-bar widgets M7a/M7b/M7c already shipped. They are
REPARENTED here, not rewritten: same classes, same constructors, same
refresh(). Only 3 of 18 tools have settings, so the empty page is the common
case and says so plainly rather than showing a blank panel.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QStackedWidget, QVBoxLayout, QWidget


class ToolSettingsPage(QWidget):
    """A stack of option bars keyed by tool, plus an empty page."""

    EMPTY_TEXT = "No settings for this tool."

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._keys: dict[str, int] = {}
        self._current: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._stack = QStackedWidget(self)
        self._empty = QLabel(self.EMPTY_TEXT, self._stack)
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setWordWrap(True)
        self._empty_index = self._stack.addWidget(self._empty)
        layout.addWidget(self._stack)
        self._stack.setCurrentIndex(self._empty_index)

    def add_bar(self, key: str, widget: QWidget) -> None:
        """Adopt an option bar. Reparenting is what moves it out of wherever
        it was; the widget itself is not modified.

        Raises `KeyError` if `key` is already registered. Deferred as a
        Minor in M7.3 (#105) when there were only three call sites (wall,
        opening, roof) and a collision was theoretical; M7.4 Task 11 brings
        four more (box, cylinder, cone, sphere), so a copy-pasted key is no
        longer a hypothetical mistake.
        """
        if key in self._keys:
            raise KeyError(f"a bar is already registered for {key!r}")
        self._keys[key] = self._stack.addWidget(widget)

    def show_bar(self, key: str | None) -> None:
        """Show `key`'s bar, or the empty page for None or an unknown key."""
        index = self._keys.get(key) if key is not None else None
        if index is None:
            self._current = None
            self._stack.setCurrentIndex(self._empty_index)
            return
        self._current = key
        self._stack.setCurrentIndex(index)

    @property
    def current_key(self) -> str | None:
        return self._current

    def current_widget(self) -> QWidget:
        return self._stack.currentWidget()

    def current_text(self) -> str:
        widget = self._stack.currentWidget()
        return widget.text() if isinstance(widget, QLabel) else ""
