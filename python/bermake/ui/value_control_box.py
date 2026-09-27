"""ValueControlBox — pure state for the Measurements box (VCB).

Holds the typed buffer + an `active` flag. The MainWindow event filter feeds
it characters; the active tool consumes `text` on Enter. No Qt dependency.
"""

from __future__ import annotations


class ValueControlBox:
    def __init__(self) -> None:
        self._buffer = ""
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    @property
    def text(self) -> str:
        return self._buffer

    def feed(self, ch: str) -> None:
        self._buffer += ch
        self._active = True

    def backspace(self) -> None:
        self._buffer = self._buffer[:-1]
        if not self._buffer:
            self._active = False

    def clear(self) -> None:
        self._buffer = ""
        self._active = False
