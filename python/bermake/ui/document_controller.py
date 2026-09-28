"""DocumentController (M6a): the document session state — path, dirty flag, and
the window-title string. No Qt widgets, so it is unit-testable headlessly.
"""

from __future__ import annotations

from pathlib import Path

_APP = "Bermake"


class DocumentController:
    def __init__(self) -> None:
        self.current_path: Path | None = None
        self.dirty: bool = False

    def mark_dirty(self) -> None:
        self.dirty = True

    def mark_clean(self) -> None:
        self.dirty = False

    def set_path(self, path) -> None:
        self.current_path = Path(path) if path else None

    def display_title(self) -> str:
        name = self.current_path.name if self.current_path else "Untitled"
        star = "*" if self.dirty else ""
        return f"{name}{star} - {_APP}"
