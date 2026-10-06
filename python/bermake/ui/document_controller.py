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
        # A recovered document has no path but is titled after the drawing it
        # came from, as "House.berm (recovered)" (M7.12, spec 4.5).
        self.recovered_name: str | None = None

    def mark_dirty(self) -> None:
        self.dirty = True

    def mark_clean(self) -> None:
        self.dirty = False

    def set_path(self, path) -> None:
        self.current_path = Path(path) if path else None
        if self.current_path is not None:
            self.recovered_name = None

    def display_title(self) -> str:
        star = "*" if self.dirty else ""
        if self.current_path is None and self.recovered_name:
            return f"{self.recovered_name} (recovered){star} - {_APP}"
        name = self.current_path.name if self.current_path else "Untitled"
        return f"{name}{star} - {_APP}"
