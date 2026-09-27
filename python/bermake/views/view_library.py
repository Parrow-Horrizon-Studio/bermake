"""ViewLibrary (M7e): owns the ordered list of SavedViews for a document.

Mirrors TagLibrary/MaterialLibrary: a plain list owner with to_records() /
from_records() for .berm persistence. Lives on Model as `model.views`.
SavedView is frozen, so rename/replace swap in a new dataclass copy.
"""

from __future__ import annotations

from dataclasses import replace

from bermake.io.document_codec import CameraState
from bermake.views.saved_view import SavedView


class ViewLibrary:
    """The document's saved Scenes, in display order."""

    def __init__(self) -> None:
        self._views: list[SavedView] = []
        self._next_id = 0

    def add(self, view: SavedView) -> SavedView:
        """Append `view` (identity preserved) and advance next_id past its id."""
        self._views.append(view)
        self._next_id = max(self._next_id, int(view.id) + 1)
        return view

    def get(self, vid: int) -> SavedView | None:
        for v in self._views:
            if v.id == vid:
                return v
        return None

    def index_of(self, vid: int) -> int:
        for i, v in enumerate(self._views):
            if v.id == vid:
                return i
        return -1

    def remove(self, vid: int) -> None:
        i = self.index_of(vid)
        if i >= 0:
            del self._views[i]

    def insert(self, index: int, view: SavedView) -> None:
        """Restore `view` at `index` (used by DeleteViewCommand undo)."""
        self._views.insert(index, view)
        self._next_id = max(self._next_id, int(view.id) + 1)

    def rename(self, vid: int, name: str) -> None:
        """Rename (no-op on empty name); replaces the frozen view with a copy."""
        i = self.index_of(vid)
        if i >= 0 and name:
            self._views[i] = replace(self._views[i], name=str(name))

    def replace_view(self, vid: int, view: SavedView) -> None:
        """Overwrite the view at vid's position (used by UpdateViewCommand)."""
        i = self.index_of(vid)
        if i >= 0:
            self._views[i] = view

    def move(self, vid: int, direction: int) -> bool:
        """Swap one place up (<0) or down (>0). Returns False if clamped at an end."""
        i = self.index_of(vid)
        if i < 0:
            return False
        j = i + (1 if direction > 0 else -1)
        if j < 0 or j >= len(self._views):
            return False
        self._views[i], self._views[j] = self._views[j], self._views[i]
        return True

    def views(self) -> list[SavedView]:
        return list(self._views)

    @property
    def next_id(self) -> int:
        return self._next_id

    def to_records(self) -> list[dict]:
        records = []
        for v in self._views:
            records.append(
                {
                    "id": int(v.id),
                    "name": str(v.name),
                    "camera": v.camera.to_dict(),
                    "tag_visibility": {str(k): bool(vis) for k, vis in v.tag_visibility.items()},
                    "face_style": str(v.face_style),
                    "xray": bool(v.xray),
                    "color_by_tag": bool(v.color_by_tag),
                }
            )
        return records

    @classmethod
    def from_records(cls, records: list[dict], next_id: int) -> ViewLibrary:
        lib = cls()
        for r in records:
            lib._views.append(
                SavedView(
                    id=int(r["id"]),
                    name=str(r["name"]),
                    camera=CameraState.from_dict(r["camera"]),
                    tag_visibility={
                        int(k): bool(v) for k, v in r.get("tag_visibility", {}).items()
                    },
                    face_style=str(r["face_style"]),
                    xray=bool(r["xray"]),
                    # Tolerant: a Scene saved before M7.5a carries no such key
                    # and must still load, with the mode off.
                    color_by_tag=bool(r.get("color_by_tag", False)),
                )
            )
        lib._next_id = int(next_id)
        return lib
