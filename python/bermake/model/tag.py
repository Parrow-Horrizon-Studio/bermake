"""Tags ('Layers') for organizing objects + per-tag visibility (M5c).

Pure Python — no GL, no Qt — so it is fully unit-testable headlessly. Tags
attach to group/component Instances via Instance.tag_id; the renderer and
picking consult TagLibrary.is_visible to hide objects on a hidden tag. The
library is serialization-ready for M6 file I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Tag:
    """A named tag. `visible` is mutable view state (not part of undo).

    `color` (M7.5a Task 11) is document state, unlike `visible`: it lives in
    to_records/from_records and is meant to be changed only through
    TagLibrary.set_color, which commands.tag_commands.SetTagColorCommand
    wraps so recoloring a tag is undoable.
    """

    id: int
    name: str
    visible: bool = True
    color: tuple[float, float, float] = (0.5, 0.5, 0.5)


class TagLibrary:
    """Owns the model's Tag objects: Untagged first, then user tags."""

    UNTAGGED_ID = 0

    # M7.5a Task 11: fixed hues a new tag cycles through, so two tags created
    # back to back never share a colour. Deliberately distinct from
    # MaterialLibrary's muted architectural _BUILTIN_PALETTE (spec 1.9) --
    # tags are a wayfinding overlay, not a material choice, and should read
    # as unmistakably different swatches. The cycle wraps once every entry
    # has been used once.
    _PALETTE: tuple[tuple[float, float, float], ...] = (
        (0.90, 0.25, 0.25),  # red
        (0.95, 0.60, 0.15),  # orange
        (0.90, 0.85, 0.20),  # yellow
        (0.30, 0.75, 0.35),  # green
        (0.20, 0.55, 0.90),  # blue
        (0.55, 0.35, 0.85),  # violet
        (0.90, 0.35, 0.65),  # magenta
        (0.25, 0.80, 0.80),  # cyan
    )
    # Untagged's colour: neutral gray, outside the cycle above -- it is
    # auto-seeded rather than user-created, so it should not consume a hue a
    # real tag would otherwise get.
    _UNTAGGED_COLOR: tuple[float, float, float] = (0.55, 0.55, 0.55)

    def __init__(self) -> None:
        self._untagged = Tag(self.UNTAGGED_ID, "Untagged", True, self._UNTAGGED_COLOR)
        self._tags: dict[int, Tag] = {self.UNTAGGED_ID: self._untagged}
        self._order: list[int] = [self.UNTAGGED_ID]
        self._next_id = 1
        self._next_palette_index = 0

    def add(self, name: str) -> Tag:
        """Append a new tag with a fresh monotonic id and return it.

        Colour cycles `_PALETTE`, so two tags added in a row never share one.
        """
        color = self._PALETTE[self._next_palette_index % len(self._PALETTE)]
        self._next_palette_index += 1
        tag = Tag(self._next_id, str(name), True, color)
        self._tags[tag.id] = tag
        self._order.append(tag.id)
        self._next_id += 1
        return tag

    def get(self, tid: int) -> Tag:
        """Return the tag for `tid`, or the Untagged tag if unknown."""
        return self._tags.get(tid, self._untagged)

    def tags(self) -> list[Tag]:
        """All tags in display order (Untagged first)."""
        return [self._tags[i] for i in self._order]

    def set_visible(self, tid: int, visible: bool) -> None:
        """Set a tag's visibility. No-op for Untagged (always visible)."""
        if tid == self.UNTAGGED_ID:
            return
        tag = self._tags.get(tid)
        if tag is not None:
            tag.visible = bool(visible)

    def rename(self, tid: int, name: str) -> None:
        """Rename a user tag. No-op for Untagged or an empty name."""
        if tid == self.UNTAGGED_ID:
            return
        tag = self._tags.get(tid)
        if tag is not None and name:
            tag.name = str(name)

    def set_color(self, tid: int, color: tuple[float, float, float]) -> None:
        """Set a tag's colour, Untagged included.

        Unlike `set_visible`/`rename`, there is no Untagged special case:
        colour is plain document state, not the "always visible" rule
        `visible` carries. Called directly by SetTagColorCommand's do()/
        undo() -- never by UI code, so recoloring stays undoable.
        """
        tag = self._tags.get(tid)
        if tag is not None:
            tag.color = (float(color[0]), float(color[1]), float(color[2]))

    def is_visible(self, tid: int) -> bool:
        """Whether entities on this tag should be drawn (Untagged always True)."""
        return self.get(tid).visible

    @property
    def next_id(self) -> int:
        return self._next_id

    def to_records(self) -> list[dict]:
        """Serialize all tags in display order (Untagged first)."""
        return [
            {"id": t.id, "name": t.name, "visible": t.visible, "color": list(t.color)}
            for t in self.tags()
        ]

    @classmethod
    def from_records(cls, records: list[dict], next_id: int) -> TagLibrary:
        """Rebuild a library authoritatively from saved records (no auto-seed).

        Schema <= 4 wrote no "color" key (Task 12's migration relies on
        this): a missing colour falls back to the same cycling palette
        `add()` uses, so an old file's tags still come up visually distinct
        rather than uniformly gray. Untagged is excluded from that cycle --
        keyed on record position it would take _PALETTE[0] (bright red) and
        make every untagged face in a pre-Task-11 file read as a real tag.
        It consumes no hue either, so the cycle a loaded library resumes at
        matches the one a freshly-built library would be on.
        """
        lib = cls()  # seeds Untagged, then we overwrite
        lib._tags = {}
        lib._order = []
        hues_used = 0
        for r in records:
            color = r.get("color")
            if color is None:
                if int(r["id"]) == cls.UNTAGGED_ID:
                    color = cls._UNTAGGED_COLOR
                else:
                    color = cls._PALETTE[hues_used % len(cls._PALETTE)]
            if int(r["id"]) != cls.UNTAGGED_ID:
                hues_used += 1
            tag = Tag(
                int(r["id"]),
                str(r["name"]),
                bool(r["visible"]),
                (float(color[0]), float(color[1]), float(color[2])),
            )
            lib._tags[tag.id] = tag
            lib._order.append(tag.id)
        lib._untagged = lib._tags.get(cls.UNTAGGED_ID, lib._untagged)
        lib._next_id = int(next_id)
        lib._next_palette_index = hues_used
        return lib
