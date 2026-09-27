"""Image assets for textured materials.

Deliberately Qt-free, like the rest of `bermake.model`. A `Texture` holds the
ORIGINAL encoded bytes rather than a decode or a re-encode: a photographic JPG
re-encoded as PNG grows by an order of magnitude, and a round trip through the
document would be generationally lossy. Decoding happens at the boundaries, in
viewport/texture_cache.py, which computes the metadata stored here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class Texture:
    """One image asset. `data` is the file's own bytes, exactly as imported."""

    id: int
    name: str
    data: bytes
    image_format: str
    width: int
    height: int
    has_transparency: bool


class TextureLibrary:
    """Per-model texture assets, ordered for display."""

    def __init__(self) -> None:
        self._textures: dict[int, Texture] = {}
        self._order: list[int] = []
        self._next_id = 1

    def add(
        self,
        name: str,
        data: bytes,
        image_format: str,
        width: int,
        height: int,
        has_transparency: bool,
    ) -> Texture:
        tex = Texture(
            self._next_id,
            str(name),
            bytes(data),
            str(image_format),
            int(width),
            int(height),
            bool(has_transparency),
        )
        self._textures[tex.id] = tex
        self._order.append(tex.id)
        self._next_id += 1
        return tex

    def get(self, tid: int) -> Texture | None:
        """The texture, or None. There is no default texture."""
        return self._textures.get(tid)

    def textures(self) -> list[Texture]:
        return [self._textures[t] for t in self._order]

    @property
    def next_id(self) -> int:
        return self._next_id

    def edit(self, tid: int, **fields) -> Texture:
        updated = replace(self._textures[tid], **fields)
        self._textures[tid] = updated
        return updated

    def index_of(self, tid: int) -> int:
        return self._order.index(tid)

    def remove(self, tid: int) -> Texture:
        record = self._textures.pop(tid)
        self._order.remove(tid)
        return record

    def restore(self, texture: Texture, index: int) -> None:
        """Put a removed texture back at its original display position."""
        self._textures[texture.id] = texture
        self._order.insert(index, texture.id)
        self._next_id = max(self._next_id, texture.id + 1)

    def to_records(self) -> list[dict]:
        """Metadata only. The bytes travel as separate container entries."""
        return [
            {
                "id": t.id,
                "name": t.name,
                "image_format": t.image_format,
                "width": t.width,
                "height": t.height,
                "has_transparency": t.has_transparency,
            }
            for t in self.textures()
        ]

    @classmethod
    def from_records(
        cls, records, blobs: dict[int, bytes], next_id: int | None = None
    ) -> TextureLibrary:
        """Rebuild from records plus the blobs the container yielded.

        A record whose blob is absent loads with empty data rather than
        raising: spec 1.8 requires a document referencing a missing entry to
        open with that material untextured, not to be refused.

        `next_id`, like `MaterialLibrary.from_records`, is normally the
        container's saved counter. When it is absent (a hand-written document,
        or records with no counter at all) it falls back to one past the
        highest id present -- but that fallback regresses across a
        remove-then-save of the highest-id texture, which is why the caller
        should always pass the saved counter when it has one.
        """
        lib = cls()
        for rec in records:
            tid = int(rec["id"])
            tex = Texture(
                tid,
                str(rec["name"]),
                bytes(blobs.get(tid, b"")),
                str(rec["image_format"]),
                int(rec["width"]),
                int(rec["height"]),
                bool(rec["has_transparency"]),
            )
            lib._textures[tid] = tex
            lib._order.append(tid)
        if next_id is None:
            next_id = max((t for t in lib._textures), default=0) + 1
        lib._next_id = int(next_id)
        return lib
