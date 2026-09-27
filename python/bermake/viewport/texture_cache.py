"""Decode texture images, and own their GL texture objects.

Split in two on purpose. `decode_image` and `sniff_format` touch no GL and are
the part with real logic, so they are tested directly. `TextureCache` is the GL
half; it takes its GL module by injection so a recorder can stand in.

Decoding uses QImage, which was verified to work with no QApplication, so
nothing here needs a Qt fixture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from OpenGL import GL as _GL
from PySide6.QtGui import QImage

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"


@dataclass(frozen=True, slots=True)
class DecodedImage:
    width: int
    height: int
    has_transparency: bool
    # TOP-DOWN: pixels[0] is the image's TOP row, which is QImage's own row
    # order and what ui/materials_page.py hands straight back to QImage to
    # build the swatch icon. Do NOT flip it here to suit OpenGL -- GL's texture
    # origin is the bottom-left, and reconciling the two is the upload path's
    # job (see gl_row_order below). Flipping here would fix the 3D surface and
    # silently mirror every swatch, reintroducing the decoder divergence Task 9
    # deliberately removed.
    pixels: np.ndarray


def sniff_format(data: bytes) -> str | None:
    """The image format from its magic bytes, or None if unrecognised."""
    if data.startswith(_PNG_MAGIC):
        return "png"
    if data.startswith(_JPEG_MAGIC):
        return "jpeg"
    return None


def decode_image(data: bytes) -> DecodedImage | None:
    """Decode to RGBA, or None if the bytes are not a readable image.

    `has_transparency` scans the alpha channel rather than trusting
    QImage.hasAlphaChannel(), which reports the FORMAT: a fully opaque RGBA PNG
    answers True to it. Trusting it would put opaque images into the sorted
    translucent pass, paying the sort and losing depth writes for nothing.
    """
    if not data:
        return None
    img = QImage()
    if not img.loadFromData(data):
        return None
    img = img.convertToFormat(QImage.Format.Format_RGBA8888)
    w, h = img.width(), img.height()
    if w == 0 or h == 0:
        return None
    flat = np.frombuffer(img.constBits(), dtype=np.uint8, count=img.sizeInBytes())
    # Rows are padded to bytesPerLine, so reshape by stride and trim.
    pixels = flat.reshape(h, img.bytesPerLine())[:, : w * 4].reshape(h, w, 4).copy()
    return DecodedImage(w, h, bool((pixels[:, :, 3] < 255).any()), pixels)


def gl_row_order(pixels: np.ndarray) -> np.ndarray:
    """Re-order a top-down image buffer for OpenGL's bottom-left origin.

    QImage numbers rows from the top; glTexImage2D reads its buffer as starting
    at `v = 0`, which GL places at the BOTTOM of the texture. Handing Qt's
    buffer over unchanged therefore paints every image upside down on the
    surface -- left/right correct, top/bottom swapped, which reads as "the
    photograph is upside down" and which no fixture that is symmetric under a
    vertical flip can see. This is the one place the two conventions meet, so
    it is the one place the reconciliation belongs: not in decode_image, whose
    pixels the Materials swatch consumes in Qt's own order, and not in
    uv_projection, where flipping `v` would invert the meaning of the
    `offset_v` stored in every saved file.

    Returns a contiguous copy, because glTexImage2D reads a raw buffer and a
    reversed numpy view has a negative stride.
    """
    return np.ascontiguousarray(pixels[::-1])


class TextureCache:
    """GL texture objects keyed by texture id, uploaded on first use."""

    def __init__(self, gl=_GL) -> None:
        self._gl = gl
        self._ids: dict[int, int] = {}
        # Ids whose bytes did not decode. Remembered, not just returned: since
        # M7.5b Task 6 this method is called per batch per frame from the draw
        # path, and without this a texture that can never decode would be run
        # through a full image decode on every one of those calls for ever.
        self._failed: set[int] = set()

    def texture_for(self, texture) -> int | None:
        """The GL texture id for this record, uploading it once, or None.

        None means there is nothing to bind: the record carries no bytes (a
        missing container entry) or the bytes did not decode. Callers render
        untextured rather than failing the frame.
        """
        existing = self._ids.get(texture.id)
        if existing is not None:
            return existing
        if texture.id in self._failed:
            return None
        img = decode_image(texture.data)
        if img is None:
            self._failed.add(texture.id)
            return None
        gl = self._gl
        tid = int(gl.glGenTextures(1))
        gl.glBindTexture(gl.GL_TEXTURE_2D, tid)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA,
            img.width,
            img.height,
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            gl_row_order(img.pixels),
        )
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        self._ids[texture.id] = tid
        return tid

    def cached_ids(self) -> frozenset[int]:
        """Every texture id holding a live GL upload or a remembered decode
        failure -- i.e. every id `invalidate` or `release_all` would affect.

        Read-only introspection for a caller that wants to reconcile this
        cache against a TextureLibrary's current contents (M7.5b Task 9)
        without reaching into the private `_ids` / `_failed` sets directly.
        """
        return frozenset(self._ids) | frozenset(self._failed)

    def invalidate(self, tid: int) -> None:
        """Drop one cached upload, so the next use re-uploads.

        Clears a remembered decode failure too: re-importing an image edits the
        record in place, so repaired bytes would otherwise never be retried.
        """
        self._failed.discard(tid)
        gl_id = self._ids.pop(tid, None)
        if gl_id is not None:
            self._gl.glDeleteTextures([gl_id])

    def release_all(self) -> None:
        for gl_id in self._ids.values():
            self._gl.glDeleteTextures([gl_id])
        self._ids.clear()
        self._failed.clear()
