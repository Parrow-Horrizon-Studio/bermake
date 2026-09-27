"""M7.5b Task 5: decoding images and caching GL textures."""

from __future__ import annotations

import struct
import zlib

import numpy as np
from bermake.model.texture import TextureLibrary
from bermake.viewport import texture_cache
from bermake.viewport.texture_cache import (
    TextureCache,
    decode_image,
    gl_row_order,
    sniff_format,
)


def _png(w: int, h: int, rgba: list[int]) -> bytes:
    raw = b"".join(b"\x00" + bytes(rgba[y * w * 4 : (y + 1) * w * 4]) for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


_OPAQUE = _png(2, 2, [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 9, 9, 9, 255])
_CUTOUT = _png(2, 2, [255, 0, 0, 255, 0, 255, 0, 128, 0, 0, 255, 255, 9, 9, 9, 0])

# Vertically ASYMMETRIC on purpose: red across the top row, blue across the
# bottom. Every earlier fixture in this milestone was symmetric under a
# vertical flip, which is exactly why a mirrored upload shipped unnoticed.
_TOP = (220, 30, 30)
_BOTTOM = (30, 30, 220)
_TOP_DOWN = _png(
    2,
    2,
    [*_TOP, 255, *_TOP, 255, *_BOTTOM, 255, *_BOTTOM, 255],
)


def test_decoding_needs_no_qapplication():
    """Decoding must not require a QApplication -- that closes the spec's
    biggest flagged risk and is the whole reason texture decoding is testable
    headlessly. Asserting on QGuiApplication.instance() in *this* process is
    the wrong mechanism: pytest's own qtbot/main_window fixtures create one,
    and a QApplication is process-global and never goes away once built, so
    the assertion fails as soon as anything Qt-backed has already run in this
    session. Prove the claim in an isolated subprocess instead, where a clean
    `QGuiApplication.instance() is None` is actually meaningful.
    """
    import subprocess
    import sys

    code = (
        "from bermake.viewport.texture_cache import decode_image\n"
        f"img = decode_image({_CUTOUT!r})\n"
        "assert img is not None\n"
        "assert (img.width, img.height) == (2, 2)\n"
        "assert [int(a) for a in img.pixels[:, :, 3].flatten()] == [255, 128, 255, 0]\n"
        "from PySide6.QtGui import QGuiApplication\n"
        "assert QGuiApplication.instance() is None\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "ok"


def test_decode_reports_size_and_rgba_pixels():
    img = decode_image(_OPAQUE)
    assert (img.width, img.height) == (2, 2)
    assert img.pixels.shape == (2, 2, 4)
    assert img.pixels.dtype == np.uint8


def test_an_rgba_image_with_no_transparent_texel_is_not_translucent():
    # The whole point. QImage.hasAlphaChannel() is True here because the FORMAT
    # has alpha, but every texel is opaque. Trusting that flag would put this
    # image in the sorted translucent pass for nothing.
    assert decode_image(_OPAQUE).has_transparency is False


def test_an_image_with_a_transparent_texel_is_translucent():
    assert decode_image(_CUTOUT).has_transparency is True


def test_corrupt_bytes_decode_to_none_rather_than_raising():
    # Spec 1.8: import fails with a message and no library change.
    assert decode_image(b"not an image at all") is None
    assert decode_image(b"") is None


def test_sniff_format_recognises_png_and_rejects_junk():
    assert sniff_format(_OPAQUE) == "png"
    assert sniff_format(b"not an image at all") is None


def test_sniff_format_recognises_jpeg():
    assert sniff_format(b"\xff\xd8\xff\xe0" + b"\x00" * 16) == "jpeg"


class _RecordingGL:
    """Stands in for the GL module, recording calls instead of making them."""

    def __init__(self):
        self.calls: list[tuple] = []
        self._next = 100

    def glGenTextures(self, _n):
        self._next += 1
        self.calls.append(("gen", self._next))
        return self._next

    def glDeleteTextures(self, ids):
        self.calls.append(("delete", tuple(ids) if hasattr(ids, "__iter__") else (ids,)))

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            self.calls.append((name,))

        return _noop


def _lib_with(data: bytes):
    lib = TextureLibrary()
    img = decode_image(data)
    return lib, lib.add("t.png", data, "png", img.width, img.height, img.has_transparency)


def test_a_texture_uploads_once_and_is_cached():
    lib, tex = _lib_with(_OPAQUE)
    gl = _RecordingGL()
    cache = TextureCache(gl=gl)

    first = cache.texture_for(tex)
    second = cache.texture_for(tex)
    assert first == second
    assert [c for c in gl.calls if c[0] == "gen"] == [("gen", 101)]


def test_invalidate_forces_a_re_upload_and_deletes_the_old_id():
    lib, tex = _lib_with(_OPAQUE)
    gl = _RecordingGL()
    cache = TextureCache(gl=gl)

    first = cache.texture_for(tex)
    cache.invalidate(tex.id)
    second = cache.texture_for(tex)
    assert second != first
    assert ("delete", (first,)) in gl.calls


def test_a_texture_with_no_bytes_yields_no_gl_id():
    # A document referencing a missing container entry loads with empty data
    # (Task 1). That must render untextured, not crash the frame.
    lib = TextureLibrary()
    tex = lib.add("gone.png", b"", "png", 4, 4, False)
    cache = TextureCache(gl=_RecordingGL())
    assert cache.texture_for(tex) is None


def test_release_all_deletes_every_uploaded_texture():
    lib, tex = _lib_with(_OPAQUE)
    other = lib.add("b.png", _CUTOUT, "png", 2, 2, True)
    gl = _RecordingGL()
    cache = TextureCache(gl=gl)
    a = cache.texture_for(tex)
    b = cache.texture_for(other)

    cache.release_all()
    deleted = {i for c in gl.calls if c[0] == "delete" for i in c[1]}
    assert {a, b} <= deleted
    assert cache.texture_for(tex) != a  # re-uploads after release


def test_a_decode_failure_is_remembered_rather_than_retried(monkeypatch):
    """M7.5b Task 6 put texture_for on the per-frame draw path.

    Before that, re-decoding undecodable bytes on every call was harmless dead
    cost. Now it is a full image decode per batch per frame, for ever, for a
    texture that will never succeed.
    """
    lib = TextureLibrary()
    tex = lib.add("junk.png", b"not an image at all", "png", 4, 4, False)
    cache = TextureCache(gl=_RecordingGL())

    calls = []
    real = texture_cache.decode_image

    def counting(data):
        calls.append(data)
        return real(data)

    monkeypatch.setattr(texture_cache, "decode_image", counting)

    assert cache.texture_for(tex) is None
    assert cache.texture_for(tex) is None
    assert cache.texture_for(tex) is None
    assert len(calls) == 1


def test_invalidating_a_failed_texture_lets_it_decode_again(monkeypatch):
    # Re-importing the image edits the record in place, so the remembered
    # failure has to be dropped by the same call that drops a successful
    # upload, or the repaired bytes would never be tried.
    lib = TextureLibrary()
    tex = lib.add("junk.png", b"not an image at all", "png", 4, 4, False)
    cache = TextureCache(gl=_RecordingGL())
    assert cache.texture_for(tex) is None

    cache.invalidate(tex.id)
    fixed = lib.edit(tex.id, data=_OPAQUE)
    assert cache.texture_for(fixed) is not None


# --- Row order: Qt's top-down vs GL's bottom-left origin ---------------------
#
# The manual visual pass found every texture rendering vertically mirrored: a
# photograph painted on a wall came out upside down, left/right correct. Nothing
# in 2211 tests saw it, because every fixture was symmetric under a vertical
# flip. These three hold that line at the seam where the two conventions meet.


def test_decode_keeps_qt_row_order_with_the_top_row_first():
    # The Materials swatch hands these pixels straight back to QImage, which
    # numbers rows from the top. Flipping here to suit GL would fix the 3D
    # surface and silently mirror every swatch.
    img = decode_image(_TOP_DOWN)
    assert tuple(img.pixels[0, 0, :3]) == _TOP
    assert tuple(img.pixels[-1, 0, :3]) == _BOTTOM


def test_gl_row_order_puts_the_bottom_row_first():
    # glTexImage2D reads its buffer starting at v = 0, which GL places at the
    # BOTTOM of the texture, so the bottom row has to travel first.
    img = decode_image(_TOP_DOWN)
    rows = gl_row_order(img.pixels)
    assert tuple(rows[0, 0, :3]) == _BOTTOM
    assert tuple(rows[-1, 0, :3]) == _TOP
    # glTexImage2D reads a raw buffer, so a negative-stride view will not do.
    assert rows.flags["C_CONTIGUOUS"]


def test_the_upload_hands_gl_the_bottom_row_first():
    # The seam itself: whatever helper is used, the buffer that actually
    # reaches glTexImage2D must be in GL order. Uploading decode_image's own
    # top-down buffer is the shipped defect this kills.
    lib = TextureLibrary()
    img = decode_image(_TOP_DOWN)
    tex = lib.add("wall.png", _TOP_DOWN, "png", img.width, img.height, img.has_transparency)

    class _CapturingGL(_RecordingGL):
        uploaded = None

        def glTexImage2D(self, _t, _l, _if, _w, _h, _b, _f, _ty, pixels):
            self.uploaded = np.asarray(pixels).reshape(_h, _w, 4).copy()

    gl = _CapturingGL()
    TextureCache(gl=gl).texture_for(tex)

    assert gl.uploaded is not None, "nothing was uploaded"
    assert tuple(gl.uploaded[0, 0, :3]) == _BOTTOM, (
        "the image's TOP row reached GL first, so it lands at v = 0 (the bottom "
        "of the texture) and every texture renders vertically mirrored"
    )
    assert tuple(gl.uploaded[-1, 0, :3]) == _TOP
