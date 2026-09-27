"""Resolving an image file referenced by a document, safely.

Both OBJ's `map_Kd` and glTF's `image.uri` name a file relative to the
document. A document someone emailed you must not be able to name a file
outside its own directory: the bytes it points at are read into the model and
then saved, shared and exported with it, so a traversal here is a read
primitive for anything the user can open.

This module also holds `sanitize_filename_stem`, the write-side counterpart:
a material or texture name that becomes part of an exported filename is just
as untrusted, since on import a texture is named after the source document's
own material name (see gltf_import._find_or_decode_texture).
"""

from __future__ import annotations

import re
from pathlib import Path

# Anything outside this set is replaced with '_'. `Path.with_name` already
# rejects a path separator outright (so this is not itself a traversal fix),
# but a bare denylist of separators still lets through characters that are
# unsafe for narrower reasons: 'a:b' names an NTFS alternate data stream, and
# '#', '?', '%' are significant in a URI and glTF's `image.uri` requires them
# percent-encoded, so passing them through unescaped writes a spec-invalid
# document even when the file itself lands fine.
_UNSAFE_STEM_CHARS = re.compile(r"[^A-Za-z0-9._-]")

# On-disk size ceiling for a sibling image, checked via stat() before ever
# reading its bytes -- the same shape as gltf_import._MAX_GLTF_BYTES, and for
# the same reason: this is an untrusted path (a document someone emailed you
# names this file), and read_gltf_scene already guards the document itself
# and every declared element count precisely so nothing downstream reads an
# unbounded amount of attacker-controlled data. A texture sitting beside a
# small, otherwise-innocuous .gltf/.obj is exactly the shape that guard would
# miss without this one: a 10 GB "texture.png" is a plausible archive
# payload. A few hundred MiB comfortably covers any legitimate texture.
_MAX_IMAGE_BYTES = 512 * 1024 * 1024  # 512 MiB


def sanitize_filename_stem(name: str, fallback: str) -> str:
    """A filesystem- and URI-safe stem for `name`, which may be arbitrary
    untrusted text (an OBJ material name, or a glTF material/texture name --
    on import a texture is minted with the name of the source document's own
    material, an arbitrary string from a file someone else authored).

    First collapses whitespace to '_' for a readable stem (this alone is the
    original, weaker behaviour), then replaces every remaining character
    outside [A-Za-z0-9._-] with '_'. Falls back -- never empty -- only when
    the result has no alphanumeric character at all, e.g. "###" (every
    character replaced) or "." / ".." (both meaningful path components on
    their own, and neither contains one). A leading or trailing underscore
    that survives sanitizing is otherwise left alone: an earlier version of
    this function unconditionally stripped them, which silently collapsed
    "_Brick" to the same stem as "Brick" -- issue #119's exact failure mode
    (two distinct names merging on write, so one material and its image
    vanish from the export) -- widened to trigger on any leading/trailing
    underscore rather than only on a literal duplicate name.
    """
    collapsed = "_".join(str(name).split())
    safe = _UNSAFE_STEM_CHARS.sub("_", collapsed)
    return safe if any(c.isalnum() for c in safe) else fallback


def read_sibling_image_bytes(base_dir, relative: str) -> bytes | None:
    """The bytes of `relative` resolved under `base_dir`, or None.

    Returns None rather than raising for every rejection, so a caller can stay
    best-effort. Both sides are resolved BEFORE comparison, so `..`, an
    absolute path and a symlink pointing outside are all caught; comparing the
    unresolved strings would catch only the first. A file over
    `_MAX_IMAGE_BYTES` is also rejected, checked via stat() before any of its
    content is read.
    """
    try:
        base = Path(base_dir).resolve()
        candidate = (base / relative).resolve()
        if not candidate.is_relative_to(base):
            return None
        if not candidate.is_file():
            return None
        if candidate.stat().st_size > _MAX_IMAGE_BYTES:
            return None
        return candidate.read_bytes()
    except (OSError, ValueError):
        return None
