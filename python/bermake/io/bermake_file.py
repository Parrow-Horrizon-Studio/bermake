"""Zip container + manifest version gate for the native .berm format (M6a).

The only part of bermake.io that touches the filesystem. A .berm file is a zip
holding manifest.json (the version gate) + document.json (the codec payload).
"""

from __future__ import annotations

import json
import os
import zipfile
from collections.abc import Callable
from pathlib import Path

from bermake._core import version as _core_version
from bermake.io.document_codec import (
    LoadedDocument,
    document_from_dict,
    document_to_dict,
)
from bermake.io.errors import BermakeFormatError, BermakeVersionError

SCHEMA_VERSION = 9  # M7.7: per-document viewport environment (background, sky, ground, ink)
# The oldest schema this build opens. Every file ever written under the Bermake
# name is format "bermake" at schema 9 (v0.14.0 onwards); anything older came
# from Pluton and is rejected by its format name before this floor is reached,
# so a lower number here means a corrupt manifest. Raising the floor drops files
# testers already hold, so it is a maintainer decision, never a tidy-up.
MIN_SCHEMA_VERSION = 9

# n -> the n-to-(n+1) upgrade of the decoded document.json dict. A schema bump
# that only adds optional keys needs no entry: the codec reads them with
# defaults and a missing entry means "pass through unchanged". Anything else
# (a renamed, moved or reinterpreted key) registers a migration here, and
# load_document applies every step from the file's own version up to
# SCHEMA_VERSION - 1, in order, before document_from_dict sees the data. A
# migration takes the dict and returns the upgraded dict. Empty today: nothing
# since schema 9 has needed one. See the file-format policy in
# docs/2026-05-16-pluton-design.md.
_MIGRATIONS: dict[int, Callable[[dict], dict]] = {}

_MANIFEST = "manifest.json"
_DOCUMENT = "document.json"
_TEXTURES_DIR = "textures/"
_THUMBNAIL = "thumbnail.png"


def save_document(
    path, model, camera, doc, render_style, *, thumbnail: bytes | None = None
) -> None:
    """Write the document to `path` atomically (temp file + os.replace).

    `thumbnail`, when given, is a PNG-encoded preview image written as a sibling
    entry for file browsers (#78). It is optional (D13): producing one needs a
    live GL context, which headless saves and tests do not have, so `None`
    (the default) writes no entry and the save still succeeds.
    """
    path = Path(path)
    data = document_to_dict(model, camera, doc, render_style)
    manifest = {
        "format": "bermake",
        "schema_version": SCHEMA_VERSION,
        "app_version": _core_version(),
    }
    tmp = path.with_name(path.name + ".tmp")
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(_MANIFEST, json.dumps(manifest, separators=(",", ":")))
            zf.writestr(_DOCUMENT, json.dumps(data, separators=(",", ":")))
            for tex in model.textures.textures():
                if tex.data:
                    zf.writestr(f"{_TEXTURES_DIR}{tex.id}.{tex.image_format}", tex.data)
            if thumbnail:
                zf.writestr(_THUMBNAIL, thumbnail)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def load_document(path) -> LoadedDocument:
    """Read a .berm file. Raises BermakeFormatError / BermakeVersionError / OSError."""
    path = Path(path)
    try:
        with zipfile.ZipFile(path, "r") as zf:
            manifest = json.loads(zf.read(_MANIFEST))
            fmt = manifest.get("format")
            if fmt == "pluton":
                raise BermakeFormatError(
                    "this file was written by Pluton, the former name of Bermake. "
                    "The format was renamed in v0.14.0 and there is no conversion "
                    "path, so it cannot be opened."
                )
            if fmt != "bermake":
                raise BermakeFormatError("not a Bermake file (bad 'format' in manifest)")
            ver = manifest.get("schema_version")
            # Older files are the accept path, down to MIN_SCHEMA_VERSION: this is a
            # range check on purpose, never `!=`, so a file written by an earlier
            # release keeps opening after SCHEMA_VERSION moves on. Only newer is
            # rejected (no forward compatibility).
            if not isinstance(ver, int) or ver > SCHEMA_VERSION:
                raise BermakeVersionError(
                    f"file schema_version {ver} is newer than supported ({SCHEMA_VERSION})"
                )
            if ver < MIN_SCHEMA_VERSION:
                raise BermakeFormatError(
                    f"file schema_version {ver} is older than any Bermake release wrote "
                    f"(oldest supported is {MIN_SCHEMA_VERSION}); the file is corrupt or "
                    "not a Bermake file"
                )
            data = json.loads(zf.read(_DOCUMENT))
            blobs: dict[int, bytes] = {}
            for name in zf.namelist():
                if not name.startswith(_TEXTURES_DIR):
                    continue
                stem = name[len(_TEXTURES_DIR) :].rsplit(".", 1)[0]
                if stem.isdigit():
                    blobs[int(stem)] = zf.read(name)
    except zipfile.BadZipFile as e:
        raise BermakeFormatError("not a valid .berm file (not a zip archive)") from e
    except KeyError as e:
        raise BermakeFormatError(f"missing entry in .berm archive: {e}") from e
    except json.JSONDecodeError as e:
        raise BermakeFormatError(f"corrupt JSON in .berm archive: {e}") from e
    try:
        for step in range(ver, SCHEMA_VERSION):
            migrate = _MIGRATIONS.get(step)
            if migrate is not None:
                data = migrate(data)
    except (KeyError, TypeError, ValueError, IndexError) as e:
        raise BermakeFormatError(f"malformed document: {e}") from e
    return document_from_dict(data, blobs)
