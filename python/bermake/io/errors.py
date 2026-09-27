"""Exception hierarchy for .berm load/save.

A BermakeIOError means 'this file is bad' (we own the message). OS-level errors
(permission, disk full) are left to propagate as OSError so the UI can tell the
two apart.
"""

from __future__ import annotations


class BermakeIOError(Exception):
    """Base for all .berm load/save errors that mean 'this file is bad'."""


class BermakeFormatError(BermakeIOError):
    """Not a valid .berm document: bad zip, missing entries, malformed structure."""


class BermakeVersionError(BermakeIOError):
    """The file's schema_version is newer than this build supports."""
