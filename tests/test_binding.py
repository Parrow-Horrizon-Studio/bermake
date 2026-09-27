"""Tests that verify the Python ↔ C++ binding pipeline."""

import re

from bermake import __version__, _core, version


def test_core_version_returns_string():
    result = _core.version()
    assert isinstance(result, str)


def test_core_version_matches_semver_pattern():
    result = _core.version()
    assert re.match(r"^\d+\.\d+\.\d+$", result), (
        f"Expected MAJOR.MINOR.PATCH format, got: {result!r}"
    )


def test_top_level_version_function_delegates_to_core():
    assert version() == _core.version()


def test_dunder_version_matches_core():
    assert __version__ == _core.version()
