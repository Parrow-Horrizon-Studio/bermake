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


def test_the_declared_and_the_compiled_version_agree():
    """Catches a version bump applied to one file and not the other.

    Not a comparison of `__version__` against `_core.version()`: those are the
    same call, since `__init__` assigns `__version__ = version()` with `version`
    imported from `_core`, so such a test cannot fail. The hand-edited version in
    pyproject.toml reaches the installed distribution metadata, and the C++
    literal reaches `_core.version()`, and nothing derives either from the other.
    Pinning the expected value as well catches a bump that was never applied at
    all, and catches an editable install that was not refreshed after the bump.
    """
    from importlib.metadata import version as distribution_version

    from bermake import _core

    assert distribution_version("bermake") == _core.version() == "0.14.0"
