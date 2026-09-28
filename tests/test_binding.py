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
    """Catches a version that reached one route and not the other.

    pyproject.toml is the single version source (M7.9, #130). It reaches the
    installed distribution metadata through scikit-build-core, and reaches
    `_core.version()` through the regex in the root CMakeLists.txt and a compile
    definition. Reading pyproject.toml as well catches an editable install that
    was not rebuilt after a bump: both routes would still agree with each other,
    on the old number.

    Not a comparison of `__version__` against `_core.version()`: those are the
    same call, since `__init__` assigns `__version__ = version()` with `version`
    imported from `_core`, so such a test cannot fail.
    """
    import tomllib
    from importlib.metadata import version as distribution_version
    from pathlib import Path

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]

    assert distribution_version("bermake") == _core.version() == declared
