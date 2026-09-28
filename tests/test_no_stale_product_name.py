"""A guard against the retired product name surviving in shipped code.

Scoped deliberately. docs/ retains thousands of occurrences on purpose (spec
D6), and tests/data/ keeps the old name as fixture provenance (spec D11), so a
repository-wide search would fail by design. This looks only at what ships.

The scope assertion is not decoration. A guard test that walks an empty tree
passes while proving nothing, which is the exact failure this project produced
three times during M7.7, twice in test code and once in a verification script.

__pycache__ is excluded deliberately rather than left to the UnicodeDecodeError
branch below. There are around 150 stale .pyc files under the package, every one
of which contains the old module path, and all 150 do currently raise on a strict
utf-8 decode, so the guard would pass either way today. Depending on that is
wrong twice over: bytecode is not shipped source, and a single .pyc that happened
to decode would turn this guard into a false positive nobody could explain. The
real scope with the exclusion is around 224 files, comfortably above the
threshold.

The Python half of the scope is derived from tool.scikit-build.wheel.packages
in pyproject.toml rather than hardcoded, because that config is the definition
of what ships in the wheel; a hardcoded list would silently stop matching a
newly added package. cpp, packaging and four root packaging/build/lint files
are added explicitly, since they are not covered by the wheel packages list but
a stale reference in any of them is a real regression (this milestone nearly
shipped exactly that in a workflow's lint path). See _shipped_roots().

One shipped file is meant to contain the retired name: the old-document
detection in bermake_file.py has to compare against the literal old format
string and explain itself by name. That is why this asserts against an
allowlist of exact per-file counts (see INTENDED) rather than against zero.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# The only shipped file that is meant to contain the retired name, and the exact
# number of times. The loader has to recognise a document written by the old
# application and say so, which the spec requires, so one comparison literal and
# one word in the explanatory message are both load-bearing and cannot be removed.
#
# Pinned by count rather than exempted by filename, deliberately. Exempting the
# file would hide a genuine leftover elsewhere in it, and would also let someone
# delete the detection or the message without any test noticing. A count fails in
# both directions: a new occurrence and a missing one.
INTENDED = {
    "python/bermake/io/bermake_file.py": {"pluton": 1, "Pluton": 1, "PLUTON": 0},
}


def _shipped_roots() -> list[Path]:
    """The trees and files that ship, derived rather than hardcoded.

    The Python packages come from the packaging config, because
    tool.scikit-build.wheel.packages IS the definition of what goes into the
    wheel. A hardcoded list cannot notice a newly added package, so the guard's
    coverage would quietly stop matching the thing it guards. Reading the config
    means a new package is covered the day it is added.

    cpp is explicit because it is compiled in rather than packaged as a tree, and
    the four root files are explicit because a build target or a lint path
    reverting to the old name is a real regression this milestone nearly shipped.
    """
    import tomllib

    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    packages = config["tool"]["scikit-build"]["wheel"]["packages"]

    roots = [ROOT / p for p in packages]
    roots.append(ROOT / "cpp")
    # M7.9: packaging/ holds the tester README and notices that ship in the zip.
    roots.append(ROOT / "packaging")
    roots.extend(
        ROOT / name
        for name in (
            "pyproject.toml",
            "CMakeLists.txt",
            "vcpkg.json",
            ".github/workflows/build.yml",
        )
    )
    return roots


def _text_files():
    for root in _shipped_roots():
        if root.is_file():
            candidates = [root]
        else:
            candidates = root.rglob("*")
        for path in candidates:
            if not path.is_file():
                continue
            if "__pycache__" in path.parts:
                continue
            if path.suffix in {".glb", ".png", ".jpg", ".ico"}:
                continue
            yield path


def test_the_guard_actually_has_a_tree_to_search():
    """Without this, every assertion below is vacuous."""
    files = list(_text_files())
    assert len(files) > 150, f"guard scope collapsed to {len(files)} files"


@pytest.mark.parametrize("needle", ["pluton", "Pluton", "PLUTON"])
def test_shipped_files_mention_the_retired_name_only_where_intended(needle):
    found = {}
    for path in _text_files():
        try:
            text = path.read_text(encoding="utf-8", errors="strict")
        except UnicodeDecodeError:
            continue
        count = text.count(needle)
        if count:
            found[path.relative_to(ROOT).as_posix()] = count

    expected = {p: n for p, counts in INTENDED.items() if (n := counts[needle])}
    assert found == expected, (
        f"{needle!r} occurrences do not match the allowlist.\n"
        f"found:    {found}\n"
        f"expected: {expected}"
    )
