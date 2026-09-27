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
real scope with the exclusion is around 220 files, comfortably above the
threshold.

One shipped file is meant to contain the retired name: the old-document
detection in bermake_file.py has to compare against the literal old format
string and explain itself by name. That is why this asserts against an
allowlist of exact per-file counts (see INTENDED) rather than against zero.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SHIPPED = (ROOT / "python" / "bermake", ROOT / "cpp")

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


def _text_files():
    for tree in SHIPPED:
        for path in tree.rglob("*"):
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
