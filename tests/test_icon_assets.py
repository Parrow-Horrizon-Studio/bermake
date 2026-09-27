"""The icon assets obey the M7.2 spec, mechanically (Task 3).

Parsing the SVGs is what keeps 35 hand-authored files consistent: a stray
<text>, gradient, or wrong viewBox fails here rather than rendering
differently on someone else's Qt build.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from importlib.resources import files

import pytest

from bermake.ui import actions, icons

SVG_NS = "http://www.w3.org/2000/svg"
ALLOWED_TAGS = frozenset(
    {"svg", "g", "path", "line", "circle", "rect", "polyline", "polygon"}
)
BANNED_SUBSTRINGS = ("<text", "linearGradient", "radialGradient", "<filter", "<image", "<style", "<script")


def _asset_paths() -> list[tuple[str, str]]:
    root = files(icons.ICON_DIR_PACKAGE)
    return sorted(
        (entry.name[:-4], entry.read_text(encoding="utf-8"))
        for entry in root.iterdir()
        if entry.name.endswith(".svg")
    )


def test_every_tool_action_has_an_asset_file():
    stems = icons.available_icon_stems()
    tools = [s for s in actions.ACTIONS if s.group == actions.TOOL_GROUP]
    missing = sorted(s.icon for s in tools if s.icon not in stems)
    assert missing == [], f"tool icons missing: {missing}"


@pytest.mark.parametrize("stem,text", _asset_paths())
def test_viewbox_is_24x24(stem, text):
    root = ET.fromstring(text)
    assert root.get("viewBox") == "0 0 24 24", stem


@pytest.mark.parametrize("stem,text", _asset_paths())
def test_only_whitelisted_elements(stem, text):
    for element in ET.fromstring(text).iter():
        tag = element.tag.removeprefix(f"{{{SVG_NS}}}")
        assert tag in ALLOWED_TAGS, f"{stem}: <{tag}> is not permitted"


@pytest.mark.parametrize("stem,text", _asset_paths())
def test_no_banned_constructs(stem, text):
    for banned in BANNED_SUBSTRINGS:
        assert banned not in text, f"{stem}: contains {banned}"


@pytest.mark.parametrize("stem,text", _asset_paths())
def test_monochrome_only(stem, text):
    lowered = text.lower()
    for element in ET.fromstring(text).iter():
        for attr in ("fill", "stroke"):
            value = element.get(attr)
            if value in (None, "none"):
                continue
            assert value == "#000000", f"{stem}: {attr}={value} is not the flat black"
    assert "#fff" not in lowered


@pytest.mark.parametrize("stem,text", _asset_paths())
def test_renders_to_a_non_empty_pixmap(stem, text, qtbot):
    from PySide6.QtGui import QColor

    pixmap = icons.icon_pixmap(stem, 24, QColor("#000000"))
    image = pixmap.toImage()
    inked = sum(
        1
        for y in range(image.height())
        for x in range(image.width())
        if image.pixelColor(x, y).alpha() > 0
    )
    assert inked > 20, f"{stem} renders nearly blank ({inked} inked pixels)"


def test_every_declared_icon_has_an_asset_file():
    stems = icons.available_icon_stems()
    declared = sorted({s.icon for s in actions.ACTIONS if s.icon is not None})
    missing = [stem for stem in declared if stem not in stems]
    assert missing == [], f"declared but missing: {missing}"
    assert len(declared) == 35, f"expected 35 declared icons, got {len(declared)}"


def test_no_orphan_asset_files():
    # M7.3: the panel's tab strip and Outliner rows reference icons that are
    # not commands, so they cannot come from ACTIONS. They are declared in
    # panel_icons.NON_ACTION_ICONS and unioned in here -- an SVG referenced by
    # nothing at all still fails.
    from bermake.ui.panel_icons import NON_ACTION_ICONS

    declared = {s.icon for s in actions.ACTIONS if s.icon is not None} | NON_ACTION_ICONS
    orphans = sorted(icons.available_icon_stems() - declared)
    assert orphans == [], f"asset files no action or panel references: {orphans}"
