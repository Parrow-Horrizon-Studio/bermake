"""The panel's non-action icons exist and render (M7.3 Task 7)."""

from __future__ import annotations

import pytest
from PySide6.QtGui import QColor

from bermake.ui import icons
from bermake.ui.panel_icons import NON_ACTION_ICONS

EXPECTED = frozenset(
    {
        "tab-tool-settings",
        "tab-entity-info",
        "tab-material",
        "tab-tags",
        "tab-scenes",
        "outliner-group",
        "outliner-component",
        "visible",
        "hidden",
    }
)


def test_the_declared_set_is_exactly_the_nine_panel_icons():
    assert NON_ACTION_ICONS == EXPECTED


@pytest.mark.parametrize("stem", sorted(EXPECTED))
def test_each_panel_icon_has_an_asset_file(stem):
    assert stem in icons.available_icon_stems()


@pytest.mark.parametrize("stem", sorted(EXPECTED))
def test_each_panel_icon_renders_to_a_non_empty_pixmap(stem, qtbot):
    # icon() is strict on a missing stem, but a file that parses and draws
    # nothing would still ship a blank button -- so check real pixels.
    pixmap = icons.icon(stem, QColor("#ffffff")).pixmap(24, 24)
    image = pixmap.toImage()
    assert any(
        image.pixelColor(x, y).alpha() > 0
        for x in range(image.width())
        for y in range(image.height())
    ), f"{stem} rendered blank"
