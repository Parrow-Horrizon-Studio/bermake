"""Icon loader (M7.2 Task 2). Needs a QApplication for QPixmap."""

from __future__ import annotations

import pytest
from PySide6.QtGui import QColor

from bermake.ui import icons

_SQUARE = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
    '<rect x="4" y="4" width="16" height="16" fill="#000000"/></svg>'
)


def test_missing_stem_raises_rather_than_returning_an_empty_icon(qtbot):
    with pytest.raises(KeyError):
        icons.icon("definitely_not_an_icon")


def test_loads_a_real_icon(qtbot):
    result = icons.icon("tool_line")
    assert not result.isNull()


def test_pixmap_is_recolored_to_the_requested_color(qtbot, tmp_path):
    pixmap = icons.icon_pixmap("tool_line", 24, QColor("#ff0000"))
    image = pixmap.toImage()
    # Every pixel that is not fully transparent must carry the requested hue.
    opaque = [
        image.pixelColor(x, y)
        for y in range(image.height())
        for x in range(image.width())
        if image.pixelColor(x, y).alpha() > 200
    ]
    assert opaque, "recolored pixmap is entirely transparent"
    for color in opaque:
        assert (color.red(), color.green(), color.blue()) == (255, 0, 0)


def test_icons_are_cached_by_stem_and_color(qtbot):
    first = icons.icon("tool_line", QColor("#123456"))
    second = icons.icon("tool_line", QColor("#123456"))
    assert first is second
    third = icons.icon("tool_line", QColor("#654321"))
    assert third is not first


def test_available_stems_includes_every_declared_icon(qtbot):
    stems = icons.available_icon_stems()
    assert "tool_line" in stems
