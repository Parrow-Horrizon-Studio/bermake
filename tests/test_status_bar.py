"""Unit tests for the bottom status bar widget."""

from __future__ import annotations


def test_status_bar_starts_empty(qtbot):
    from bermake.ui.status_bar import StatusBar

    bar = StatusBar()
    qtbot.addWidget(bar)
    assert bar.prompt_text() == ""


def test_status_bar_shows_tool_only_when_no_snap(qtbot):
    from bermake.ui.status_bar import StatusBar

    bar = StatusBar()
    qtbot.addWidget(bar)
    bar.set_tool("Line")
    bar.set_snap("")
    assert bar.prompt_text() == "Line · —"


def test_status_bar_shows_tool_and_snap(qtbot):
    from bermake.ui.status_bar import StatusBar

    bar = StatusBar()
    qtbot.addWidget(bar)
    bar.set_tool("Line")
    bar.set_snap("Endpoint")
    assert bar.prompt_text() == "Line · Endpoint"


def test_status_bar_clear_tool_blanks_everything(qtbot):
    from bermake.ui.status_bar import StatusBar

    bar = StatusBar()
    qtbot.addWidget(bar)
    bar.set_tool("Line")
    bar.set_snap("Grid")
    bar.set_tool("")  # no active tool
    assert bar.prompt_text() == ""


def test_the_status_chip_is_opaque_enough_to_read_over_a_white_viewport():
    """The status bar is an overlay on the viewport, not window chrome.

    At the pre-M7.7 alpha of 0.5, black over a white background resolves to
    about mid grey behind #dddddd text, roughly 2.3:1, which is unreadable.
    0.72 lands near 6.6:1 and still reads over the dark environment.

    The alpha is deliberately NOT tied to doc.environment: one value that works
    over white, dark and sky beats a second dependency from the status bar to
    document state (spec D10).
    """
    from bermake.ui.status_bar import _BOX_ALPHA, _BOX_STYLE, _CHIP_ALPHA, _FIELD_STYLE

    assert _CHIP_ALPHA >= 0.70
    assert _BOX_ALPHA >= _CHIP_ALPHA
    # The constants must actually reach the stylesheets, not sit beside them.
    assert f"{_CHIP_ALPHA}" in _FIELD_STYLE
    assert f"{_BOX_ALPHA}" in _BOX_STYLE
