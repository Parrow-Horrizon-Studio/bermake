from __future__ import annotations

from bermake.ui.value_control_box import ValueControlBox


def test_starts_empty_inactive():
    v = ValueControlBox()
    assert not v.active
    assert v.text == ""


def test_feed_activates_and_appends():
    v = ValueControlBox()
    v.feed("1")
    v.feed("5")
    v.feed("0")
    v.feed("0")
    v.feed("m")
    v.feed("m")
    assert v.active
    assert v.text == "1500mm"


def test_backspace_edits_and_deactivates_when_empty():
    v = ValueControlBox()
    v.feed("4")
    v.feed("2")
    v.backspace()
    assert v.text == "4" and v.active
    v.backspace()
    assert v.text == "" and not v.active


def test_clear():
    v = ValueControlBox()
    v.feed("9")
    v.clear()
    assert v.text == "" and not v.active
