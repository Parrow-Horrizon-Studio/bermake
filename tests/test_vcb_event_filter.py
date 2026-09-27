"""The VCB event filter: digit activates + consumes; Enter applies; Esc clears."""

from __future__ import annotations

from bermake.ui.main_window import MainWindow
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent


def _key(key, text=""):
    return QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier, text)


def test_digit_activates_and_is_consumed(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    win._tool_manager.activate_by_shortcut("M")  # a tool that supports typed entry
    consumed = win._vcb_handle_key(_key(Qt.Key.Key_1, "1"))
    assert consumed
    assert win._vcb.active and win._vcb.text == "1"


def test_letter_while_inactive_not_consumed(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    win._tool_manager.activate_by_shortcut("M")
    assert win._vcb_handle_key(_key(Qt.Key.Key_L, "l")) is False  # 'L' falls through to shortcut


def test_letter_while_active_is_consumed(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    win._tool_manager.activate_by_shortcut("M")
    win._vcb_handle_key(_key(Qt.Key.Key_1, "1"))
    assert win._vcb_handle_key(_key(Qt.Key.Key_M, "m")) is True
    assert win._vcb.text == "1m"


def test_enter_clears_and_esc_clears(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    win._tool_manager.activate_by_shortcut("M")
    win._vcb_handle_key(_key(Qt.Key.Key_1, "1"))
    # apply (no gesture → tool returns False, but VCB clears)
    win._vcb_handle_key(_key(Qt.Key.Key_Return, "\r"))
    assert not win._vcb.active
    win._vcb_handle_key(_key(Qt.Key.Key_5, "5"))
    win._vcb_handle_key(_key(Qt.Key.Key_Escape, "\x1b"))
    assert not win._vcb.active
