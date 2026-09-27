def test_unit_change_refreshes_the_wall_options_bar(qtbot):
    from bermake.ui.main_window import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    win._activate("wall")                       # Wall tool -> its options bar is shown

    calls = []
    original = win._refresh_tool_options
    win._refresh_tool_options = lambda: (calls.append(1), original())[1]

    win._set_units_metric("mm")
    assert calls, "switching units must refresh the active tool's options bar"

    calls.clear()
    win._set_units_imperial()
    assert calls, "switching to imperial must refresh the active tool's options bar"
