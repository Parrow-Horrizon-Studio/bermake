"""Window/toolbar state persistence (M7.2 Task 8)."""

from __future__ import annotations

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QMainWindow, QToolBar

from bermake.ui import window_state


def _settings(tmp_path) -> QSettings:
    return QSettings(str(tmp_path / "bermake_test.ini"), QSettings.Format.IniFormat)


def _window(qtbot) -> QMainWindow:
    window = QMainWindow()
    bar = QToolBar("Standard", window)
    bar.setObjectName("standard")
    window.addToolBar(bar)
    qtbot.addWidget(window)
    return window


def test_restore_returns_false_when_nothing_is_saved(qtbot, tmp_path):
    assert window_state.restore_window_state(_window(qtbot), _settings(tmp_path)) is False


def test_save_then_restore_round_trips(qtbot, tmp_path):
    settings = _settings(tmp_path)
    saved = _window(qtbot)
    # 720x560 stays comfortably inside the 800x800 offscreen virtual screen
    # that CI uses (see tests/conftest.py), with real headroom -- beyond
    # whatever margin QMainWindow.restoreGeometry() reserves for the window
    # frame -- when clamping to the available screen. It also differs from
    # BOTH of QMainWindow's own defaults: 640x480 for a never-shown window
    # (what qtbot.addWidget leaves us with, since it does not call .show())
    # and 200x100 for a shown one. Matching either default by coincidence
    # would let this assertion pass even if restore_window_state() restored
    # nothing at all.
    saved.resize(720, 560)
    window_state.save_window_state(saved, settings)

    restored = _window(qtbot)
    assert window_state.restore_window_state(restored, settings) is True
    assert restored.size().width() == 720
    assert restored.size().height() == 560


def test_restore_survives_a_corrupt_blob(qtbot, tmp_path):
    settings = _settings(tmp_path)
    settings.setValue(window_state.GEOMETRY_KEY, b"not-a-qt-blob")
    settings.setValue(window_state.STATE_KEY, b"also-garbage")

    # Must return False rather than raising -- a bad settings store can never
    # prevent the application from starting.
    assert window_state.restore_window_state(_window(qtbot), settings) is False


def test_restore_rejects_a_state_from_a_different_version(qtbot, tmp_path):
    settings = _settings(tmp_path)
    saved = _window(qtbot)
    settings.setValue(window_state.GEOMETRY_KEY, saved.saveGeometry())
    settings.setValue(
        window_state.STATE_KEY, saved.saveState(window_state.WINDOW_STATE_VERSION + 1)
    )

    assert window_state.restore_window_state(_window(qtbot), settings) is False


def test_restore_is_atomic_when_state_restore_fails(qtbot, tmp_path):
    settings = _settings(tmp_path)
    saved = _window(qtbot)
    settings.setValue(window_state.GEOMETRY_KEY, saved.saveGeometry())
    # A state blob saved under a different version always fails Qt's own
    # version check inside restoreState(), even though the geometry blob
    # right above it is perfectly valid.
    settings.setValue(
        window_state.STATE_KEY, saved.saveState(window_state.WINDOW_STATE_VERSION + 1)
    )

    restored = _window(qtbot)
    restored.resize(555, 444)
    size_before = restored.size()

    assert window_state.restore_window_state(restored, settings) is False
    # The geometry half must not have been left applied -- either both parts
    # restore or neither does.
    assert restored.size() == size_before


def test_restore_survives_a_wrong_type_value(qtbot, tmp_path):
    settings = _settings(tmp_path)
    # An INI-format QSettings can hand back a plain str where a QByteArray
    # was written (e.g. hand-edited, or written by a non-Qt tool). That's a
    # distinct failure mode from garbage bytes: restoreGeometry() raises
    # instead of returning False, which is what the except clause guards.
    settings.setValue(window_state.GEOMETRY_KEY, "not-a-blob")
    settings.setValue(window_state.STATE_KEY, "also-not-a-blob")

    assert window_state.restore_window_state(_window(qtbot), settings) is False


def test_reset_clears_both_keys(qtbot, tmp_path):
    settings = _settings(tmp_path)
    window_state.save_window_state(_window(qtbot), settings)
    assert settings.value(window_state.STATE_KEY) is not None

    window_state.reset_window_state(settings)

    assert settings.value(window_state.GEOMETRY_KEY) is None
    assert settings.value(window_state.STATE_KEY) is None


def test_toolbar_visibility_survives_a_round_trip(qtbot, tmp_path):
    settings = _settings(tmp_path)
    saved = _window(qtbot)
    saved.findChild(QToolBar, "standard").setVisible(False)
    window_state.save_window_state(saved, settings)

    restored = _window(qtbot)
    window_state.restore_window_state(restored, settings)

    assert restored.findChild(QToolBar, "standard").isHidden()


def test_properties_dock_has_an_object_name_so_state_can_persist(qtbot, main_window):
    # QMainWindow.saveState() silently drops docks without an object name,
    # exactly like toolbars -- see
    # tests/test_ui_builder_toolbars.py::test_every_toolbar_has_an_object_name_so_state_can_persist.
    # Materials/Tags/Scenes were docks through this exact scenario before
    # M7.3 Task 11 turned them into plain Properties-panel pages; Properties
    # itself (M7.3 Task 8) is the sole surviving real dock.
    assert main_window._properties_dock.objectName() != ""


def test_dock_visibility_and_floating_state_survive_a_round_trip(qtbot, tmp_path, main_window):
    # Reproduces the Task 11 (M7.2) review finding directly: before a dock
    # had an object name, restore_window_state() still returned True (Qt's
    # restoreState() drops unnamed widgets silently, with no error), but the
    # dock came back in its default state instead of the saved one. Both
    # isHidden() and isFloating() default to False on a never-shown window
    # (verified separately), so a passing assertion below cannot be an
    # accidental match against the default -- it proves a genuine restore of
    # dock-specific state, not just the toolbar/geometry halves. Properties
    # is the only real dock left after M7.3 Task 11.
    settings = _settings(tmp_path)

    main_window._properties_dock.setVisible(False)
    main_window._properties_dock.setFloating(True)
    window_state.save_window_state(main_window, settings)

    from bermake.ui.main_window import MainWindow

    restored = MainWindow()
    qtbot.addWidget(restored)

    assert window_state.restore_window_state(restored, settings) is True
    assert restored._properties_dock.isHidden()
    assert restored._properties_dock.isFloating()


def test_a_version_1_layout_is_refused(qtbot, tmp_path):
    """A blob written before the dock consolidation must be discarded whole.

    Qt returns False for a version mismatch rather than partially applying,
    which is exactly why the constant exists -- without the bump, restoreState
    would try to place three docks that no longer exist.
    """
    settings = _settings(tmp_path)
    window = _window(qtbot)
    settings.setValue(window_state.GEOMETRY_KEY, window.saveGeometry())
    settings.setValue(window_state.STATE_KEY, window.saveState(1))

    assert window_state.restore_window_state(_window(qtbot), settings) is False
