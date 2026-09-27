from __future__ import annotations

from bermake.document import DocumentSettings
from bermake.units import UnitSystem


def test_default_is_metric_meters():
    d = DocumentSettings()
    assert d.units.system is UnitSystem.METRIC
    assert d.units.metric_unit == "m"


def test_set_units_replaces():
    d = DocumentSettings()
    d.set_metric("mm")
    assert d.units.metric_unit == "mm"
    d.set_imperial()
    assert d.units.system is UnitSystem.IMPERIAL


def test_switching_preserves_other_systems_prefs():
    d = DocumentSettings()
    d.set_metric("mm")
    d.set_imperial()                 # switching to imperial keeps metric_unit
    assert d.units.metric_unit == "mm"
    d.set_metric("cm")               # back to metric keeps imperial_denominator (default 16)
    assert d.units.imperial_denominator == 16


def test_main_window_has_doc_and_units_menu(qtbot):
    from bermake.ui.main_window import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    assert win._doc.units.system is UnitSystem.METRIC
    assert win._units_menu is not None
    assert "Units" in win._units_menu.title()
