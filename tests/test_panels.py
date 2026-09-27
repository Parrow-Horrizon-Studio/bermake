"""The Properties tab table (M7.3 Task 8)."""

from __future__ import annotations

import subprocess
import sys

from bermake.ui import icons
from bermake.ui.panel_icons import TAB_ICONS
from bermake.ui.panels import PROPERTIES_TABS


def test_importing_the_table_loads_no_qt():
    # Same rule actions.py follows: the UI taxonomy must be inspectable
    # without a QApplication.
    code = "import sys; import bermake.ui.panels; print(any(m.startswith('PySide6') for m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_the_five_tabs_are_declared_in_order():
    assert [t.id for t in PROPERTIES_TABS] == [
        "tool_settings",
        "entity_info",
        "material",
        "tags",
        "scenes",
    ]


def test_tab_ids_are_unique():
    ids = [t.id for t in PROPERTIES_TABS]
    assert len(ids) == len(set(ids))


def test_every_tab_icon_resolves_to_an_asset():
    # The single reason this table exists rather than living in the widget.
    stems = icons.available_icon_stems()
    missing = [t.icon for t in PROPERTIES_TABS if t.icon not in stems]
    assert missing == []


def test_every_tab_icon_is_declared_as_a_panel_icon():
    assert {t.icon for t in PROPERTIES_TABS} == TAB_ICONS


def test_every_tab_has_a_title():
    assert all(t.title.strip() for t in PROPERTIES_TABS)
