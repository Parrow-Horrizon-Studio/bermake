"""Icon stems the M7.3 panel references directly, not through an ActionSpec.

tests/test_icon_assets.py enforces that no asset file is an orphan. Every
other icon in the project is reachable from ACTIONS, so that test derived its
whitelist from the registry alone. The panel's tab strip and Outliner rows
need icons that are not commands, so they are declared here and the orphan
test unions this set in -- the guard keeps working, and adding an unused SVG
still fails it.

Qt-free on purpose: this is a set of strings, importable by the asset test
without a QApplication.
"""

from __future__ import annotations

TAB_ICONS = frozenset(
    {
        "tab-tool-settings",
        "tab-entity-info",
        "tab-material",
        "tab-tags",
        "tab-scenes",
    }
)

OUTLINER_ICONS = frozenset(
    {
        "outliner-group",
        "outliner-component",
        "visible",
        "hidden",
    }
)

NON_ACTION_ICONS = TAB_ICONS | OUTLINER_ICONS
