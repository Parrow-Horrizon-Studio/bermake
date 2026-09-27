"""The Properties editor's tab table (M7.3).

Qt-free, like actions.py, and carrying the same "importing this loads no
PySide6" test.

Be clear-eyed about what this buys. Unlike actions.py -- where one declaration
feeds a menu, a toolbar, a shortcut and a context menu -- a tab appears in
exactly ONE surface, so this table collapses no duplication. It exists for a
narrower reason: it makes "every tab id maps to an icon asset that exists" a
test rather than a blank button someone notices in a screenshot. M7.2 hit that
class of bug, which is why ui_builder.build_action carries a KeyError fallback.

Page construction stays in properties_dock.py: a factory here would need Qt
and defeat the point.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TabSpec:
    """One Properties tab. `icon` is an asset stem under bermake/ui/icons."""

    id: str
    title: str
    icon: str


PROPERTIES_TABS: tuple[TabSpec, ...] = (
    TabSpec("tool_settings", "Tool Settings", "tab-tool-settings"),
    TabSpec("entity_info", "Entity Info", "tab-entity-info"),
    TabSpec("material", "Material", "tab-material"),
    TabSpec("tags", "Tags", "tab-tags"),
    TabSpec("scenes", "Scenes", "tab-scenes"),
)
