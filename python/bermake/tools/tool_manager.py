"""ToolManager: holds the registered tool roster and the currently-active tool."""

from __future__ import annotations

from bermake.tools.tool import Tool, ToolContext


class ToolManager:
    """One active tool at a time, addressed by stable id or by keyboard shortcut.

    Registration used to key solely on `shortcut.upper()`, which made the
    shortcut a tool's identity: two tools sharing no shortcut (both `""`)
    would silently overwrite each other in the registry, and neither could be
    armed from a menu or toolbar. Tools are now looked up by `id` first; the
    shortcut index below is kept only for `activate_by_shortcut`, which the
    keyboard-shortcut path still needs (M7.4 Task 5).
    """

    def __init__(self, ctx: ToolContext | None = None) -> None:
        self._ctx = ctx
        self._tools_by_id: dict[str, Tool] = {}
        self._tools_by_shortcut: dict[str, Tool] = {}
        self._active: Tool | None = None

    def set_context(self, ctx: ToolContext) -> None:
        """MainWindow calls this once the Scene exists."""
        self._ctx = ctx

    def register(self, tool: Tool) -> None:
        """Register `tool` under its id, and under its shortcut if it has one.

        Registering a second tool under an id (or shortcut) already in use
        replaces the first -- last registration wins, the same silent-replace
        semantics a plain dict assignment always had here.
        """
        self._tools_by_id[tool.id] = tool
        shortcut = tool.shortcut.upper()
        if shortcut:
            self._tools_by_shortcut[shortcut] = tool

    def tool_ids(self) -> set[str]:
        """The set of currently-registered tool ids."""
        return set(self._tools_by_id)

    def activate_by_id(self, tool_id: str) -> bool:
        """Arm the tool registered under `tool_id`. False if the id is unknown."""
        return self._arm(self._tools_by_id.get(tool_id))

    def activate_by_shortcut(self, key: str) -> bool:
        """Arm the tool registered under keyboard shortcut `key`.

        An empty key never matches -- a tool with no shortcut is not indexed
        here at all, but the early return also guards against a stray empty
        string being mistaken for "no filter" and matching arbitrarily.
        """
        if not key:
            return False
        return self._arm(self._tools_by_shortcut.get(key.upper()))

    def _arm(self, target: Tool | None) -> bool:
        if target is None:
            return False
        if self._active is target:
            return True
        if self._ctx is None:
            raise RuntimeError("ToolManager has no ToolContext; call set_context() first")
        if self._active is not None:
            self._active.deactivate()
        target.activate(self._ctx)
        self._active = target
        return True

    def deactivate_current(self) -> None:
        if self._active is not None:
            self._active.deactivate()
            self._active = None

    @property
    def active(self) -> Tool | None:
        return self._active
