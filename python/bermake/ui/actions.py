"""The declarative action registry (M7.2).

Every user-invokable command is declared here exactly once, as plain data.
Menus, toolbars, shortcuts, and context menus all reference actions by id, so
label / icon / shortcut are identical in every surface by construction.

This module imports no Qt: the entire UI taxonomy is therefore testable
without a QApplication. `bermake.viewport.render_style` is pure and safe to
import.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from bermake.viewport.render_style import FaceStyle

TOOL_GROUP = "tool"
FACE_STYLE_GROUP = "face_style"
UNITS_GROUP = "units"
ENVIRONMENT_GROUP = "environment"
THEME_GROUP = "theme"
AUTOSAVE_GROUP = "autosave"


class CursorStyle(Enum):
    """Which base a tool's cursor is composited onto.

    CROSSHAIR: the tool places a point and needs a precise hotspot.
    ARROW: the tool picks an existing entity, where a crosshair would imply
    an accuracy that is not being used.
    """

    CROSSHAIR = "crosshair"
    ARROW = "arrow"


class ContextTarget(Enum):
    """What sits under the cursor when the right mouse button is released."""

    FACE = "face"
    EDGE = "edge"
    INSTANCE = "instance"
    ANNOTATION = "annotation"
    EMPTY = "empty"


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """One user-invokable command.

    handler is a MainWindow method name, resolved with getattr at build time.
    handler_arg, when set, is passed as the method's single positional
    argument -- used by the tools (which all share `_activate`), the face
    styles, and the metric unit entries.
    """

    id: str
    label: str
    handler: str
    icon: str | None = None
    shortcut: str | None = None
    extra_shortcuts: tuple[str, ...] = ()
    checkable: bool = False
    group: str | None = None
    cursor: CursorStyle | None = None
    handler_arg: object | None = None
    tooltip: str | None = None

    @property
    def effective_tooltip(self) -> str:
        if self.tooltip is not None:
            return self.tooltip
        if self.shortcut is not None:
            return f"{self.label} ({self.shortcut})"
        return self.label


@dataclass(frozen=True, slots=True)
class ToolbarSpec:
    """One QToolBar. A None entry in action_ids is a separator."""

    id: str
    title: str
    action_ids: tuple[str | None, ...]


@dataclass(frozen=True, slots=True)
class SubmenuSpec:
    """A nested menu, placed as an entry inside MenuSpec.action_ids (M7.12).

    A None entry in action_ids is a separator. Submenus do not nest further.
    """

    title: str
    action_ids: tuple[str | None, ...]


@dataclass(frozen=True, slots=True)
class MenuSpec:
    """One top-level menu. A None entry in action_ids is a separator, and a
    SubmenuSpec entry is a nested menu."""

    title: str
    action_ids: tuple[str | SubmenuSpec | None, ...]


def _tool(action_id: str, label: str, shortcut: str | None, cursor: CursorStyle) -> ActionSpec:
    """All tools share the same shape: checkable, grouped, and dispatched
    through MainWindow._activate with their tool id as the argument.

    shortcut is optional. It used to double as the dispatch key, which meant
    a tool without one could not be armed at all (M7.4 Task 5).
    """
    return ActionSpec(
        id=action_id,
        label=label,
        handler="_activate",
        handler_arg=action_id.removeprefix("tool_"),
        icon=action_id,
        shortcut=shortcut,
        checkable=True,
        group=TOOL_GROUP,
        cursor=cursor,
    )


_CH = CursorStyle.CROSSHAIR
_AR = CursorStyle.ARROW

ACTIONS: tuple[ActionSpec, ...] = (
    # --- File ---------------------------------------------------------
    ActionSpec("file_new", "New", "_on_file_new", icon="file_new", shortcut="Ctrl+N"),
    ActionSpec("file_open", "Open…", "_on_file_open", icon="file_open", shortcut="Ctrl+O"),
    ActionSpec("file_save", "Save", "_on_file_save", icon="file_save", shortcut="Ctrl+S"),
    ActionSpec("file_save_as", "Save As…", "_on_file_save_as", shortcut="Ctrl+Shift+S"),
    # File > Autosave (M7.12): the interval in minutes, 0 meaning Off.
    ActionSpec(
        "file_autosave_off",
        "Off",
        "_set_autosave_interval",
        handler_arg=0,
        checkable=True,
        group=AUTOSAVE_GROUP,
    ),
    ActionSpec(
        "file_autosave_1",
        "Every Minute",
        "_set_autosave_interval",
        handler_arg=1,
        checkable=True,
        group=AUTOSAVE_GROUP,
    ),
    ActionSpec(
        "file_autosave_5",
        "Every 5 Minutes",
        "_set_autosave_interval",
        handler_arg=5,
        checkable=True,
        group=AUTOSAVE_GROUP,
    ),
    ActionSpec(
        "file_autosave_10",
        "Every 10 Minutes",
        "_set_autosave_interval",
        handler_arg=10,
        checkable=True,
        group=AUTOSAVE_GROUP,
    ),
    ActionSpec(
        "file_autosave_30",
        "Every 30 Minutes",
        "_set_autosave_interval",
        handler_arg=30,
        checkable=True,
        group=AUTOSAVE_GROUP,
    ),
    ActionSpec("file_import_obj", "Import OBJ…", "_on_import_obj"),
    ActionSpec("file_export_obj", "Export OBJ…", "_on_export_obj"),
    ActionSpec("file_import_gltf", "Import glTF…", "_on_import_gltf"),
    ActionSpec("file_export_gltf", "Export glTF…", "_on_export_gltf"),
    # --- Edit ---------------------------------------------------------
    ActionSpec("edit_undo", "Undo", "_on_undo", icon="edit_undo", shortcut="Ctrl+Z"),
    ActionSpec(
        "edit_redo",
        "Redo",
        "_on_redo",
        icon="edit_redo",
        shortcut="Ctrl+Y",
        extra_shortcuts=("Ctrl+Shift+Z",),
    ),
    ActionSpec("edit_select_all", "Select All", "_on_select_all", shortcut="Ctrl+A"),
    ActionSpec("edit_select_none", "Select None", "_on_select_none"),
    # M7.6c. Invert lives on Edit because it needs no entity under the
    # cursor; the other four are relative to a click and live on the
    # right-click Select submenu, which is where a SketchUp user reaches
    # for them.
    ActionSpec("edit_invert_selection", "Invert Selection", "_on_invert_selection"),
    ActionSpec("select_grow", "Grow", "_on_grow_selection"),
    ActionSpec("select_shrink", "Shrink", "_on_shrink_selection"),
    ActionSpec("select_same_material", "All with Same Material", "_on_select_same_material"),
    ActionSpec("select_same_tag", "All with Same Tag", "_on_select_same_tag"),
    ActionSpec(
        "edit_erase",
        "Erase",
        "_on_delete_selection",
        shortcut="Del",
        extra_shortcuts=("Backspace",),
    ),
    # M7.3. No icon: menu and context-menu entries only, never toolbar
    # buttons, and test_icon_assets pins the declared-icon count at 29.
    ActionSpec("edit_hide", "Hide", "_on_hide", shortcut="H"),
    ActionSpec("edit_unhide", "Unhide", "_on_unhide", shortcut="Shift+H"),
    ActionSpec("edit_unhide_all", "Unhide All", "_on_unhide_all"),
    ActionSpec("edit_make_group", "Make Group", "_on_make_group", shortcut="Ctrl+G"),
    ActionSpec(
        "edit_make_component",
        "Make Component…",
        "_on_make_component",
        shortcut="Ctrl+Shift+G",
    ),
    ActionSpec("edit_explode", "Explode", "_on_explode", shortcut="Ctrl+Shift+E"),
    ActionSpec("edit_make_unique", "Make Unique", "_on_make_unique"),
    ActionSpec("edit_clear_context", "Clear Active Context", "_on_clear_scene"),
    ActionSpec("edit_edit_group", "Edit Group", "_on_edit_group"),
    ActionSpec("edit_close_group", "Close Group", "_on_close_group"),
    ActionSpec("edit_paint_selection", "Paint with Active Material", "_on_paint_selection"),
    ActionSpec("edit_label_text", "Edit Text…", "_on_edit_label_text"),
    # M7.6b Task 7. No icon: menu entry only, like edit_hide/edit_unhide above.
    ActionSpec("edit_delete_guides", "Delete Guides", "_on_delete_guides"),
    # --- Units --------------------------------------------------------
    ActionSpec(
        "units_metric_m",
        "Metric (m)",
        "_set_units_metric",
        handler_arg="m",
        checkable=True,
        group=UNITS_GROUP,
    ),
    ActionSpec(
        "units_metric_cm",
        "Metric (cm)",
        "_set_units_metric",
        handler_arg="cm",
        checkable=True,
        group=UNITS_GROUP,
    ),
    ActionSpec(
        "units_metric_mm",
        "Metric (mm)",
        "_set_units_metric",
        handler_arg="mm",
        checkable=True,
        group=UNITS_GROUP,
    ),
    ActionSpec(
        "units_imperial",
        "Imperial (architectural)",
        "_set_units_imperial",
        checkable=True,
        group=UNITS_GROUP,
    ),
    # --- Environment (M7.7) -------------------------------------------------
    # Flat entries in View rather than a submenu: they predate SubmenuSpec
    # (M7.12), when a MenuSpec could not nest. Labels carry no dash.
    ActionSpec(
        "view_env_sky_ground",
        "Sky and Ground",
        "_set_environment",
        handler_arg="sky_and_ground",
        checkable=True,
        group=ENVIRONMENT_GROUP,
    ),
    ActionSpec(
        "view_env_neutral_grey",
        "Neutral Grey",
        "_set_environment",
        handler_arg="neutral_grey",
        checkable=True,
        group=ENVIRONMENT_GROUP,
    ),
    ActionSpec(
        "view_env_landscape",
        "Landscape",
        "_set_environment",
        handler_arg="landscape",
        checkable=True,
        group=ENVIRONMENT_GROUP,
    ),
    ActionSpec(
        "view_env_plain_white",
        "Plain White",
        "_set_environment",
        handler_arg="plain_white",
        checkable=True,
        group=ENVIRONMENT_GROUP,
    ),
    ActionSpec(
        "view_env_studio",
        "Studio",
        "_set_environment",
        handler_arg="studio",
        checkable=True,
        group=ENVIRONMENT_GROUP,
    ),
    # --- Theme (M7.7) -------------------------------------------------------
    ActionSpec(
        "view_theme_light",
        "Light Theme",
        "_set_theme",
        handler_arg="light",
        checkable=True,
        group=THEME_GROUP,
    ),
    ActionSpec(
        "view_theme_dark",
        "Dark Theme",
        "_set_theme",
        handler_arg="dark",
        checkable=True,
        group=THEME_GROUP,
    ),
    # --- Tools --------------------------------------------------------
    _tool("tool_select", "Select", "Space", _AR),
    _tool("tool_eraser", "Eraser", "E", _AR),
    _tool("tool_paint", "Paint", "B", _AR),
    _tool("tool_line", "Line", "L", _CH),
    _tool("tool_rectangle", "Rectangle", "R", _CH),
    _tool("tool_circle", "Circle", "C", _CH),
    _tool("tool_polygon", "Polygon", "G", _CH),
    _tool("tool_arc", "Arc", "A", _CH),
    _tool("tool_push_pull", "Push/Pull", "P", _AR),
    _tool("tool_offset", "Offset", "F", _CH),
    # No shortcut (M7.4 Task 5's registry re-key is what makes this legal):
    # the path is preselected with Select before this tool is ever armed, so
    # there is no natural single letter left free that reads as "sweep".
    _tool("tool_follow_me", "Follow Me", None, _AR),
    _tool("tool_move", "Move", "M", _CH),
    _tool("tool_rotate", "Rotate", "Q", _CH),
    _tool("tool_scale", "Scale", "S", _CH),
    _tool("tool_tape_measure", "Tape Measure", "T", _CH),
    _tool("tool_dimension", "Dimension", "I", _CH),
    _tool("tool_text", "Text", "N", _CH),
    _tool("tool_wall", "Wall", "W", _CH),
    _tool("tool_door_window", "Door/Window", "D", _CH),
    _tool("tool_roof", "Roof", "O", _CH),
    # No shortcut (M7.4 Task 5's registry re-key, same as Follow Me above):
    # four new tools competing for single letters against an already-full
    # roster isn't worth the naming fights, and each is reachable from the
    # Primitives toolbar/menu regardless.
    _tool("tool_box", "Box", None, _CH),
    _tool("tool_cylinder", "Cylinder", None, _CH),
    _tool("tool_cone", "Cone", None, _CH),
    _tool("tool_sphere", "Sphere", None, _CH),
    # --- View ---------------------------------------------------------
    ActionSpec(
        "view_style_wireframe",
        "Wireframe",
        "_on_set_face_style",
        icon="view_style_wireframe",
        handler_arg=FaceStyle.WIREFRAME,
        checkable=True,
        group=FACE_STYLE_GROUP,
    ),
    ActionSpec(
        "view_style_hidden_line",
        "Hidden Line",
        "_on_set_face_style",
        icon="view_style_hidden_line",
        handler_arg=FaceStyle.HIDDEN_LINE,
        checkable=True,
        group=FACE_STYLE_GROUP,
    ),
    ActionSpec(
        "view_style_monochrome",
        "Monochrome",
        "_on_set_face_style",
        icon="view_style_monochrome",
        handler_arg=FaceStyle.MONOCHROME,
        checkable=True,
        group=FACE_STYLE_GROUP,
    ),
    ActionSpec(
        "view_style_shaded",
        "Shaded",
        "_on_set_face_style",
        icon="view_style_shaded",
        handler_arg=FaceStyle.SHADED,
        checkable=True,
        group=FACE_STYLE_GROUP,
    ),
    ActionSpec("view_xray", "X-Ray", "_on_toggle_xray", icon="view_xray", checkable=True),
    # M7.5a Task 11. No icon: a menu entry only, like the units toggles above,
    # not a toolbar button.
    ActionSpec("view_color_by_tag", "Color by Tag", "_on_toggle_color_by_tag", checkable=True),
    # M7.6b Task 7. No icon, same reason. Starts checked (see MainWindow.__init__):
    # guides are visible by default, matching SketchUp.
    ActionSpec("view_guides", "Guides", "_on_toggle_guides", checkable=True),
    ActionSpec(
        "view_select_vertices",
        "Select Vertices",
        "_on_toggle_select_vertices",
        checkable=True,
    ),
    ActionSpec(
        "view_zoom_extents",
        "Zoom Extents",
        "_on_zoom_extents",
        icon="view_zoom_extents",
        shortcut="Shift+Z",
    ),
    ActionSpec("view_reset_toolbars", "Reset Toolbars", "_on_reset_toolbars"),
    # Panel focus (M7.3). These replace the toggleViewAction()s that vanished
    # with the Materials/Tags/Scenes docks. No icon: they are menu entries,
    # never toolbar buttons, and test_icon_assets pins the declared-icon count.
    ActionSpec("view_materials", "Materials", "_on_show_material_tab"),
    ActionSpec("view_tags", "Tags", "_on_show_tags_tab"),
    ActionSpec("view_scenes", "Scenes", "_on_show_scenes_tab"),
    # --- Help (M7.7) --------------------------------------------------------
    ActionSpec("help_welcome", "Welcome to Bermake", "_on_show_welcome"),
    ActionSpec(
        "help_compatibility_rendering",
        "Use Compatibility Rendering",
        "_on_toggle_compatibility_rendering",
        checkable=True,
        tooltip="Draw through the bundled Mesa instead of the graphics driver (restart needed)",
    ),
    ActionSpec("help_about", "About Bermake", "_on_show_about"),
)

_BY_ID: dict[str, ActionSpec] = {spec.id: spec for spec in ACTIONS}


def action_by_id(action_id: str) -> ActionSpec:
    """Look up a declared action. Raises KeyError if the id is unknown, so a
    typo in a menu or toolbar table fails a test rather than silently
    producing an empty menu."""
    return _BY_ID[action_id]


TOOLBARS: tuple[ToolbarSpec, ...] = (
    ToolbarSpec(
        "standard",
        "Standard",
        (
            "file_new",
            "file_open",
            "file_save",
            None,
            "edit_undo",
            "edit_redo",
            None,
            "view_zoom_extents",
        ),
    ),
    ToolbarSpec("principal", "Principal", ("tool_select", "tool_eraser", "tool_paint")),
    ToolbarSpec(
        "drawing",
        "Drawing",
        (
            "tool_line",
            "tool_rectangle",
            "tool_circle",
            "tool_polygon",
            "tool_arc",
            "tool_box",
            "tool_cylinder",
            "tool_cone",
            "tool_sphere",
        ),
    ),
    ToolbarSpec(
        "modification",
        "Modification",
        (
            "tool_push_pull",
            "tool_offset",
            "tool_follow_me",
            "tool_move",
            "tool_rotate",
            "tool_scale",
        ),
    ),
    ToolbarSpec(
        "construction",
        "Construction",
        ("tool_tape_measure", "tool_dimension", "tool_text"),
    ),
    ToolbarSpec(
        "architecture",
        "Architecture",
        ("tool_wall", "tool_door_window", "tool_roof"),
    ),
    ToolbarSpec(
        "styles",
        "Styles",
        (
            "view_style_wireframe",
            "view_style_hidden_line",
            "view_style_monochrome",
            "view_style_shaded",
            None,
            "view_xray",
        ),
    ),
)

MENUS: tuple[MenuSpec, ...] = (
    MenuSpec(
        "File",
        (
            "file_new",
            "file_open",
            None,
            "file_save",
            "file_save_as",
            SubmenuSpec(
                "Autosave",
                (
                    "file_autosave_off",
                    "file_autosave_1",
                    "file_autosave_5",
                    "file_autosave_10",
                    "file_autosave_30",
                ),
            ),
            None,
            "file_import_obj",
            "file_export_obj",
            "file_import_gltf",
            "file_export_gltf",
        ),
    ),
    MenuSpec(
        "Edit",
        (
            "edit_undo",
            "edit_redo",
            None,
            "edit_select_all",
            "edit_select_none",
            "edit_invert_selection",
            None,
            "edit_make_group",
            "edit_make_component",
            None,
            "edit_explode",
            "edit_make_unique",
            None,
            "edit_hide",
            "edit_unhide",
            "edit_unhide_all",
            None,
            "edit_delete_guides",
            None,
            "edit_clear_context",
        ),
    ),
    MenuSpec(
        "Units",
        ("units_metric_m", "units_metric_cm", "units_metric_mm", "units_imperial"),
    ),
    MenuSpec(
        "Tools",
        (
            "tool_select",
            "tool_eraser",
            "tool_paint",
            None,
            "tool_line",
            "tool_rectangle",
            "tool_circle",
            "tool_polygon",
            "tool_arc",
            "tool_box",
            "tool_cylinder",
            "tool_cone",
            "tool_sphere",
            None,
            "tool_push_pull",
            "tool_offset",
            "tool_follow_me",
            "tool_move",
            "tool_rotate",
            "tool_scale",
            None,
            "tool_tape_measure",
            "tool_dimension",
            "tool_text",
            None,
            "tool_wall",
            "tool_door_window",
            "tool_roof",
        ),
    ),
    MenuSpec(
        "View",
        (
            "view_style_wireframe",
            "view_style_hidden_line",
            "view_style_monochrome",
            "view_style_shaded",
            None,
            "view_xray",
            "view_color_by_tag",
            "view_guides",
            "view_select_vertices",
            None,
            "view_env_sky_ground",
            "view_env_neutral_grey",
            "view_env_landscape",
            "view_env_plain_white",
            "view_env_studio",
            None,
            "view_theme_light",
            "view_theme_dark",
            None,
            "view_zoom_extents",
            None,
            "view_materials",
            "view_tags",
            "view_scenes",
        ),
    ),
    MenuSpec("Help", ("help_welcome", None, "help_compatibility_rendering", None, "help_about")),
)

# Right-click menus. `Assign Tag` is deliberately absent: it is a dynamic
# submenu built from the live TagLibrary (see Task 13), and it appears on
# INSTANCE only -- M5c shipped instances-only tagging, so offering it on a
# face or edge would present a control that cannot work (#69).
CONTEXT_MENUS: dict[ContextTarget, tuple[str | None, ...]] = {
    ContextTarget.FACE: (
        "edit_erase",
        None,
        "edit_make_group",
        "edit_make_component",
        None,
        "edit_paint_selection",
    ),
    ContextTarget.EDGE: (
        "edit_erase",
        None,
        "edit_make_group",
        "edit_make_component",
    ),
    ContextTarget.INSTANCE: (
        "edit_edit_group",
        "edit_erase",
        None,
        "edit_hide",
        "edit_unhide",
        None,
        "edit_explode",
        "edit_make_unique",
    ),
    ContextTarget.ANNOTATION: ("edit_erase", "edit_label_text"),
    ContextTarget.EMPTY: (
        "edit_select_all",
        "edit_select_none",
        None,
        "edit_close_group",
        # A hidden object cannot be right-clicked, so this is the only place
        # the way back can live.
        "edit_unhide_all",
        None,
        "view_zoom_extents",
    ),
}
