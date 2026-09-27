"""The main application window — hosts the viewport, status bar, ToolManager, and CommandStack."""

from __future__ import annotations

import math
from pathlib import Path
from typing import ClassVar

import numpy as np
from PySide6.QtCore import QBuffer, QIODevice, QPoint, QSettings, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QWidget

from bermake.commands import CommandStack
from bermake.commands.scene_commands import ClearSceneCommand
from bermake.commands.view_commands import (
    CreateViewCommand,
    DeleteViewCommand,
    RenameViewCommand,
    ReorderViewCommand,
    UpdateViewCommand,
)
from bermake.document import DocumentSettings
from bermake.io import (
    BermakeIOError,
    export_gltf,
    export_obj,
    load_document,
    read_gltf_scene,
    read_gltf_texture_bytes,
    read_obj_document,
    save_document,
)
from bermake.io.document_codec import CameraState
from bermake.io.obj_io import read_obj_texture_bytes
from bermake.model import Model
from bermake.model.model_queries import instance_path
from bermake.model.tag import TagLibrary
from bermake.selection import Selection
from bermake.tools import (
    ArcTool,
    CircleTool,
    EraserTool,
    LineTool,
    MoveTool,
    PolygonTool,
    PushPullTool,
    RectangleTool,
    RotateTool,
    ScaleTool,
    SelectTool,
    TapeMeasureTool,
    ToolContext,
    ToolManager,
)
from bermake.tools.dimension_tool import DimensionTool
from bermake.tools.follow_me_tool import FollowMeTool
from bermake.tools.offset_tool import OffsetTool
from bermake.tools.opening_tool import DoorWindowTool
from bermake.tools.paint_tool import PaintTool
from bermake.tools.primitive_tool import BoxTool, ConeTool, CylinderTool, SphereTool
from bermake.tools.roof_tool import RoofTool
from bermake.tools.text_tool import TextTool
from bermake.tools.wall_tool import WallTool
from bermake.ui import preferences, selection_controller
from bermake.ui.cursors import cursor_for
from bermake.ui.document_controller import DocumentController
from bermake.ui.entity_info_page import EntityInfoPage
from bermake.ui.materials_page import MaterialsPage
from bermake.ui.opening_options_bar import OpeningOptionsBar
from bermake.ui.outliner_tree import OutlinerTree
from bermake.ui.primitive_options_bar import PrimitiveOptionsBar
from bermake.ui.properties_dock import PropertiesDock
from bermake.ui.roof_options_bar import RoofOptionsBar
from bermake.ui.scenes_page import ScenesPage
from bermake.ui.status_bar import StatusBar
from bermake.ui.tags_page import TagsPage
from bermake.ui.tool_settings_page import ToolSettingsPage
from bermake.ui.value_control_box import ValueControlBox
from bermake.ui.wall_options_bar import WallOptionsBar
from bermake.ui.window_state import (
    WINDOW_STATE_VERSION,
    reset_window_state,
    restore_window_state,
    save_window_state,
)
from bermake.viewport import environment as env_module
from bermake.viewport.render_style import FaceStyle, RenderStyle
from bermake.viewport.texture_cache import decode_image, sniff_format
from bermake.viewport.view_animator import ViewAnimator
from bermake.viewport.viewport_widget import _GUIDE_KINDS, ViewportWidget
from bermake.views.capture import apply_tags_and_style, capture_view

# M7.5b Task 11 (#78): container thumbnails are for a file browser preview, so
# the long edge is capped well below full viewport resolution.
_THUMBNAIL_MAX_EDGE = 512

# M7.6b: arrow keys arm an inference lock on the matching axis, unless the
# active tool claims Up/Down for itself (Tool.consumes_arrow_keys).
_LOCK_AXIS_BY_KEY = {
    Qt.Key.Key_Up: 2,  # blue / Z
    Qt.Key.Key_Left: 1,  # green / Y
    Qt.Key.Key_Right: 0,  # red / X
}


def _texture_decoder(data: bytes):
    """Adapt texture_cache's decoder to the Qt-free shape bermake/io expects."""
    decoded = decode_image(data)
    if decoded is None:
        return None
    return (
        sniff_format(data) or "png",
        decoded.width,
        decoded.height,
        decoded.has_transparency,
    )


class MainWindow(QMainWindow):
    """Top-level Bermake window."""

    # Maps the current render style back to the registry action id that
    # represents it (M7.2, Task 10) -- used to seed the face-style radio
    # group's initial checked state from self._render_style.
    _FACE_STYLE_ACTION_IDS: ClassVar[dict[FaceStyle, str]] = {
        FaceStyle.WIREFRAME: "view_style_wireframe",
        FaceStyle.HIDDEN_LINE: "view_style_hidden_line",
        FaceStyle.MONOCHROME: "view_style_monochrome",
        FaceStyle.SHADED: "view_style_shaded",
    }

    def _face_style_action_id(self) -> str:
        return self._FACE_STYLE_ACTION_IDS[self._render_style.face_style]

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Bermake")
        self.resize(1280, 800)

        # Per-document settings (units etc.)
        self._doc = DocumentSettings()

        # Measurements box (VCB) — pure state, no Qt dependency.
        self._vcb = ValueControlBox()

        # Scene graph + tool manager + command stack
        self._model = Model()
        self._render_style = RenderStyle()
        self._selection = Selection()
        self._command_stack = CommandStack()
        self._tool_manager = ToolManager()
        self._tool_manager.register(LineTool())
        self._tool_manager.register(RectangleTool())
        self._tool_manager.register(PushPullTool())
        self._tool_manager.register(OffsetTool())
        self._follow_me_tool = FollowMeTool()
        self._tool_manager.register(self._follow_me_tool)
        self._tool_manager.register(CircleTool())
        self._tool_manager.register(PolygonTool())
        self._tool_manager.register(ArcTool())
        self._tool_manager.register(SelectTool())
        self._tool_manager.register(EraserTool())
        self._paint_tool = PaintTool()
        self._tool_manager.register(self._paint_tool)
        self._tool_manager.register(MoveTool())
        self._tool_manager.register(RotateTool())
        self._tool_manager.register(ScaleTool())
        self._tool_manager.register(TapeMeasureTool())
        self._wall_tool = WallTool()
        self._tool_manager.register(self._wall_tool)
        self._opening_tool = DoorWindowTool()
        self._tool_manager.register(self._opening_tool)
        self._roof_tool = RoofTool()
        self._tool_manager.register(self._roof_tool)
        self._dimension_tool = DimensionTool()
        self._tool_manager.register(self._dimension_tool)
        self._text_tool = TextTool()
        self._tool_manager.register(self._text_tool)
        self._box_tool = BoxTool()
        self._tool_manager.register(self._box_tool)
        self._cylinder_tool = CylinderTool()
        self._tool_manager.register(self._cylinder_tool)
        self._cone_tool = ConeTool()
        self._tool_manager.register(self._cone_tool)
        self._sphere_tool = SphereTool()
        self._tool_manager.register(self._sphere_tool)

        # Viewport + status bar (created BEFORE setting ToolContext so we can
        # wire the camera + widget_size_provider into the context).
        self._viewport = ViewportWidget(self._model, self._tool_manager, self)
        self._viewport.selection = self._selection
        # M7d, Task 13 (Part B): wire the annotation units provider (Task 4
        # carry-over) -- without this, _paint_annotations silently falls back
        # to a default Units() regardless of the document's unit setting.
        self._viewport.set_units_provider(lambda: self._doc.units)
        # M7.6b Task 9: the cursor readout stands down while the VCB's typed
        # buffer is active, so the two numeric sinks never show conflicting
        # values at once.
        self._viewport.set_vcb_active_provider(lambda: self._vcb.active)
        self._status_bar = StatusBar()
        # Follow Me is the first tool that must report a refusal (a forked
        # preselection, or a corner too tight for the profile) from inside
        # its own commit path rather than through a MainWindow handler, so
        # it needs a standing status-bar reference of its own -- wired here,
        # outside ToolContext, once.
        self._follow_me_tool.set_status_bar(self._status_bar)

        # Materials page — must exist BEFORE _rebuild_tool_context() so the
        # lambda `set_active_material=self._materials_page.set_active` captures
        # a live reference.
        self._active_material_id = self._model.materials.DEFAULT_ID
        self._materials_page = MaterialsPage(
            self._model.materials, self, command_stack=self._command_stack, model=self._model
        )
        self._materials_page.active_material_changed.connect(self._on_active_material_changed)

        # Tags page. Not referenced by the ToolContext (tag assignment uses the
        # existing Select tool + Selection).
        self._active_tag_id = TagLibrary.UNTAGGED_ID
        self._tags_page = TagsPage(
            self._model.tags, self, command_stack=self._command_stack, model=self._model
        )
        self._tags_page.active_tag_changed.connect(self._on_active_tag_changed)
        self._tags_page.visibility_changed.connect(self._viewport.update)
        # Spec 1.7: "Tag visibility toggled -> full rebuild (tag_hidden
        # moved)" -- a highlight-only sync would never move the dimming.
        self._tags_page.visibility_changed.connect(self._rebuild_outliner)
        self._tags_page.assign_to_selection_requested.connect(self._on_assign_tag)

        # Scenes page (M7e).
        self._scenes_page = ScenesPage(self._model.views, self)
        self._scenes_page.create_requested.connect(self._on_create_view)
        self._scenes_page.update_requested.connect(self._on_update_view)
        self._scenes_page.delete_requested.connect(self._on_delete_view)
        self._scenes_page.rename_requested.connect(self._on_rename_view)
        self._scenes_page.reorder_requested.connect(self._on_reorder_view)
        self._scenes_page.recall_requested.connect(self._on_recall_view)

        # Properties panel (M7.3) — Outliner over an icon-tabbed editor.
        # Must be added BEFORE restore_window_state() below: restoreState()
        # only reattaches docks it can find by object name at the time it runs.
        self._properties_dock = PropertiesDock(
            self, command_stack=self._command_stack, model=self._model
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._properties_dock)
        self._outliner = OutlinerTree(self._properties_dock)
        self._properties_dock.set_outliner(self._outliner)
        self._outliner.instance_clicked.connect(self._on_outliner_clicked)
        self._outliner.instance_activated.connect(self._on_outliner_activated)
        self._outliner.hide_toggled.connect(self._on_outliner_hide_toggled)
        self._outliner.rename_requested.connect(self._on_outliner_rename)
        self._command_stack.add_change_listener(self._rebuild_outliner)
        self._rebuild_outliner()

        # Materials/Tags/Scenes pages (M7.3, Task 11) — installed into the
        # Properties panel's tab stack right after the Outliner above.
        self._properties_dock.set_page("material", self._materials_page)
        self._properties_dock.set_page("tags", self._tags_page)
        self._properties_dock.set_page("scenes", self._scenes_page)

        # Entity Info page (M7.3, Task 13) -- reports what is selected and
        # edits the few properties that already have commands behind them.
        # The page emits intent only; MainWindow routes each signal through
        # the command stack below, exactly as the pages above do.
        self._entity_info_page = EntityInfoPage(self._properties_dock)
        self._properties_dock.set_page("entity_info", self._entity_info_page)
        self._entity_info_page.rename_requested.connect(self._on_entity_rename)
        self._entity_info_page.hidden_requested.connect(self._on_entity_hidden)
        self._entity_info_page.tag_requested.connect(self._assign_tag)
        self._entity_info_page.material_requested.connect(self._on_entity_material)
        # NOT wired as a blanket self._command_stack.add_change_listener the way
        # _rebuild_outliner is (d1491bc) -- tried that first, and it crashes:
        # MakeGroupCommand/MakeComponentCommand fire the change listener (inside
        # execute(), via _fire_change()) BEFORE their caller reassigns
        # self._selection to the new instance, so entity_summary() would run
        # against a selection of edges/faces the command just consumed,
        # chasing a face id no longer live (KeyError from Scene.face_loop).
        # _rebuild_outliner is immune because outliner_rows() reads model
        # structure only, never the selection. Hide/Unhide never change what a
        # selected id refers to (only instance.hidden), so those three call
        # sites refresh explicitly below instead. Undo/redo already refresh
        # correctly via _on_after_undo_redo -> _refresh_selection_status().

        # Camera tween animator (M7e). Owns the same live camera object the
        # viewport mutates in place; a manual camera move cancels a running tween.
        self._view_animator = ViewAnimator(self._viewport.camera, self._viewport.update, self)
        self._viewport.set_camera_input_callback(self._view_animator.cancel)

        # Document session state (path / dirty / title) + dirty signal sources.
        self._doc_controller = DocumentController()
        self._command_stack.add_change_listener(self._on_document_changed)
        self._materials_page.library_changed.connect(self._on_document_changed)
        self._tags_page.library_changed.connect(self._on_document_changed)
        self._tags_page.visibility_changed.connect(self._on_document_changed)
        # _on_document_changed only marks dirty and retitles, but a library
        # edit can change what is on screen right now -- a tag recolored under
        # Color-by-Tag (Task 11), a material recolored under any face style --
        # so both pages also repaint instead of waiting for the next
        # incidental one. visibility_changed is already wired to update above.
        self._materials_page.library_changed.connect(self._viewport.update)
        self._tags_page.library_changed.connect(self._viewport.update)
        # M7.5b final review, item 1: the very same hazard one row down. A
        # texture-placement edit from the Properties spin boxes moves the image
        # on a face that is on screen right now, and the command stack's own
        # listeners only mark dirty, retitle and rebuild the Outliner -- so the
        # panel the spec calls "the primary mechanism" (design line 172) has to
        # ask for the repaint too, exactly as the two library pages above do.
        self._properties_dock.placement_changed.connect(self._on_placement_committed)
        # Task 8 (M7.5c): the button only asks; MainWindow owns the command
        # stack and the selection, so it builds the ResetFaceUvsCommand.
        self._properties_dock.reset_uvs_requested.connect(self._on_reset_uvs_requested)

        # NOW we can build the ToolContext that includes the viewport refs.
        self._rebuild_tool_context()
        # Task 15: initialize breadcrumb (clears it since we start at root).
        # _status_bar exists — it was created just above.
        self._refresh_breadcrumb()

        # Wall options bar (M7a, Task 5) — thickness/height fields for the Wall
        # tool; shown only while the Wall tool is active (see _refresh_tool_options).
        self._wall_options_bar = WallOptionsBar(
            self._wall_tool, units_provider=lambda: self._doc.units
        )

        # Opening options bar (M7b, Task 7) — Door/Window toggle + size fields
        # for the Door/Window tool; shown only while that tool is active (see
        # _refresh_tool_options).
        self._opening_options_bar = OpeningOptionsBar(
            self._opening_tool, units_provider=lambda: self._doc.units
        )

        # Roof options bar (M7c, Task 6) — kind toggle + slope field for the
        # Roof tool (M7c is flush; no overhang field); shown only while that
        # tool is active (see _refresh_tool_options).
        self._roof_options_bar = RoofOptionsBar(
            self._roof_tool, units_provider=lambda: self._doc.units
        )

        # Primitive option bars (M7.4, Task 11) — segments (cylinder/cone/
        # sphere) and rings (sphere only) fields; Box's bar has neither
        # field (its footprint and height come from the drag gesture), but
        # still gets a key below so arming it behaves like the other three.
        self._box_options_bar = PrimitiveOptionsBar(
            self._box_tool, has_segments=False, has_rings=False
        )
        self._cylinder_options_bar = PrimitiveOptionsBar(
            self._cylinder_tool, has_segments=True, has_rings=False
        )
        self._cone_options_bar = PrimitiveOptionsBar(
            self._cone_tool, has_segments=True, has_rings=False
        )
        self._sphere_options_bar = PrimitiveOptionsBar(
            self._sphere_tool, has_segments=True, has_rings=True
        )

        # Tool Settings tab (M7.3, Task 12) — hosts the option bars above in
        # a QStackedWidget; _refresh_tool_options picks which one (if any)
        # is visible.
        self._tool_settings_page = ToolSettingsPage(self._properties_dock)
        self._tool_settings_page.add_bar("wall", self._wall_options_bar)
        self._tool_settings_page.add_bar("opening", self._opening_options_bar)
        self._tool_settings_page.add_bar("roof", self._roof_options_bar)
        self._tool_settings_page.add_bar("box", self._box_options_bar)
        self._tool_settings_page.add_bar("cylinder", self._cylinder_options_bar)
        self._tool_settings_page.add_bar("cone", self._cone_options_bar)
        self._tool_settings_page.add_bar("sphere", self._sphere_options_bar)
        self._properties_dock.set_page("tool_settings", self._tool_settings_page)

        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._viewport, stretch=1)
        layout.addWidget(self._status_bar, stretch=0)
        self.setCentralWidget(container)

        self._viewport.set_status_bar(self._status_bar)
        self._viewport.set_event_finished_callback(self._refresh_status_text)
        # M7.2 Task 14: real right-clicks always show the menu (exec_menu=True);
        # _on_context_menu_requested defaults exec_menu to False so tests can
        # call it directly without entering QMenu.exec()'s modal loop.
        self._viewport.context_menu_requested.connect(
            lambda x, y: self._on_context_menu_requested(x, y, exec_menu=True)
        )

        # Clear selection after any undo or redo.
        self._command_stack.add_undo_listener(self._on_after_undo_redo)
        self._command_stack.add_redo_listener(self._on_after_undo_redo)

        # Keyboard shortcuts not covered by the action registry (M7.2, Task 10)
        # -- these are not commands and have no menu entry.
        QShortcut(
            QKeySequence(Qt.Key.Key_Up), self, activated=lambda: self._on_tool_key(Qt.Key.Key_Up)
        )
        QShortcut(
            QKeySequence(Qt.Key.Key_Down),
            self,
            activated=lambda: self._on_tool_key(Qt.Key.Key_Down),
        )
        QShortcut(
            QKeySequence(Qt.Key.Key_Left),
            self,
            activated=lambda: self._on_tool_key(Qt.Key.Key_Left),
        )
        QShortcut(
            QKeySequence(Qt.Key.Key_Right),
            self,
            activated=lambda: self._on_tool_key(Qt.Key.Key_Right),
        )
        QShortcut(QKeySequence("Esc"), self, activated=self._on_escape)
        QShortcut(QKeySequence(Qt.Key.Key_Return), self, activated=self._on_finish_gesture)
        QShortcut(QKeySequence(Qt.Key.Key_Enter), self, activated=self._on_finish_gesture)

        # Install the VCB event filter on the QApplication so it intercepts
        # key events (and ShortcutOverride) before any QShortcut fires.
        # Guard for None so headless unit tests without a QApplication still work.
        from PySide6.QtWidgets import QApplication

        _app = QApplication.instance()
        if _app is not None:
            _app.installEventFilter(self)

        # Menu bar + actions + shortcuts, built from the declarative registry
        # in bermake.ui.actions (M7.2, Task 10).
        from bermake.ui.ui_builder import build_all_actions, build_menubar, build_shortcuts

        build_all_actions(self)
        self._menus = build_menubar(self)
        build_shortcuts(self)

        # Reflect current document state in the exclusive groups.
        self._actions[self._face_style_action_id()].setChecked(True)
        self._face_style_actions = {
            style: self._actions[action_id]
            for style, action_id in self._FACE_STYLE_ACTION_IDS.items()
        }
        self._xray_action = self._actions["view_xray"]
        self._environment_actions = {
            key: self._actions[action_id]
            for key, action_id in (
                (env_module.SKY_AND_GROUND_KEY, "view_env_sky_ground"),
                (env_module.NEUTRAL_GREY_KEY, "view_env_neutral_grey"),
                (env_module.LANDSCAPE_KEY, "view_env_landscape"),
                (env_module.PLAIN_WHITE_KEY, "view_env_plain_white"),
                (env_module.STUDIO_KEY, "view_env_studio"),
            )
        }
        self._theme_actions = {
            "light": self._actions["view_theme_light"],
            "dark": self._actions["view_theme_dark"],
        }
        self._color_by_tag_action = self._actions["view_color_by_tag"]
        # M7.6b Task 7: guides are visible by default (SketchUp), matching
        # ViewportWidget.show_guides's own default -- this just reflects that
        # starting state into the menu's checkmark.
        self._guides_action = self._actions["view_guides"]
        self._guides_action.setChecked(True)

        # Dynamic View entries: the Properties panel is still a real dock, so
        # it keeps a genuine toggleViewAction. Materials/Tags/Scenes are now
        # tabs inside it and are declared actions instead (see actions.py).
        self._view_menu = self._menus["View"]
        self._view_menu.addSeparator()
        self._properties_dock_action = self._properties_dock.toggleViewAction()
        self._properties_dock_action.setText("Properties Panel")
        self._view_menu.addAction(self._properties_dock_action)

        # Seven dockable toolbars (M7.2, Task 11), sharing the same QAction
        # objects the menus above were built from.
        from bermake.ui.ui_builder import build_toolbars

        self._toolbars = build_toolbars(self)

        # View ▸ Toolbars -- one checkbox per toolbar, plus the escape hatch.
        # Reset Toolbars lives in this menu (not on a toolbar) because
        # restoreState() can hide every toolbar but cannot touch the menu bar.
        self._view_menu.addSeparator()
        self._toolbars_menu = self._view_menu.addMenu("Toolbars")
        for toolbar in self._toolbars.values():
            self._toolbars_menu.addAction(toolbar.toggleViewAction())
        self._toolbars_menu.addSeparator()
        self._toolbars_menu.addAction(self._actions["view_reset_toolbars"])

        # Window/toolbar-layout persistence (M7.2, Task 8 + 11). Must run
        # AFTER every toolbar and dock above exists: QMainWindow.restoreState()
        # only reattaches toolbars/docks it can find by object name at the
        # time it runs, so restoring any earlier would silently restore
        # nothing. `_default_window_state` captures the just-built default
        # arrangement before any saved layout is applied, so Reset Toolbars
        # has a known-good state to fall back to.
        self._settings = QSettings()
        self._default_window_state = self.saveState(WINDOW_STATE_VERSION)
        restore_window_state(self, self._settings)

        # Reflect the saved theme choice (M7.7). Deferred to here, after
        # self._settings exists, rather than beside self._theme_actions above.
        self._theme_actions[preferences.read_theme(self._settings)].setChecked(True)

        # Back-compat aliases for tests that read a named menu by attribute.
        self._file_menu = self._menus["File"]
        self._units_menu = self._menus["Units"]

        self._update_window_title()
        self._sync_environment_ui()

    # --- Material slot ---------------------------------------------------

    def _on_active_material_changed(self, material) -> None:
        self._active_material_id = material.id

    # --- Tag slots -------------------------------------------------------

    def _on_active_tag_changed(self, tag_id: int) -> None:
        self._active_tag_id = tag_id

    def _update_selection_tag_indicator(self) -> None:
        self._tags_page.set_selection_tag(
            selection_controller.selection_tag_label(self._model, self._selection)
        )

    def _on_assign_tag(self) -> None:
        """TagsPage's "Assign to Selection" button -- assigns the page's active tag."""
        self._assign_tag(self._active_tag_id)

    def _assign_tag(self, tag_id: int) -> None:
        """Assign `tag_id` to every selected instance via TagInstancesCommand.

        Shared by the Tags panel's "Assign to Selection" button (_on_assign_tag)
        and the right-click Assign Tag submenu -- one command construction, not
        two. The controller decides; this method owns the Qt half.
        """
        did_assign, message = selection_controller.assign_tag(
            self._model, self._selection, self._command_stack, tag_id
        )
        self._status_bar.set_message(message)
        if did_assign:
            self._update_selection_tag_indicator()
            self._viewport.update()

    # --- Scene graph back-compat property --------------------------------

    @property
    def scene(self):
        """Back-compat: the active scene from the model (root mesh by default)."""
        return self._model.active_scene

    # --- Tool context -----------------------------------------------------

    def _tool_context(self) -> ToolContext:
        """Build a fresh ToolContext pointing at the active scene.

        Split out of `_rebuild_tool_context` (M7.4 Task 8) so a test can
        activate a tool directly -- the same way ToolManager does internally
        when arming one -- without duplicating this construction. Building
        one does not install it anywhere or touch the currently-active tool;
        see `_rebuild_tool_context` for that.
        """
        return ToolContext(
            scene=self._model.active_scene,
            command_stack=self._command_stack,
            camera=self._viewport.camera,
            widget_size_provider=lambda: (self._viewport.width(), self._viewport.height()),
            selection=self._selection,
            units_provider=lambda: self._doc.units,
            model=self._model,
            request_context_rebuild=self._on_active_context_changed,
            active_material_provider=lambda: self._model.materials.get(self._active_material_id),
            set_active_material=self._materials_page.set_active,
            # M7.5b Task 12 fix round 1: PaintTool.end_placement_drag() calls
            # this once per committed drag so Task 10's Properties placement
            # fields don't go stale when the drag lands on the currently
            # selected face -- _refresh_entity_info is the same single call
            # site every other placement-touching action already uses.
            # Final review, item 1: the repaint is now asked for explicitly
            # rather than relying on the mouse-release handler that happens to
            # follow a drag -- see _on_placement_drag_committed.
            on_placement_committed=self._on_placement_drag_committed,
            # M7.6b Task 7 fix round 1: a hidden guide must be unpickable too
            # (Select/Erase), not just unpainted -- this is how that reaches
            # them without handing a picking tool the whole viewport.
            show_guides_provider=lambda: self._viewport.show_guides,
            # M7.6c Task 7: vertex picking is opt-in (View > Select Vertices),
            # off by default -- mirrors show_guides_provider above.
            select_vertices_provider=lambda: self._viewport.select_vertices,
        )

    def _rebuild_tool_context(self) -> None:
        """Build a fresh ToolContext pointing at the active scene and install it.

        Called from __init__ and after any active-context change (undo/redo,
        enter/exit group).  If a tool is already active it is re-activated so
        it picks up the new scene reference.
        """
        ctx = self._tool_context()
        self._tool_manager.set_context(ctx)
        # Re-activate the current tool so it picks up the new context/scene.
        active = self._tool_manager.active
        if active is not None:
            active.deactivate()
            active.activate(ctx)
        if active is not self._paint_tool:
            # M7.5b Task 12: PaintTool.begin_placement_drag/update/end are
            # exercised directly (e.g. by tests reaching `_paint_tool`)
            # without Paint necessarily being the tool the user has armed,
            # so keep its scene/model refs fresh on every rebuild too --
            # mirrors what ToolManager._arm already does the moment Paint
            # truly becomes active.
            self._paint_tool.activate(ctx)

    # --- Slots -----------------------------------------------------------

    def _refresh_status_text(self) -> None:
        active = self._tool_manager.active
        if self._vcb.active:
            self._status_bar.set_status(self._vcb.text + "▏")
        elif active is None:
            self._status_bar.set_status("")
        else:
            # M7.6b Task 9: the Measurements box shows the tool's bare numeric
            # value, not the prose status_text -- the same condition (a tool
            # is active, the VCB is idle) also drives the on-canvas cursor
            # readout in ViewportWidget._paint_annotations, so the two sinks
            # never disagree.
            self._status_bar.set_status(active.measurement_text or "")
        self._refresh_selection_status()
        self._update_selection_tag_indicator()

    def _vcb_handle_key(self, event) -> bool:
        """Pure VCB key logic. Returns True if the key was consumed."""
        active_tool = self._tool_manager.active
        if active_tool is None:
            return False
        key = event.key()
        text = event.text()
        if self._vcb.active:
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if self._vcb.text:
                    active_tool.apply_typed_value(self._vcb.text, self._doc.units)
                self._vcb.clear()
                self._refresh_status_text()
                self._viewport.update()
                return True
            if key == Qt.Key.Key_Escape:
                self._vcb.clear()
                self._refresh_status_text()
                self._viewport.update()
                return True
            if key == Qt.Key.Key_Backspace:
                self._vcb.backspace()
                self._refresh_status_text()
                self._viewport.update()
                return True
            if text and text.isprintable() and text not in ("\r", "\n"):
                self._vcb.feed(text)
                self._refresh_status_text()
                self._viewport.update()
                return True
            return False
        # inactive: only a digit activates the box.
        if text in set("0123456789"):
            self._vcb.feed(text)
            self._refresh_status_text()
            self._viewport.update()
            return True
        return False

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent

        if event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            if event.key() == Qt.Key.Key_Shift:
                # Same arming-order rule as _on_tool_key: a Shift lock formed
                # right after a key-driven gesture end would otherwise be
                # destroyed by the stale edge on the next mouse move.
                if event.type() == QEvent.Type.KeyPress:
                    self._viewport.settle_gesture_lifecycle()
                self._viewport.inference.set_shift_lock(
                    event.type() == QEvent.Type.KeyPress, self._viewport.last_snap
                )
                return False

        if event.type() in (QEvent.Type.KeyPress, QEvent.Type.ShortcutOverride):
            if self._vcb.active or (
                event.type() == QEvent.Type.KeyPress and event.text() in set("0123456789")
            ):
                if self._vcb_handle_key(event):
                    event.accept()
                    return True
        return super().eventFilter(obj, event)

    def _activate(self, tool_id: str) -> None:
        if self._tool_manager.activate_by_id(tool_id):
            active = self._tool_manager.active
            self._status_bar.set_tool(active.name if active else "")
            self._status_bar.set_snap("")
            self._refresh_status_text()
            self._refresh_tool_options()
            self._viewport.update()
            # Keep the toolbar/menu button in sync when the tool was armed by
            # a raw keyboard shortcut rather than by triggering its QAction
            # (Qt only auto-checks a QActionGroup member when the action
            # itself fires) -- otherwise the toolbar would lie about which
            # tool is active (M7.2, Task 11).
            #
            # This used to re-derive the action id by scanning ACTIONS for a
            # spec whose *shortcut* matched the active tool's shortcut, which
            # silently no-opped for any shortcut-less tool (M7.4 Task 5's own
            # invariant -- action_id == f"tool_{tool_id}" for all 18 shipped
            # tools -- makes that indirection unnecessary; `tool_id` is the
            # argument this method was already called with).
            if active is not None:
                action_id = f"tool_{tool_id}"
                self._actions[action_id].setChecked(True)
                # Viewport only: the tool cursor must not leak over docks,
                # the menu bar, or the per-tool option bars. The ratio
                # comes from the widget itself (not a global) since a
                # window can move between monitors with different
                # scaling (M7.2, Task 12).
                dpr = self._viewport.devicePixelRatioF()
                self._viewport.setCursor(cursor_for(action_id, dpr=dpr))

    def _disarm_tool_ui(self) -> None:
        """Undo _activate's visible effects when no tool is armed.

        Arming a tool checks its action (depressing the toolbar button) and
        puts that tool's cursor on the viewport. Disarming has to undo both,
        or the toolbar and the cursor keep advertising a tool that is no
        longer active -- the exact ambiguity the toolbars exist to remove.
        QActionGroup is exclusive, so nothing unchecks the member for us.
        """
        from bermake.ui.actions import TOOL_GROUP

        group = self._action_groups.get(TOOL_GROUP)
        if group is not None and group.checkedAction() is not None:
            group.setExclusive(False)
            group.checkedAction().setChecked(False)
            group.setExclusive(True)
        self._viewport.unsetCursor()

    def _refresh_tool_options(self) -> None:
        """Point the Tool Settings tab at the active tool's option bar.

        Arming a tool that HAS settings also focuses the tab: before M7.3 the
        Wall tool's thickness and height simply appeared, and a panel sitting
        on another tab would silently not show them. Tools without settings
        never steal the tab -- otherwise every tool switch would yank the user
        out of whatever they were inspecting.

        This SWITCHES the tab (spec 1.6) rather than calling
        _show_properties_tab, which also un-hides the whole dock -- arming a
        tool must not reopen a panel the user deliberately closed.
        """
        active = self._tool_manager.active
        key = None
        if isinstance(active, WallTool):
            key = "wall"
        elif isinstance(active, DoorWindowTool):
            key = "opening"
        elif isinstance(active, RoofTool):
            key = "roof"
        elif isinstance(active, BoxTool):
            key = "box"
        elif isinstance(active, CylinderTool):
            key = "cylinder"
        elif isinstance(active, ConeTool):
            key = "cone"
        elif isinstance(active, SphereTool):
            key = "sphere"

        if key == "wall":
            self._wall_options_bar.refresh()
        elif key == "opening":
            self._opening_options_bar.refresh()
        elif key == "roof":
            self._roof_options_bar.refresh()
        elif key == "box":
            self._box_options_bar.refresh()
        elif key == "cylinder":
            self._cylinder_options_bar.refresh()
        elif key == "cone":
            self._cone_options_bar.refresh()
        elif key == "sphere":
            self._sphere_options_bar.refresh()

        self._tool_settings_page.show_bar(key)
        if key is not None:
            self._properties_dock.show_tab("tool_settings")

    def _on_escape(self) -> None:
        self._viewport.inference.release_lock()
        active = self._tool_manager.active
        if active is None:
            return
        if active.has_active_gesture:
            from PySide6.QtGui import QKeyEvent

            ev = QKeyEvent(
                QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier
            )
            active.on_key_press(ev)
        elif self._model.active_path:
            # M4e: inside a group/component with no active gesture — Esc exits one
            # level. Handled here (not in the tool) so Esc exits regardless of which
            # tool is active and even when the selection is empty (entering clears it).
            self._model.exit_one()
            self._selection.clear()
            self._on_active_context_changed()
        else:
            self._tool_manager.deactivate_current()
            self._status_bar.set_tool("")
            self._status_bar.set_snap("")
            self._status_bar.set_coordinates("")
            self._disarm_tool_ui()
            self._refresh_tool_options()
        self._refresh_status_text()
        self._viewport.update()

    def _on_finish_gesture(self) -> None:
        active = self._tool_manager.active
        if active is None or not active.has_active_gesture:
            return
        from PySide6.QtGui import QKeyEvent

        ev = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier)
        active.on_key_press(ev)
        self._refresh_status_text()
        self._viewport.update()

    def _on_tool_key(self, qt_key) -> None:
        """Route an arrow key to inference locking, or to the tool that claims it.

        Left and Right always lock. Up and Down lock only when the active tool does
        not consume them; PolygonTool, RoofTool and RotateTool do.
        """
        active = self._tool_manager.active
        claimed = active is not None and active.consumes_arrow_keys
        if qt_key in (Qt.Key.Key_Up, Qt.Key.Key_Down) and claimed:
            if active.has_active_gesture:
                from PySide6.QtGui import QKeyEvent

                ev = QKeyEvent(QKeyEvent.Type.KeyPress, qt_key, Qt.KeyboardModifier.NoModifier)
                active.on_key_press(ev)
                self._refresh_status_text()
                self._viewport.update()
            return

        inference = self._viewport.inference
        # Settle any pending gesture-end edge BEFORE arming. A gesture that
        # ended on a key (Enter, Escape, a typed value) leaves that edge
        # undetected until the next mouse event, which would otherwise fire
        # it AFTER this keypress and release the lock it just armed. D6: "a
        # lock often outlives the keypress".
        self._viewport.settle_gesture_lifecycle()
        if qt_key == Qt.Key.Key_Down:
            inference.toggle_edge_lock()
        else:
            inference.toggle_axis_lock(_LOCK_AXIS_BY_KEY[qt_key])
        self._refresh_status_text()
        self._viewport.update()

    def _on_clear_scene(self) -> None:
        self._command_stack.execute(ClearSceneCommand(), self._model.active_scene)
        self._refresh_status_text()
        self._viewport.update()

    # --- Edit-menu handlers -----------------------------------------------

    def _grouped_vertex_ids(self, sel) -> list[int]:
        """The vertices Make Group / Make Component move into the new
        definition: every vertex of a selected edge or face, and nothing else.

        Final review I5: both handlers used to call `selection_vertices(scene,
        sel)` directly, which also returns bare selected vertices -- ones
        belonging to no selected edge or face. Both handlers gate on
        `sel.edges or sel.faces`, so a face plus an unrelated bare vertex
        passed the gate and handed that vertex to the command, which copied it
        into the group and then failed to remove it from the parent
        (`remove_vertex` raises on a vertex still used by an edge, and
        group_commands swallows that), leaving a duplicated, orphaned vertex.
        A window box-select reaches this in one gesture: it takes every vertex
        whose point falls in the rect while taking only fully-enclosed faces.

        `selection_vertices` itself is deliberately left alone -- Move, Rotate
        and Scale genuinely want to drag a bare vertex. What differs is what
        "group this" means: the geometry the user selected, and a vertex that
        belongs to no selected edge or face is not geometry to group.
        """
        from types import SimpleNamespace

        from bermake.tools.transform_support import selection_vertices

        geometry_only = SimpleNamespace(edges=sel.edges, faces=sel.faces, vertices=())
        return selection_vertices(self._model.active_scene, geometry_only)

    def _on_make_group(self) -> None:
        from bermake.commands.group_commands import MakeGroupCommand

        sel = self._selection
        if not (sel.edges or sel.faces):
            self._status_bar.set_message("Select edges or faces to group.")
            return
        vertex_ids = self._grouped_vertex_ids(sel)
        edge_ids = list(sel.edges)
        face_ids = list(sel.faces)
        cmd = MakeGroupCommand(
            self._model.active_context, vertex_ids, edge_ids, face_ids, tag_id=self._active_tag_id
        )
        self._command_stack.execute(cmd, self._model)
        sel.replace(instances=[cmd.created_instance.id])
        self._refresh_selection_status()
        self._refresh_breadcrumb()
        self._viewport.update()

    def _prompt_component_name(self, default: str) -> str | None:
        """Show a dialog to get a component name. Overridable for testing."""
        from PySide6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(self, "Make Component", "Component name:", text=default)
        return name if ok else None

    def _on_make_component(self) -> None:
        from bermake.commands.group_commands import MakeComponentCommand

        sel = self._selection
        if not (sel.edges or sel.faces):
            self._status_bar.set_message("Select edges or faces to make a component.")
            return
        default = f"Component #{self._model._next_def_id}"
        name = self._prompt_component_name(default)
        if name is None:
            return
        vertex_ids = self._grouped_vertex_ids(sel)
        edge_ids = list(sel.edges)
        face_ids = list(sel.faces)
        cmd = MakeComponentCommand(
            self._model.active_context,
            vertex_ids,
            edge_ids,
            face_ids,
            name=name,
            tag_id=self._active_tag_id,
        )
        self._command_stack.execute(cmd, self._model)
        sel.replace(instances=[cmd.created_instance.id])
        self._refresh_selection_status()
        self._refresh_breadcrumb()
        self._viewport.update()

    def _on_explode(self) -> None:
        from bermake.commands import CompositeCommand
        from bermake.commands.explode_command import ExplodeInstanceCommand
        from bermake.commands.instance_lifecycle_commands import MakeUniqueCommand

        sel = self._selection
        if not sel.instances:
            self._status_bar.set_message("Select an instance to explode.")
            return
        inst_id = next(iter(sel.instances))
        inst = next((c for c in self._model.active_context.children if c.id == inst_id), None)
        if inst is None:
            return
        if len(inst.definition.instances) > 1:
            cmd = CompositeCommand(
                name="Explode",
                children=[
                    MakeUniqueCommand(inst),
                    ExplodeInstanceCommand(self._model.active_context, inst),
                ],
            )
        else:
            cmd = ExplodeInstanceCommand(self._model.active_context, inst)
        self._command_stack.execute(cmd, self._model)
        sel.clear()
        self._refresh_selection_status()
        self._rebuild_tool_context()
        self._viewport.update()

    def _on_make_unique(self) -> None:
        from bermake.commands.instance_lifecycle_commands import MakeUniqueCommand

        sel = self._selection
        if not sel.instances:
            self._status_bar.set_message("Select an instance to make unique.")
            return
        inst_id = next(iter(sel.instances))
        inst = next((c for c in self._model.active_context.children if c.id == inst_id), None)
        if inst is None:
            return
        cmd = MakeUniqueCommand(inst)
        self._command_stack.execute(cmd, self._model)
        self._refresh_selection_status()
        self._viewport.update()

    def _on_delete_selection(self) -> None:
        from bermake.commands import CompositeCommand
        from bermake.commands.annotation_commands import DeleteAnnotationsCommand
        from bermake.commands.instance_lifecycle_commands import DeleteInstanceCommand
        from bermake.commands.scene_commands import RemoveEdgeCommand, RemoveFaceCommand

        sel = self._selection
        if sel.is_empty():
            return

        # M7.6c D6: a selected vertex is NOT deletable. Scene.remove_vertex
        # exists, but a vertex with incident edges cannot go without cascading
        # into those edges and the faces they bound, and a Delete that
        # destroys two faces because one corner was selected is the same trap
        # M7.6b closed for hidden guides. A vertex-only selection is a no-op
        # here; a mixed selection deletes the edges and faces and leaves the
        # vertices, which prune_to_live then drops if they died with their
        # geometry.
        if not (sel.edges or sel.faces or sel.instances or sel.annotations):
            return

        # Fix wave: build the annotation delete up front but do NOT execute it
        # standalone -- it must ride along as one child of whichever single
        # command below ends up on the undo stack, so a mixed selection (any
        # combination of annotations + instances/edges/faces) is always
        # exactly ONE undo-stack entry instead of two. DeleteAnnotationsCommand
        # ignores the `target` argument passed to do()/undo() (it operates on
        # the target_context captured here), so it composes safely whether the
        # enclosing command executes against self._model (instance branch) or
        # self._model.active_scene (edge/face branch below).
        ann_cmd = None
        if sel.annotations:
            ann_cmd = DeleteAnnotationsCommand(list(sel.annotations), self._model.active_context)

        # Instance delete takes priority when instances are selected.
        if sel.instances:
            children_by_id = {c.id: c for c in self._model.active_context.children}
            cmds = []
            for inst_id in list(sel.instances):
                inst = children_by_id.get(inst_id)
                if inst is None:
                    continue
                cmds.append(DeleteInstanceCommand(self._model.active_context, inst))
            if ann_cmd is not None:
                cmds.append(ann_cmd)
            if not cmds:
                sel.clear()
                return
            if len(cmds) == 1:
                cmd = cmds[0]
            else:
                cmd = CompositeCommand(name="Delete Instances")
                cmd.children = cmds
            self._command_stack.execute(cmd, self._model)
            sel.clear()
            self._refresh_selection_status()
            self._viewport.update()
            return

        composite = CompositeCommand(name="Delete Selection")
        removed_faces: set[int] = set()
        for e_id in list(sel.edges):
            try:
                self._model.active_scene.edge(e_id)
            except KeyError:
                continue
            for f_id in self._model.active_scene.edge_faces(e_id):
                if f_id is None or f_id in removed_faces:
                    continue
                fc = RemoveFaceCommand(f_id)
                fc.do(self._model.active_scene)
                composite.children.append(fc)
                removed_faces.add(f_id)
            ec = RemoveEdgeCommand(e_id)
            ec.do(self._model.active_scene)
            composite.children.append(ec)
        for f_id in list(sel.faces):
            if f_id in removed_faces:
                continue
            try:
                self._model.active_scene.face_loop(f_id)
            except KeyError:
                continue
            fc = RemoveFaceCommand(f_id)
            fc.do(self._model.active_scene)
            composite.children.append(fc)
            removed_faces.add(f_id)
        if ann_cmd is not None:
            ann_cmd.do(self._model.active_scene)
            composite.children.append(ann_cmd)
        if composite.children:
            self._command_stack.push_executed(composite, self._model.active_scene)
        sel.clear()
        self._refresh_selection_status()
        self._viewport.update()

    def _on_delete_guides(self) -> None:
        """Remove every guide in the active context, as one undoable step."""
        from bermake.commands.annotation_commands import DeleteAnnotationsCommand

        ctx = self._model.active_context
        ids = [a.id for a in ctx.annotations if a.kind in ("guide", "guide_point")]
        if not ids:
            return
        self._command_stack.execute(DeleteAnnotationsCommand(ids, ctx), self._model)
        self._viewport.update()

    # --- M7.2 selection + view commands ------------------------------------

    def _on_select_all(self) -> None:
        """Select every entity in the active editing context."""
        selection_controller.select_all(self._model, self._selection)
        self._refresh_selection_status()
        self._viewport.update()

    def _on_select_none(self) -> None:
        """Clear the selection."""
        selection_controller.select_none(self._selection)
        self._refresh_selection_status()
        self._viewport.update()

    def _on_invert_selection(self) -> None:
        from bermake.selection_ops import invert

        edges, faces, instances, vertices = invert(
            self._model, self._selection, select_vertices=self._viewport.select_vertices
        )
        self._selection.replace(edges=edges, faces=faces, instances=instances, vertices=vertices)
        self._refresh_selection_status()
        self._viewport.update()

    def _on_grow_selection(self) -> None:
        from bermake.selection_ops import grow

        edges, faces, vertices = grow(
            self._model.active_scene,
            edges=self._selection.edges,
            faces=self._selection.faces,
            vertices=self._selection.vertices,
        )
        self._selection.replace(
            edges=edges,
            faces=faces,
            vertices=vertices,
            instances=set(self._selection.instances),
            annotations=set(self._selection.annotations),
        )
        self._refresh_selection_status()
        self._viewport.update()

    def _on_shrink_selection(self) -> None:
        from bermake.selection_ops import shrink

        edges, faces, vertices = shrink(
            self._model.active_scene,
            edges=self._selection.edges,
            faces=self._selection.faces,
            vertices=self._selection.vertices,
        )
        self._selection.replace(
            edges=edges,
            faces=faces,
            vertices=vertices,
            instances=set(self._selection.instances),
            annotations=set(self._selection.annotations),
        )
        self._refresh_selection_status()
        self._viewport.update()

    def _on_select_same_material(self) -> None:
        from bermake.selection_ops import same_material

        faces = same_material(self._model.active_scene, self._selection.faces)
        if not faces:
            return
        self._selection.replace(faces=faces)
        self._refresh_selection_status()
        self._viewport.update()

    def _on_select_same_tag(self) -> None:
        from bermake.selection_ops import same_tag

        instances = same_tag(self._model, self._selection.instances)
        if not instances:
            return
        self._selection.replace(instances=instances)
        self._refresh_selection_status()
        self._viewport.update()

    def _on_toggle_select_vertices(self, checked: bool) -> None:
        """M7.6c: View > Select Vertices.

        Turning the mode off also drops any selected vertex. Same reasoning as
        _on_toggle_guides: a selection the user can no longer see, but which
        Move would still drag, is the trap that method's docstring describes
        on the other half of the path.
        """
        self._viewport.select_vertices = bool(checked)
        if not checked and self._selection.vertices:
            self._selection.remove(vertices=set(self._selection.vertices))
            self._refresh_selection_status()
        self._viewport.update()

    def _on_zoom_extents(self) -> None:
        """Frame everything visible without changing the view direction."""
        from bermake.model.model_queries import model_bounds
        from bermake.viewport.camera_framing import frame_bounds

        bounds = model_bounds(self._model)
        if bounds is None:
            return  # Nothing visible -- leave the camera exactly where it is.

        camera = self._viewport.camera
        width, height = self._viewport.width(), self._viewport.height()
        aspect = width / height if height else 1.0

        position = np.asarray(camera.position, dtype=np.float64)
        target = np.asarray(camera.target, dtype=np.float64)
        new_position, new_target = frame_bounds(
            bounds[0], bounds[1], target - position, aspect, math.radians(camera.fov_y_deg)
        )
        camera.position = new_position.astype(np.float32)
        camera.target = new_target.astype(np.float32)
        self._viewport.update()

    def _on_edit_group(self) -> None:
        """Enter the selected group/component for editing."""
        instance_ids = set(self._selection.instances)
        if len(instance_ids) != 1:
            return
        target_id = next(iter(instance_ids))
        for child in self._model.active_context.children:
            if child.id == target_id:
                self._model.enter(child)
                self._selection.clear()
                self._on_active_context_changed()
                self._refresh_selection_status()
                return

    def _on_close_group(self) -> None:
        """Step out one editing context. A no-op at the root."""
        if not self._model.active_path:
            return
        self._model.exit_one()
        self._selection.clear()
        self._on_active_context_changed()
        self._refresh_selection_status()

    def _on_paint_selection(self) -> None:
        """Apply the active material to every selected face."""
        from bermake.commands import CompositeCommand
        from bermake.commands.material_commands import PaintFaceCommand

        face_ids = sorted(self._selection.faces)
        if not face_ids:
            return
        children = [PaintFaceCommand(f_id, self._active_material_id) for f_id in face_ids]
        composite = CompositeCommand(name="Paint Selection", children=children)
        self._command_stack.execute(composite, self._model.active_scene)
        self._viewport.update()

    def _on_edit_label_text(self) -> None:
        """Retype the selected text annotation's label."""
        from PySide6.QtWidgets import QInputDialog

        from bermake.commands.annotation_commands import EditLabelTextCommand

        annotation_ids = sorted(self._selection.annotations)
        if len(annotation_ids) != 1:
            return
        context = self._model.active_context
        ann = next((a for a in context.annotations if a.id == annotation_ids[0]), None)
        if ann is None or getattr(ann, "kind", None) != "label":
            return
        text, ok = QInputDialog.getText(self, "Text", "Label:", text=ann.text)
        if not ok or not text.strip() or text.strip() == ann.text:
            return
        self._command_stack.execute(
            EditLabelTextCommand(annotation_ids[0], text.strip(), context), self._model
        )
        self._viewport.update()

    # --- Context menu (M7.2 Task 14) --------------------------------------

    def _on_context_menu_requested(self, x: int, y: int, exec_menu: bool = False) -> None:
        """Resolve a right-click at viewport pixel (x, y), select the target
        (if any), and show its context menu.

        exec_menu defaults to False so tests can drive resolution and the
        selection guard directly without entering QMenu.exec()'s modal event
        loop; the real signal connection (see __init__) always passes
        exec_menu=True.
        """
        from bermake.ui.actions import ContextTarget
        from bermake.ui.context_menu import build_context_menu, resolve_context_target

        target, entity_id = resolve_context_target(
            self._model,
            self._viewport.camera,
            x,
            y,
            self._viewport.width(),
            self._viewport.height(),
            self._doc.units,
            show_guides=self._viewport.show_guides,
        )

        # Right-clicking an unselected entity selects it first (SketchUp's
        # behaviour); right-clicking empty space leaves the selection alone --
        # Select None is right there for the explicit case. An already-selected
        # entity keeps a multi-entity selection intact rather than collapsing it.
        if target is not ContextTarget.EMPTY and entity_id is not None:
            if not self._selection_contains(target, entity_id):
                self._select_only(target, entity_id)

        enablement = {aid: a.isEnabled() for aid, a in self._actions.items()}
        try:
            # Inside the try, not before it: build_context_menu disables
            # entries as it goes, so a raise partway through would otherwise
            # leave whatever it had already disabled stuck that way.
            menu = build_context_menu(self, target, entity_id)
            if exec_menu:
                self._exec_context_menu(menu, self._viewport.mapToGlobal(QPoint(x, y)))
        finally:
            # build_context_menu disables entries on the SAME QAction objects
            # the menu bar holds (Task 13's design: context menus, the menu
            # bar, and the toolbars all share one action registry so label /
            # icon / shortcut can never drift between surfaces). Re-enabling
            # unconditionally in a finally -- not a step after menu.exec()
            # that an early return could skip -- is what stops a disabled
            # context entry from leaving the matching menu-bar item
            # permanently greyed out.
            self._restore_action_enablement(enablement)

    def _exec_context_menu(self, menu, global_pos) -> None:
        """Pop the context menu modally at `global_pos`.

        Its own method purely so a test can replace it on the instance.
        QMenu.exec cannot be monkeypatched: PySide6 dispatches the call
        straight to C++, so assigning QMenu.exec is accepted but has no
        effect, and the real modal loop runs. Under the offscreen platform
        CI uses there is nothing to dismiss it, so such a test hangs
        forever rather than failing.
        """
        menu.exec(global_pos)

    def _select_only(self, target, entity_id: int) -> None:
        """Replace the selection with the single right-clicked entity."""
        from bermake.ui.actions import ContextTarget

        self._selection.clear()
        if target is ContextTarget.FACE:
            self._selection.toggle_face(entity_id)
        elif target is ContextTarget.EDGE:
            self._selection.toggle_edge(entity_id)
        elif target is ContextTarget.INSTANCE:
            self._selection.toggle_instance(entity_id)
        elif target is ContextTarget.ANNOTATION:
            self._selection.toggle_annotation(entity_id)
        self._refresh_selection_status()
        self._viewport.update()

    def _selection_contains(self, target, entity_id: int) -> bool:
        """Whether the right-clicked entity is already part of the selection
        (dispatches to the matching Selection.contains_* predicate the same
        way _select_only dispatches to toggle_*)."""
        from bermake.ui.actions import ContextTarget

        if target is ContextTarget.FACE:
            return self._selection.contains_face(entity_id)
        if target is ContextTarget.EDGE:
            return self._selection.contains_edge(entity_id)
        if target is ContextTarget.INSTANCE:
            return self._selection.contains_instance(entity_id)
        if target is ContextTarget.ANNOTATION:
            return self._selection.contains_annotation(entity_id)
        return False

    def _restore_action_enablement(self, snapshot: dict[str, bool]) -> None:
        """Undo build_context_menu's per-entry disabling.

        Context menus reuse the same QAction objects the menu bar holds, so a
        disabled context entry would otherwise leave the matching menu-bar
        item permanently greyed out.

        This restores the state captured before the menu was built rather
        than forcing everything enabled. Forcing works only while nothing
        else in the app ever disables a registry action -- true today, but
        it would break silently the moment something reasonable (greying out
        Undo on an empty stack, Save when the document is clean) arrives,
        and any right-click would be the trigger.
        """
        for action_id, was_enabled in snapshot.items():
            self._actions[action_id].setEnabled(was_enabled)

    def _refresh_selection_status(self) -> None:
        self._status_bar.set_selection(selection_controller.selection_status_text(self._selection))
        self._sync_outliner_selection()
        self._refresh_entity_info()

    def _refresh_breadcrumb(self) -> None:
        """Task 15: rebuild the breadcrumb from model.active_path and push to status bar.

        Format: ``Model ▸ <name> ▸ …``  Built from the root definition's name
        (model.root.name) followed by each entered instance's definition name.
        When at root (active_path is empty), the breadcrumb is cleared so the
        status bar is uncluttered.
        """
        if not self._model.active_path:
            self._status_bar.set_breadcrumb("")
            return
        parts = [self._model.root.name]
        for inst in self._model.active_path:
            parts.append(inst.definition.name)
        self._status_bar.set_breadcrumb(" ▸ ".join(parts))

    def _on_active_context_changed(self) -> None:
        """Called by SelectTool after enter/exit to rebuild the tool context
        for the new active editing context and update the breadcrumb."""
        self._viewport.on_active_context_changed()
        self._rebuild_tool_context()
        self._refresh_breadcrumb()
        self._rebuild_outliner()
        self._viewport.update()

    # --- Outliner (M7.3, Task 10) ----------------------------------------

    def _rebuild_outliner(self) -> None:
        """Repopulate the tree from the model. Structural changes only."""
        from bermake.model.model_queries import outliner_rows

        self._outliner.set_rows(outliner_rows(self._model), self._model.root.name)
        self._sync_outliner_selection()

    def _sync_outliner_selection(self) -> None:
        """Push the current selection into the tree as highlight, no rebuild."""
        self._outliner.set_selected_ids(self._selection.instances)

    def _outliner_instance(self, instance_id: int):
        """The Instance for a row id, or None if it no longer resolves."""
        path = instance_path(self._model, instance_id)
        return path[-1] if path else None

    def _on_outliner_clicked(self, instance_id: int) -> None:
        """Select the row's instance, re-rooting the active path to its parents.

        Selection is only meaningful inside the active context -- every tool,
        the dim pass and pick_selectable assume it -- so selecting a nested row
        without moving the context would leave a highlighted object no tool
        could touch. The breadcrumb and dim pass both update, so the context
        change is visible rather than silent.
        """
        path = instance_path(self._model, instance_id)
        if path is None:
            return
        parents = list(path[:-1])
        if [i.id for i in self._model.active_path] != [i.id for i in parents]:
            self._model.active_path = parents
            self._on_active_context_changed()
        self._selection.replace(instances=[instance_id])
        self._refresh_selection_status()
        self._viewport.update()

    def _on_outliner_activated(self, instance_id: int) -> None:
        """Double-click: enter that instance, like double-clicking it in 3D."""
        path = instance_path(self._model, instance_id)
        if path is None:
            return
        self._model.active_path = list(path)
        self._selection.clear()
        # _on_active_context_changed() already rebuilds the outliner -- a
        # second explicit rebuild here just repopulated the same tree twice.
        self._on_active_context_changed()
        self._refresh_selection_status()

    def _on_outliner_hide_toggled(self, instance_id: int, hidden: bool) -> None:
        from bermake.commands.visibility_commands import HideInstancesCommand

        instance = self._outliner_instance(instance_id)
        if instance is None:
            return
        self._command_stack.execute(HideInstancesCommand([instance], hidden), self._model)
        # A hide command does not touch the selection, so
        # _refresh_selection_status's usual selection-changed trigger never
        # fires -- without this, Entity Info's Hidden checkbox goes stale
        # the moment the toggled row is (or overlaps) the current selection.
        self._refresh_entity_info()
        self._viewport.update()

    def _on_outliner_rename(self, instance_id: int, new_name: str) -> None:
        from bermake.commands.naming_commands import RenameInstanceCommand

        instance = self._outliner_instance(instance_id)
        if instance is None:
            return
        if instance.name == new_name.strip():
            # No command runs, so nothing fires _rebuild_outliner -- without
            # this, the row keeps whatever raw text the user typed (padding
            # whitespace included) instead of the canonical stored name.
            self._rebuild_outliner()
            return
        self._command_stack.execute(RenameInstanceCommand(instance, new_name), self._model)

    # --- Hide / Unhide (M7.3, Task 14) ------------------------------------

    def _set_hidden_on_selection(self, hidden: bool) -> None:
        from bermake.commands.visibility_commands import HideInstancesCommand

        instances = selection_controller.selected_instances(self._model, self._selection)
        if not instances:
            self._status_bar.set_message("Select objects to hide or unhide.")
            return
        self._command_stack.execute(HideInstancesCommand(instances, hidden), self._model)
        # Same reasoning as _on_outliner_hide_toggled: Edit > Hide/Unhide acts
        # on the current selection without changing it, so nothing else
        # refreshes Entity Info's Hidden checkbox.
        self._refresh_entity_info()
        self._viewport.update()

    def _on_hide(self) -> None:
        self._set_hidden_on_selection(True)

    def _on_unhide(self) -> None:
        self._set_hidden_on_selection(False)

    def _on_unhide_all(self) -> None:
        """Reveal everything hidden in the active context.

        Without this there is no way back for an object you cannot
        right-click, other than the Outliner -- which the user may have closed.
        """
        from bermake.commands.visibility_commands import HideInstancesCommand

        hidden = [inst for inst in self._model.active_context.children if inst.hidden]
        if not hidden:
            self._status_bar.set_message("Nothing is hidden here.")
            return
        self._command_stack.execute(HideInstancesCommand(hidden, False), self._model)
        # Same reasoning as _on_outliner_hide_toggled: a currently-selected
        # instance may be among those just revealed, and nothing else would
        # refresh Entity Info's Hidden checkbox for it.
        self._refresh_entity_info()
        self._viewport.update()

    # --- Entity Info (M7.3, Task 13) --------------------------------------

    def _refresh_entity_info(self) -> None:
        from bermake.model.entity_info import entity_summary

        self._entity_info_page.refresh(
            entity_summary(self._model, self._selection),
            self._doc.units,
            self._model.tags,
            self._model.materials,
        )
        # M7.5b Task 10 fix round 1: the Properties panel's texture-position
        # fields are "the primary placement mechanism ... for the selected
        # face and side" (design line 172), so they must track the selection
        # the same way Entity Info does -- reusing this method's ~15 call
        # sites (including the two that deliberately bypass
        # _refresh_selection_status, and _on_after_undo_redo, so an undo that
        # changes a placement is reflected too) rather than adding a second,
        # parallel refresh hook.
        self._properties_dock.set_selected_face(self._single_selected_face_id())

    def _single_selected_face_id(self) -> int | None:
        """The one selected face, or None -- for the texture-position group,
        which only makes sense for a single, unambiguous target (spec: "the
        selected face", singular). Selection carries no notion of "just one
        face and nothing else" itself, so this mirrors entity_summary's own
        cross-type-Mixed check rather than adding that to Selection, which
        nothing else needs.
        """
        sel = self._selection
        if len(sel.faces) == 1 and not sel.edges and not sel.instances and not sel.annotations:
            return next(iter(sel.faces))
        return None

    def _on_entity_rename(self, new_name: str) -> None:
        from bermake.commands.naming_commands import RenameInstanceCommand

        instances = selection_controller.selected_instances(self._model, self._selection)
        if len(instances) != 1 or instances[0].name == new_name.strip():
            # No command runs on a no-op commit, so nothing repopulates the
            # page -- without this, the Name field keeps whatever raw text
            # the user typed (padding whitespace included).
            self._refresh_entity_info()
            return
        self._command_stack.execute(RenameInstanceCommand(instances[0], new_name), self._model)

    def _on_entity_hidden(self, hidden: bool) -> None:
        from bermake.commands.visibility_commands import HideInstancesCommand

        instances = selection_controller.selected_instances(self._model, self._selection)
        if not instances:
            return
        self._command_stack.execute(HideInstancesCommand(instances, hidden), self._model)
        self._viewport.update()

    def _on_entity_material(self, material_id: int) -> None:
        from bermake.commands import CompositeCommand
        from bermake.commands.material_commands import PaintFaceCommand

        face_ids = sorted(self._selection.faces)
        if not face_ids:
            return
        # PaintFaceCommand targets the SCENE, not the model, and paints one
        # face -- so a multi-face selection composes into a single undo step.
        composite = CompositeCommand(
            name="Paint", children=[PaintFaceCommand(f_id, material_id) for f_id in face_ids]
        )
        self._command_stack.execute(composite, self._model.active_scene)
        self._viewport.update()

    # --- texture placement (M7.5b) ----------------------------------------

    def _on_placement_committed(self) -> None:
        """A face's texture placement just changed through the command stack.

        Repaint only. The panel resynced its own fields before it emitted, and
        the stack's change listeners already marked the document dirty, so the
        one thing nothing else does is put the moved image on screen.
        """
        self._viewport.update()

    def _on_reset_uvs_requested(self) -> None:
        """Properties' "Reset to projection" button (Task 8, M7.5c).

        Clears the stored UVs for the selected face on the placement group's
        current side, never the placement (spec D11: projection is the base,
        placement is a separate layer on top). Re-syncs the panel afterward
        so the button's own enabled state -- the stored-UV indicator, spec
        D12 -- reflects the clear, then repaints through the same path
        _on_placement_committed uses rather than a second mechanism.
        """
        from bermake.commands.material_commands import ResetFaceUvsCommand

        face_id = self._single_selected_face_id()
        if face_id is None:
            return
        side = self._properties_dock.target_side
        self._command_stack.execute(ResetFaceUvsCommand(face_id, side), self._model.active_scene)
        self._refresh_entity_info()
        self._on_placement_committed()

    def _on_placement_drag_committed(self) -> None:
        """Task 12's Shift-drag committed a placement (ToolContext hook).

        The drag also has to repopulate the Properties fields, which the
        numeric path does for itself. The repaint is asked for explicitly here
        even though a mouse release repaints anyway: relying on that made the
        drag work and the numeric path look broken, which is precisely the
        asymmetry this call removes.
        """
        self._refresh_entity_info()
        self._on_placement_committed()

    def _on_after_undo_redo(self) -> None:
        """Called by CommandStack listeners after every successful undo or redo."""
        # revalidate_active_path() must run FIRST: _prune_selection() reads
        # self._model.active_context, which is only meaningful once the active
        # path has been walked back to the nearest still-reachable instance.
        self._model.revalidate_active_path()
        self._prune_selection()
        self._refresh_selection_status()
        self._rebuild_tool_context()
        self._refresh_breadcrumb()
        self._scenes_page.refresh()
        # Material add / edit / delete became undoable in M7.5a Task 9, so the
        # swatch grid and editor can now go stale behind an undo the same way
        # the Scenes list can.
        self._materials_page.refresh()
        # M7.5b Task 9: undoing a texture import removes the Texture record
        # (AddTextureCommand.undo) -- reconcile the renderer's GL cache so it
        # does not keep that upload alive until the whole document closes.
        self._viewport.scene_renderer.evict_stale_textures(self._model.textures)
        # Tag colour became undoable in M7.5a Task 11 -- same staleness risk.
        self._tags_page.refresh()

    def _prune_selection(self) -> None:
        """Keep only selected entities still live in the active context (#46)."""
        selection_controller.prune_to_live(self._model, self._selection)

    def _on_set_face_style(self, style: FaceStyle) -> None:
        self._render_style.face_style = style
        self._face_style_actions[style].setChecked(True)
        self._viewport.set_render_style(self._render_style)

    def _on_toggle_xray(self, checked: bool) -> None:
        self._render_style.xray = bool(checked)
        self._viewport.set_render_style(self._render_style)

    def _on_toggle_color_by_tag(self, checked: bool) -> None:
        self._render_style.color_by_tag = bool(checked)
        self._viewport.set_render_style(self._render_style)

    def _on_toggle_guides(self, checked: bool) -> None:
        """Hide or show every guide without deleting any.

        Hiding also deselects any guide that was selected. Task 7 made a
        hidden guide unpickable and unsnappable on the reasoning that a
        hidden thing the user can still act on is a trap; a selected guide
        left selected is that same trap on the other half of the path, and
        the sharpest form of it -- Delete then destroys an annotation with
        nothing on screen to say what went.
        """
        self._viewport.show_guides = bool(checked)
        if not checked:
            hidden = {
                ann.id
                for ann in self._model.active_context.annotations
                if getattr(ann, "kind", None) in _GUIDE_KINDS
                and ann.id in self._selection.annotations
            }
            if hidden:
                self._selection.remove(annotations=hidden)
                self._refresh_selection_status()
        self._viewport.update()

    # --- Toolbars (M7.2, Task 11) -----------------------------------------

    def _on_reset_toolbars(self) -> None:
        """Forget the saved layout and reapply the default arrangement live."""
        reset_window_state(self._settings)
        self.restoreState(self._default_window_state, WINDOW_STATE_VERSION)

    # --- Properties tabs (M7.3, Task 11) ----------------------------------

    def _show_properties_tab(self, tab_id: str) -> None:
        """Reveal the panel (it may be closed) and focus one tab."""
        self._properties_dock.show()
        self._properties_dock.raise_()
        self._properties_dock.show_tab(tab_id)

    def _on_show_material_tab(self) -> None:
        self._show_properties_tab("material")

    def _on_show_tags_tab(self) -> None:
        self._show_properties_tab("tags")

    def _on_show_scenes_tab(self) -> None:
        self._show_properties_tab("scenes")

    # --- Scenes (M7e) ----------------------------------------------------

    def _sync_render_style_ui(self) -> None:
        """Reflect self._render_style in the View menu + viewport (no signal echo)."""
        for st, action in self._face_style_actions.items():
            action.setChecked(st == self._render_style.face_style)
        self._xray_action.blockSignals(True)
        self._xray_action.setChecked(self._render_style.xray)
        self._xray_action.blockSignals(False)
        self._color_by_tag_action.blockSignals(True)
        self._color_by_tag_action.setChecked(self._render_style.color_by_tag)
        self._color_by_tag_action.blockSignals(False)
        self._viewport.set_render_style(self._render_style)

    def _on_create_view(self) -> None:
        name = f"Scene {len(self._model.views.views()) + 1}"
        view = capture_view(
            self._model.views.next_id,
            name,
            self._viewport.camera,
            self._model.tags,
            self._render_style,
        )
        self._command_stack.execute(CreateViewCommand(view), self._model)
        self._scenes_page.refresh(select_id=view.id)

    def _on_update_view(self, view_id: int) -> None:
        old = self._model.views.get(int(view_id))
        if old is None:
            return
        new_view = capture_view(
            old.id, old.name, self._viewport.camera, self._model.tags, self._render_style
        )
        self._command_stack.execute(UpdateViewCommand(old.id, new_view), self._model)
        self._scenes_page.refresh(select_id=old.id)

    def _on_delete_view(self, view_id: int) -> None:
        if self._model.views.get(int(view_id)) is None:
            return
        self._command_stack.execute(DeleteViewCommand(int(view_id)), self._model)
        self._scenes_page.refresh()

    def _on_rename_view(self, view_id: int, name: str) -> None:
        name = str(name).strip()
        old = self._model.views.get(int(view_id))
        if old is None or not name or name == old.name:
            self._scenes_page.refresh(select_id=int(view_id))
            return
        self._command_stack.execute(RenameViewCommand(int(view_id), name), self._model)
        self._scenes_page.refresh(select_id=int(view_id))

    def _on_reorder_view(self, view_id: int, direction: int) -> None:
        vid = int(view_id)
        index = self._model.views.index_of(vid)
        target = index + (1 if int(direction) > 0 else -1)
        if index < 0 or target < 0 or target >= len(self._model.views.views()):
            return
        self._command_stack.execute(ReorderViewCommand(vid, int(direction)), self._model)
        self._scenes_page.refresh(select_id=vid)

    def _on_recall_view(self, view_id: int) -> None:
        view = self._model.views.get(int(view_id))
        if view is None:
            return
        apply_tags_and_style(view, self._model.tags, self._render_style)
        self._sync_render_style_ui()
        self._tags_page.refresh()
        from_state = CameraState.from_camera(self._viewport.camera)
        self._view_animator.start(from_state, view.camera)

    def _on_undo(self) -> None:
        if self._command_stack.undo():
            self._refresh_status_text()
            self._viewport.update()

    def _on_redo(self) -> None:
        if self._command_stack.redo():
            self._refresh_status_text()
            self._viewport.update()

    def _set_units_metric(self, unit: str) -> None:
        self._doc.set_metric(unit)
        self._refresh_status_text()
        self._refresh_tool_options()
        self._viewport.update()
        self._on_document_changed()

    def _set_units_imperial(self) -> None:
        self._doc.set_imperial()
        self._refresh_status_text()
        self._refresh_tool_options()
        self._viewport.update()
        self._on_document_changed()

    def _set_environment(self, key: str) -> None:
        self._doc.set_environment(env_module.PRESETS[key])
        self._viewport.set_environment(self._doc.environment)
        self._on_document_changed()

    def _sync_environment_ui(self) -> None:
        """Reflect self._doc.environment in the View menu + viewport.

        preset_key returns None for a value no preset matches, which leaves every
        entry unchecked rather than lying about which one is active.
        """
        active = env_module.preset_key(self._doc.environment)
        for key, action in self._environment_actions.items():
            action.setChecked(key == active)
        self._viewport.set_environment(self._doc.environment)

    def _set_theme(self, name: str) -> None:
        from PySide6.QtWidgets import QApplication

        from bermake.ui.theme import apply_theme, theme_for_name

        app = QApplication.instance()
        if app is not None:
            apply_theme(app, theme_for_name(name))
        preferences.write_theme(self._settings, name)

    def changeEvent(self, event):
        """Rebuild icons when the palette changes (#101).

        The icons are monochrome SVGs recoloured from the palette's text colour
        at load time and cached on (stem, colour), so they already serve both
        themes. What was missing was the refresh: nothing ever cleared the cache
        or re-requested them, so a theme change left every toolbar and menu icon
        in the previous theme's ink until restart.
        """
        from PySide6.QtCore import QEvent

        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange:
            self._rebuild_all_icons()

    def _rebuild_all_icons(self) -> None:
        """Re-request every icon at the current palette's text colour.

        Every source, and that is the whole point. The action icons come from the
        ACTIONS registry. The properties dock's tab strip is built once and holds
        its own buttons, so it needs re-tinting. The Outliner sets row icons
        during set_rows, so it needs rebuilding. Neither is reachable from an
        ActionSpec, and a refresh that walked only the registry would leave every
        tab and every Outliner row in the old ink while looking complete from the
        toolbar.

        Read the colour from self.palette() at call time, never from a value
        captured earlier: the cache is keyed on the colour, so re-requesting at
        the old one would re-cache the stale icon and make the switch appear to
        do nothing.
        """
        from bermake.ui import actions as actions_module
        from bermake.ui import icons as icons_module

        icons_module.clear_icon_cache()
        color = self.palette().windowText().color()
        for spec in actions_module.ACTIONS:
            if spec.icon is None:
                continue
            action = self._actions.get(spec.id)
            if action is None:
                continue
            try:
                action.setIcon(icons_module.icon(spec.icon, color))
            except KeyError:
                # Same reasoning as ui_builder.build_action: a damaged install
                # must not take the window down over one missing glyph.
                continue
        self._properties_dock.refresh_icons(color)
        self._rebuild_outliner()

    # --- File I/O --------------------------------------------------------

    def _on_document_changed(self) -> None:
        self._doc_controller.mark_dirty()
        self._update_window_title()

    def _update_window_title(self) -> None:
        self.setWindowTitle(self._doc_controller.display_title())

    def _on_export_obj(self) -> None:
        path = self._prompt_save_path("OBJ files (*.obj)", "Export OBJ")
        if not path:
            return
        path = str(path)
        if not path.endswith(".obj"):
            path += ".obj"
        try:
            export_obj(path, self._model)
        except OSError as e:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.critical(self, "Export failed", str(e))
            return
        self._status_bar.set_message(f"Exported {Path(path).name}")

    def _on_import_obj(self) -> None:
        path = self._prompt_open_path("OBJ files (*.obj)", "Import OBJ")
        if not path:
            return
        try:
            doc = read_obj_document(path)
        except (BermakeIOError, OSError) as e:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.critical(self, "Import failed", str(e))
            return
        texture_bytes = read_obj_texture_bytes(path, doc)
        from bermake.commands.obj_commands import ImportObjCommand

        cmd = ImportObjCommand(
            doc,
            self._model.active_context,
            texture_bytes=texture_bytes,
            decoder=_texture_decoder,
        )
        self._command_stack.execute(cmd, self._model)
        s = cmd.summary
        msg = f"Imported {s.faces_imported} faces"
        if s.objects:
            msg += f" in {s.objects} object(s)"
        if s.faces_skipped:
            msg += f" (skipped {s.faces_skipped} faces)"
        self._status_bar.set_message(msg)
        self._refresh_breadcrumb()
        self._viewport.update()

    def _on_export_gltf(self) -> None:
        path = self._prompt_save_path("glTF Binary (*.glb);;glTF (*.gltf)", "Export glTF")
        if not path:
            return
        path = str(path)
        if not (path.endswith(".glb") or path.endswith(".gltf")):
            path += ".glb"
        try:
            export_gltf(self._model, path)
        except OSError as e:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.critical(self, "Export failed", str(e))
            return
        self._status_bar.set_message(f"Exported {Path(path).name}")

    def _on_import_gltf(self) -> None:
        path = self._prompt_open_path("glTF (*.glb *.gltf)", "Import glTF")
        if not path:
            return
        try:
            scene = read_gltf_scene(path)
        except (BermakeIOError, OSError) as e:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.critical(self, "Import failed", str(e))
            return
        texture_bytes = read_gltf_texture_bytes(path, scene)
        from bermake.commands.gltf_commands import ImportGltfCommand

        cmd = ImportGltfCommand(
            scene,
            self._model.active_context,
            root_name=Path(path).stem,
            texture_bytes=texture_bytes,
            decoder=_texture_decoder,
        )
        self._command_stack.execute(cmd, self._model)
        s = cmd.summary
        msg = f"Imported {s.faces_imported} faces in {s.nodes} object(s)"
        if s.faces_skipped:
            msg += f" (skipped {s.faces_skipped} faces)"
        if s.faces_without_uvs:
            msg += f" ({s.faces_without_uvs} without UVs)"
        if s.images_skipped:
            msg += f" ({s.images_skipped} image(s) skipped)"
        self._status_bar.set_message(msg)
        self._refresh_breadcrumb()
        self._viewport.update()

    def _prompt_save_path(
        self, file_filter: str = "Bermake files (*.berm)", title: str = "Save As"
    ) -> str | None:
        """Return a chosen save path (or None). Overridable for testing."""
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(self, title, "", file_filter)
        return path or None

    def _capture_thumbnail(self) -> bytes | None:
        """Grab the viewport framebuffer and encode a capped-size PNG preview.

        QOpenGLWidget.grabFramebuffer() manages its own GL context (it calls
        makeCurrent() internally), so unlike the raw GL calls in
        scene_renderer.py -- which only run safely inside initializeGL/
        resizeGL/paintGL -- this is safe to call from an ordinary slot such as
        the save path. If the viewport has never been painted (no GL context
        created yet, e.g. a headless test that never shows the window),
        grabFramebuffer() returns a null QImage instead of raising; that null
        image is treated as "no thumbnail" rather than guessed at.

        A preview image is never worth losing a document over, so a missing
        GL context degrades to None instead of failing the save. PNG encoding
        failure is reported through QImage.save()'s boolean return, not an
        exception, so that path is handled by checking the return value
        rather than by catching anything.
        """
        try:
            image = self._viewport.grabFramebuffer()
        except (RuntimeError, OSError):
            return None
        if image.isNull() or image.width() <= 0 or image.height() <= 0:
            return None
        long_edge = max(image.width(), image.height())
        if long_edge > _THUMBNAIL_MAX_EDGE:
            if image.width() >= image.height():
                image = image.scaledToWidth(
                    _THUMBNAIL_MAX_EDGE, Qt.TransformationMode.SmoothTransformation
                )
            else:
                image = image.scaledToHeight(
                    _THUMBNAIL_MAX_EDGE, Qt.TransformationMode.SmoothTransformation
                )
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        saved = image.save(buffer, "PNG")
        data = bytes(buffer.data()) if saved else None
        buffer.close()
        return data or None

    def _save_to(self, path) -> bool:
        path = str(path)
        if not path.endswith(".berm"):
            path += ".berm"
        try:
            save_document(
                path,
                self._model,
                self._viewport.camera,
                self._doc,
                self._render_style,
                thumbnail=self._capture_thumbnail(),
            )
        except OSError as e:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.critical(self, "Save failed", str(e))
            return False
        self._doc_controller.set_path(path)
        self._doc_controller.mark_clean()
        self._update_window_title()
        self._status_bar.set_message(f"Saved {Path(path).name}")
        return True

    def _on_file_save(self) -> bool:
        if self._doc_controller.current_path is None:
            return self._on_file_save_as()
        return self._save_to(self._doc_controller.current_path)

    def _on_file_save_as(self) -> bool:
        path = self._prompt_save_path()
        if not path:
            return False
        return self._save_to(path)

    def _prompt_discard(self) -> str:
        """Return 'save' | 'discard' | 'cancel'. Overridable for testing."""
        from PySide6.QtWidgets import QMessageBox

        name = (
            self._doc_controller.current_path.name
            if self._doc_controller.current_path
            else "Untitled"
        )
        box = QMessageBox(self)
        box.setWindowTitle("Unsaved changes")
        box.setText(f"Save changes to {name}?")
        box.setStandardButtons(
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel
        )
        box.setDefaultButton(QMessageBox.StandardButton.Save)
        choice = box.exec()
        if choice == QMessageBox.StandardButton.Save:
            return "save"
        if choice == QMessageBox.StandardButton.Discard:
            return "discard"
        return "cancel"

    def _confirm_discard_if_dirty(self) -> bool:
        """True if it's safe to proceed (discard a New/Open/close)."""
        if not self._doc_controller.dirty:
            return True
        choice = self._prompt_discard()
        if choice == "save":
            return self._on_file_save()
        if choice == "discard":
            return True
        return False

    def closeEvent(self, event):
        # Persist before the unsaved-changes guard runs, so the layout is
        # remembered even if the user then cancels the close.
        save_window_state(self, self._settings)
        if self._confirm_discard_if_dirty():
            event.accept()
        else:
            event.ignore()

    def _reset_document(self, model, camera_state, units, style, path, *, environment) -> None:
        """Adopt a (model, camera, units, render style, environment) into the live window."""
        from dataclasses import replace

        # M7.5b Task 9: the incoming Model's TextureLibrary restarts id
        # numbering from 1, same as every other document's -- so without
        # this, the renderer's GL texture cache (keyed by that same int)
        # would hand the new document's material the PREVIOUS document's
        # texture the first time it is drawn, not merely leak it.
        self._viewport.scene_renderer.release_all_textures()
        self._model.load_from(model)
        self._materials_page.set_library(self._model.materials)
        self._tags_page.set_library(self._model.tags)
        self._scenes_page.set_library(self._model.views)
        camera_state.apply_to(self._viewport.camera)
        self._view_animator.cancel()
        self._render_style = replace(style)
        self._sync_render_style_ui()
        self._doc.set_units(units)
        self._doc.set_environment(environment)
        self._sync_environment_ui()
        self._active_material_id = self._model.materials.DEFAULT_ID
        self._active_tag_id = self._model.tags.UNTAGGED_ID
        self._selection.clear()
        self._command_stack.clear()
        self._doc_controller.set_path(path)
        self._doc_controller.mark_clean()
        self._rebuild_tool_context()
        self._refresh_breadcrumb()
        # CommandStack.clear() fires no listeners (by design -- it is not an
        # undoable edit), so without this the Outliner keeps showing the
        # PREVIOUS document's hierarchy under the previous root label (spec
        # 1.7: "Document New / Open -> rebuild and rebind libraries").
        self._rebuild_outliner()
        self._refresh_status_text()
        self._update_window_title()
        self._viewport.update()

    def _on_file_new(self) -> None:
        if not self._confirm_discard_if_dirty():
            return
        from bermake.templates import template_for_key
        from bermake.viewport.camera import Camera

        template = template_for_key(preferences.read_default_template(self._settings))
        self._reset_document(
            Model(),
            CameraState.from_camera(Camera()),
            template.units,
            RenderStyle(),
            None,
            environment=template.environment,
        )

    def _prompt_open_path(
        self, file_filter: str = "Bermake files (*.berm);;All files (*)", title: str = "Open"
    ) -> str | None:
        """Return a chosen open path (or None). Overridable for testing."""
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(self, title, "", file_filter)
        return path or None

    def _on_file_open(self) -> None:
        if not self._confirm_discard_if_dirty():
            return
        path = self._prompt_open_path()
        if not path:
            return
        try:
            loaded = load_document(path)
        except (BermakeIOError, OSError) as e:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.critical(self, "Open failed", str(e))
            return
        self._reset_document(
            loaded.model,
            loaded.camera_state,
            loaded.units,
            loaded.style,
            path,
            environment=loaded.environment,
        )

    def _on_show_welcome(self) -> None:
        """Show the welcome dialog on demand (Help > Welcome to Bermake).

        The same entry point app.py uses at startup, so the do-not-show
        checkbox is recoverable without editing a settings file.
        """
        from bermake.ui.welcome_dialog import WelcomeDialog, apply_template
        from bermake.viewport.camera import Camera

        dialog = WelcomeDialog(self._settings, self)
        if dialog.exec() != WelcomeDialog.DialogCode.Accepted:
            return
        preferences.write_show_welcome(self._settings, dialog.show_on_startup)
        preferences.write_default_template(self._settings, dialog.selected_template.key)
        if dialog.wants_open:
            self._on_file_open()
            return
        if not self._confirm_discard_if_dirty():
            return
        self._reset_document(
            Model(),
            CameraState.from_camera(Camera()),
            dialog.selected_template.units,
            RenderStyle(),
            None,
            environment=dialog.selected_template.environment,
        )
        apply_template(self._doc, dialog.selected_template, dialog.selected_unit)
        self._sync_environment_ui()
        self._refresh_status_text()

    def show_welcome_dialog(self) -> None:
        """Public entry point for app.py's startup call."""
        self._on_show_welcome()
