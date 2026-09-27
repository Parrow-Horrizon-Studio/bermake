"""The M7.3 right panel: an Outliner over an icon-tabbed Properties editor.

One dock, split vertically. The Outliner and every Properties page are
INJECTED (set_outliner / set_page) rather than constructed here, so this shell
is testable on its own and the pages that replace three former docks can land
one task at a time.

An un-injected tab shows a placeholder, which is also what a tab with nothing
to show looks like -- so the empty state needed no extra machinery.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from bermake.commands.material_commands import SetFacePlacementCommand
from bermake.scene.scene import Side, TexturePlacement
from bermake.ui.icons import icon
from bermake.ui.panels import PROPERTIES_TABS

_TAB_ICON_SIZE = 20


class PropertiesDock(QDockWidget):
    """Outliner (top) + icon-tabbed Properties editor (bottom)."""

    tab_changed = Signal(str)

    # M7.5b final review, item 1: a placement edit made HERE just landed on the
    # command stack. Neither of the stack's own change listeners
    # (_rebuild_outliner, _on_document_changed) repaints, and ViewportWidget
    # has no subscription of its own, so without this the spin boxes -- "the primary
    # mechanism" of design line 172 -- changed the scene and the screen kept
    # showing the old texture until some unrelated event happened to repaint.
    # Same shape as MaterialsPage/TagsPage.library_changed, which MainWindow
    # already wires to self._viewport.update for exactly this hazard.
    placement_changed = Signal()

    # Task 8 (M7.5c): the button only asks -- MainWindow owns the command
    # stack and the selection, so it is the one that knows which face is
    # targeted and builds the ResetFaceUvsCommand, the same division as every
    # other *_requested signal on this dock's sibling pages (EntityInfoPage's
    # rename_requested, etc).
    reset_uvs_requested = Signal()

    # QMainWindow.saveState() silently drops docks without an object name.
    # This is a persistence key: never rename it, or saved layouts silently
    # lose this dock's state.
    OBJECT_NAME = "properties_dock"

    def __init__(self, parent=None, *, command_stack=None, model=None) -> None:
        super().__init__("Properties", parent)
        self.setObjectName(self.OBJECT_NAME)

        # Injected the same way MaterialsPage takes them (Task 8/M7.5a): both
        # are None in the plain PropertiesDock() unit tests, and every
        # placement-mutating method below is a documented no-op in that case.
        self._command_stack = command_stack
        self._model = model
        self._target_face_id: int | None = None
        self._target_side: Side = Side.FRONT
        # Guards set_placement_target()'s setValue() calls from looping back
        # into the spin boxes' valueChanged handlers below, the same way
        # MaterialsPage._syncing guards _sync_editor() -- without it,
        # populating the fields from a face's stored placement would issue a
        # command for the value just read FROM the scene.
        self._syncing = False

        self._buttons: dict[str, QToolButton] = {}
        self._page_index: dict[str, int] = {}
        self._outliner: QWidget | None = None

        self._splitter = QSplitter(Qt.Orientation.Vertical, self)

        self._outliner_host = QWidget(self._splitter)
        self._outliner_layout = QVBoxLayout(self._outliner_host)
        self._outliner_layout.setContentsMargins(0, 0, 0, 0)
        self._outliner_placeholder: QWidget | None = QLabel("Outliner", self._outliner_host)
        self._outliner_layout.addWidget(self._outliner_placeholder)
        self._splitter.addWidget(self._outliner_host)

        properties = QWidget(self._splitter)
        properties_layout = QVBoxLayout(properties)
        properties_layout.setContentsMargins(0, 0, 0, 0)
        properties_layout.setSpacing(0)

        strip = QWidget(properties)
        strip_layout = QHBoxLayout(strip)
        strip_layout.setContentsMargins(2, 2, 2, 2)
        strip_layout.setSpacing(2)
        # Exclusive without being a QActionGroup: these are view switches, not
        # commands, so they never belong in the action registry.
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._stack = QStackedWidget(properties)

        palette_color = self.palette().windowText().color()
        for spec in PROPERTIES_TABS:
            button = QToolButton(strip)
            button.setCheckable(True)
            button.setAutoRaise(True)
            button.setToolTip(spec.title)
            button.setAccessibleName(spec.title)
            button.setIconSize(QSize(_TAB_ICON_SIZE, _TAB_ICON_SIZE))
            button.setIcon(icon(spec.icon, palette_color))
            button.clicked.connect(lambda _checked=False, tid=spec.id: self.show_tab(tid))
            strip_layout.addWidget(button)
            self._group.addButton(button)
            self._buttons[spec.id] = button

            placeholder = QLabel(spec.title, self._stack)
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._page_index[spec.id] = self._stack.addWidget(placeholder)

        strip_layout.addStretch(1)
        properties_layout.addWidget(strip, stretch=0)
        properties_layout.addWidget(self._stack, stretch=1)

        # Texture position (M7.5b Task 10): per-face-side placement, kept
        # OUTSIDE the tab stack because it applies to whichever face+side is
        # targeted regardless of which tab is showing, exactly like the
        # Material tab it most often accompanies. Disabled -- never hidden,
        # matching EntityInfoPage's rule -- until set_placement_target() has
        # a face to edit.
        self._placement_group = self._build_placement_group(properties)
        self._placement_group.setEnabled(False)
        properties_layout.addWidget(self._placement_group, stretch=0)

        self._splitter.addWidget(properties)

        self.setWidget(self._splitter)

        self._current = PROPERTIES_TABS[0].id
        self._buttons[self._current].setChecked(True)
        self._stack.setCurrentIndex(self._page_index[self._current])

    # --- tabs ------------------------------------------------------------
    def tab_ids(self) -> tuple[str, ...]:
        return tuple(self._buttons)

    def tab_button(self, tab_id: str) -> QToolButton:
        return self._buttons[tab_id]

    def refresh_icons(self, color) -> None:
        """Re-tint the tab strip after a palette change (#101).

        The strip is built once in __init__, so nothing else would ever
        re-request these icons and every tab would keep the old theme's ink.
        """
        for spec in PROPERTIES_TABS:
            button = self._buttons.get(spec.id)
            if button is not None:
                button.setIcon(icon(spec.icon, color))

    @property
    def current_tab_id(self) -> str:
        return self._current

    def current_page(self) -> QWidget:
        return self._stack.currentWidget()

    def show_tab(self, tab_id: str) -> None:
        """Switch to `tab_id`. An unknown id is ignored, not an error: a stale
        saved id or a typo must not take the panel down."""
        if tab_id not in self._page_index:
            return
        self._current = tab_id
        self._buttons[tab_id].setChecked(True)
        self._stack.setCurrentIndex(self._page_index[tab_id])
        self.tab_changed.emit(tab_id)

    def set_page(self, tab_id: str, widget: QWidget) -> None:
        """Replace a tab's placeholder (or prior page) with `widget`.

        Safe to call more than once for the same id: the widget currently
        installed there -- placeholder or a previous real page -- is removed
        and scheduled for deletion first.
        """
        if tab_id not in self._page_index:
            raise KeyError(f"unknown properties tab {tab_id!r}")
        old_index = self._page_index[tab_id]
        old = self._stack.widget(old_index)
        self._stack.removeWidget(old)
        old.deleteLater()
        self._page_index[tab_id] = self._stack.insertWidget(old_index, widget)
        if self._current == tab_id:
            self._stack.setCurrentIndex(self._page_index[tab_id])

    # --- texture placement (M7.5b Task 10) --------------------------------
    def _build_placement_group(self, parent: QWidget) -> QGroupBox:
        group = QGroupBox("Texture position", parent)
        form = QFormLayout(group)

        # There is no selected-side concept anywhere in the app (Selection
        # carries only edges/faces/instances/annotations) and adding one
        # would touch a core type nothing else needs -- so the group owns
        # which side it edits itself, defaulting to Front. Populated with
        # addItem BEFORE the signal is connected: an empty-to-first-item
        # QComboBox fires currentIndexChanged on the first addItem, and
        # connecting first would run _on_side_changed while the placement
        # spin boxes below do not exist yet.
        self._side_combo = QComboBox(group)
        self._side_combo.addItem("Front", Side.FRONT)
        self._side_combo.addItem("Back", Side.BACK)
        form.addRow("Side", self._side_combo)

        self._offset_u_spin = self._make_placement_spin(group, -1000.0, 1000.0, 4)
        form.addRow("Offset U", self._offset_u_spin)

        self._offset_v_spin = self._make_placement_spin(group, -1000.0, 1000.0, 4)
        form.addRow("Offset V", self._offset_v_spin)

        self._scale_spin = self._make_placement_spin(group, 0.001, 1000.0, 4)
        form.addRow("Scale", self._scale_spin)

        self._rotation_spin = self._make_placement_spin(group, -3600.0, 3600.0, 2)
        self._rotation_spin.setSuffix("°")
        form.addRow("Rotation", self._rotation_spin)

        # Task 8 (M7.5c): enabled state IS the stored-UV indicator (spec
        # D12) -- disabled means the face follows the projection, enabled
        # means it has its own coordinates. Kept inside the same group as
        # the placement fields it accompanies, not a tab, for the same
        # reason the group itself lives outside the tab stack.
        self._reset_uvs_btn = QPushButton("Reset to projection", group)
        self._reset_uvs_btn.setToolTip(
            "Drop this face's imported texture coordinates and use the plane "
            "projection again. Leaves offset, scale and rotation unchanged."
        )
        self._reset_uvs_btn.clicked.connect(self.reset_uvs_requested.emit)
        form.addRow(self._reset_uvs_btn)

        # Connected only now that every widget it can touch (via
        # set_placement_target, from _on_side_changed) already exists.
        self._side_combo.currentIndexChanged.connect(self._on_side_changed)

        return group

    def _make_placement_spin(
        self, parent: QWidget, minimum: float, maximum: float, decimals: int
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox(parent)
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(0.1)
        spin.valueChanged.connect(self._on_placement_field_changed)
        return spin

    @property
    def target_side(self) -> Side:
        """The side (Front/Back) the placement group currently edits.

        Read by MainWindow's reset_uvs_requested handler (Task 8, M7.5c),
        which knows the selected face but not which side this dock's own
        combo box is pointed at.
        """
        return self._target_side

    def set_placement_target(self, face_id: int | None, side: Side = Side.FRONT) -> None:
        """Point the group at `face_id`'s placement for `side` and repopulate
        the fields from `scene.face_placement` (the identity, for a face
        never adjusted -- there is no None case to handle)."""
        self._target_face_id = face_id
        self._target_side = side
        self._placement_group.setEnabled(face_id is not None)

        placement = TexturePlacement()
        stored_uvs = None
        if face_id is not None and self._model is not None:
            placement = self._model.active_scene.face_placement(face_id, side)
            stored_uvs = self._model.active_scene.face_uvs(face_id, side)

        self._syncing = True
        try:
            self._side_combo.setCurrentIndex(0 if side == Side.FRONT else 1)
            self._offset_u_spin.setValue(placement.offset_u)
            self._offset_v_spin.setValue(placement.offset_v)
            self._scale_spin.setValue(placement.scale)
            self._rotation_spin.setValue(math.degrees(placement.rotation))
        finally:
            self._syncing = False

        # Task 8 (M7.5c): the button's enabled state IS the stored-UV
        # indicator (spec D12), read fresh here on every retarget so it
        # tracks selection changes, side flips, and a reset's own aftermath
        # (MainWindow re-syncs this same target after executing the command).
        self._reset_uvs_btn.setEnabled(stored_uvs is not None)

    def set_selected_face(self, face_id: int | None) -> None:
        """MainWindow's selection hook (M7.5b Task 10 fix round 1).

        `face_id` is the one selected face, or None -- MainWindow decides
        which of those it is (it already owns Selection; spec: enable only
        for a selection of exactly one face and nothing else), this widget
        only decides how to react. The currently chosen Front/Back side is
        kept as-is: switching which face is targeted must not silently flip
        which side the user was looking at.
        """
        self.set_placement_target(face_id, self._target_side)

    def _rotation_widget_value(self) -> float:
        """The rotation spin box's current value, in degrees."""
        return self._rotation_spin.value()

    def _on_side_changed(self, index: int) -> None:
        if self._syncing:
            return
        side = self._side_combo.itemData(index)
        self.set_placement_target(self._target_face_id, side)

    def _on_placement_field_changed(self, _value: float) -> None:
        if self._syncing:
            return
        self._apply_placement_from_widgets(
            offset_u=self._offset_u_spin.value(),
            offset_v=self._offset_v_spin.value(),
            scale=self._scale_spin.value(),
            rotation_deg=self._rotation_spin.value(),
        )

    def _apply_placement_from_widgets(
        self, *, offset_u: float, offset_v: float, scale: float, rotation_deg: float
    ) -> None:
        """Build the full `TexturePlacement` the fields currently describe
        and apply it. Rotation is converted to radians here, at the widget
        boundary -- `TexturePlacement.rotation` is radians because that is
        what `uv_projection.apply_placement` wants, but users think in
        degrees."""
        self._apply_placement(
            TexturePlacement(
                offset_u=offset_u,
                offset_v=offset_v,
                scale=scale,
                rotation=math.radians(rotation_deg),
            )
        )

    def _apply_placement(self, placement: TexturePlacement) -> None:
        """Set the targeted face+side's placement through the command stack.

        A no-op with nothing targeted or uninjected, the same as every
        MaterialsPage mutator with self._command_stack is None.
        """
        if self._target_face_id is None or self._command_stack is None or self._model is None:
            return
        cmd = SetFacePlacementCommand(self._target_face_id, placement, side=self._target_side)
        self._command_stack.execute(cmd, self._model.active_scene)
        # Re-read from the scene rather than trusting `placement` verbatim:
        # setting the identity CLEARS the entry (scene.set_face_placement),
        # so the fields must reflect that, not just echo what was passed in.
        self.set_placement_target(self._target_face_id, self._target_side)
        # Emitted only on the path that actually executed a command -- the
        # early return above is a genuine no-op and must not make the viewport
        # repaint. Flipping the Front/Back combo issues no command either
        # (_on_side_changed only repopulates the fields), so it deliberately
        # does not emit: it changes which side the panel EDITS, never the scene.
        self.placement_changed.emit()

    # --- outliner --------------------------------------------------------
    def outliner(self) -> QWidget | None:
        return self._outliner

    def set_outliner(self, widget: QWidget) -> None:
        """Install the Outliner above the tab strip (Task 9)."""
        if self._outliner_placeholder is not None:
            self._outliner_layout.removeWidget(self._outliner_placeholder)
            self._outliner_placeholder.deleteLater()
            self._outliner_placeholder = None
        if self._outliner is not None:
            self._outliner_layout.removeWidget(self._outliner)
            self._outliner.deleteLater()
        self._outliner = widget
        self._outliner_layout.addWidget(widget)
