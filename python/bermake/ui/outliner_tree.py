"""The Outliner tree (M7.3).

Renders OutlinerRow data and emits instance ids back. It never walks the model
itself -- that is model_queries.outliner_rows' job -- so everything decidable
about which rows exist is tested without a QApplication, and this file holds
only what Qt genuinely requires.

Two columns: the label (editable, selects on click) and the eye (toggles
visibility on click). Every programmatic mutation runs under `_applying` so
Qt's itemChanged / itemSelectionChanged do not fire signals back at whoever
just called in.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QMouseEvent
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTreeWidget, QTreeWidgetItem

from bermake.ui.icons import icon

_ID_ROLE = Qt.ItemDataRole.UserRole
_HIDDEN_ROLE = Qt.ItemDataRole.UserRole + 1
_DIMMED_ROLE = Qt.ItemDataRole.UserRole + 2

_EYE_WIDTH = 28
_DIM_ALPHA = 110


class OutlinerTree(QTreeWidget):
    """The model hierarchy: one row per instance, with rename and an eye toggle."""

    instance_clicked = Signal(int)
    instance_activated = Signal(int)
    hide_toggled = Signal(int, bool)
    rename_requested = Signal(int, str)

    LABEL_COLUMN = 0
    EYE_COLUMN = 1

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._applying = False
        self._items: dict[int, QTreeWidgetItem] = {}
        # Set for the duration of a single mousePressEvent, so a real click on
        # the eye column can be told apart from any other selection change
        # (see mousePressEvent / _on_selection_changed).
        self._press_column: int | None = None

        self.setColumnCount(2)
        self.setHeaderHidden(True)
        # Single-selection only: _on_selection_changed emits one instance id
        # (MainWindow replaces the whole Selection with it), so advertising
        # ExtendedSelection promised a multi-select the rest of the pipeline
        # could not honour -- a shift-click range visually snapped back to
        # one row the moment the click landed. Carrying a full id set through
        # is real M7.4 scope, not a fix for this contract.
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        header = self.header()
        header.setSectionResizeMode(self.LABEL_COLUMN, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(self.EYE_COLUMN, QHeaderView.ResizeMode.Fixed)
        self.setColumnWidth(self.EYE_COLUMN, _EYE_WIDTH)

        self.itemClicked.connect(self._on_item_clicked)
        self.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.itemChanged.connect(self._on_item_changed)
        self.itemSelectionChanged.connect(self._on_selection_changed)

    # --- building --------------------------------------------------------
    def set_rows(self, rows, root_label: str) -> None:
        """Rebuild from a depth-ordered row list (parents before children).

        The root is not a row -- it has no Instance -- so it is rendered here
        from `root_label` and everything at depth 0 hangs beneath it.
        """
        self._applying = True
        try:
            self.clear()
            self._items = {}

            root = QTreeWidgetItem([root_label, ""])
            root.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.addTopLevelItem(root)
            root.setExpanded(True)

            # rows are parents-before-children, so a stack indexed by depth is
            # enough to find each row's parent without any lookup by id.
            stack: list[QTreeWidgetItem] = [root]
            for row in rows:
                del stack[row.depth + 1 :]
                parent = stack[row.depth]
                item = QTreeWidgetItem([row.label, ""])
                item.setData(self.LABEL_COLUMN, _ID_ROLE, row.instance_id)
                item.setData(self.LABEL_COLUMN, _HIDDEN_ROLE, bool(row.hidden))
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                item.setIcon(
                    self.LABEL_COLUMN,
                    icon(
                        "outliner-component" if row.is_component else "outliner-group",
                        self.palette().windowText().color(),
                    ),
                )
                self._apply_visibility(item, row)
                parent.addChild(item)
                stack.append(item)
                self._items[row.instance_id] = item
                if row.on_active_path:
                    item.setExpanded(True)
                    ancestor = parent
                    while ancestor is not None:
                        ancestor.setExpanded(True)
                        ancestor = ancestor.parent()
        finally:
            self._applying = False

    def _apply_visibility(self, item: QTreeWidgetItem, row) -> None:
        """Eye icon from the row's OWN hidden; dimming from the inherited states.

        These are separate on purpose. Clicking the eye writes `hidden` and
        nothing else, so a row greyed because an ancestor is hidden -- or
        because its tag is off -- would not come back. Showing one appearance
        for both would misrepresent what the click does.
        """
        item.setIcon(
            self.EYE_COLUMN,
            icon(
                "hidden" if row.hidden else "visible",
                self.palette().windowText().color(),
            ),
        )
        dimmed = bool(row.inherited_hidden or row.tag_hidden)
        item.setData(self.LABEL_COLUMN, _DIMMED_ROLE, dimmed)
        if dimmed:
            color = self.palette().windowText().color()
            color.setAlpha(_DIM_ALPHA)
            item.setForeground(self.LABEL_COLUMN, QBrush(color))

    # --- queries ---------------------------------------------------------
    def item_for(self, instance_id: int) -> QTreeWidgetItem | None:
        return self._items.get(int(instance_id))

    def is_dimmed(self, item: QTreeWidgetItem) -> bool:
        return bool(item.data(self.LABEL_COLUMN, _DIMMED_ROLE))

    @staticmethod
    def instance_id_of(item: QTreeWidgetItem) -> int | None:
        value = item.data(OutlinerTree.LABEL_COLUMN, _ID_ROLE)
        return None if value is None else int(value)

    # --- selection -------------------------------------------------------
    def set_selected_ids(self, instance_ids) -> None:
        """Highlight exactly `instance_ids`. Emits nothing and rebuilds nothing.

        Called on every selection change, including many times a second during
        a box-select drag, which is why rows carry no selection state.
        """
        wanted = {int(i) for i in instance_ids}
        self._applying = True
        try:
            for instance_id, item in self._items.items():
                item.setSelected(instance_id in wanted)
        finally:
            self._applying = False

    # --- mouse -------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Record which column the press landed in for _on_selection_changed.

        Qt updates the current item / selection (firing itemSelectionChanged)
        synchronously inside the base class's mousePressEvent, BEFORE
        itemClicked fires on release -- so a press on the eye column would
        otherwise re-root the active context (via instance_clicked) a moment
        before the eye toggle itself is handled. Cleared again once the base
        implementation returns so an unrelated selection change (keyboard
        navigation, a programmatic set_selected_ids) is never suppressed by a
        stale value from an earlier click.
        """
        self._press_column = self.columnAt(int(event.position().x()))
        try:
            super().mousePressEvent(event)
        finally:
            self._press_column = None

    # --- signals ---------------------------------------------------------
    def _on_selection_changed(self) -> None:
        if self._applying:
            return
        if self._press_column == self.EYE_COLUMN:
            # The eye toggle only ever writes `hidden`; it must not also move
            # the current selection / active context.
            return
        item = self.currentItem()
        if item is None:
            return
        instance_id = self.instance_id_of(item)
        if instance_id is not None:
            self.instance_clicked.emit(instance_id)

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        if self._applying or column != self.EYE_COLUMN:
            return
        instance_id = self.instance_id_of(item)
        if instance_id is None:
            return
        currently_hidden = bool(item.data(self.LABEL_COLUMN, _HIDDEN_ROLE))
        self.hide_toggled.emit(instance_id, not currently_hidden)

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        if self._applying or column != self.LABEL_COLUMN:
            return
        instance_id = self.instance_id_of(item)
        if instance_id is not None:
            self.instance_activated.emit(instance_id)

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._applying or column != self.LABEL_COLUMN:
            return
        instance_id = self.instance_id_of(item)
        if instance_id is None:
            return
        self.rename_requested.emit(instance_id, item.text(self.LABEL_COLUMN))
