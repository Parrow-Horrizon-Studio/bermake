# python/bermake/ui/tags_page.py
"""The Tags page (M5c, a Properties tab since M7.3): a list panel for object
tags + per-tag visibility."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QColorDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from bermake.commands.tag_commands import SetTagColorCommand
from bermake.model.tag import TagLibrary

_SWATCH_SIZE = 12


def _swatch_icon(color: tuple[float, float, float]) -> QIcon:
    """A small solid-colour icon for a tag row, echoing its `color` field."""
    r, g, b = (round(max(0.0, min(1.0, c)) * 255) for c in color)
    pix = QPixmap(_SWATCH_SIZE, _SWATCH_SIZE)
    pix.fill(QColor(r, g, b))
    return QIcon(pix)


class TagsPage(QWidget):
    """Tag list (checkbox = visibility, selected row = active tag) + Add/Assign
    buttons. A Properties tab page since M7.3 -- it was a QDockWidget through
    v0.4.0, which is why its layout was built into a child `container`.

    Recoloring a tag (M7.5a Task 11) goes through the command stack -- never
    TagLibrary.set_color directly -- so it is undoable, the same way
    MaterialsPage's editor drives commands instead of MaterialLibrary
    (Task 9). The stack and model are injected by MainWindow after
    construction; when neither is supplied (as in the standalone
    TagsPage(lib) tests), _on_edit_color is a documented no-op.
    """

    active_tag_changed = Signal(int)
    visibility_changed = Signal()
    assign_to_selection_requested = Signal()
    library_changed = Signal()

    def __init__(
        self,
        library: TagLibrary,
        parent=None,
        *,
        command_stack=None,
        model=None,
    ) -> None:
        super().__init__(parent)
        self._library = library
        self._command_stack = command_stack
        self._model = model
        self._active_id = TagLibrary.UNTAGGED_ID
        self._rebuilding = False

        container = self
        layout = QVBoxLayout(container)
        self._selection_label = QLabel("Selection: —", container)
        layout.addWidget(self._selection_label)
        self._list = QListWidget(container)
        self._list.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        self._list.itemChanged.connect(self._on_item_changed)
        self._list.currentItemChanged.connect(self._on_current_changed)
        layout.addWidget(self._list)
        add_btn = QPushButton("Add Tag", container)
        add_btn.clicked.connect(self._on_add)
        layout.addWidget(add_btn)
        assign_btn = QPushButton("Assign to Selection", container)
        assign_btn.clicked.connect(self._on_assign)
        layout.addWidget(assign_btn)
        color_btn = QPushButton("Tag Color…", container)
        color_btn.clicked.connect(self._on_edit_color)
        layout.addWidget(color_btn)

        self._rebuild()

    def _rebuild(self) -> None:
        self._rebuilding = True
        self._list.clear()
        for tag in self._library.tags():
            item = QListWidgetItem(tag.name)
            item.setIcon(_swatch_icon(tag.color))
            item.setData(Qt.ItemDataRole.UserRole, tag.id)
            if tag.id == TagLibrary.UNTAGGED_ID:
                # Untagged: always visible — checkbox shown checked, not user-toggleable.
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setFlags(
                    item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEditable
                )
                item.setCheckState(
                    Qt.CheckState.Checked if tag.visible else Qt.CheckState.Unchecked
                )
            self._list.addItem(item)
            if tag.id == self._active_id:
                self._list.setCurrentItem(item)
        self._rebuilding = False

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        if self._rebuilding:
            return
        tid = int(item.data(Qt.ItemDataRole.UserRole))
        tag = self._library.get(tid)
        # Visibility (checkbox). Untagged stays always-visible.
        visible = item.checkState() == Qt.CheckState.Checked
        if self._library.is_visible(tid) != visible:
            self._library.set_visible(tid, visible)
            self.visibility_changed.emit()
        # Rename (inline text edit). Untagged is not renamable; empty is rejected.
        new_name = item.text().strip()
        if tid != TagLibrary.UNTAGGED_ID and new_name and new_name != tag.name:
            self._library.rename(tid, new_name)
            self.library_changed.emit()
        elif item.text() != tag.name:
            # Reject empty/invalid edit -> restore display without re-triggering.
            self._list.blockSignals(True)
            item.setText(tag.name)
            self._list.blockSignals(False)

    def _on_current_changed(self, current, _previous) -> None:
        if self._rebuilding or current is None:
            return
        self._active_id = int(current.data(Qt.ItemDataRole.UserRole))
        self.active_tag_changed.emit(self._active_id)

    def _on_add(self) -> None:
        tag = self._library.add(f"Tag {len(self._library.tags())}")
        self.library_changed.emit()
        self._rebuild()
        self.set_active(tag.id)

    def _on_assign(self) -> None:
        self.assign_to_selection_requested.emit()

    def _on_edit_color(self) -> None:
        """Recolor the active tag through the command stack, via QColorDialog.

        A documented no-op when no command_stack was injected (the
        standalone TagsPage(lib) tests) -- mirrors MaterialsPage's
        `_apply_edit` guard, which the same lack of a stack short-circuits.
        """
        if self._command_stack is None:
            return
        tag = self._library.get(self._active_id)
        r, g, b = (round(c * 255) for c in tag.color)
        qc = QColorDialog.getColor(QColor(r, g, b), self)
        if not qc.isValid():
            return
        picked = (qc.redF(), qc.greenF(), qc.blueF())
        # Re-picking the colour the dialog opened on is not an edit; issuing
        # a command for it would push a no-op undo entry (same guard as
        # MaterialsPage._on_edit_color, compared on the picker's 8-bit
        # round-trip rather than the stored floats).
        if tuple(round(c * 255) for c in picked) == (r, g, b):
            return
        cmd = SetTagColorCommand(self._library, self._active_id, picked)
        self._command_stack.execute(cmd, self._model)
        self._rebuild()
        self.library_changed.emit()

    def set_selection_tag(self, name: str | None) -> None:
        """Show the tag of the current selection (None -> em-dash placeholder)."""
        self._selection_label.setText(f"Selection: {name}" if name else "Selection: —")

    def set_active(self, tag_id: int) -> None:
        """Select the row for `tag_id` (used to set the active tag programmatically)."""
        self._active_id = tag_id
        for i in range(self._list.count()):
            item = self._list.item(i)
            if int(item.data(Qt.ItemDataRole.UserRole)) == tag_id:
                self._list.setCurrentItem(item)
                break

    def set_library(self, library: TagLibrary) -> None:
        """Rebind to a new library (after file Open / New) and rebuild the list."""
        self._library = library
        self._active_id = TagLibrary.UNTAGGED_ID
        self._rebuild()

    def refresh(self) -> None:
        """Rebuild the list from the current library (reflects programmatic changes)."""
        self._rebuild()

    @property
    def active_tag_id(self) -> int:
        return self._active_id
