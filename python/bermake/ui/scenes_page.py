"""The Scenes page (M7e, a Properties tab since M7.3): saved Scenes with recall
+ management.

Clicking a row recalls that Scene; Add captures the current view; Update
overwrites the selected Scene; Delete removes it; the arrows reorder. Rename is
inline (double-click). A near-clone of TagsPage. The page only emits intent
signals — MainWindow routes them through CommandStack / the view animator.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ScenesPage(QWidget):
    """Saved-Scene list + Add/Update/Delete/reorder controls."""

    create_requested = Signal()
    update_requested = Signal(int)
    delete_requested = Signal(int)
    rename_requested = Signal(int, str)
    reorder_requested = Signal(int, int)
    recall_requested = Signal(int)

    def __init__(self, library, parent=None) -> None:
        super().__init__(parent)
        self._library = library
        self._rebuilding = False

        container = self
        layout = QVBoxLayout(container)

        self._list = QListWidget(container)
        self._list.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        self._list.itemClicked.connect(self._on_item_clicked)
        self._list.itemChanged.connect(self._on_item_changed)
        self._list.currentItemChanged.connect(lambda *_: self._update_button_states())
        layout.addWidget(self._list)

        self._add_btn = QPushButton("Add", container)
        self._add_btn.clicked.connect(lambda: self.create_requested.emit())
        self._update_btn = QPushButton("Update", container)
        self._update_btn.clicked.connect(self._on_update)
        self._delete_btn = QPushButton("Delete", container)
        self._delete_btn.clicked.connect(self._on_delete)
        row = QHBoxLayout()
        row.addWidget(self._add_btn)
        row.addWidget(self._update_btn)
        row.addWidget(self._delete_btn)
        layout.addLayout(row)

        self._up_btn = QPushButton("Move Up", container)
        self._up_btn.clicked.connect(lambda: self._on_reorder(-1))
        self._down_btn = QPushButton("Move Down", container)
        self._down_btn.clicked.connect(lambda: self._on_reorder(1))
        arrows = QHBoxLayout()
        arrows.addWidget(self._up_btn)
        arrows.addWidget(self._down_btn)
        layout.addLayout(arrows)

        self._rebuild()

    # --- current-row helpers ---------------------------------------------

    def _current_id(self):
        item = self._list.currentItem()
        return None if item is None else int(item.data(Qt.ItemDataRole.UserRole))

    def _update_button_states(self) -> None:
        has = self._list.currentItem() is not None
        self._update_btn.setEnabled(has)
        self._delete_btn.setEnabled(has)
        self._up_btn.setEnabled(has)
        self._down_btn.setEnabled(has)

    # --- rebuild ----------------------------------------------------------

    def _rebuild(self, select_id=None) -> None:
        self._rebuilding = True
        self._list.clear()
        for view in self._library.views():
            item = QListWidgetItem(view.name)
            item.setData(Qt.ItemDataRole.UserRole, view.id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            self._list.addItem(item)
            if view.id == select_id:
                self._list.setCurrentItem(item)
        self._rebuilding = False
        self._update_button_states()

    # --- slots ------------------------------------------------------------

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        if self._rebuilding or item is None:
            return
        self.recall_requested.emit(int(item.data(Qt.ItemDataRole.UserRole)))

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        if self._rebuilding or item is None:
            return
        vid = int(item.data(Qt.ItemDataRole.UserRole))
        self.rename_requested.emit(vid, item.text().strip())

    def _on_update(self) -> None:
        vid = self._current_id()
        if vid is not None:
            self.update_requested.emit(vid)

    def _on_delete(self) -> None:
        vid = self._current_id()
        if vid is not None:
            self.delete_requested.emit(vid)

    def _on_reorder(self, direction: int) -> None:
        vid = self._current_id()
        if vid is not None:
            self.reorder_requested.emit(vid, direction)

    # --- public API -------------------------------------------------------

    def refresh(self, select_id=None) -> None:
        """Rebuild rows from the current library (never fires recall)."""
        self._rebuild(select_id=select_id)

    def set_library(self, library) -> None:
        """Rebind to a new library (after file Open / New) and rebuild."""
        self._library = library
        self._rebuild()
