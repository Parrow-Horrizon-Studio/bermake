"""The Entity Info tab (M7.3): what is selected, and the four things you can
change about it.

The page emits intent (rename_requested, hidden_requested, tag_requested,
material_requested) and owns no commands -- MainWindow routes each through the
command stack, exactly as ScenesPage already does. That keeps this widget
testable against a plain EntitySummary with no model in sight.

Fields are DISABLED, never hidden, when the selection does not support them:
a control that keeps its position is learnable, one that reflows between
selections is not. Same rule context_menu.is_enabled follows.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from bermake.model.material import MaterialLibrary
from bermake.model.tag import TagLibrary
from bermake.units import format_area, format_length

_PLACEHOLDER = "—"


class EntityInfoPage(QWidget):
    """A read-out of the selection plus four editable fields."""

    rename_requested = Signal(str)
    hidden_requested = Signal(bool)
    tag_requested = Signal(int)
    material_requested = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._applying = False

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)
        layout.addStretch(1)

        self._kind = QLabel(_PLACEHOLDER, self)
        self._definition = QLabel(_PLACEHOLDER, self)
        self._counts = QLabel(_PLACEHOLDER, self)
        self._size = QLabel(_PLACEHOLDER, self)
        self._measure = QLabel(_PLACEHOLDER, self)
        self._name = QLineEdit(self)
        self._hidden = QCheckBox(self)
        self._tag = QComboBox(self)
        self._material = QComboBox(self)

        form.addRow("Type", self._kind)
        form.addRow("Name", self._name)
        form.addRow("Definition", self._definition)
        form.addRow("Contents", self._counts)
        form.addRow("Dimensions", self._size)
        form.addRow("Measure", self._measure)
        form.addRow("Tag", self._tag)
        form.addRow("Material", self._material)
        form.addRow("Hidden", self._hidden)

        self._name.editingFinished.connect(self._on_name_committed)
        self._hidden.clicked.connect(self._on_hidden_clicked)
        self._tag.activated.connect(self._on_tag_activated)
        self._material.activated.connect(self._on_material_activated)

    # --- accessors used by tests and by MainWindow -----------------------
    def name_field(self) -> QLineEdit:
        return self._name

    def hidden_field(self) -> QCheckBox:
        return self._hidden

    def tag_field(self) -> QComboBox:
        return self._tag

    def material_field(self) -> QComboBox:
        return self._material

    def kind_text(self) -> str:
        return self._kind.text()

    def definition_text(self) -> str:
        return self._definition.text()

    def counts_text(self) -> str:
        return self._counts.text()

    def size_text(self) -> str:
        return self._size.text()

    def measure_text(self) -> str:
        return self._measure.text()

    # --- refresh ---------------------------------------------------------
    def refresh(self, summary, units, tag_library, material_library) -> None:
        """Repopulate from a summary. Emits nothing: this runs on every
        selection change, and a signal here would fire commands at ourselves."""
        self._applying = True
        try:
            self._kind.setText(
                summary.kind if summary.count <= 1 else f"{summary.kind} x{summary.count}"
            )
            self._definition.setText(summary.definition_name or _PLACEHOLDER)
            self._counts.setText(self._format_counts(summary))
            self._size.setText(self._format_size(summary, units))
            self._measure.setText(self._format_measure(summary, units))

            self._name.setText(summary.name or "")
            self._name.setEnabled(summary.name is not None)

            self._hidden.setTristate(summary.hidden is None)
            if summary.hidden is None:
                self._hidden.setCheckState(Qt.CheckState.PartiallyChecked)
            else:
                self._hidden.setChecked(bool(summary.hidden))
            # instances_only covers Group, Component, AND a same-type but
            # heterogeneous Group+Component selection ("Mixed") -- Hidden
            # applies to any instance selection (spec 1.4), not just a
            # homogeneous one. A cross-type Mixed (instances + faces, say)
            # leaves instances_only False and stays disabled.
            self._hidden.setEnabled(summary.instances_only)

            self._fill_tags(summary, tag_library)
            self._fill_materials(summary, material_library)
        finally:
            self._applying = False

    def _format_counts(self, summary) -> str:
        parts = [
            (summary.child_count, "object"),
            (summary.edge_count, "edge"),
            (summary.face_count, "face"),
        ]
        text = ", ".join(
            f"{n} {word}" + ("s" if n != 1 else "") for n, word in parts if n is not None
        )
        return text or _PLACEHOLDER

    def _format_size(self, summary, units) -> str:
        if summary.size is None:
            return _PLACEHOLDER
        return " x ".join(format_length(v, units) for v in summary.size)

    def _format_measure(self, summary, units) -> str:
        if summary.area is not None:
            return format_area(summary.area, units)
        if summary.length is not None:
            return format_length(summary.length, units)
        return _PLACEHOLDER

    def _fill_tags(self, summary, tag_library) -> None:
        self._tag.clear()
        for tag in tag_library.tags():
            self._tag.addItem(tag.name, tag.id)
        # See the comment on _hidden's setEnabled above -- Tag applies to any
        # instance selection, including a heterogeneous Group+Component one.
        self._tag.setEnabled(summary.instances_only)
        if summary.tag_id is None:
            self._tag.setCurrentIndex(-1)
            return
        self._tag.setCurrentIndex(self._tag.findData(summary.tag_id))

    def _fill_materials(self, summary, material_library) -> None:
        self._material.clear()
        for material in material_library.materials():
            self._material.addItem(material.name, material.id)
        self._material.setEnabled(summary.kind == "Face")
        if summary.material_id is None:
            self._material.setCurrentIndex(-1)
            return
        self._material.setCurrentIndex(self._material.findData(summary.material_id))

    # --- intents -----------------------------------------------------------
    def _on_name_committed(self) -> None:
        if self._applying:
            return
        self.rename_requested.emit(self._name.text())

    def _on_hidden_clicked(self) -> None:
        if self._applying:
            return
        # A click always resolves a tri-state into a definite request.
        self.hidden_requested.emit(self._hidden.checkState() != Qt.CheckState.Unchecked)

    def _on_tag_activated(self, index: int) -> None:
        if self._applying or index < 0:
            return
        value = self._tag.itemData(index)
        self.tag_requested.emit(int(TagLibrary.UNTAGGED_ID if value is None else value))

    def _on_material_activated(self, index: int) -> None:
        if self._applying or index < 0:
            return
        value = self._material.itemData(index)
        self.material_requested.emit(int(MaterialLibrary.DEFAULT_ID if value is None else value))
