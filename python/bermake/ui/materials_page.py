"""The Materials page (M5b, a Properties tab since M7.3; editor added M7.5a
Task 9): a swatch grid for choosing the active material, plus an editor below
it for the active material's fields.

Every mutation (add / edit / delete) goes through a command on the command
stack -- never MaterialLibrary directly -- so it is undoable. The stack and
the model (for the current scene) are injected by MainWindow after
construction the same way ToolContext injects them into a Tool; when neither
is supplied (as in the standalone MaterialsPage(lib) tests) every mutating
method below is a documented no-op, and the widget behaves exactly as the
M5b swatch-only page did.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QColorDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from bermake.commands.command import CompositeCommand
from bermake.commands.material_commands import (
    AddMaterialCommand,
    AddTextureCommand,
    DeleteMaterialCommand,
    EditMaterialCommand,
    SetMaterialTextureCommand,
)
from bermake.model.material import MaterialLibrary
from bermake.viewport.texture_cache import decode_image, sniff_format

_COLUMNS = 4
_SWATCH_ICON_SIZE = QSize(30, 22)  # inset from _swatch_style's 36x28 min size


def _swatch_style(color: tuple[float, float, float], alpha: float, active: bool) -> str:
    """Stylesheet for one swatch, alpha-composited over whatever is behind it.

    Qt style sheet rgba() alpha is 0..255, not the CSS3 0..1 -- using
    `alpha` directly here would make every translucent material nearly
    invisible instead of see-through.
    """
    r, g, b = (round(c * 255) for c in color)
    a = round(max(0.0, min(1.0, alpha)) * 255)
    border = "3px solid #2f8fff" if active else "1px solid #555"
    return (
        f"background-color: rgba({r},{g},{b},{a}); border: {border}; "
        "min-width: 36px; min-height: 28px;"
    )


class MaterialsPage(QWidget):
    """Swatch grid + editor for the active material. Emits
    active_material_changed(Material)."""

    active_material_changed = Signal(object)  # emits a Material
    library_changed = Signal()

    def __init__(
        self,
        library: MaterialLibrary,
        parent=None,
        *,
        command_stack=None,
        model=None,
    ) -> None:
        super().__init__(parent)
        self._library = library
        self._command_stack = command_stack
        self._model = model
        self._active_id = MaterialLibrary.DEFAULT_ID
        self._buttons: dict[int, QPushButton] = {}
        # QIcon per texture id, built once and reused across every
        # _rebuild_swatches()/_restyle() call -- both fire on every keystroke
        # in the alpha/metallic/roughness/size spin boxes (M7.5b fix round 1),
        # and decoding the same PNG/JPEG per textured material per keystroke
        # would be needless work. Kept in sync with the live TextureLibrary by
        # set_library() (wipes it -- a new/reopened document's ids restart
        # from 1 with different bytes) and refresh() (drops entries for ids
        # an undo/redo removed).
        self._texture_icon_cache: dict[int, QIcon] = {}
        self._syncing = False

        container = self
        outer = QVBoxLayout(container)
        self._grid = QGridLayout()
        outer.addLayout(self._grid)
        custom = QPushButton("Custom color…", container)
        custom.clicked.connect(self._on_custom)
        outer.addWidget(custom)

        outer.addLayout(self._build_editor(container))
        outer.addStretch(1)

        self._rebuild_swatches()
        self._sync_editor()

    def _build_editor(self, container: QWidget) -> QFormLayout:
        form = QFormLayout()

        self._name_edit = QLineEdit(container)
        self._name_edit.editingFinished.connect(self._on_name_committed)
        form.addRow("Name", self._name_edit)

        self._color_btn = QPushButton(container)
        self._color_btn.clicked.connect(self._on_edit_color)
        form.addRow("Base color", self._color_btn)

        self._alpha_spin = self._make_unit_spin(container, self._on_alpha_changed)
        form.addRow("Alpha", self._alpha_spin)

        self._metallic_spin = self._make_unit_spin(container, self._on_metallic_changed)
        form.addRow("Metallic", self._metallic_spin)

        self._roughness_spin = self._make_unit_spin(container, self._on_roughness_changed)
        form.addRow("Roughness", self._roughness_spin)

        self._texture_btn = QPushButton("Texture…", container)
        self._texture_btn.clicked.connect(lambda: self._choose_texture())
        form.addRow(self._texture_btn)

        self._clear_texture_btn = QPushButton("Clear Texture", container)
        self._clear_texture_btn.clicked.connect(lambda: self._clear_texture())
        form.addRow(self._clear_texture_btn)

        self._texture_width_spin = self._make_size_spin(container, self._on_texture_size_changed)
        form.addRow("Texture width", self._texture_width_spin)

        self._texture_height_spin = self._make_size_spin(container, self._on_texture_size_changed)
        form.addRow("Texture height", self._texture_height_spin)

        self._delete_btn = QPushButton("Delete Material", container)
        self._delete_btn.clicked.connect(lambda: self._delete_active())
        form.addRow(self._delete_btn)

        return form

    def _make_unit_spin(self, container: QWidget, on_changed) -> QDoubleSpinBox:
        spin = QDoubleSpinBox(container)
        spin.setRange(0.0, 1.0)
        spin.setSingleStep(0.05)
        spin.setDecimals(2)
        spin.valueChanged.connect(on_changed)
        return spin

    def _make_size_spin(self, container: QWidget, on_changed) -> QDoubleSpinBox:
        """A spin box for texture_size, in model units -- unlike the 0..1
        unit spins above, real-world texture dimensions can be any positive
        size."""
        spin = QDoubleSpinBox(container)
        spin.setRange(0.01, 1000.0)
        spin.setSingleStep(0.1)
        spin.setDecimals(3)
        spin.valueChanged.connect(on_changed)
        return spin

    # --- swatch grid -------------------------------------------------------

    def _rebuild_swatches(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._buttons.clear()
        for idx, mat in enumerate(self._library.materials()):
            btn = QPushButton(self)
            btn.setToolTip(mat.name)
            self._apply_swatch_appearance(btn, mat)
            btn.clicked.connect(lambda _checked=False, mid=mat.id: self._on_pick(mid))
            self._grid.addWidget(btn, idx // _COLUMNS, idx % _COLUMNS)
            self._buttons[mat.id] = btn

    def _restyle(self) -> None:
        for mid, btn in self._buttons.items():
            mat = self._library.get(mid)
            self._apply_swatch_appearance(btn, mat)

    def _apply_swatch_appearance(self, btn: QPushButton, mat) -> None:
        style, icon = self._swatch_appearance(mat)
        btn.setStyleSheet(style)
        btn.setIcon(icon if icon is not None else QIcon())
        btn.setIconSize(_SWATCH_ICON_SIZE)

    def _swatch_appearance(self, mat) -> tuple[str, QIcon | None]:
        """The (stylesheet, icon) pair a swatch button should show for `mat`.

        Kept as one method the tests can call directly (M7.5b Task 9) so a
        textured material can be asserted to look different from a plain one.
        Qt style sheets have no reliable way to alpha-composite an arbitrary
        image UNDER a tint colour, so the texture is carried as a QIcon
        instead of a `background-image` -- the tint + border still come from
        `_swatch_style`, unchanged from M7.5a.
        """
        style = _swatch_style(mat.base_color, mat.alpha, mat.id == self._active_id)
        icon = self._texture_icon(mat.texture_id)
        return style, icon

    def _texture_icon(self, texture_id: int | None) -> QIcon | None:
        """The cached swatch QIcon for `texture_id`, decoding + building it
        at most once.

        Uses `decode_image` -- the SAME decoder `_choose_texture` already
        used to accept the bytes into the library -- rather than a second,
        independent one (`QPixmap.loadFromData`, M7.5b fix round 1). Two
        decoders accepting different formats would show as a silently
        icon-less swatch with no error for whatever the swatch's decoder
        rejects and the picker's accepts.

        Caching matters because `_rebuild_swatches()` runs after every
        `_apply_edit`/`_apply_texture_size`, and `QDoubleSpinBox.valueChanged`
        fires per keystroke -- without this, editing metallic on an unrelated
        material would re-decode every textured material's image on every
        keystroke.
        """
        if texture_id is None or self._model is None:
            return None
        cached = self._texture_icon_cache.get(texture_id)
        if cached is not None:
            return cached
        tex = self._model.textures.get(texture_id)
        if tex is None:
            return None
        img = decode_image(tex.data)
        if img is None:
            return None
        qimage = QImage(
            img.pixels.tobytes(),
            img.width,
            img.height,
            img.width * 4,
            QImage.Format.Format_RGBA8888,
        )
        icon = QIcon(QPixmap.fromImage(qimage))
        self._texture_icon_cache[texture_id] = icon
        return icon

    def _prune_texture_icon_cache(self) -> None:
        """Drop cached icons for texture ids no longer in the live library.

        Without this an icon built for an id an undo removed (or a future
        DeleteTextureCommand removes) would outlive its Texture record --
        harmless as long as ids are never reused for different bytes within
        one document, but that invariant is `TextureLibrary`'s to keep, not
        this page's to assume forever.
        """
        if self._model is None:
            self._texture_icon_cache.clear()
            return
        live = {t.id for t in self._model.textures.textures()}
        for tid in [t for t in self._texture_icon_cache if t not in live]:
            del self._texture_icon_cache[tid]

    def _on_pick(self, material_id: int) -> None:
        self._active_id = material_id
        self._restyle()
        self._sync_editor()
        self.active_material_changed.emit(self._library.get(material_id))

    def _on_custom(self) -> None:
        qc = QColorDialog.getColor(parent=self)
        if not qc.isValid():
            return
        color = (qc.redF(), qc.greenF(), qc.blueF())
        self._add_material(qc.name(), color)  # name == hex "#rrggbb"

    def set_active(self, material_id: int) -> None:
        """Update the highlighted swatch (used by the Paint tool's eyedropper)."""
        self._active_id = material_id
        self._restyle()
        self._sync_editor()
        self.active_material_changed.emit(self._library.get(material_id))

    def set_library(self, library: MaterialLibrary) -> None:
        """Rebind to a new library (after file Open / New) and rebuild swatches."""
        self._library = library
        self._active_id = MaterialLibrary.DEFAULT_ID
        # A new or reopened document's TextureLibrary restarts id numbering
        # from 1 with different bytes -- the same collision release_all()
        # fixes for the renderer's GL cache, one layer up in this page's own
        # icon cache. A stale id here would otherwise show the PREVIOUS
        # document's image under the new one's swatch.
        self._texture_icon_cache.clear()
        self._rebuild_swatches()
        self._sync_editor()

    def refresh(self) -> None:
        """Rebuild swatches + editor from the current library (same library).

        Called after every undo/redo: material add / edit / delete are all
        undoable (M7.5a Task 9), so rewinding the stack can change the
        library under a page that would otherwise keep painting the
        pre-undo state. If the active material is gone (an undone Add), fall
        back to Default rather than leaving `_active_id` dangling --
        MaterialLibrary.get() would silently resolve it to Default anyway.
        """
        dropped = not any(m.id == self._active_id for m in self._library.materials())
        if dropped:
            self._active_id = MaterialLibrary.DEFAULT_ID
        # An undone import (or a future DeleteTextureCommand) can remove a
        # Texture record between refreshes; don't keep its icon around under
        # a ghost key.
        self._prune_texture_icon_cache()
        self._rebuild_swatches()
        self._sync_editor()
        if dropped:
            # Tell MainWindow too: it caches the active material id for the
            # Paint tool, which would otherwise keep painting the dead one.
            self.active_material_changed.emit(self._library.get(self._active_id))

    @property
    def active_material_id(self) -> int:
        return self._active_id

    # --- editor: read side ---------------------------------------------------

    def _sync_editor(self) -> None:
        """Repopulate the editor fields from the active material. Guarded by
        `_syncing` so setValue()/setText() below don't loop back into the
        `_on_*_changed` handlers and re-issue a command for a value we
        just read from the library."""
        mat = self._library.get(self._active_id)
        self._syncing = True
        try:
            self._name_edit.setText(mat.name)
            self._color_btn.setStyleSheet(_swatch_style(mat.base_color, 1.0, False))
            self._alpha_spin.setValue(mat.alpha)
            self._metallic_spin.setValue(mat.metallic)
            self._roughness_spin.setValue(mat.roughness)
            self._texture_width_spin.setValue(mat.texture_size[0])
            self._texture_height_spin.setValue(mat.texture_size[1])
        finally:
            self._syncing = False
        self._delete_btn.setEnabled(self._can_delete_active())
        # M7.5b final review, item 5: the size spins get the same treatment as
        # Clear Texture, for the same reason. Editing them on a material that
        # samples nothing pushes an undoable command and invalidates the
        # renderer's uv_key -- a re-upload of every definition carrying the
        # material -- to change a number that cannot alter a single pixel.
        has_texture = mat.texture_id is not None
        self._clear_texture_btn.setEnabled(has_texture)
        self._texture_width_spin.setEnabled(has_texture)
        self._texture_height_spin.setEnabled(has_texture)

    def _current_scene(self):
        return None if self._model is None else self._model.active_scene

    # --- editor: write side (commands only) -----------------------------

    def _apply_edit(self, **fields) -> None:
        """Edit the active material through the command stack."""
        mid = self.active_material_id
        if mid is None or self._command_stack is None:
            return
        cmd = EditMaterialCommand(self._library, mid, **fields)
        self._command_stack.execute(cmd, self._current_scene())
        self._rebuild_swatches()
        self._sync_editor()
        self.library_changed.emit()

    def _add_material(self, name: str, color: tuple[float, float, float]) -> None:
        """Add a material through the command stack."""
        if self._command_stack is None:
            return
        cmd = AddMaterialCommand(self._library, name, color)
        self._command_stack.execute(cmd, self._current_scene())
        self._rebuild_swatches()
        self.library_changed.emit()
        self.set_active(cmd.material_id)

    def _can_delete_active(self) -> bool:
        return self.active_material_id != MaterialLibrary.DEFAULT_ID

    def _delete_active(self, confirm=None) -> None:
        """Delete the active material through the command stack.

        `confirm` must be SYNCHRONOUS: it has to block until the user has
        answered and return a bool, the way the default modal
        QMessageBox.question does. It is called between constructing `cmd`
        (which snapshots the affected faces model-wide, so `affected_count`
        can be quoted) and executing it. A non-blocking `confirm` that
        returned to the event loop in between would let a paint gesture land
        on the canvas mid-decision. That contract is enforced rather than
        merely documented: DeleteMaterialCommand.do() re-snapshots before it
        mutates, so the geometry it clears is always the geometry as it
        actually is -- only the quoted count could go stale.

        `confirm` also lets tests supply a stub instead of a modal dialog.
        """
        mid = self.active_material_id
        if not self._can_delete_active() or self._command_stack is None or self._model is None:
            return
        cmd = DeleteMaterialCommand(self._library, mid, self._model)
        ask = confirm if confirm is not None else self._confirm_delete
        if not ask(cmd.affected_count):
            return
        self._command_stack.execute(cmd, self._model)
        self.set_active(MaterialLibrary.DEFAULT_ID)
        self._rebuild_swatches()
        self.library_changed.emit()

    def _confirm_delete(self, count: int) -> bool:
        mat = self._library.get(self.active_material_id)
        noun = "face side" if count == 1 else "face sides"
        text = (
            f'Delete material "{mat.name}"? {count} {noun} painted with it will revert to Default.'
        )
        result = QMessageBox.question(
            self,
            "Delete Material",
            text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return result == QMessageBox.StandardButton.Yes

    def _choose_texture(self, chooser=None, on_error=None) -> None:
        """Import an image and point the active material at it, as one undo step.

        `chooser` and `on_error` are seams so a test never opens a real modal.
        Decode happens BEFORE any command is built, so a corrupt file leaves
        the document untouched (spec 1.8).
        """
        mid = self.active_material_id
        if mid is None or self._command_stack is None or self._model is None:
            return
        pick = chooser if chooser is not None else self._ask_for_image
        path = pick()
        if not path:
            return

        report = on_error if on_error is not None else self._report_error
        try:
            data = Path(path).read_bytes()
        except OSError as exc:
            report(f"Could not read {path}: {exc}")
            return

        fmt = sniff_format(data)
        img = decode_image(data) if fmt else None
        if img is None:
            report(f"{Path(path).name} is not a readable PNG or JPEG image.")
            return

        add = AddTextureCommand(
            self._model.textures,
            Path(path).name,
            data,
            fmt,
            img.width,
            img.height,
            img.has_transparency,
        )
        scene = self._current_scene()
        add.do(scene)
        assign = SetMaterialTextureCommand(self._library, mid, add.texture_id)
        assign.do(scene)
        self._command_stack.push_executed(
            CompositeCommand(name="Set Texture", children=[add, assign]), scene
        )
        self._rebuild_swatches()
        self._sync_editor()
        self.library_changed.emit()

    def _ask_for_image(self) -> str | None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose Texture", "", "Images (*.png *.jpg *.jpeg)"
        )
        return path or None

    def _report_error(self, message: str) -> None:
        QMessageBox.warning(self, "Texture", message)

    def _clear_texture(self) -> None:
        """Clear the active material's texture through the command stack.

        Only the material's pointer is cleared -- the Texture asset itself is
        kept in the library (spec D11): undoing this must be able to restore
        the image, and a delete-the-asset action is a separate, destructive
        operation (DeleteTextureCommand) with its own confirmation.
        """
        mid = self.active_material_id
        if mid is None or self._command_stack is None:
            return
        cmd = SetMaterialTextureCommand(self._library, mid, None)
        self._command_stack.execute(cmd, self._current_scene())
        self._rebuild_swatches()
        self._sync_editor()
        self.library_changed.emit()

    def _apply_texture_size(self, width: float, height: float) -> None:
        """Edit the active material's real-world texture size through the stack."""
        mid = self.active_material_id
        if mid is None or self._command_stack is None:
            return
        mat = self._library.get(mid)
        cmd = SetMaterialTextureCommand(self._library, mid, mat.texture_id, (width, height))
        self._command_stack.execute(cmd, self._current_scene())
        self._rebuild_swatches()
        self._sync_editor()
        self.library_changed.emit()

    # --- editor: signal handlers (thin wrappers over the seams above) ------

    def _on_name_committed(self) -> None:
        if self._syncing:
            return
        name = self._name_edit.text().strip()
        mat = self._library.get(self.active_material_id)
        if name and name != mat.name:
            self._apply_edit(name=name)

    def _on_edit_color(self) -> None:
        mat = self._library.get(self.active_material_id)
        r, g, b = (round(c * 255) for c in mat.base_color)
        qc = QColorDialog.getColor(QColor(r, g, b), self)
        if not qc.isValid():
            return
        picked = (qc.redF(), qc.greenF(), qc.blueF())
        # Re-picking the colour the dialog opened on is not an edit; issuing
        # a command for it would push a no-op undo entry. Compare on the
        # round-tripped 8-bit values the dialog actually offers, not on the
        # stored floats, which carry precision the picker cannot express.
        # (Matches the guard _on_name_committed already has.)
        if tuple(round(c * 255) for c in picked) == (r, g, b):
            return
        self._apply_edit(base_color=picked)

    def _on_alpha_changed(self, value: float) -> None:
        if self._syncing:
            return
        self._apply_edit(alpha=value)

    def _on_metallic_changed(self, value: float) -> None:
        if self._syncing:
            return
        self._apply_edit(metallic=value)

    def _on_roughness_changed(self, value: float) -> None:
        if self._syncing:
            return
        self._apply_edit(roughness=value)

    def _on_texture_size_changed(self, _value: float) -> None:
        if self._syncing:
            return
        self._apply_texture_size(
            self._texture_width_spin.value(), self._texture_height_spin.value()
        )
