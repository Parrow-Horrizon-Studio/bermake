"""Material commands (M7.5a): add, edit, delete, and per-side paint."""

from __future__ import annotations

import numpy as np

from bermake.commands.command import Command
from bermake.scene.scene import Side, TexturePlacement

_DEFAULT_MATERIAL_ID = 0  # == MaterialLibrary.DEFAULT_ID (the unpainted sentinel)


def _apply(scene, f_id: int, material_id: int, side: Side) -> None:
    if material_id == _DEFAULT_MATERIAL_ID:
        scene.clear_face_material(f_id, side)
    else:
        scene.set_face_material(f_id, material_id, side)


class PaintFaceCommand(Command):
    """Assign a material to one side of one face; undo restores the prior one.

    Captures the previous material at do() time (id-preserving undo). Painting
    the Default material (id 0) clears any paint; undo restores it exactly.
    """

    name = "Paint Face"

    def __init__(self, face_id: int, new_material_id: int, side: Side = Side.FRONT) -> None:
        self._fid = face_id
        self._new = new_material_id
        self._side = side
        self._old: int | None = None

    def do(self, scene) -> None:
        self._old = scene.face_material(self._fid, self._side)
        _apply(scene, self._fid, self._new, self._side)

    def undo(self, scene) -> None:
        _apply(
            scene,
            self._fid,
            self._old if self._old is not None else _DEFAULT_MATERIAL_ID,
            self._side,
        )


class AddMaterialCommand(Command):
    """Add a custom material. Undo removes it again."""

    name = "Add Material"

    def __init__(self, library, name: str, base_color: tuple[float, float, float]) -> None:
        self._lib = library
        self._name = name
        self._color = base_color
        self._mid: int | None = None
        self._removed = None

    @property
    def material_id(self) -> int:
        assert self._mid is not None, "material_id read before do()"
        return self._mid

    def do(self, scene) -> None:
        if self._mid is None:
            self._mid = self._lib.add_custom(self._name, self._color).id
        else:
            # redo: put the same record back at the end, preserving its id
            self._lib.restore(self._removed, len(self._lib.materials()))

    def undo(self, scene) -> None:
        self._removed = self._lib.remove(self._mid)


class EditMaterialCommand(Command):
    """Edit any subset of a material's fields; undo restores the whole record."""

    name = "Edit Material"

    def __init__(self, library, material_id: int, **fields) -> None:
        self._lib = library
        self._mid = material_id
        self._fields = fields
        self._before = None

    def do(self, scene) -> None:
        self._before = self._lib.get(self._mid)
        self._lib.edit(self._mid, **self._fields)

    def undo(self, scene) -> None:
        self._lib.edit(
            self._mid,
            name=self._before.name,
            base_color=self._before.base_color,
            alpha=self._before.alpha,
            metallic=self._before.metallic,
            roughness=self._before.roughness,
        )


class DeleteMaterialCommand(Command):
    """Remove a material and repaint every side using it back to Default.

    MODEL-WIDE, not scene-wide. A MaterialLibrary belongs to the Model, but
    each Definition owns its own Scene -- so a material can be painted in any
    definition, not just the one currently entered. Scanning only the entered
    scene would leave faces elsewhere tagged with an id the library no longer
    holds: MaterialLibrary.get() falls back to Default for unknown ids, so
    they would render as Default with no error while document_codec still
    wrote the dangling id to disk.

    `affected_count` is readable before do() so the confirmation dialog can
    quote it and the user can still cancel.

    Caveat: Model.traverse() walks from `root` through instances, so a
    Definition that exists but is instanced nowhere is invisible to this
    scan. Closing that gap needs a model-wide definition registry.
    """

    name = "Delete Material"

    def __init__(self, library, material_id: int, model) -> None:
        if material_id == 0:
            raise ValueError("the Default material cannot be deleted")
        self._lib = library
        self._mid = material_id
        self._model = model
        # Eager snapshot: affected_count must be quotable before do() so the
        # confirmation dialog can offer a real number and still be cancelled.
        self._affected = self._snapshot()
        self._record = None
        self._index = -1

    def _snapshot(self) -> list[tuple[object, int, Side]]:
        """Every (definition, face, side) painted with this material, model-wide.

        traverse() yields a definition once per instance of it, so a
        definition instanced N times would otherwise be scanned -- and
        counted -- N times. Dedupe by definition id.
        """
        seen: set[int] = set()
        found: list[tuple[object, int, Side]] = []
        for definition, _world in self._model.traverse():
            if definition.id in seen:
                continue
            seen.add(definition.id)
            for f_id, side in definition.mesh.faces_with_material(self._mid):
                found.append((definition, f_id, side))
        return found

    @property
    def affected_count(self) -> int:
        return len(self._affected)

    def do(self, model) -> None:
        # Re-snapshot rather than trusting the construction-time list. The
        # confirmation dialog runs between __init__ and do(); it is modal
        # today, but this makes the synchronicity contract enforced instead
        # of assumed -- whatever the geometry actually looks like now is what
        # gets cleared, and undo() therefore restores exactly that. (An
        # assertion was the alternative; it would turn a currently-impossible
        # race into a crash in the user's session rather than fixing it, and
        # it would also be wrong on redo, where the set can legitimately have
        # grown since the first do().)
        self._affected = self._snapshot()
        self._index = self._lib.index_of(self._mid)
        self._record = self._lib.remove(self._mid)
        for definition, f_id, side in self._affected:
            definition.mesh.clear_face_material(f_id, side)

    def undo(self, model) -> None:
        self._lib.restore(self._record, self._index)
        for definition, f_id, side in self._affected:
            definition.mesh.set_face_material(f_id, self._mid, side)


class AddTextureCommand(Command):
    """Add an image asset to the texture library."""

    name = "Add Texture"

    def __init__(
        self,
        library,
        name: str,
        data: bytes,
        image_format: str,
        width: int,
        height: int,
        has_transparency: bool,
    ) -> None:
        self._lib = library
        self._args = (name, data, image_format, width, height, has_transparency)
        self._record = None

    @property
    def texture_id(self) -> int | None:
        """The id this command created; None before do()."""
        return None if self._record is None else self._record.id

    def do(self, scene) -> None:
        if self._record is None:
            self._record = self._lib.add(*self._args)
        else:
            # Redo must reuse the SAME id, or every material pointing at the
            # old one is left dangling.
            self._lib.restore(self._record, len(self._lib.textures()))

    def undo(self, scene) -> None:
        self._record = self._lib.remove(self._record.id)


class SetMaterialTextureCommand(Command):
    """Point a material at a texture, or clear it, with its real-world size."""

    name = "Set Material Texture"

    def __init__(
        self,
        materials,
        material_id: int,
        texture_id: int | None,
        texture_size: tuple[float, float] | None = None,
    ) -> None:
        self._materials = materials
        self._mid = material_id
        self._new = (texture_id, texture_size)
        before = materials.get(material_id)
        self._before = (before.texture_id, before.texture_size)

    def do(self, scene) -> None:
        tid, size = self._new
        fields = {"texture_id": tid}
        if size is not None:
            fields["texture_size"] = size
        elif tid is None:
            fields["texture_size"] = (1.0, 1.0)
        self._materials.edit(self._mid, **fields)

    def undo(self, scene) -> None:
        tid, size = self._before
        self._materials.edit(self._mid, texture_id=tid, texture_size=size)


class DeleteTextureCommand(Command):
    """Remove a texture and clear every material that referenced it.

    References live on materials, not faces, so the scan is over the whole
    MaterialLibrary. M7.5a's DeleteMaterialCommand originally scanned a single
    Scene and left dangling ids elsewhere that survived save and load; this is
    the same hazard one level up.
    """

    name = "Delete Texture"

    def __init__(self, textures, materials, texture_id: int) -> None:
        self._textures = textures
        self._materials = materials
        self._tid = texture_id
        self._affected = [m.id for m in materials.materials() if m.texture_id == texture_id]
        self._record = None
        self._index = -1

    @property
    def affected_material_count(self) -> int:
        """How many materials lose their texture. Readable before do()."""
        return len(self._affected)

    def do(self, scene) -> None:
        self._index = self._textures.index_of(self._tid)
        self._record = self._textures.remove(self._tid)
        for mid in self._affected:
            self._materials.edit(mid, texture_id=None)

    def undo(self, scene) -> None:
        self._textures.restore(self._record, self._index)
        for mid in self._affected:
            self._materials.edit(mid, texture_id=self._tid)


class SetFacePlacementCommand(Command):
    """Adjust one face side's texture placement."""

    name = "Set Face Texture Placement"

    def __init__(self, face_id: int, placement: TexturePlacement, side: Side = Side.FRONT) -> None:
        self._fid = face_id
        self._side = side
        self._new = placement
        self._before: TexturePlacement | None = None

    def do(self, scene) -> None:
        if self._before is None:
            self._before = scene.face_placement(self._fid, self._side)
        scene.set_face_placement(self._fid, self._new, self._side)

    def undo(self, scene) -> None:
        scene.set_face_placement(self._fid, self._before, self._side)


class ResetFaceUvsCommand(Command):
    """Drop a face's stored UVs so it follows the plane projection again.

    Clears the stored array ONLY and never the placement (spec D11): the
    projection is the base and the placement is a separate layer composed on
    top, so clearing both under one name would be two actions in one control.
    """

    name = "Reset to Projection"

    def __init__(self, face_id: int, side: Side = Side.FRONT) -> None:
        self._fid = int(face_id)
        self._side = side
        self._before: np.ndarray | None = None

    def do(self, scene) -> None:
        self._before = scene.face_uvs(self._fid, self._side)
        scene.clear_face_uvs(self._fid, self._side)

    def undo(self, scene) -> None:
        if self._before is not None:
            scene.set_face_uvs(self._fid, self._before, self._side)
