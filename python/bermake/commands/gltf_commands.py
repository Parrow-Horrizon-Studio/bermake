"""ImportGltfCommand (M6c): undoable wrapper around build_gltf_into_model."""

from __future__ import annotations

from bermake.commands.command import Command
from bermake.io.gltf_import import build_gltf_into_model


class ImportGltfCommand(Command):
    """Import a GltfSceneData into the model, undoably. do() builds the wrapped
    group; undo() detaches the single wrapper instance (the whole subtree becomes
    unreachable). Materials added to the library are NOT undone (parity with
    ImportObjCommand)."""

    name = "Import glTF"

    def __init__(
        self, scene, target_context, root_name="glTF", texture_bytes=None, decoder=None
    ) -> None:
        self._scene = scene
        self._target = target_context
        self._root_name = root_name
        self._texture_bytes = texture_bytes
        self._decoder = decoder
        self._result = None
        self.summary = None

    def do(self, model) -> None:
        self._result = build_gltf_into_model(
            self._scene,
            model,
            self._target,
            root_name=self._root_name,
            texture_bytes=self._texture_bytes,
            decoder=self._decoder,
        )
        self.summary = self._result.summary

    def undo(self, model) -> None:
        if self._result is None:
            return
        root = self._result.root_instance
        if root in self._target.children:
            self._target.children.remove(root)
        model.revalidate_active_path()
        self._result = None
