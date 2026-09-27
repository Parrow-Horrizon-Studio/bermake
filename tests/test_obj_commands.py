# tests/test_obj_commands.py
from bermake.commands.obj_commands import ImportObjCommand
from bermake.io.obj_codec import ObjDocument, ObjFace, ObjObject
from bermake.model.model import Model


def _grouped_doc():
    return ObjDocument(
        vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        objects=(ObjObject("A", (ObjFace((0, 1, 2), None),)),),
        materials={},
        has_object_tags=True,
    )


def _merge_doc():
    return ObjDocument(
        vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        objects=(ObjObject("default", (ObjFace((0, 1, 2), None),)),),
        materials={},
        has_object_tags=False,
    )


def test_import_command_group_do_undo_redo():
    model = Model()
    cmd = ImportObjCommand(_grouped_doc(), model.active_context)
    cmd.do(model)
    assert cmd.summary.faces_imported == 1
    assert len(model.active_context.children) == 1
    cmd.undo(model)
    assert len(model.active_context.children) == 0        # group removed
    cmd.do(model)                                          # redo
    assert len(model.active_context.children) == 1


def test_import_command_merge_do_undo():
    model = Model()
    ctx = model.active_context
    cmd = ImportObjCommand(_merge_doc(), ctx)
    cmd.do(model)
    assert len(list(ctx.mesh.faces_iter())) == 1
    assert len(list(ctx.mesh.vertices_iter())) == 3
    cmd.undo(model)
    assert len(list(ctx.mesh.faces_iter())) == 0          # geometry removed
    assert len(list(ctx.mesh.vertices_iter())) == 0
    assert len(list(ctx.mesh.edges_iter())) == 0          # fully locks the empty-mesh invariant
