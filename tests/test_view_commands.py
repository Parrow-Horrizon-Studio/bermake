from bermake.commands.command_stack import CommandStack
from bermake.commands.view_commands import (
    CreateViewCommand,
    DeleteViewCommand,
    RenameViewCommand,
    ReorderViewCommand,
    UpdateViewCommand,
)
from bermake.io.document_codec import CameraState
from bermake.model.model import Model
from bermake.views.saved_view import SavedView


def _view(vid, name="V", fov=45.0):
    cam = CameraState(
        position=(float(vid), 0.0, 0.0),
        target=(0.0, 0.0, 0.0),
        up=(0.0, 0.0, 1.0),
        fov_y_deg=fov,
    )
    return SavedView(vid, name, cam, {}, "SHADED", False)


def _names(model):
    return [v.name for v in model.views.views()]


def test_create_do_undo_redo():
    model = Model()
    stack = CommandStack()
    view = _view(model.views.next_id, "A")
    stack.execute(CreateViewCommand(view), model)
    assert _names(model) == ["A"]
    stack.undo()
    assert _names(model) == []
    stack.redo()
    assert _names(model) == ["A"]
    assert model.views.get(0) is view   # same object re-attached on redo


def test_delete_restores_at_original_index():
    model = Model()
    stack = CommandStack()
    for i, n in enumerate("ABC"):
        model.views.add(_view(i, n))
    stack.execute(DeleteViewCommand(1), model)   # delete "B" (middle)
    assert _names(model) == ["A", "C"]
    stack.undo()
    assert _names(model) == ["A", "B", "C"]       # back in the middle
    stack.redo()
    assert _names(model) == ["A", "C"]


def test_rename_undo_restores_true_original():
    model = Model()
    stack = CommandStack()
    model.views.add(_view(0, "Old"))
    cmd = RenameViewCommand(0, "New")
    stack.execute(cmd, model)
    assert model.views.get(0).name == "New"
    # Mutate live state between do and undo — undo must still restore "Old":
    model.views.rename(0, "Externally Changed")
    stack.undo()
    assert model.views.get(0).name == "Old"


def test_reorder_do_undo():
    model = Model()
    stack = CommandStack()
    for i, n in enumerate("ABC"):
        model.views.add(_view(i, n))
    stack.execute(ReorderViewCommand(0, +1), model)  # move "A" down
    assert _names(model) == ["B", "A", "C"]
    stack.undo()
    assert _names(model) == ["A", "B", "C"]


def test_reorder_clamped_move_is_a_noop_on_undo():
    model = Model()
    stack = CommandStack()
    model.views.add(_view(0, "A"))
    model.views.add(_view(1, "B"))
    stack.execute(ReorderViewCommand(0, -1), model)  # "A" already first → no move
    assert _names(model) == ["A", "B"]
    stack.undo()                                      # must NOT move anything
    assert _names(model) == ["A", "B"]


def test_update_undo_restores_prior_snapshot():
    model = Model()
    stack = CommandStack()
    model.views.add(_view(0, "V", fov=30.0))
    new_view = _view(0, "V", fov=90.0)               # same id/name, new camera
    cmd = UpdateViewCommand(0, new_view)
    stack.execute(cmd, model)
    assert model.views.get(0).camera.fov_y_deg == 90.0
    # Mutate live between do and undo — undo restores the fov=30 snapshot:
    model.views.replace_view(0, _view(0, "V", fov=12.0))
    stack.undo()
    assert model.views.get(0).camera.fov_y_deg == 30.0


def test_delete_first_do_capture_survives_redo():
    model = Model()
    stack = CommandStack()
    for i, n in enumerate("ABC"):
        model.views.add(_view(i, n))
    cmd = DeleteViewCommand(1)                     # delete "B" (middle)
    stack.execute(cmd, model)
    assert _names(model) == ["A", "C"]
    stack.undo()                                   # "B" back at index 1
    assert _names(model) == ["A", "B", "C"]
    model.views.replace_view(1, _view(1, "B-MUTATED"))   # external change at id 1
    stack.redo()                                   # do() again: guard must skip recapture
    assert _names(model) == ["A", "C"]
    stack.undo()                                   # restores the ORIGINAL "B", not "B-MUTATED"
    assert model.views.get(1).name == "B"


def test_rename_first_do_capture_survives_redo():
    model = Model()
    stack = CommandStack()
    model.views.add(_view(0, "Old"))
    cmd = RenameViewCommand(0, "New")
    stack.execute(cmd, model)
    assert model.views.get(0).name == "New"
    stack.undo()
    assert model.views.get(0).name == "Old"
    model.views.rename(0, "Externally Changed")    # external change between undo and redo
    stack.redo()                                   # do() again: guard must skip recapture
    assert model.views.get(0).name == "New"
    stack.undo()                                   # restores the true original "Old"
    assert model.views.get(0).name == "Old"


def test_update_first_do_capture_survives_redo():
    model = Model()
    stack = CommandStack()
    model.views.add(_view(0, "V", fov=30.0))
    new_view = _view(0, "V", fov=90.0)
    cmd = UpdateViewCommand(0, new_view)
    stack.execute(cmd, model)
    assert model.views.get(0).camera.fov_y_deg == 90.0
    stack.undo()
    assert model.views.get(0).camera.fov_y_deg == 30.0
    model.views.replace_view(0, _view(0, "V", fov=12.0))   # external change
    stack.redo()                                   # do() again: guard must skip recapture
    assert model.views.get(0).camera.fov_y_deg == 90.0
    stack.undo()                                   # restores the true original fov 30
    assert model.views.get(0).camera.fov_y_deg == 30.0
