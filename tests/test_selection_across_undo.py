import numpy as np


def _win(qtbot):
    from bermake.ui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def test_selection_survives_transform_undo(qtbot):
    from bermake.commands.instance_commands import TransformInstanceCommand

    win = _win(qtbot)
    model = win._model
    defn = model.new_definition("Box", is_group=True)
    inst = model.new_instance(defn, np.eye(4, dtype=np.float64))
    model.root.children.append(inst)

    win._selection.replace(instances={inst.id})
    assert inst.id in win._selection.instances

    move = np.eye(4, dtype=np.float64)
    move[0, 3] = 1.0
    # TransformInstanceCommand takes a single instance, not a list (confirmed
    # against python/bermake/commands/instance_commands.py's real signature).
    win._command_stack.execute(TransformInstanceCommand(inst, move), model)
    win._command_stack.undo()

    # The instance still exists, so it must still be selected.
    assert inst.id in win._selection.instances


def test_selection_drops_entities_the_undo_destroyed(qtbot):
    win = _win(qtbot)
    model = win._model
    defn = model.new_definition("Box", is_group=True)
    inst = model.new_instance(defn, np.eye(4, dtype=np.float64))
    model.root.children.append(inst)
    win._selection.replace(instances={inst.id})

    # Simulate the entity going away, then the post-undo hook running.
    model.root.children.remove(inst)
    win._on_after_undo_redo()

    assert inst.id not in win._selection.instances
