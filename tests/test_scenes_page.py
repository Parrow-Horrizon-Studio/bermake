from bermake.io.document_codec import CameraState
from bermake.ui.scenes_page import ScenesPage
from bermake.views.saved_view import SavedView
from bermake.views.view_library import ViewLibrary


def _cam():
    return CameraState(position=(1.0, 0.0, 0.0), target=(0.0, 0.0, 0.0),
                       up=(0.0, 0.0, 1.0), fov_y_deg=45.0)


def _lib(names=("Front", "Top")):
    lib = ViewLibrary()
    for i, n in enumerate(names):
        lib.add(SavedView(i, n, _cam(), {}, "SHADED", False))
    return lib


def test_lists_scene_names_with_ids(qtbot):
    dock = ScenesPage(_lib(), None)
    qtbot.addWidget(dock)
    assert dock._list.count() == 2
    from PySide6.QtCore import Qt
    assert dock._list.item(0).text() == "Front"
    assert int(dock._list.item(0).data(Qt.ItemDataRole.UserRole)) == 0


def test_add_button_emits_create(qtbot):
    dock = ScenesPage(_lib(), None)
    qtbot.addWidget(dock)
    with qtbot.waitSignal(dock.create_requested, timeout=500):
        dock._add_btn.click()


def test_delete_emits_selected_id(qtbot):
    dock = ScenesPage(_lib(), None)
    qtbot.addWidget(dock)
    dock._list.setCurrentRow(1)                 # select "Top" (id 1)
    with qtbot.waitSignal(dock.delete_requested, timeout=500) as blocker:
        dock._delete_btn.click()
    assert blocker.args == [1]


def test_reorder_buttons_emit_direction(qtbot):
    dock = ScenesPage(_lib(), None)
    qtbot.addWidget(dock)
    dock._list.setCurrentRow(0)
    with qtbot.waitSignal(dock.reorder_requested, timeout=500) as down:
        dock._down_btn.click()
    assert down.args == [0, 1]
    with qtbot.waitSignal(dock.reorder_requested, timeout=500) as up:
        dock._up_btn.click()
    assert up.args == [0, -1]


def test_click_recalls(qtbot):
    dock = ScenesPage(_lib(), None)
    qtbot.addWidget(dock)
    item = dock._list.item(1)
    with qtbot.waitSignal(dock.recall_requested, timeout=500) as blocker:
        dock._list.itemClicked.emit(item)
    assert blocker.args == [1]


def test_double_click_rename_emits(qtbot):
    dock = ScenesPage(_lib(), None)
    qtbot.addWidget(dock)
    item = dock._list.item(0)
    with qtbot.waitSignal(dock.rename_requested, timeout=500) as blocker:
        item.setText("Renamed")                 # triggers itemChanged
    assert blocker.args == [0, "Renamed"]


def test_buttons_disabled_when_empty(qtbot):
    dock = ScenesPage(ViewLibrary(), None)
    qtbot.addWidget(dock)
    assert dock._add_btn.isEnabled()
    assert not dock._delete_btn.isEnabled()
    assert not dock._update_btn.isEnabled()
    assert not dock._up_btn.isEnabled()
    assert not dock._down_btn.isEnabled()


def test_set_library_rebinds(qtbot):
    dock = ScenesPage(_lib(("A",)), None)
    qtbot.addWidget(dock)
    assert dock._list.count() == 1
    dock.set_library(_lib(("X", "Y", "Z")))
    assert dock._list.count() == 3
    assert dock._list.item(0).text() == "X"
