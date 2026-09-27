from bermake.io.document_codec import CameraState
from bermake.views.saved_view import SavedView
from bermake.views.view_library import ViewLibrary


def _view(vid, name="V"):
    cam = CameraState(position=(float(vid), 0.0, 0.0), target=(0.0, 0.0, 0.0),
                      up=(0.0, 0.0, 1.0), fov_y_deg=45.0)
    return SavedView(vid, name, cam, {vid: False}, "SHADED", False)


def test_add_advances_next_id_past_the_view():
    lib = ViewLibrary()
    assert lib.next_id == 0
    lib.add(_view(0))
    assert lib.next_id == 1
    lib.add(_view(5))
    assert lib.next_id == 6  # jumps past the highest id seen


def test_get_and_index_of():
    lib = ViewLibrary()
    lib.add(_view(0, "A"))
    lib.add(_view(1, "B"))
    assert lib.get(1).name == "B"
    assert lib.get(99) is None
    assert lib.index_of(1) == 1
    assert lib.index_of(99) == -1


def test_remove_then_insert_restores_position():
    lib = ViewLibrary()
    a, b, c = _view(0, "A"), _view(1, "B"), _view(2, "C")
    lib.add(a); lib.add(b); lib.add(c)
    lib.remove(1)
    assert [v.name for v in lib.views()] == ["A", "C"]
    lib.insert(1, b)
    assert [v.name for v in lib.views()] == ["A", "B", "C"]


def test_rename_replaces_frozen_view():
    lib = ViewLibrary()
    lib.add(_view(0, "Old"))
    lib.rename(0, "New")
    assert lib.get(0).name == "New"
    lib.rename(0, "")  # empty rejected
    assert lib.get(0).name == "New"


def test_move_up_swaps_returns_true():
    lib = ViewLibrary()
    lib.add(_view(0, "A")); lib.add(_view(1, "B")); lib.add(_view(2, "C"))
    assert lib.move(1, -1) is True          # move "B" up one
    assert [v.name for v in lib.views()] == ["B", "A", "C"]


def test_move_clamps_at_ends_returns_false():
    lib = ViewLibrary()
    lib.add(_view(0, "A")); lib.add(_view(1, "B"))
    assert lib.move(0, -1) is False         # "A" already first
    assert lib.move(1, +1) is False         # "B" already last
    assert [v.name for v in lib.views()] == ["A", "B"]  # unchanged


def test_records_round_trip_preserves_order_and_next_id():
    lib = ViewLibrary()
    lib.add(_view(0, "A")); lib.add(_view(3, "B"))
    records = lib.to_records()
    assert records[0]["name"] == "A"
    assert records[0]["tag_visibility"] == {"0": False}  # keys stringified for JSON
    rebuilt = ViewLibrary.from_records(records, lib.next_id)
    assert [v.name for v in rebuilt.views()] == ["A", "B"]
    assert rebuilt.get(0).tag_visibility == {0: False}   # keys back to int
    assert rebuilt.next_id == 4


def test_color_by_tag_round_trips_through_records():
    # A Scene's job is reproducing a view, so every RenderStyle field it
    # captures has to survive persistence. color_by_tag was written nowhere
    # (the final-review finding), so a Scene saved in the mode came back
    # without it.
    cam = CameraState(position=(0.0, 0.0, 0.0), target=(0.0, 0.0, 0.0),
                      up=(0.0, 0.0, 1.0), fov_y_deg=45.0)
    lib = ViewLibrary()
    lib.add(SavedView(0, "Tagged", cam, {}, "SHADED", False, True))

    records = lib.to_records()
    assert records[0]["color_by_tag"] is True

    rebuilt = ViewLibrary.from_records(records, lib.next_id)
    assert rebuilt.get(0).color_by_tag is True


def test_a_record_without_color_by_tag_still_loads():
    # Every Scene written before M7.5a lacks the key and must still open.
    records = [
        {
            "id": 0,
            "name": "Old",
            "camera": {"position": [0.0, 0.0, 0.0], "target": [0.0, 0.0, 0.0],
                       "up": [0.0, 0.0, 1.0], "fov_y_deg": 45.0},
            "tag_visibility": {},
            "face_style": "SHADED",
            "xray": False,
        }
    ]
    lib = ViewLibrary.from_records(records, 1)
    assert lib.get(0).color_by_tag is False
