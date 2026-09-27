from bermake.selection import Selection


def test_selection_tracks_instances():
    s = Selection()
    s.replace(instances=[3, 5])
    assert s.instances == {3, 5}
    assert not s.is_empty()
    s.toggle_instance(3)
    assert s.instances == {5}
    s.clear()
    assert s.is_empty()
    assert s.counts() == (0, 0, 0, 0, 0)
