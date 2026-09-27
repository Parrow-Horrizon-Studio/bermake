"""The Entity Info page renders a summary and emits edit intents (M7.3 Task 13)."""

from __future__ import annotations

import numpy as np
from bermake.model.entity_info import EntitySummary
from bermake.model.material import MaterialLibrary
from bermake.model.tag import TagLibrary
from bermake.ui.entity_info_page import EntityInfoPage
from bermake.units import UnitSystem, Units


def _page(qtbot):
    page = EntityInfoPage()
    qtbot.addWidget(page)
    return page


def _refresh(page, summary):
    page.refresh(summary, Units(), TagLibrary(), MaterialLibrary())


def _square(window):
    scene = window._model.active_context.mesh
    v = [
        scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32)),
    ]
    scene.add_face_from_loop(v)


def _two_squares(window):
    # Two disjoint quads (no shared vertices/edges) so face_ids has length 2
    # -- long enough to tell a CompositeCommand apart from N separate pushes.
    scene = window._model.active_context.mesh

    def quad(x0):
        return [
            scene.add_vertex(np.array([x0 + 0.0, 0.0, 0.0], dtype=np.float32)),
            scene.add_vertex(np.array([x0 + 1.0, 0.0, 0.0], dtype=np.float32)),
            scene.add_vertex(np.array([x0 + 1.0, 1.0, 0.0], dtype=np.float32)),
            scene.add_vertex(np.array([x0 + 0.0, 1.0, 0.0], dtype=np.float32)),
        ]

    scene.add_face_from_loop(quad(0.0))
    scene.add_face_from_loop(quad(2.0))


def test_nothing_selected_reports_so(qtbot):
    page = _page(qtbot)
    _refresh(page, EntitySummary(kind="Nothing", count=0))
    assert "Nothing" in page.kind_text()


def test_every_field_is_disabled_with_no_selection(qtbot):
    # Disabled rather than hidden: an entry that keeps its position is
    # learnable, one that reflows between selections is not.
    page = _page(qtbot)
    _refresh(page, EntitySummary(kind="Nothing", count=0))
    assert not page.name_field().isEnabled()
    assert not page.hidden_field().isEnabled()
    assert not page.tag_field().isEnabled()
    assert not page.material_field().isEnabled()


def test_disabled_fields_stay_present_not_removed(qtbot):
    # "Disabled, not hidden": every isEnabled()-is-False assertion above is
    # satisfied just as well by removeRow()/setVisible(False), since a
    # widget removed from the layout also reports isEnabled() == False (Qt
    # defaults) or simply isn't there to ask. Nothing before this test
    # actually proves the field is still IN the form. isHidden() reports the
    # widget's own explicit hidden flag (set only by hide()/setVisible(False))
    # independent of ancestor visibility, so it stays meaningful even though
    # this offscreen-platform page is never shown -- unlike isVisible(),
    # which would read False regardless just because the top-level window
    # isn't shown. Pairing it with parentWidget() rules out removeRow(),
    # which (without takeRow()) deletes the widget outright.
    page = _page(qtbot)
    _refresh(page, EntitySummary(kind="Nothing", count=0))

    for field in (page.name_field(), page.hidden_field(), page.tag_field(), page.material_field()):
        assert not field.isEnabled()
        assert not field.isHidden()
        assert field.parentWidget() is page


def test_a_single_group_populates_name_and_counts(qtbot):
    page = _page(qtbot)
    _refresh(
        page,
        EntitySummary(
            kind="Group",
            count=1,
            name="North",
            definition_name="Wall",
            child_count=0,
            edge_count=4,
            face_count=1,
            size=(2.0, 1.0, 3.0),
            tag_id=0,
            hidden=False,
        ),
    )

    assert page.name_field().text() == "North"
    assert page.name_field().isEnabled()
    assert "Wall" in page.definition_text()
    assert "4" in page.counts_text()


def test_dimensions_are_formatted_with_the_document_units(qtbot):
    page = _page(qtbot)
    _refresh(page, EntitySummary(kind="Group", count=1, size=(2.0, 1.0, 3.0)))
    text = page.size_text()
    assert "2 m" in text and "1 m" in text and "3 m" in text


def test_switching_units_relabels_the_panel_with_no_model_involvement(qtbot):
    # Spec line 190: entity_summary returns numbers, never formatted strings;
    # formatting happens in the widget via format_length/format_area, so
    # switching metric<->imperial re-labels with the SAME summary object --
    # no re-fetch from the model needed. Prove it by refreshing twice with
    # one summary instance and two different Units.
    page = _page(qtbot)
    summary = EntitySummary(kind="Group", count=1, size=(2.0, 1.0, 3.0))

    page.refresh(summary, Units(), TagLibrary(), MaterialLibrary())
    metric_text = page.size_text()
    assert "2 m" in metric_text and "1 m" in metric_text and "3 m" in metric_text

    imperial_units = Units(system=UnitSystem.IMPERIAL)
    page.refresh(summary, imperial_units, TagLibrary(), MaterialLibrary())
    imperial_text = page.size_text()

    assert imperial_text != metric_text
    # format_length's imperial branch renders feet/inches with ' and " marks;
    # each of the three sizes (2m, 1m, 3m) is at least a foot, so every one
    # of them carries a foot mark.
    assert imperial_text.count("'") == 3


def test_area_is_formatted_for_a_face(qtbot):
    page = _page(qtbot)
    _refresh(page, EntitySummary(kind="Face", count=1, area=4.0, material_id=0))
    assert "4.00 m²" in page.measure_text()


def test_length_is_formatted_for_an_edge(qtbot):
    page = _page(qtbot)
    _refresh(page, EntitySummary(kind="Edge", count=1, length=2.5))
    assert "2.5 m" in page.measure_text()


def test_the_name_field_is_disabled_for_a_multi_selection(qtbot):
    page = _page(qtbot)
    _refresh(
        page,
        EntitySummary(kind="Group", count=2, name=None, hidden=False, instances_only=True),
    )
    assert not page.name_field().isEnabled()
    assert page.hidden_field().isEnabled()


def test_a_mixed_hidden_state_leaves_the_checkbox_partially_checked(qtbot):
    from PySide6.QtCore import Qt

    page = _page(qtbot)
    _refresh(page, EntitySummary(kind="Group", count=2, hidden=None))
    assert page.hidden_field().checkState() == Qt.CheckState.PartiallyChecked


def test_committing_a_name_emits_the_intent(qtbot):
    page = _page(qtbot)
    _refresh(page, EntitySummary(kind="Group", count=1, name="Old", hidden=False))

    page.name_field().setText("New")
    with qtbot.waitSignal(page.rename_requested) as blocker:
        page.name_field().editingFinished.emit()

    assert blocker.args == ["New"]


def test_refreshing_does_not_emit(qtbot):
    # refresh() runs on every selection change; it must not look like a user
    # edit or the panel would fire commands at itself.
    page = _page(qtbot)
    seen = []
    page.rename_requested.connect(seen.append)
    page.hidden_requested.connect(seen.append)
    page.tag_requested.connect(seen.append)
    page.material_requested.connect(seen.append)

    _refresh(page, EntitySummary(kind="Group", count=1, name="A", tag_id=0, hidden=True))
    _refresh(page, EntitySummary(kind="Group", count=1, name="B", tag_id=0, hidden=False))

    assert seen == []


def test_toggling_hidden_emits_the_intent(qtbot):
    page = _page(qtbot)
    _refresh(page, EntitySummary(kind="Group", count=1, hidden=False, instances_only=True))

    with qtbot.waitSignal(page.hidden_requested) as blocker:
        page.hidden_field().click()

    assert blocker.args == [True]


def test_the_page_is_installed_and_renames_through_a_command(main_window):
    _square(main_window)
    main_window._on_select_all()
    main_window._on_make_group()
    instance = main_window._model.root.children[-1]
    main_window._selection.replace(instances=[instance.id])
    main_window._refresh_selection_status()

    main_window._entity_info_page.rename_requested.emit("Renamed")

    assert instance.name == "Renamed"
    main_window._command_stack.undo()
    assert instance.name == ""


def test_a_no_op_entity_rename_restores_the_pages_canonical_name(main_window):
    # No command runs on a no-op commit, so nothing used to repopulate the
    # page -- the Name field kept whatever raw text (padding whitespace
    # included) the user had just typed.
    _square(main_window)
    main_window._on_select_all()
    main_window._on_make_group()
    instance = main_window._model.root.children[-1]
    main_window._selection.replace(instances=[instance.id])
    main_window._refresh_selection_status()
    main_window._entity_info_page.rename_requested.emit("Wall")
    assert instance.name == "Wall"

    main_window._entity_info_page.name_field().setText("Wall  ")
    main_window._entity_info_page.rename_requested.emit("Wall  ")

    assert main_window._entity_info_page.name_field().text() == "Wall"


def test_hiding_from_the_page_goes_through_a_command(main_window):
    _square(main_window)
    main_window._on_select_all()
    main_window._on_make_group()
    instance = main_window._model.root.children[-1]
    main_window._selection.replace(instances=[instance.id])
    main_window._refresh_selection_status()

    main_window._entity_info_page.hidden_requested.emit(True)

    assert instance.hidden is True
    main_window._command_stack.undo()
    assert instance.hidden is False


def test_assigning_a_tag_from_the_page_goes_through_a_command(main_window):
    _square(main_window)
    main_window._on_select_all()
    main_window._on_make_group()
    instance = main_window._model.root.children[-1]
    main_window._selection.replace(instances=[instance.id])
    main_window._refresh_selection_status()
    tag = main_window._model.tags.add("Walls")

    main_window._entity_info_page.tag_requested.emit(tag.id)

    assert instance.tag_id == tag.id
    main_window._command_stack.undo()
    assert instance.tag_id == 0


def test_a_multi_face_selections_area_renders_as_unavailable_not_zero(qtbot):
    # Task 6: area is None (not summed) whenever more than one face is
    # selected. The panel must show that as unavailable -- distinct from a
    # real "0.00 m2" reading -- not silently coerce None into zero.
    page = _page(qtbot)
    _refresh(page, EntitySummary(kind="Face", count=2, area=None, material_id=None))
    text = page.measure_text()
    assert "0" not in text
    assert text != ""


def test_a_disagreeing_material_across_faces_shows_no_selection(qtbot):
    # Task 6: material_id collapses to None when a multi-face selection
    # disagrees. The combo must not fall back to showing "Default" (id 0)
    # as if every face agreed on it -- that would misreport the selection.
    page = _page(qtbot)
    _refresh(page, EntitySummary(kind="Face", count=2, area=None, material_id=None))
    assert page.material_field().currentIndex() == -1
    assert page.material_field().isEnabled()


def test_an_annotation_selection_disables_hidden_and_tag(qtbot):
    # The annotation-kind branch of entity_summary (Label/Dimension) was
    # noted as untested in Task 6. Hidden and Tag have no meaning for an
    # annotation -- entity_summary leaves both None -- so this page must
    # disable rather than leave them clickable with a stale value.
    page = _page(qtbot)
    _refresh(page, EntitySummary(kind="Label", count=1))
    assert not page.hidden_field().isEnabled()
    assert not page.tag_field().isEnabled()
    assert not page.name_field().isEnabled()


def test_a_mixed_group_and_component_selection_enables_tag_and_hidden(qtbot):
    # A same-type (instances-only) but heterogeneous selection -- a Group and
    # a Component together -- reports kind="Mixed" just like a cross-type one
    # does, but both underlying commands (TagInstancesCommand,
    # HideInstancesCommand) accept it and the right-click Hide entry is
    # enabled for it too (spec 1.4: "any instance selection"). instances_only
    # is what lets this page tell the two "Mixed" cases apart.
    page = _page(qtbot)
    _refresh(page, EntitySummary(kind="Mixed", count=2, hidden=None, instances_only=True))
    assert page.hidden_field().isEnabled()
    assert page.tag_field().isEnabled()


def test_a_cross_type_mixed_selection_still_disables_tag_and_hidden(qtbot):
    # Instances plus faces (say) is the OTHER "Mixed": instances_only is
    # False here (the default), so Tag and Hidden must stay disabled -- there
    # is no coherent instance to apply either to.
    page = _page(qtbot)
    _refresh(page, EntitySummary(kind="Mixed", count=2))
    assert not page.hidden_field().isEnabled()
    assert not page.tag_field().isEnabled()


def test_painting_from_the_page_goes_through_one_undoable_command(main_window):
    # A multi-face selection must compose into a SINGLE undo step, not one
    # per face -- PaintFaceCommand paints exactly one face. Two disjoint
    # faces are required to discriminate: with only one face, a broken
    # implementation that pushes one PaintFaceCommand per face (no
    # CompositeCommand at all) grows the stack by 1 too -- identical to the
    # correct behavior. With two faces the two decompositions diverge: a
    # per-face pusher grows the stack by 2, and a single undo() afterward
    # would only pop the most-recently-pushed command, reverting just the
    # second face and leaving the first one painted.
    _two_squares(main_window)
    scene = main_window._model.active_context.mesh
    face_ids = [f.id for f in scene.faces_iter()]
    assert len(face_ids) == 2
    main_window._selection.replace(faces=face_ids)
    main_window._refresh_selection_status()
    material = main_window._model.materials.add_custom("Brick", (0.6, 0.3, 0.2))
    depth = len(main_window._command_stack._undo)

    main_window._entity_info_page.material_requested.emit(material.id)

    assert scene.face_material(face_ids[0]) == material.id
    assert scene.face_material(face_ids[1]) == material.id
    assert len(main_window._command_stack._undo) == depth + 1

    main_window._command_stack.undo()

    assert scene.face_material(face_ids[0]) == 0
    assert scene.face_material(face_ids[1]) == 0


def test_the_outliner_eye_toggle_keeps_entity_info_in_sync(main_window):
    # A hide command does not touch the selection, so _refresh_selection_status
    # (which only fires on selection changes) never runs -- Entity Info's
    # checkbox went stale while the Outliner's own row updated correctly.
    # No explicit _refresh_entity_info() call here: that is what would mask
    # the bug this test exists to catch.
    from PySide6.QtCore import Qt

    _square(main_window)
    main_window._on_select_all()
    main_window._on_make_group()
    instance = main_window._model.root.children[-1]
    main_window._selection.replace(instances=[instance.id])
    main_window._refresh_selection_status()
    assert main_window._entity_info_page.hidden_field().checkState() == Qt.CheckState.Unchecked

    main_window._outliner.hide_toggled.emit(instance.id, True)

    assert instance.hidden is True
    assert main_window._entity_info_page.hidden_field().checkState() == Qt.CheckState.Checked


def test_the_edit_hide_action_keeps_entity_info_in_sync(main_window):
    from PySide6.QtCore import Qt

    _square(main_window)
    main_window._on_select_all()
    main_window._on_make_group()
    instance = main_window._model.root.children[-1]
    main_window._selection.replace(instances=[instance.id])
    main_window._refresh_selection_status()
    assert main_window._entity_info_page.hidden_field().checkState() == Qt.CheckState.Unchecked

    main_window._on_hide()

    assert instance.hidden is True
    assert main_window._entity_info_page.hidden_field().checkState() == Qt.CheckState.Checked


def test_the_edit_unhide_action_keeps_entity_info_in_sync(main_window):
    from PySide6.QtCore import Qt

    _square(main_window)
    main_window._on_select_all()
    main_window._on_make_group()
    instance = main_window._model.root.children[-1]
    main_window._selection.replace(instances=[instance.id])
    main_window._refresh_selection_status()
    main_window._on_hide()
    assert main_window._entity_info_page.hidden_field().checkState() == Qt.CheckState.Checked

    main_window._on_unhide()

    assert instance.hidden is False
    assert main_window._entity_info_page.hidden_field().checkState() == Qt.CheckState.Unchecked


def test_undoing_a_hide_keeps_entity_info_in_sync(main_window):
    from PySide6.QtCore import Qt

    _square(main_window)
    main_window._on_select_all()
    main_window._on_make_group()
    instance = main_window._model.root.children[-1]
    main_window._selection.replace(instances=[instance.id])
    main_window._refresh_selection_status()
    main_window._on_hide()
    assert main_window._entity_info_page.hidden_field().checkState() == Qt.CheckState.Checked

    main_window._command_stack.undo()

    assert instance.hidden is False
    assert main_window._entity_info_page.hidden_field().checkState() == Qt.CheckState.Unchecked
