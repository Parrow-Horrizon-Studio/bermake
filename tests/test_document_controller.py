from bermake.model.model import Model
from bermake.ui.document_controller import DocumentController


def test_controller_dirty_transitions_and_title():
    c = DocumentController()
    assert c.current_path is None
    assert c.dirty is False
    assert c.display_title() == "Untitled - Bermake"

    c.mark_dirty()
    assert c.display_title() == "Untitled* - Bermake"

    c.set_path("/tmp/house.berm")
    c.mark_clean()
    assert c.display_title() == "house.berm - Bermake"
    c.mark_dirty()
    assert c.display_title() == "house.berm* - Bermake"


def test_model_load_from_keeps_identity_swaps_contents():
    target = Model()
    target_id = id(target)
    other = Model()
    g = other.new_definition("Grp", is_group=True)
    inst = other.new_instance(g)
    other.root.children.append(inst)

    target.load_from(other)
    assert id(target) == target_id  # same object
    assert target.root is other.root  # contents swapped
    assert target.materials is other.materials
    assert target.tags is other.tags
    assert target.active_path == []


# --- M7.12 Task 6: a recovered document's title -------------------------------


def test_a_recovered_document_is_titled_with_its_recovered_name():
    c = DocumentController()
    c.recovered_name = "House.berm"
    c.mark_dirty()
    assert c.display_title() == "House.berm (recovered)* - Bermake"
    c.mark_clean()
    assert c.display_title() == "House.berm (recovered) - Bermake"


def test_saving_under_a_real_path_clears_the_recovered_name():
    c = DocumentController()
    c.recovered_name = "House.berm"
    c.set_path(None)
    assert c.recovered_name == "House.berm"
    c.set_path("/tmp/elsewhere.berm")
    assert c.recovered_name is None
    assert c.display_title() == "elsewhere.berm - Bermake"


def test_a_new_controller_has_no_recovered_name():
    assert DocumentController().recovered_name is None
