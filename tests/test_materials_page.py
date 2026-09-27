from __future__ import annotations

import pytest
from bermake.model.material import MaterialLibrary
from bermake.ui.materials_page import MaterialsPage


@pytest.fixture
def lib():
    return MaterialLibrary()


def test_page_builds_a_swatch_per_material(qtbot, lib):
    dock = MaterialsPage(lib)
    qtbot.addWidget(dock)
    assert len(dock._buttons) == len(lib.materials())


def test_pick_changes_active_and_emits(qtbot, lib):
    dock = MaterialsPage(lib)
    qtbot.addWidget(dock)
    brick = next(m for m in lib.materials() if m.name == "Brick Red")
    with qtbot.waitSignal(dock.active_material_changed, timeout=500) as blocker:
        dock._on_pick(brick.id)
    assert dock.active_material_id == brick.id
    assert blocker.args[0].id == brick.id


def test_set_active_highlights(qtbot, lib):
    dock = MaterialsPage(lib)
    qtbot.addWidget(dock)
    forest = next(m for m in lib.materials() if m.name == "Forest Green")
    dock.set_active(forest.id)
    assert dock.active_material_id == forest.id


def test_add_custom_then_rebuild_grows_grid(qtbot, lib):
    dock = MaterialsPage(lib)
    qtbot.addWidget(dock)
    n = len(dock._buttons)
    mat = lib.add_custom("#abcdef", (0.67, 0.80, 0.94))
    dock._rebuild_swatches()
    assert len(dock._buttons) == n + 1
    assert mat.id in dock._buttons
