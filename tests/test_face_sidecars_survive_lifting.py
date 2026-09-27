"""Lifting geometry between definitions must carry its face sidecars (#114).

Make Group moves faces from a parent Scene into a fresh child Scene, and
Explode moves them back. Both replay the geometry structurally through
`add_face_from_loop`, which mints a brand new face id in a different Scene, so
nothing carries `_face_materials_*`, `_face_placements_*` or `_face_uvs_*`
across unless something copies them.

This is the third and fourth site in the family: `Scene.split_edge` was fixed
in M7.5c stage 1, and `HalfEdgeMesh::dissolve_edge` is deliberately left alone
(design decision D4a: a merge has two candidate sources per sidecar and no
defensible winner). Unlike dissolve, the mapping here is one to one, so a
verbatim copy is unambiguous.

Every test below paints TWO faces DIFFERENTLY. A fixture that paints one face,
or paints both the same, passes against an implementation that crosses the
mapping, and this milestone has had four separate cases where a fixture's
incidental symmetry hid the distinction the test existed to prove.
"""

import numpy as np

from bermake.commands.command_stack import CommandStack
from bermake.commands.explode_command import ExplodeInstanceCommand
from bermake.commands.group_commands import MakeGroupCommand
from bermake.commands.instance_lifecycle_commands import MakeUniqueCommand
from bermake.model.model import Model
from bermake.scene.scene import Side, TexturePlacement

# Two quads, side by side in the XY plane, sharing no vertices.
_QUAD_A = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
_QUAD_B = [(2, 0, 0), (3, 0, 0), (3, 1, 0), (2, 1, 0)]

# Distinct per corner AND distinct between the two faces, so a crossed mapping
# or a rotated loop is a visible failure rather than a coincidence.
_UVS_A = [(0.10, 0.20), (0.30, 0.25), (0.35, 0.45), (0.15, 0.40)]
_UVS_B = [(0.60, 0.70), (0.80, 0.75), (0.85, 0.95), (0.65, 0.90)]

_PLACE_A = TexturePlacement(0.25, 0.50, 2.0, 30.0)
_PLACE_B = TexturePlacement(-0.75, 0.125, 0.5, 210.0)


def _painted_pair(scene, model):
    """Two differently painted quads. Returns (face ids, vertex ids, material ids)."""
    red = model.materials.add_custom("Red", (1.0, 0.0, 0.0))
    blue = model.materials.add_custom("Blue", (0.0, 0.0, 1.0))

    vids = []
    fids = []
    for corners in (_QUAD_A, _QUAD_B):
        ids = [scene.add_vertex(np.array(p, dtype=np.float32)) for p in corners]
        vids.extend(ids)
        fids.append(scene.add_face_from_loop(ids))

    scene.set_face_material(fids[0], red.id)
    scene.set_face_material(fids[1], blue.id)
    scene.set_face_placement(fids[0], _PLACE_A, Side.FRONT)
    scene.set_face_placement(fids[1], _PLACE_B, Side.FRONT)
    scene.set_face_uvs(fids[0], _UVS_A, Side.FRONT)
    scene.set_face_uvs(fids[1], _UVS_B, Side.FRONT)
    # One BACK-side entry, because the sidecars are per (face, side) and a fix
    # that loops only over FRONT would otherwise pass everything here.
    scene.set_face_material(fids[1], red.id, Side.BACK)
    scene.set_face_uvs(fids[0], _UVS_B, Side.BACK)
    return fids, vids, (red.id, blue.id)


def _faces_by_first_corner_x(scene):
    """Face ids keyed by the x of their loop's first vertex, so a test can name
    which quad it is looking at without depending on face id ordering."""
    out = {}
    for f in scene.faces_iter():
        first = scene.vertex(f.loop_vertex_ids[0]).position
        out[round(float(first[0]))] = f.id
    return out


def _assert_pair_intact(scene, mat_ids):
    """Both quads still carry exactly what they were painted with."""
    red, blue = mat_ids
    by_x = _faces_by_first_corner_x(scene)
    assert set(by_x) == {0, 2}, f"expected the two quads, got {sorted(by_x)}"
    fa, fb = by_x[0], by_x[2]

    assert scene.face_material(fa) == red
    assert scene.face_material(fb) == blue
    assert scene.face_placement(fa, Side.FRONT) == _PLACE_A
    assert scene.face_placement(fb, Side.FRONT) == _PLACE_B
    np.testing.assert_allclose(scene.face_uvs(fa, Side.FRONT), _UVS_A, atol=1e-6)
    np.testing.assert_allclose(scene.face_uvs(fb, Side.FRONT), _UVS_B, atol=1e-6)

    assert scene.face_material(fb, Side.BACK) == red, "the BACK side must transfer too"
    np.testing.assert_allclose(scene.face_uvs(fa, Side.BACK), _UVS_B, atol=1e-6)


def test_make_group_carries_the_face_sidecars_into_the_new_definition():
    model = Model()
    fids, vids, mat_ids = _painted_pair(model.root.mesh, model)

    CommandStack().execute(MakeGroupCommand(model.root, vids, [], fids), model)

    child = model.root.children[-1].definition.mesh
    _assert_pair_intact(child, mat_ids)


def test_explode_carries_the_face_sidecars_back_into_the_parent():
    """The inverse lift, and the same gap. Flagged in #114 as worth checking in
    the same pass rather than being found later as a fourth instance."""
    model = Model()
    defn = model.new_definition("G", is_group=True)
    _fids, _vids, mat_ids = _painted_pair(defn.mesh, model)
    inst = model.new_instance(defn)
    model.root.children.append(inst)

    CommandStack().execute(ExplodeInstanceCommand(model.root, inst), model)

    _assert_pair_intact(model.root.mesh, mat_ids)


def test_group_then_explode_round_trips_the_paint():
    """The composed operation a user actually performs. Neither half may lose
    anything, and a fix to only one of them fails here."""
    model = Model()
    fids, vids, mat_ids = _painted_pair(model.root.mesh, model)
    stack = CommandStack()

    stack.execute(MakeGroupCommand(model.root, vids, [], fids), model)
    inst = model.root.children[-1]
    stack.execute(ExplodeInstanceCommand(model.root, inst), model)

    _assert_pair_intact(model.root.mesh, mat_ids)


def test_undoing_a_group_restores_the_parents_paint():
    """Undo restores the ORIGINAL face ids, whose sidecar entries were never
    removed (design decision D5: entries on dead faces are left alone because
    ids are never reused, which is exactly what makes undo work). This pins
    that the fix does not break that by deleting the source entries."""
    model = Model()
    fids, vids, mat_ids = _painted_pair(model.root.mesh, model)
    stack = CommandStack()

    stack.execute(MakeGroupCommand(model.root, vids, [], fids), model)
    assert stack.undo()

    _assert_pair_intact(model.root.mesh, mat_ids)


def test_clone_definition_deep_copies_the_face_sidecars():
    """`clone_definition` promises "deep-copy a definition's geometry", and a
    face's paint is part of what a user means by that. Found by sweeping every
    `add_face_from_loop` call site while fixing #114, rather than as a fifth
    instance later."""
    model = Model()
    defn = model.new_definition("Comp", is_group=False)
    _fids, _vids, mat_ids = _painted_pair(defn.mesh, model)

    clone = model.clone_definition(defn)

    _assert_pair_intact(clone.mesh, mat_ids)


def test_make_unique_keeps_the_paint_on_the_copy():
    """The user-visible half of the above. Make Unique means "give me my own
    copy of this component", and a copy that comes back unpainted is the same
    failure as #114 wearing a different name."""
    model = Model()
    defn = model.new_definition("Comp", is_group=False)
    _fids, _vids, mat_ids = _painted_pair(defn.mesh, model)
    first = model.new_instance(defn)
    second = model.new_instance(defn)
    model.root.children.extend([first, second])

    CommandStack().execute(MakeUniqueCommand(second), model)

    assert second.definition is not defn, "the instance should now have its own definition"
    _assert_pair_intact(second.definition.mesh, mat_ids)
    _assert_pair_intact(defn.mesh, mat_ids)  # the original must be untouched


def test_an_unpainted_face_gains_no_sidecar_entries():
    """Discriminates against copying a default rather than copying what is
    there. Writing Default explicitly would make every grouped face look
    painted to `faces_with_material`, which drives the delete-material count."""
    model = Model()
    scene = model.root.mesh
    ids = [scene.add_vertex(np.array(p, dtype=np.float32)) for p in _QUAD_A]
    fid = scene.add_face_from_loop(ids)

    CommandStack().execute(MakeGroupCommand(model.root, ids, [], [fid]), model)

    child = model.root.children[-1].definition.mesh
    cf = next(f.id for f in child.faces_iter())
    for material in model.materials.materials():
        assert child.faces_with_material(material.id) == []
    assert child.faces_with_uvs() == []
    assert child.faces_with_placement() == []
    assert child.face_uvs(cf, Side.FRONT) is None
