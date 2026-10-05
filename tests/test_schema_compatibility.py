"""Files testers already have must keep opening in later versions (#134, D4).

The v0.14.0 hard break made every real file `format: "bermake"` with schema 9,
and the loader accepts anything from MIN_SCHEMA_VERSION up to SCHEMA_VERSION.
The compatibility story rests on one frozen file, written by the v0.16.0 save
path and never regenerated (tools/make_schema9_fixture.py records what is in
it). These tests prove three things about it:

1. it still loads, with every feature it carries intact;
2. it still loads after the schema version moves on, which is the test that
   fails if someone tightens the gate from `>` to `!=`;
3. the migration registry a future non-additive change will use is wired in.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from bermake.io import bermake_file
from bermake.io.bermake_file import (
    MIN_SCHEMA_VERSION,
    SCHEMA_VERSION,
    load_document,
)
from bermake.io.errors import BermakeFormatError, BermakeVersionError
from bermake.model.annotation import Dimension, Guide, Label
from bermake.scene.scene import Side, TexturePlacement
from bermake.units import Units
from bermake.viewport.environment import LANDSCAPE
from bermake.viewport.render_style import FaceStyle

DATA = Path(__file__).parent / "data"
FIXTURE = DATA / "berm" / "v0.16.0-schema9.berm"


def _rewrite_manifest(tmp_path: Path, **manifest_fields) -> Path:
    """Copy the fixture with some manifest fields replaced, leaving the rest as is."""
    out = tmp_path / "rewritten.berm"
    with zipfile.ZipFile(FIXTURE) as zin, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            payload = zin.read(item.filename)
            if item.filename == "manifest.json":
                manifest = json.loads(payload)
                manifest.update(manifest_fields)
                payload = json.dumps(manifest).encode()
            zout.writestr(item.filename, payload)
    return out


def _material_named(model, name):
    return next(m for m in model.materials.materials() if m.name == name)


def test_the_schema_version_and_floor_are_pinned():
    """The one place that pins the numbers.

    Bumping SCHEMA_VERSION is a deliberate act: it means adding a frozen
    fixture for the new version and, if the change is not additive, a
    migration. Raising MIN_SCHEMA_VERSION drops files testers already hold and
    needs an explicit decision.
    """
    assert SCHEMA_VERSION == 9
    assert MIN_SCHEMA_VERSION == 9


def test_every_registered_migration_sits_between_the_floor_and_the_current_version():
    for n in bermake_file._MIGRATIONS:
        assert MIN_SCHEMA_VERSION <= n < SCHEMA_VERSION


def test_the_fixture_is_what_it_claims_to_be():
    with zipfile.ZipFile(FIXTURE) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        names = zf.namelist()
    assert manifest == {"format": "bermake", "schema_version": 9, "app_version": "0.16.0"}
    assert "thumbnail.png" in names
    assert "textures/1.png" in names


def test_the_fixture_loads_with_every_feature_intact():
    loaded = load_document(FIXTURE)
    model = loaded.model

    # Document settings: units, camera, render style, environment.
    assert loaded.units == Units(metric_unit="cm", metric_precision=2)
    assert loaded.camera_state.position == (10.0, -8.0, 6.0)
    assert loaded.camera_state.target == (2.0, 2.0, 0.5)
    assert loaded.camera_state.fov_y_deg == 50.0
    assert loaded.style.face_style is FaceStyle.MONOCHROME
    assert loaded.style.xray is True
    assert loaded.style.color_by_tag is True
    assert loaded.environment == LANDSCAPE

    # Custom materials, one textured; the texture blob is the checker image.
    oak = _material_named(model, "Oak")
    slate = _material_named(model, "Slate")
    assert oak.base_color == (0.76, 0.60, 0.42)
    assert oak.texture_size == (2.0, 3.0)
    texture = model.textures.get(oak.texture_id)
    assert texture is not None
    assert (texture.name, texture.image_format, texture.width, texture.height) == (
        "uv_checker.png",
        "png",
        256,
        256,
    )
    assert texture.data == (DATA / "uv_checker.png").read_bytes()

    # Root floor: paint, stored UVs and texture placement on both sides.
    root_mesh = model.root.mesh
    faces = list(root_mesh.faces_iter())
    assert len(faces) == 1
    floor = faces[0].id
    assert root_mesh.face_material(floor, Side.FRONT) == oak.id
    assert root_mesh.face_material(floor, Side.BACK) == slate.id
    np.testing.assert_allclose(
        root_mesh.face_uvs(floor, Side.FRONT),
        [(0.0, 0.0), (0.5, 0.0), (0.5, 0.75), (0.0, 0.75)],
    )
    np.testing.assert_allclose(
        root_mesh.face_uvs(floor, Side.BACK),
        [(1.0, 0.0), (0.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
    )
    assert root_mesh.face_placement(floor, Side.FRONT) == TexturePlacement(0.25, -0.5, 2.0, 0.5)
    assert root_mesh.face_placement(floor, Side.BACK) == TexturePlacement(scale=3.0)

    # Tags.
    assert [t.name for t in model.tags.tags()] == ["Untagged", "Walls", "Furniture"]
    walls, furniture = model.tags.tags()[1], model.tags.tags()[2]
    assert (walls.color, furniture.color) == ((0.9, 0.25, 0.25), (0.95, 0.6, 0.15))

    # Nested group.
    outer_inst, chair_a, chair_b = model.root.children
    assert outer_inst.name == "North Wing"
    assert outer_inst.tag_id == walls.id
    assert outer_inst.definition.is_group is True
    assert outer_inst.definition.name == "Outer"
    assert len(list(outer_inst.definition.mesh.faces_iter())) == 1
    assert outer_inst.transform[:3, 3].tolist() == [5.0, 0.0, 0.0]
    (inner_inst,) = outer_inst.definition.children
    assert inner_inst.definition.name == "Inner"
    assert inner_inst.definition.is_group is True
    assert inner_inst.name == "Inner box"
    assert inner_inst.hidden is True
    assert len(list(inner_inst.definition.mesh.faces_iter())) == 1

    # Component with two instances of one shared definition.
    assert chair_a.definition is chair_b.definition
    assert chair_a.definition.name == "Chair"
    assert chair_a.definition.is_group is False
    assert chair_a.tag_id == 0
    assert chair_b.tag_id == furniture.id
    assert chair_a.transform[:3, 3].tolist() == [1.0, 1.0, 0.0]
    assert chair_b.transform[:3, 3].tolist() == [2.0, 1.0, 0.0]

    # Annotations: one dimension, one label, one guide.
    dimension, label, guide = model.root.annotations
    assert dimension == Dimension(0, (0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, -0.5, 0.0))
    assert label == Label(1, (4.0, 4.0, 0.0), (4.5, 4.5, 0.5), "Floor slab")
    assert guide == Guide(2, (2.0, 2.0, 0.0), (0.0, 0.0, 1.0))

    # Scene.
    (view,) = model.views.views()
    assert view.name == "Iso"
    assert view.camera.position == (9.0, -9.0, 7.0)
    assert view.camera.fov_y_deg == 40.0
    assert view.camera.target == (2.0, 2.0, 0.5)
    assert view.tag_visibility == {furniture.id: False}
    assert view.face_style == "HIDDEN_LINE"
    assert view.xray is True
    assert view.color_by_tag is True


def test_the_fixture_still_loads_after_the_schema_version_moves_on(monkeypatch):
    """A schema 9 file must open in a build whose SCHEMA_VERSION is 10.

    This is what fails if someone turns the version gate into `!=`, or tightens
    the floor to the current version. It has to keep passing for every release.
    """
    monkeypatch.setattr(bermake_file, "SCHEMA_VERSION", SCHEMA_VERSION + 1)
    loaded = load_document(FIXTURE)
    assert len(loaded.model.root.children) == 3
    assert loaded.environment == LANDSCAPE


def test_a_file_below_the_floor_is_rejected(tmp_path):
    path = _rewrite_manifest(tmp_path, schema_version=MIN_SCHEMA_VERSION - 1)
    with pytest.raises(BermakeFormatError) as excinfo:
        load_document(path)
    message = str(excinfo.value)
    assert str(MIN_SCHEMA_VERSION - 1) in message
    assert "corrupt" in message
    assert "not a Bermake file" in message


def test_a_file_one_version_newer_than_supported_is_rejected(tmp_path):
    path = _rewrite_manifest(tmp_path, schema_version=SCHEMA_VERSION + 1)
    with pytest.raises(BermakeVersionError) as excinfo:
        load_document(path)
    assert str(excinfo.value) == (
        f"file schema_version {SCHEMA_VERSION + 1} is newer than supported ({SCHEMA_VERSION})"
    )


def test_a_registered_migration_runs_once_and_its_output_reaches_the_codec(monkeypatch):
    calls = []

    def migrate_9(data):
        calls.append(sorted(data))
        return {**data, "migration_marker": 9}

    captured = []
    real_from_dict = bermake_file.document_from_dict

    def spy(data, blobs=None):
        captured.append(data)
        return real_from_dict(data, blobs)

    monkeypatch.setattr(bermake_file, "SCHEMA_VERSION", 10)
    monkeypatch.setattr(bermake_file, "_MIGRATIONS", {9: migrate_9})
    monkeypatch.setattr(bermake_file, "document_from_dict", spy)

    load_document(FIXTURE)

    assert len(calls) == 1
    assert "migration_marker" not in calls[0]  # it received the decoded file, unmigrated
    assert len(captured) == 1
    assert captured[0]["migration_marker"] == 9
    assert "model" in captured[0]  # the rest of the document came through


def test_migrations_apply_in_order_from_the_files_own_version(monkeypatch, tmp_path):
    order = []

    def step(n):
        def migrate(data):
            order.append(n)
            return {**data, f"after_{n}": True}

        return migrate

    captured = []
    real_from_dict = bermake_file.document_from_dict

    def spy(data, blobs=None):
        captured.append(data)
        return real_from_dict(data, blobs)

    monkeypatch.setattr(bermake_file, "SCHEMA_VERSION", 11)
    monkeypatch.setattr(bermake_file, "_MIGRATIONS", {9: step(9), 10: step(10)})
    monkeypatch.setattr(bermake_file, "document_from_dict", spy)

    load_document(FIXTURE)
    assert order == [9, 10]
    assert captured[-1]["after_9"] is True
    assert captured[-1]["after_10"] is True

    # A file already at schema 10 skips the 9 to 10 step.
    order.clear()
    load_document(_rewrite_manifest(tmp_path, schema_version=10))
    assert order == [10]


def test_a_version_with_no_migration_is_an_additive_bump(monkeypatch):
    """No entry for a version means its change was additive; the file passes through."""
    monkeypatch.setattr(bermake_file, "SCHEMA_VERSION", 11)
    monkeypatch.setattr(bermake_file, "_MIGRATIONS", {10: lambda data: data})
    assert len(load_document(FIXTURE).model.root.children) == 3


def test_a_migration_that_chokes_on_the_document_reports_a_bad_file(monkeypatch):
    def migrate_9(data):
        return {k: v for k, v in data.items() if k != "model"} | {"x": data["no_such_key"]}

    monkeypatch.setattr(bermake_file, "SCHEMA_VERSION", 10)
    monkeypatch.setattr(bermake_file, "_MIGRATIONS", {9: migrate_9})
    with pytest.raises(BermakeFormatError):
        load_document(FIXTURE)
