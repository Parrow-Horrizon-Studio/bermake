"""Write tests/data/berm/v0.16.0-schema9.berm, the frozen schema 9 compatibility file.

FROZEN. This script produced the fixture once, from the v0.16.0 save path, and
it must never be rerun: the whole point of the file is that it was written by
the code that shipped to testers, so regenerating it with newer code would make
the compatibility test compare the loader against itself. A future schema gets
a new fixture beside this one (tests/data/berm/v<release>-schema<n>.berm) and
the old file stays.

The script is kept for the record of exactly what the fixture contains.
tests/test_schema_compatibility.py asserts every feature listed here.

Usage (once, ever): .venv/Scripts/python tools/make_schema9_fixture.py
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
from bermake.document import DocumentSettings
from bermake.io.bermake_file import save_document
from bermake.io.document_codec import CameraState
from bermake.model.annotation import Dimension, Guide, Label
from bermake.model.model import Model
from bermake.scene.scene import Side, TexturePlacement
from bermake.units import Units
from bermake.viewport.camera import Camera
from bermake.viewport.environment import LANDSCAPE
from bermake.viewport.render_style import FaceStyle, RenderStyle
from bermake.views.saved_view import SavedView

ROOT = Path(__file__).resolve().parent.parent
TEXTURE_SOURCE = ROOT / "tests" / "data" / "uv_checker.png"
OUTPUT = ROOT / "tests" / "data" / "berm" / "v0.16.0-schema9.berm"


def _png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """A tiny solid-colour PNG, used only as the document's thumbnail entry."""

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    row = b"\x00" + bytes(rgb) * width
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(row * height))
        + chunk(b"IEND", b"")
    )


def _quad(scene, x: float, y: float, z: float, size: float) -> int:
    ids = [
        scene.add_vertex(np.array(p, dtype=np.float32))
        for p in (
            (x, y, z),
            (x + size, y, z),
            (x + size, y + size, z),
            (x, y + size, z),
        )
    ]
    return scene.add_face_from_loop(ids)


def _translation(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[:3, 3] = (x, y, z)
    return m


def build_model() -> Model:
    model = Model()

    # Custom materials: Oak carries the texture, Slate paints the back side.
    texture = model.textures.add(
        "uv_checker.png", TEXTURE_SOURCE.read_bytes(), "png", 256, 256, False
    )
    oak = model.materials.add_custom("Oak", (0.76, 0.60, 0.42))
    model.materials.edit(oak.id, texture_id=texture.id, texture_size=(2.0, 3.0))
    slate = model.materials.add_custom("Slate", (0.30, 0.34, 0.38))

    # Root floor: front Oak (textured), back Slate, stored per-corner UVs and a
    # texture placement on both sides.
    floor = _quad(model.root.mesh, 0.0, 0.0, 0.0, 4.0)
    model.root.mesh.set_face_material(floor, oak.id, Side.FRONT)
    model.root.mesh.set_face_material(floor, slate.id, Side.BACK)
    model.root.mesh.set_face_uvs(
        floor, [(0.0, 0.0), (0.5, 0.0), (0.5, 0.75), (0.0, 0.75)], Side.FRONT
    )
    model.root.mesh.set_face_placement(floor, TexturePlacement(0.25, -0.5, 2.0, 0.5), Side.FRONT)
    model.root.mesh.set_face_uvs(floor, [(1.0, 0.0), (0.0, 0.0), (0.0, 1.0), (1.0, 1.0)], Side.BACK)
    model.root.mesh.set_face_placement(floor, TexturePlacement(scale=3.0), Side.BACK)

    # Two tags.
    walls = model.tags.add("Walls")
    furniture = model.tags.add("Furniture")

    # Nested group: an outer group holding an inner group, each with geometry.
    outer = model.new_definition("Outer", is_group=True)
    _quad(outer.mesh, 0.0, 0.0, 1.0, 2.0)
    inner = model.new_definition("Inner", is_group=True)
    _quad(inner.mesh, 0.0, 0.0, 2.0, 1.0)
    inner_inst = model.new_instance(inner, _translation(0.5, 0.5, 0.0))
    inner_inst.name = "Inner box"
    inner_inst.hidden = True
    outer.children.append(inner_inst)
    outer_inst = model.new_instance(outer, _translation(5.0, 0.0, 0.0))
    outer_inst.tag_id = walls.id
    outer_inst.name = "North Wing"
    model.root.children.append(outer_inst)

    # Component with two instances sharing one definition.
    chair = model.new_definition("Chair", is_group=False)
    _quad(chair.mesh, 0.0, 0.0, 0.0, 0.5)
    chair_a = model.new_instance(chair, _translation(1.0, 1.0, 0.0))
    chair_b = model.new_instance(chair, _translation(2.0, 1.0, 0.0))
    chair_b.tag_id = furniture.id
    model.root.children.append(chair_a)
    model.root.children.append(chair_b)

    # Annotations: one dimension, one label, one guide.
    model.root.annotations.append(
        Dimension(model.new_annotation_id(), (0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, -0.5, 0.0))
    )
    model.root.annotations.append(
        Label(model.new_annotation_id(), (4.0, 4.0, 0.0), (4.5, 4.5, 0.5), "Floor slab")
    )
    model.root.annotations.append(
        Guide(model.new_annotation_id(), (2.0, 2.0, 0.0), (0.0, 0.0, 1.0))
    )

    # One saved scene.
    model.views.add(
        SavedView(
            0,
            "Iso",
            CameraState(
                position=(9.0, -9.0, 7.0),
                target=(2.0, 2.0, 0.5),
                up=(0.0, 0.0, 1.0),
                fov_y_deg=40.0,
            ),
            {furniture.id: False},
            "HIDDEN_LINE",
            True,
            color_by_tag=True,
        )
    )
    return model


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(
            f"{OUTPUT} already exists; the fixture is frozen and must not be rewritten"
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    camera = Camera()
    camera.position = np.array([10.0, -8.0, 6.0], dtype=np.float32)
    camera.target = np.array([2.0, 2.0, 0.5], dtype=np.float32)
    camera.fov_y_deg = 50.0

    doc = DocumentSettings()
    doc.set_units(Units(metric_unit="cm", metric_precision=2))
    doc.set_environment(LANDSCAPE)

    style = RenderStyle(face_style=FaceStyle.MONOCHROME, xray=True, color_by_tag=True)

    save_document(OUTPUT, build_model(), camera, doc, style, thumbnail=_png(8, 8, (120, 160, 200)))
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
