"""#118: map_Kd options never leak into the texture filename."""

from __future__ import annotations

import pytest
from bermake.io.obj_codec import _map_kd_filename, parse_obj


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ("-s 2 2 1 brick.png", "brick.png"),
        ("-o 0.5 0.5 0 brick.png", "brick.png"),
        ("-clamp on brick.png", "brick.png"),
        ("-imfchan r brick.png", "brick.png"),
        ("-type sphere brick.png", "brick.png"),
        ("-blendu off -blendv off brick.png", "brick.png"),
        ("-s 2 brick.png", "brick.png"),
        ("-s 2 3 brick.png", "brick.png"),
        ("-o 0.5 my brick.png", "my brick.png"),
        ("-mm 0 1 brick.png", "brick.png"),
        ("-bm 1 brick.png", "brick.png"),
        ("-cc on -texres 256 brick.png", "brick.png"),
        ("-unknownflag 3 brick.png", "brick.png"),
        ("-s 1 1 1 -clamp on -type sphere my brick.png", "my brick.png"),
        ("brick.png", "brick.png"),
        ("textures/brick.png", "textures/brick.png"),
    ],
)
def test_map_kd_options_never_reach_the_filename(args, expected):
    assert _map_kd_filename(["map_Kd", *args.split()]) == expected


def test_a_flag_with_no_filename_after_it_yields_none():
    assert _map_kd_filename(["map_Kd", "-clamp", "on"]) is None
    assert _map_kd_filename(["map_Kd"]) is None


def test_word_valued_options_survive_a_full_mtl_parse():
    mtl = "newmtl brick\nmap_Kd -clamp on -blendu off brick.png\n"
    doc = parse_obj("v 0 0 0\n", mtl)
    assert doc.material_textures == {"brick": "brick.png"}
