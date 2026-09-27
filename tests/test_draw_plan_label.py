from __future__ import annotations

import numpy as np
import pytest
from bermake.annotations.draw_plan import (
    _ARROW_PX,
    _ARROW_SPREAD,
    _LANDING_PX,
    _TEXT_GAP_PX,
    plan_annotation,
)
from bermake.model.annotation import Label
from bermake.units import Units


class _FlatCamera:
    """Orthographic-ish stand-in: x,y map straight to pixels, z is depth."""

    def world_to_screen(self, world_xyz, width, height):
        x, y, z = float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2])
        if z < 0.0:
            return None
        return (100.0 + x * 10.0, 200.0 - y * 10.0, 1.0 + z)


def _plan(lab, camera=None):
    return plan_annotation(lab, np.eye(4), camera or _FlatCamera(), 640, 480, Units())


def test_label_plan_has_leader_landing_arrow_and_text():
    """Basic structure: leader + landing + 2 arrowhead strokes + text."""
    lab = Label(1, (0.0, 0.0, 0.0), (5.0, 3.0, 0.0), "Load-bearing")
    plan = _plan(lab)
    assert plan is not None
    assert len(plan.segments_px) == 4  # leader + landing + 2 arrows
    assert len(plan.texts) == 1
    assert plan.texts[0].text == "Load-bearing"


def test_landing_is_horizontal():
    """Exactly one segment is perfectly horizontal (the landing)."""
    lab = Label(1, (0.0, 0.0, 0.0), (5.0, 3.0, 0.0), "note")
    plan = _plan(lab)
    horiz = [s for s in plan.segments_px if abs(s[1] - s[3]) < 1e-9]
    assert len(horiz) == 1


def test_text_side_flips_when_text_is_left_of_the_anchor():
    """Text to the right of anchor aligns 'left'; to the left aligns 'right'."""
    right = _plan(Label(1, (0.0, 0.0, 0.0), (5.0, 3.0, 0.0), "note"))
    left = _plan(Label(2, (5.0, 0.0, 0.0), (0.0, 3.0, 0.0), "note"))
    assert right.texts[0].align == "left"
    assert left.texts[0].align == "right"


def test_label_hit_boxes_cover_text_and_leader():
    """At least text box and leader box."""
    lab = Label(1, (0.0, 0.0, 0.0), (5.0, 3.0, 0.0), "note")
    plan = _plan(lab)
    assert len(plan.hit_boxes) >= 2


def test_label_behind_camera_yields_no_plan():
    """If anchor or text is behind camera (z<0), return None."""

    class _Behind:
        def world_to_screen(self, world_xyz, width, height):
            return None

    assert _plan(Label(1, (0.0, 0.0, 0.0), (5.0, 3.0, 0.0), "n"), _Behind()) is None


def test_style_constants_pin_layout():
    """Pin the label layout constants so a stray edit is caught."""
    assert _LANDING_PX == pytest.approx(26.0)
    assert _ARROW_PX == pytest.approx(9.0)
    assert _ARROW_SPREAD == pytest.approx(0.42)


def test_label_geometry_axis_aligned_text_right():
    """Pin exact endpoints for an axis-aligned fixture with text to the right.

    Anchor at (0,0,0) -> screen (100,200)
    Text at (5,3,0) -> screen (150,170)
    Landing runs left from text end: (150-26, 170) = (124, 170)
    Leader from anchor to elbow: (100,200) to (124,170)
    Landing from elbow to text: (124,170) to (150,170)
    Arrowhead strokes from anchor in direction of leader.
    """
    lab = Label(1, (0.0, 0.0, 0.0), (5.0, 3.0, 0.0), "text")
    plan = _plan(lab)
    assert plan is not None

    leader, landing, arr1, arr2 = plan.segments_px

    anchor_px = np.array([100.0, 200.0])
    elbow_px = np.array([150.0 - _LANDING_PX, 170.0])
    text_px = np.array([150.0, 170.0])

    # leader from anchor to elbow
    assert leader == pytest.approx((float(anchor_px[0]), float(anchor_px[1]),
                                    float(elbow_px[0]), float(elbow_px[1])))

    # landing from elbow to text (horizontal)
    assert landing == pytest.approx((float(elbow_px[0]), float(elbow_px[1]),
                                     float(text_px[0]), float(text_px[1])))

    # arrowhead strokes symmetric about leader direction, tip at anchor
    direction = (elbow_px - anchor_px) / np.linalg.norm(elbow_px - anchor_px)
    for spread, arrow_seg in [(+_ARROW_SPREAD, arr1), (-_ARROW_SPREAD, arr2)]:
        c = float(np.cos(spread))
        s = float(np.sin(spread))
        rotated = np.array([direction[0] * c - direction[1] * s,
                            direction[0] * s + direction[1] * c])
        tail = anchor_px + rotated * _ARROW_PX
        expected = (float(anchor_px[0]), float(anchor_px[1]),
                    float(tail[0]), float(tail[1]))
        assert arrow_seg == pytest.approx(expected)


def test_label_geometry_axis_aligned_text_left():
    """Pin exact endpoints when text is to the LEFT of anchor.

    Anchor at (5,0,0) -> screen (150,200)
    Text at (0,3,0) -> screen (100,170)
    Landing runs right from text end: (100+26, 170) = (126, 170)
    Leader from anchor to elbow: (150,200) to (126,170)
    Landing from elbow to text: (126,170) to (100,170)
    """
    lab = Label(2, (5.0, 0.0, 0.0), (0.0, 3.0, 0.0), "text")
    plan = _plan(lab)
    assert plan is not None

    leader, landing, _, _ = plan.segments_px

    anchor_px = np.array([150.0, 200.0])
    text_px = np.array([100.0, 170.0])
    elbow_px = np.array([100.0 + _LANDING_PX, 170.0])

    # leader
    assert leader == pytest.approx((float(anchor_px[0]), float(anchor_px[1]),
                                    float(elbow_px[0]), float(elbow_px[1])))

    # landing
    assert landing == pytest.approx((float(elbow_px[0]), float(elbow_px[1]),
                                     float(text_px[0]), float(text_px[1])))


def test_label_geometry_diagonal():
    """Non-axis-aligned case: hardcoded 'always-right' logic must fail here.

    Anchor at (3,4,0) -> screen (130,160)
    Text at (8,1,0) -> screen (180,190)
    Text is to the right: landing extends left from text.
    """
    lab = Label(3, (3.0, 4.0, 0.0), (8.0, 1.0, 0.0), "corner")
    plan = _plan(lab)
    assert plan is not None

    leader, landing, _, _ = plan.segments_px

    anchor_px = np.array([130.0, 160.0])
    text_px = np.array([180.0, 190.0])
    # text is to the right; landing runs left
    elbow_px = np.array([180.0 - _LANDING_PX, 190.0])

    # landing is horizontal
    assert abs(landing[1] - landing[3]) < 1e-9

    # landing endpoints match expectations
    assert landing == pytest.approx((float(elbow_px[0]), float(elbow_px[1]),
                                     float(text_px[0]), float(text_px[1])))

    # leader connects anchor to elbow
    assert leader == pytest.approx((float(anchor_px[0]), float(anchor_px[1]),
                                    float(elbow_px[0]), float(elbow_px[1])))


def test_arrowhead_is_at_anchor_symmetric_and_sized():
    """Arrowhead strokes are symmetric about leader direction, tip at anchor."""
    lab = Label(1, (0.0, 0.0, 0.0), (5.0, 3.0, 0.0), "x")
    plan = _plan(lab)

    anchor_px = np.array([100.0, 200.0])
    elbow_px = np.array([150.0 - _LANDING_PX, 170.0])

    arr1, arr2 = plan.segments_px[2:4]

    # Both arrows start at the anchor
    assert arr1[:2] == pytest.approx((float(anchor_px[0]), float(anchor_px[1])))
    assert arr2[:2] == pytest.approx((float(anchor_px[0]), float(anchor_px[1])))

    # Both arrows have length _ARROW_PX
    for arrow_seg in [arr1, arr2]:
        dx = arrow_seg[2] - arrow_seg[0]
        dy = arrow_seg[3] - arrow_seg[1]
        length = float(np.sqrt(dx**2 + dy**2))
        assert length == pytest.approx(_ARROW_PX)

    # Arrows are symmetric about the leader direction
    direction = (elbow_px - anchor_px) / np.linalg.norm(elbow_px - anchor_px)

    tail1 = np.array([arr1[2], arr1[3]]) - anchor_px
    tail2 = np.array([arr2[2], arr2[3]]) - anchor_px

    # Angle between leader and each tail should be _ARROW_SPREAD
    cos1 = float(np.dot(tail1, direction)) / (np.linalg.norm(tail1) * np.linalg.norm(direction))
    cos2 = float(np.dot(tail2, direction)) / (np.linalg.norm(tail2) * np.linalg.norm(direction))

    expected_cos = float(np.cos(_ARROW_SPREAD))
    assert cos1 == pytest.approx(expected_cos)
    assert cos2 == pytest.approx(expected_cos)


def test_text_sits_on_landing_with_correct_align():
    """Text x/y position and align depend on side."""
    lab = Label(1, (0.0, 0.0, 0.0), (5.0, 3.0, 0.0), "label")
    plan = _plan(lab)

    text = plan.texts[0]
    text_px = np.array([150.0, 170.0])

    # Text x matches text position
    assert text.x == pytest.approx(float(text_px[0]))

    # Text y is slightly below the landing (landing is at 170.0)
    # Based on implementation: y = float(text_px[1]) - _TEXT_GAP_PX * 0.4
    expected_y = 170.0 - _TEXT_GAP_PX * 0.4
    assert text.y == pytest.approx(expected_y)

    # Text aligns left when on the right
    assert text.align == "left"

    # For the left case, align should be "right"
    lab_left = Label(2, (5.0, 0.0, 0.0), (0.0, 3.0, 0.0), "label")
    plan_left = _plan(lab_left)
    assert plan_left.texts[0].align == "right"


def test_landing_length_is_constant():
    """The landing segment always has length _LANDING_PX."""
    lab = Label(1, (0.0, 0.0, 0.0), (5.0, 3.0, 0.0), "x")
    plan = _plan(lab)

    landing = plan.segments_px[1]
    dx = landing[2] - landing[0]
    dy = landing[3] - landing[1]
    length = float(np.sqrt(dx**2 + dy**2))
    assert length == pytest.approx(_LANDING_PX)

    # Also test diagonal case
    lab_diag = Label(3, (3.0, 4.0, 0.0), (8.0, 1.0, 0.0), "x")
    plan_diag = _plan(lab_diag)
    landing_diag = plan_diag.segments_px[1]
    dx = landing_diag[2] - landing_diag[0]
    dy = landing_diag[3] - landing_diag[1]
    length = float(np.sqrt(dx**2 + dy**2))
    assert length == pytest.approx(_LANDING_PX)


def test_landing_elbow_position_with_hardcoded_values():
    """Hardcoded test to catch _LANDING_PX changes: elbow must be at (124, 170)."""
    lab = Label(1, (0.0, 0.0, 0.0), (5.0, 3.0, 0.0), "text")
    plan = _plan(lab)

    leader, landing, _, _ = plan.segments_px

    # With text at (150, 170) and _LANDING_PX=26, elbow must be at (124, 170)
    # This hardcoded value will fail if _LANDING_PX changes
    assert leader == pytest.approx((100.0, 200.0, 124.0, 170.0))
    assert landing == pytest.approx((124.0, 170.0, 150.0, 170.0))


def test_arrow_endpoints_with_hardcoded_values():
    """Hardcoded test to catch _ARROW_PX/_ARROW_SPREAD changes."""
    lab = Label(1, (0.0, 0.0, 0.0), (5.0, 3.0, 0.0), "x")
    plan = _plan(lab)

    _, _, arr1, arr2 = plan.segments_px

    # Both arrows start at anchor (100, 200)
    assert arr1[:2] == pytest.approx((100.0, 200.0))
    assert arr2[:2] == pytest.approx((100.0, 200.0))

    # Check arrow lengths are _ARROW_PX
    for arrow_seg in [arr1, arr2]:
        dx = arrow_seg[2] - arrow_seg[0]
        dy = arrow_seg[3] - arrow_seg[1]
        length = float(np.sqrt(dx**2 + dy**2))
        assert length == pytest.approx(_ARROW_PX)

    # Hardcoded arrow tail coordinates with _ARROW_SPREAD=0.42
    # These will fail if _ARROW_SPREAD changes
    assert arr1 == pytest.approx((100.0, 200.0, 108.0, 195.88), abs=0.1)
    assert arr2 == pytest.approx((100.0, 200.0, 102.27, 191.29), abs=0.1)


def test_label_geometry_with_non_identity_world_transform():
    """Non-identity world_transform must be applied to produce screen geometry.

    Tests that both translation and scale in the world_transform are applied
    to the anchor and text positions before projection. This catches regressions
    where world_transform handling is partially or completely removed.

    Transform: 2x scale + translation by (1, 1, 0)
    Anchor (0,0,0) local → (1,1,0) world → screen (110,190)
    Text (5,3,0) local → (11,7,0) world → screen (210,130)
    """
    transform = np.array([
        [2.0, 0.0, 0.0, 1.0],
        [0.0, 2.0, 0.0, 1.0],
        [0.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)

    lab = Label(1, (0.0, 0.0, 0.0), (5.0, 3.0, 0.0), "scaled")
    plan = plan_annotation(lab, transform, _FlatCamera(), 640, 480, Units())
    assert plan is not None

    leader, landing, _, _ = plan.segments_px

    # With the transform, world coordinates differ from local coordinates
    # anchor_w = (1, 1, 0), text_w = (11, 7, 0)
    # After projection: anchor_px = (110, 190), text_px = (210, 130)
    anchor_px_expected = np.array([110.0, 190.0])
    text_px_expected = np.array([210.0, 130.0])
    elbow_px_expected = np.array([210.0 - _LANDING_PX, 130.0])

    # Leader connects anchor to elbow
    assert leader == pytest.approx((
        float(anchor_px_expected[0]), float(anchor_px_expected[1]),
        float(elbow_px_expected[0]), float(elbow_px_expected[1])
    ))

    # Landing is horizontal at text y
    assert landing == pytest.approx((
        float(elbow_px_expected[0]), float(elbow_px_expected[1]),
        float(text_px_expected[0]), float(text_px_expected[1])
    ))

    # Verify landing is indeed horizontal
    assert abs(landing[1] - landing[3]) < 1e-9


def test_label_degenerate_anchor_and_text_coincident():
    """When anchor and text are very close, leader direction is near-zero.

    In this case, _unit returns None and the arrowhead strokes are gracefully
    skipped, resulting in only 2 segments (leader + landing, no arrows).
    No exception or NaN should occur.

    With camera projection (100 + x*10, 200 - y*10):
    Anchor (0,0,0) → screen (100,200)
    Text (2.6,0,0) → screen (126,200)
    Elbow = (126 - 26, 200) = (100,200)
    Leader vector = (0,0) → _unit returns None, no arrows drawn
    """
    lab = Label(1, (0.0, 0.0, 0.0), (2.6, 0.0, 0.0), "degenerate")
    plan = _plan(lab)
    assert plan is not None

    # With elbow ≈ anchor in screen space, leader vector is near-zero
    # The _unit function returns None, so only leader + landing are drawn
    assert len(plan.segments_px) == 2

    # Verify no NaN in any coordinate
    for seg in plan.segments_px:
        for coord in seg:
            assert not np.isnan(coord)

    # Text should still be present
    assert len(plan.texts) == 1
    assert plan.texts[0].text == "degenerate"
