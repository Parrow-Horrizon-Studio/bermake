"""Shared helpers for the M7d annotation tools (Dimension, and Text in M7d T9).

Both tools snap points in WORLD space -- the renderer draws
`ToolOverlay.rubber_band_segments` in world space with no model matrix
applied, so a tool's live preview must be built from world coordinates -- but
the annotations they create (`bermake.model.annotation.Dimension` / `Label`)
store CONTEXT-LOCAL coordinates, so they ride along correctly when their
group/component moves. `world_to_active_local` is the single conversion point
both tools call when committing a click to storage, so the two tools cannot
drift out of sync on how that world-to-local conversion is done.
"""

from __future__ import annotations

import numpy as np

from bermake.viewport.picking import world_to_local_point

# Shared preview colour for every M7d annotation tool's rubber-band overlay: a
# neutral light grey that reads as "not yet real geometry" regardless of the
# snap kind driving the click (mirrors the drawing tools' _NEUTRAL).
NEUTRAL_PREVIEW_COLOR: tuple[float, float, float] = (0.85, 0.85, 0.85)


def world_to_active_local(model, world_point) -> np.ndarray:
    """Convert a world-space point (3,) into `model`'s active context's local
    frame, using `model.active_world_transform` (identity at the document
    root; the accumulated instance transform when entered into a
    group/component). `model` may be None (e.g. a bare unit test with no
    Model wired up), in which case the point passes through unchanged.
    Always returns a float64 (3,) array.
    """
    wt = model.active_world_transform if model is not None else None
    return np.asarray(world_to_local_point(world_point, wt), dtype=np.float64).reshape(3)
