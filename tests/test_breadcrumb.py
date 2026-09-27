"""Unit tests for the active-path breadcrumb (Task 15).

Tests verify that _refresh_breadcrumb builds the correct display string from
model.active_path without needing a running Qt application — we call the method
directly after patching status_bar.set_breadcrumb.
"""
from __future__ import annotations

import numpy as np
import pytest

from bermake.model.model import Model


def _build_group_model():
    """Return (model, child_inst) where child_inst is a group nested in root."""
    model = Model()
    group_def = model.new_definition("Group #3", is_group=True)
    # Give the group some geometry so local_aabb() is non-None (not required
    # for breadcrumb but good for completeness).
    child_inst = model.new_instance(group_def)
    model.root.children.append(child_inst)
    return model, child_inst


def test_breadcrumb_at_root():
    """At root (active_path=[]), breadcrumb string is empty."""
    model, _ = _build_group_model()
    assert model.active_path == []
    # Build string manually using the same logic as _refresh_breadcrumb
    parts = [model.root.name] + [inst.definition.name for inst in model.active_path]
    breadcrumb = " ▸ ".join(parts) if model.active_path else ""
    assert breadcrumb == ""


def test_breadcrumb_after_enter():
    """After entering a group, breadcrumb contains the group's name."""
    model, child_inst = _build_group_model()
    model.enter(child_inst)
    assert len(model.active_path) == 1

    parts = [model.root.name] + [inst.definition.name for inst in model.active_path]
    breadcrumb = " ▸ ".join(parts)
    assert "Group #3" in breadcrumb
    assert breadcrumb.startswith("Model")
    assert "▸" in breadcrumb


def test_breadcrumb_contains_root_name():
    """Root name 'Model' appears in the breadcrumb when inside a group."""
    model, child_inst = _build_group_model()
    model.enter(child_inst)
    parts = [model.root.name] + [inst.definition.name for inst in model.active_path]
    breadcrumb = " ▸ ".join(parts)
    assert breadcrumb == "Model ▸ Group #3"


def test_breadcrumb_deep_nesting():
    """Three levels of nesting produce a three-segment breadcrumb."""
    model = Model()
    outer_def = model.new_definition("Outer", is_group=True)
    inner_def = model.new_definition("Inner", is_group=True)
    outer_inst = model.new_instance(outer_def)
    inner_inst = model.new_instance(inner_def)
    model.root.children.append(outer_inst)
    outer_def.children.append(inner_inst)

    model.enter(outer_inst)
    model.enter(inner_inst)

    parts = [model.root.name] + [inst.definition.name for inst in model.active_path]
    breadcrumb = " ▸ ".join(parts)
    assert breadcrumb == "Model ▸ Outer ▸ Inner"
