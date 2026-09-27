"""Selection operations, lifted out of MainWindow (M7.3 Task 1).

Every function here takes its collaborators explicitly and imports no Qt, so
the selection rules are testable without a QApplication. MainWindow keeps the
Qt half -- status bar text, viewport repaint, dock indicators -- and delegates
the decisions here.

assign_tag returns (did_assign, message) rather than touching a status bar,
so the same call serves the Tags panel button and the right-click Assign Tag
submenu without either one owning the other's UI.
"""

from __future__ import annotations

from bermake.commands.tag_commands import TagInstancesCommand
from bermake.model.model_queries import select_all_ids


def select_all(model, selection) -> None:
    """Select every entity in the active editing context."""
    edges, faces, instances = select_all_ids(model)
    selection.replace(edges=edges, faces=faces, instances=instances)


def select_none(selection) -> None:
    selection.clear()


def prune_to_live(model, selection) -> None:
    """Keep only selected entities still live in the active context (#46).

    An undo/redo can leave previously-selected ids dangling (undoing a create,
    redoing a delete) while leaving others untouched (a transform). Intersect
    each of the four sets with what is still live rather than blanket-clearing,
    which would drop a selection nothing touched.
    """
    context = model.active_context
    live_instances = {inst.id for inst in context.children}
    live_edges = {e.id for e in context.mesh.edges_iter()}
    live_faces = {f.id for f in context.mesh.faces_iter()}
    live_annotations = {a.id for a in context.annotations}
    live_vertices = {v.id for v in context.mesh.vertices_iter()}
    selection.replace(
        edges=selection.edges & live_edges,
        faces=selection.faces & live_faces,
        instances=selection.instances & live_instances,
        annotations=selection.annotations & live_annotations,
        vertices=selection.vertices & live_vertices,
    )


def selection_status_text(selection) -> str:
    """Summarize the selection, e.g. "4 edges, 1 face selected", or "" when empty."""
    counts = selection.counts()
    if not any(counts):
        return ""
    labels = ("edge", "face", "instance", "annotation", "vertex")
    plurals = ("edges", "faces", "instances", "annotations", "vertices")
    parts = [
        f"{n} {label if n == 1 else plural}"
        for n, label, plural in zip(counts, labels, plurals, strict=True)
        if n
    ]
    return ", ".join(parts) + " selected"


def selected_instances(model, selection) -> list:
    """The selected instances that are children of the active context."""
    return [inst for inst in model.active_context.children if inst.id in selection.instances]


def selection_tag_label(model, selection) -> str | None:
    """The selection's common tag name, "(multiple)", or None when no
    instance is selected."""
    instances = selected_instances(model, selection)
    if not instances:
        return None
    tag_ids = {inst.tag_id for inst in instances}
    if len(tag_ids) == 1:
        return model.tags.get(next(iter(tag_ids))).name
    return "(multiple)"


def assign_tag(model, selection, command_stack, tag_id: int) -> tuple[bool, str]:
    """Assign `tag_id` to every selected instance. Returns (did_assign, message)."""
    instances = selected_instances(model, selection)
    if not instances:
        return False, "Select objects to assign a tag."
    command_stack.execute(TagInstancesCommand(instances, tag_id), model)
    name = model.tags.get(tag_id).name
    return True, f"Assigned tag '{name}' to {len(instances)} object(s)."
