"""Right-click menu composition (M7.2).

context_menu_ids and is_enabled are pure: given what is under the cursor and
a little model state, they say what the menu contains and which entries are
live. Task 14 (resolve_context_target / build_context_menu below) turns that
into a hit-test and a QMenu.

Entries are disabled rather than hidden when they do not apply. A menu whose
items keep their position is learnable; one that reflows between right-clicks
is not.

build_context_menu reuses the menu bar's own QAction objects (see
ui_builder.ACTION_REGISTRY_ATTR), so it mutates their shared enabled state.
MainWindow._on_context_menu_requested is responsible for unconditionally
re-enabling every action once the menu closes -- see the comment there for
why that must not be conditional.
"""

from __future__ import annotations

from bermake.ui.actions import CONTEXT_MENUS, ContextTarget


def resolve_context_target(
    model, camera, x: int, y: int, width: int, height: int, units, show_guides: bool = True
) -> tuple[ContextTarget, int | None]:
    """What is under the cursor: (ContextTarget, entity_id or None).

    Uses the exact picking calls SelectTool.on_mouse_release makes --
    pick_annotation, Model.pick_instance, pick_selectable -- so a right-click
    hits exactly what a left-click would hit. The correspondence is
    guaranteed by sharing the code, not by keeping two implementations in
    step.

    Precedence mirrors SelectTool: annotations draw on top of everything, so
    they are hit-tested first; then whole instances; then the active
    context's own edges/faces.

    `show_guides` (M7.6b Task 7 fix round 1): a hidden guide must not be
    right-clickable either -- defaults to True (the pre-Task-7 behaviour, and
    ViewportWidget.show_guides's own default) so every existing caller keeps
    working unchanged.
    """
    from bermake.annotations.picking import pick_annotation
    from bermake.viewport.picking import pick_selectable

    annotation_id = pick_annotation(
        (x, y),
        model.active_context.annotations,
        model.active_world_transform,
        camera,
        width,
        height,
        units,
        show_guides=show_guides,
    )
    if annotation_id is not None:
        return ContextTarget.ANNOTATION, annotation_id

    origin, direction = camera.ray_from_screen(x, y, width, height)

    instance = model.pick_instance(origin, direction)
    if instance is not None:
        return ContextTarget.INSTANCE, instance.id

    hit = pick_selectable(
        (x, y),
        (width, height),
        camera,
        model.active_scene,
        world_transform=model.active_world_transform,
    )
    if hit is not None:
        kind, entity_id = hit
        if kind == "face":
            return ContextTarget.FACE, entity_id
        return ContextTarget.EDGE, entity_id

    return ContextTarget.EMPTY, None


def _entity_is_component(window, target: ContextTarget, entity_id) -> bool:
    """True only for a component instance -- a plain group cannot be made
    unique, so Make Unique must be disabled for one.

    Definition has no separate is_component flag: MakeComponentCommand sets
    is_group=False and MakeGroupCommand sets is_group=True (see
    bermake.commands.group_commands), so "is a component" is `not is_group`.
    """
    if target is not ContextTarget.INSTANCE or entity_id is None:
        return False
    for child in window._model.active_context.children:
        if child.id == entity_id:
            return not child.definition.is_group
    return False


def _selection_is_one_label(window) -> bool:
    """True when exactly one annotation is selected and it is a Label.

    Mirrors _on_edit_label_text's own guards: it returns silently for a
    Dimension (which has no editable text) and for any selection that is
    not exactly one annotation.
    """
    annotation_ids = window._selection.annotations
    if len(annotation_ids) != 1:
        return False
    only = next(iter(annotation_ids))
    for annotation in window._model.active_context.annotations:
        if annotation.id == only:
            return getattr(annotation, "kind", None) == "label"
    return False


def _add_assign_tag_submenu(window, menu, entity_id) -> None:
    """One checkable entry per tag, assigning it to the current selection."""
    submenu = menu.addMenu("Assign Tag")
    current = None
    for child in window._model.active_context.children:
        if child.id == entity_id:
            current = child.tag_id
            break

    for tag in window._model.tags.tags():
        entry = submenu.addAction(tag.name)
        entry.setCheckable(True)
        entry.setChecked(tag.id == current)
        entry.triggered.connect(lambda _checked=False, tag_id=tag.id: window._assign_tag(tag_id))


def _add_select_submenu(window, menu) -> None:
    """M7.6c: the operations that derive a selection from the current one.

    A submenu rather than five top-level entries, because the context menu is
    already long and these four share one idea. Entries grey out rather than
    hide (the same rule the rest of this module follows): a menu whose shape
    changes under the cursor is harder to learn than one whose entries are
    sometimes unavailable.
    """
    submenu = menu.addMenu("Select")
    has_faces = bool(window._selection.faces)
    has_instances = bool(window._selection.instances)
    has_any = not window._selection.is_empty()
    for action_id, enabled in (
        ("select_grow", has_any),
        ("select_shrink", has_any),
        ("select_same_material", has_faces),
        ("select_same_tag", has_instances),
    ):
        action = window._actions[action_id]
        action.setEnabled(enabled)
        submenu.addAction(action)


def build_context_menu(window, target: ContextTarget, entity_id):
    """A QMenu for `target`, reusing the window's existing QActions.

    Reusing the same QAction objects the menu bar and toolbars hold means an
    entry cannot drift in label, icon, or shortcut between surfaces. It also
    means this function mutates their shared enabled state -- the caller
    (MainWindow._on_context_menu_requested) must re-enable everything once
    the menu closes.
    """
    from PySide6.QtWidgets import QMenu

    menu = QMenu(window)
    in_group = bool(window._model.active_path)
    has_selection = not window._selection.is_empty()
    is_component = _entity_is_component(window, target, entity_id)
    single_instance = len(window._selection.instances) == 1
    editable_label = _selection_is_one_label(window)
    has_hidden = any(inst.hidden for inst in window._model.active_context.children)
    # Distinct from has_hidden above: this is about what is *selected*, not
    # what merely exists in the context, and it is what edit_unhide needs --
    # per the spec, Unhide is enabled only when the selection itself
    # contains a hidden instance, not whenever anything at all is selected.
    has_hidden_selected = any(
        inst.hidden
        for inst in window._model.active_context.children
        if inst.id in window._selection.instances
    )

    for action_id in context_menu_ids(target, in_group=in_group, has_selection=has_selection):
        if action_id is None:
            menu.addSeparator()
            continue
        action = window._actions[action_id]
        action.setEnabled(
            is_enabled(
                action_id,
                target=target,
                in_group=in_group,
                is_component=is_component,
                has_selection=has_selection,
                single_instance=single_instance,
                editable_label=editable_label,
                has_hidden=has_hidden,
                has_hidden_selected=has_hidden_selected,
            )
        )
        menu.addAction(action)

    _add_select_submenu(window, menu)

    if target is ContextTarget.INSTANCE:
        _add_assign_tag_submenu(window, menu, entity_id)

    return menu


def context_menu_ids(
    target: ContextTarget, *, in_group: bool, has_selection: bool
) -> tuple[str | None, ...]:
    """The entries for a right-click on `target`. None is a separator.

    The composition is fixed per target; in_group and has_selection affect
    enablement (see is_enabled), not membership.
    """
    del in_group, has_selection  # Membership is fixed; only enablement varies.
    return CONTEXT_MENUS[target]


def is_enabled(
    action_id: str,
    *,
    target: ContextTarget,
    in_group: bool,
    is_component: bool,
    has_selection: bool,
    single_instance: bool = False,
    editable_label: bool = False,
    has_hidden: bool = False,
    has_hidden_selected: bool = False,
) -> bool:
    """Whether a context-menu entry should be clickable.

    is_component distinguishes a component instance (which can be made
    unique) from a plain group (which cannot).

    single_instance and editable_label exist because their handlers guard
    the same conditions and return silently when they fail. An entry that
    is clickable but does nothing is worse than a greyed-out one: this
    module's rule is to disable rather than hide, so the entry stays
    visible and its unavailability is legible.

    has_hidden says at least one instance in the active context is hidden --
    Unhide All is the only route back for an object you cannot right-click,
    so it is offered exactly when it would do something.

    has_hidden_selected says at least one instance in the *selection* is
    hidden. This is a different signal from has_hidden: has_hidden asks
    about the whole context, has_hidden_selected asks about what is
    selected. edit_unhide needs the latter -- a hidden instance can enter
    the selection through the Outliner even though it cannot be
    right-clicked, so a visible-only selection must not offer Unhide.
    """
    if action_id == "edit_close_group":
        return in_group
    if action_id == "edit_make_unique":
        return is_component
    if action_id == "edit_edit_group":
        # _on_edit_group returns early unless exactly one instance is selected.
        return single_instance
    if action_id == "edit_label_text":
        # _on_edit_label_text returns early for a Dimension, or for anything
        # other than a single selected annotation.
        return editable_label
    if action_id in ("edit_select_none", "edit_erase", "edit_paint_selection"):
        return has_selection
    if action_id == "edit_hide":
        return has_selection
    if action_id == "edit_unhide":
        return has_hidden_selected
    if action_id == "edit_unhide_all":
        # The escape hatch is pointless when nothing is hidden, and offering
        # it anyway would train people to click a no-op.
        return has_hidden
    return True
