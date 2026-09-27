from __future__ import annotations

import numpy as np

from bermake.model.definition import Definition
from bermake.model.instance import Instance
from bermake.model.material import MaterialLibrary
from bermake.model.tag import TagLibrary
from bermake.model.texture import TextureLibrary


class Model:
    """The scene graph: a root Definition + the active editing path."""

    def __init__(self) -> None:
        self._next_def_id = 0
        self._next_inst_id = 0
        self.root = self.new_definition("Model", is_group=False)
        self.active_path: list[Instance] = []
        self.materials = MaterialLibrary()
        self.textures = TextureLibrary()
        self.tags = TagLibrary()
        self.opening_definitions = {}  # M7b: (kind, w, h, depth) -> shared Component Definition
        self._next_annotation_id = 0  # M7d: model-wide unique annotation ids
        # M7e: saved Scenes (camera + tags + style). Imported here, not at module
        # top, to avoid a model <-> io.document_codec import cycle.
        from bermake.views.view_library import ViewLibrary

        self.views = ViewLibrary()

    # --- construction ---
    def new_definition(self, name: str, is_group: bool) -> Definition:
        d = Definition(self._next_def_id, name, is_group)
        self._next_def_id += 1
        return d

    def new_instance(self, definition: Definition, transform=None) -> Instance:
        inst = Instance(self._next_inst_id, definition, transform)
        self._next_inst_id += 1
        definition.instances.append(inst)
        return inst

    def new_annotation_id(self) -> int:
        """Allocate a model-wide unique annotation id."""
        annotation_id = self._next_annotation_id
        self._next_annotation_id += 1
        return annotation_id

    # --- active context ---
    @property
    def active_context(self) -> Definition:
        return self.active_path[-1].definition if self.active_path else self.root

    @property
    def active_scene(self):
        return self.active_context.mesh

    @property
    def active_world_transform(self) -> np.ndarray:
        m = np.eye(4, dtype=np.float64)
        for inst in self.active_path:
            m = m @ inst.transform
        return m

    def definition_is_dimmed(self, definition) -> bool:
        """True when `definition` should render dimmed (recede) -- i.e. you
        are inside a group (active_path is non-empty) and this definition is
        not the active editing context. At the root context, nothing is
        dimmed. This is the single source of truth for the dim rule: both
        geometry (scene_renderer) and annotations (draw_plan) dim exactly
        when this returns True (#95)."""
        return bool(self.active_path) and definition is not self.active_context

    def enter(self, instance: Instance) -> None:
        self.active_path.append(instance)

    def exit_one(self) -> None:
        if self.active_path:
            self.active_path.pop()

    def traverse(self):
        """Yield (definition, world_transform) depth-first from the root."""
        yield from self._traverse(self.root, np.eye(4, dtype=np.float64))

    def _traverse(self, definition, world):
        yield definition, world
        for inst in definition.children:
            yield from self._traverse(inst.definition, world @ inst.transform)

    def traverse_visible(self):
        """Like traverse(), but prunes any instance that is hidden or on a hidden
        tag — and its whole subtree (hiding an object hides its contents).
        Instances on the active editing path are always kept (you're editing
        inside them)."""
        active_ids = {inst.id for inst in self.active_path}
        yield from self._traverse_visible(self.root, np.eye(4, dtype=np.float64), active_ids)

    def _traverse_visible(self, definition, world, active_ids):
        yield definition, world
        for inst in definition.children:
            if inst.id not in active_ids and (inst.hidden or not self.tags.is_visible(inst.tag_id)):
                continue
            yield from self._traverse_visible(inst.definition, world @ inst.transform, active_ids)

    def clone_definition(self, definition):
        """Deep-copy a definition's geometry + child instances into a fresh def."""
        clone = self.new_definition(definition.name, definition.is_group)
        idmap = {}
        for v in definition.mesh.vertices_iter():
            idmap[v.id] = clone.mesh.add_vertex(v.position)
        for e in definition.mesh.edges_iter():
            clone.mesh.add_edge(idmap[e.v1_id], idmap[e.v2_id])
        for f in definition.mesh.faces_iter():
            nf = clone.mesh.add_face_from_loop([idmap[v] for v in f.loop_vertex_ids])
            # A deep copy includes the face's material, texture placement and
            # stored UVs. add_face_from_loop mints a fresh id and carries none
            # of them, so without this Make Unique hands the user an
            # unpainted copy of a painted component (#114's family).
            clone.mesh.copy_face_attributes_from(definition.mesh, f.id, nf)
        for child in definition.children:
            new_child = self.new_instance(child.definition, child.transform)
            new_child.tag_id = child.tag_id
            clone.children.append(new_child)
        return clone

    def pick_instance(self, origin, direction):
        """Return the nearest Instance (among active context's children) hit by the ray.

        The world ray is transformed into each instance's local frame via mat_invert.
        Returns the Instance with the smallest hit.t, or None if no hit.

        Instances whose definition has no faces under the cursor (e.g. an
        edge-only group or component) fall back to a ray/edge proximity test
        in the same local frame, so they remain pickable. A face hit always
        wins over an edge hit within the same instance (edges are only
        tested when the face test misses); across instances the nearest hit
        wins.
        """
        from bermake.geometry.ray import closest_point_on_segment_to_ray
        from bermake.geometry.transforms import mat_invert

        best, best_t = None, float("inf")
        world0 = self.active_world_transform
        for inst in self.active_context.children:
            # No active-path bypass here (unlike _traverse_visible): both loops
            # iterate active_context.children, and a child of the active context
            # cannot itself be entered -- entering it would make it the context.
            if inst.hidden or not self.tags.is_visible(inst.tag_id):
                continue
            world = world0 @ inst.transform
            inv = mat_invert(world)
            o = (inv @ np.append(origin, 1.0))[:3]
            d = inv[:3, :3] @ np.asarray(direction, np.float64)
            hit = inst.definition.mesh.ray_pick_face(o, d)
            if hit is not None:
                if hit.t < best_t:
                    best, best_t = inst, hit.t
                continue

            # Face miss: fall back to edge proximity in the same local frame.
            mesh = inst.definition.mesh
            for e in mesh.edges_iter():
                a = np.asarray(mesh.vertex(e.v1_id).position, np.float64)
                b = np.asarray(mesh.vertex(e.v2_id).position, np.float64)
                on_pt, _seg_t = closest_point_on_segment_to_ray(o, d, a, b)
                on_pt = np.asarray(on_pt, np.float64)
                ray_t = float(np.dot(on_pt - o, d) / np.dot(d, d))
                if ray_t <= 0:
                    continue
                dist = float(np.linalg.norm(on_pt - (o + ray_t * d)))
                tol = 0.05 * ray_t  # world-space tolerance, a ~3deg cone
                if dist <= tol and ray_t < best_t:
                    best, best_t = inst, ray_t
        return best

    def pick_face_local(self, origin, direction):
        """Nearest child-instance face hit for a WORLD ray, as (point, normal)
        in the active-context-local frame. `normal` faces the ray origin
        (viewer-facing). None if nothing is hit.

        The normal is transformed by the inverse-transpose of the instance
        transform's linear block (#92), which is exact under any invertible
        linear map -- including a non-uniform scale, where the linear block
        itself would tilt the normal off-perpendicular."""
        from bermake.geometry.transforms import mat_invert

        w = self.active_world_transform
        w_inv = mat_invert(w)
        o_a = (w_inv @ np.append(np.asarray(origin, np.float64), 1.0))[:3]
        d_a = w_inv[:3, :3] @ np.asarray(direction, np.float64)

        best = None
        best_t = float("inf")
        for inst in self.active_context.children:
            # No active-path bypass here (unlike _traverse_visible): both loops
            # iterate active_context.children, and a child of the active context
            # cannot itself be entered -- entering it would make it the context.
            if inst.hidden or not self.tags.is_visible(inst.tag_id):
                continue
            t_inv = mat_invert(inst.transform)
            o_c = (t_inv @ np.append(o_a, 1.0))[:3]
            d_c = t_inv[:3, :3] @ d_a
            hit = inst.definition.mesh.ray_pick_face(o_c, d_c)
            if hit is None or hit.t >= best_t:
                continue
            n_c = np.asarray(inst.definition.mesh.face_normal(hit.face_id), np.float64)
            p_c = np.asarray(hit.point, np.float64)
            p_a = (inst.transform @ np.append(p_c, 1.0))[:3]
            # t_inv[:3, :3] is the inverse of the transform's linear block
            # (t_inv is the inverse of the whole affine 4x4); its transpose
            # maps normals correctly under any invertible linear map,
            # including a non-uniform scale (#92).
            n_a = t_inv[:3, :3].T @ n_c
            n_a_norm = float(np.linalg.norm(n_a))
            if n_a_norm > 1e-12:
                n_a = n_a / n_a_norm
            if np.dot(n_a, d_a) > 0.0:  # orient toward the viewer (against the ray)
                n_a = -n_a
            best = (p_a, n_a)
            best_t = hit.t
        return best

    def load_from(self, other: Model) -> None:
        """Replace this model's contents with another's, in place (keeps identity).

        Lets the viewport / tool context keep their existing Model reference while
        the whole document is swapped underneath (file Open / New).
        """
        self.root = other.root
        self.active_path = []
        self._next_def_id = other._next_def_id
        self._next_inst_id = other._next_inst_id
        self.materials = other.materials
        self.textures = other.textures
        self.tags = other.tags
        self.opening_definitions = other.opening_definitions
        self._next_annotation_id = other._next_annotation_id
        self.views = other.views

    def revalidate_active_path(self) -> None:
        """Pop the active path to the nearest still-reachable instance.

        After an undo/redo destroys a group, the entered instance may no longer
        be a child of its parent context. Walk the path from the root; truncate
        at the first instance that isn't in its parent's children list.
        """
        valid: list[Instance] = []
        parent = self.root
        for inst in self.active_path:
            if inst in parent.children:
                valid.append(inst)
                parent = inst.definition
            else:
                break
        self.active_path = valid
