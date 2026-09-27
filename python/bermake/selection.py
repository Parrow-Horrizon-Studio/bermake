"""Shared, transient selection state: which edges, faces, instances and
annotations are selected.

Owned by MainWindow, handed to tools (via ToolContext) and the renderer. NOT
part of the Scene geometry model and NOT on the undo stack (selecting is not an
undoable action; only deletions are). `version` bumps on every mutation so the
renderer can cheaply detect changes.
"""

from __future__ import annotations

from collections.abc import Iterable


class Selection:
    __slots__ = ("_annotations", "_edges", "_faces", "_instances", "_version", "_vertices")

    def __init__(self) -> None:
        self._edges: set[int] = set()
        self._faces: set[int] = set()
        self._instances: set[int] = set()
        self._annotations: set[int] = set()
        self._version: int = 0
        self._vertices: set[int] = set()

    @property
    def edges(self) -> set[int]:
        return self._edges

    @property
    def faces(self) -> set[int]:
        return self._faces

    @property
    def instances(self) -> set[int]:
        return self._instances

    @property
    def annotations(self) -> set[int]:
        return self._annotations

    @property
    def vertices(self) -> set[int]:
        return self._vertices

    @property
    def version(self) -> int:
        return self._version

    def _bump(self) -> None:
        self._version += 1

    def replace(
        self,
        *,
        edges: Iterable[int] = (),
        faces: Iterable[int] = (),
        instances: Iterable[int] = (),
        annotations: Iterable[int] = (),
        vertices: Iterable[int] = (),
    ) -> None:
        self._edges = set(edges)
        self._faces = set(faces)
        self._instances = set(instances)
        self._annotations = set(annotations)
        self._vertices = set(vertices)
        self._bump()

    def add(
        self,
        *,
        edges: Iterable[int] = (),
        faces: Iterable[int] = (),
        instances: Iterable[int] = (),
        annotations: Iterable[int] = (),
        vertices: Iterable[int] = (),
    ) -> None:
        self._edges |= set(edges)
        self._faces |= set(faces)
        self._instances |= set(instances)
        self._annotations |= set(annotations)
        self._vertices |= set(vertices)
        self._bump()

    def remove(
        self,
        *,
        edges: Iterable[int] = (),
        faces: Iterable[int] = (),
        instances: Iterable[int] = (),
        annotations: Iterable[int] = (),
        vertices: Iterable[int] = (),
    ) -> None:
        self._edges -= set(edges)
        self._faces -= set(faces)
        self._instances -= set(instances)
        self._annotations -= set(annotations)
        self._vertices -= set(vertices)
        self._bump()

    def toggle_edge(self, e_id: int) -> None:
        self._edges.symmetric_difference_update({e_id})
        self._bump()

    def toggle_face(self, f_id: int) -> None:
        self._faces.symmetric_difference_update({f_id})
        self._bump()

    def toggle_instance(self, i_id: int) -> None:
        self._instances.symmetric_difference_update({i_id})
        self._bump()

    def toggle_annotation(self, a_id: int) -> None:
        self._annotations.symmetric_difference_update({a_id})
        self._bump()

    def toggle_vertex(self, v_id: int) -> None:
        self._vertices.symmetric_difference_update({v_id})
        self._bump()

    def clear(self) -> None:
        if self._edges or self._faces or self._instances or self._annotations or self._vertices:
            self._edges.clear()
            self._faces.clear()
            self._instances.clear()
            self._annotations.clear()
            self._vertices.clear()
            self._bump()

    def contains_edge(self, e_id: int) -> bool:
        return e_id in self._edges

    def contains_face(self, f_id: int) -> bool:
        return f_id in self._faces

    def contains_instance(self, i_id: int) -> bool:
        return i_id in self._instances

    def contains_annotation(self, a_id: int) -> bool:
        return a_id in self._annotations

    def is_empty(self) -> bool:
        return (
            not self._edges
            and not self._faces
            and not self._instances
            and not self._annotations
            and not self._vertices
        )

    def counts(self) -> tuple[int, int, int, int, int]:
        return (
            len(self._edges),
            len(self._faces),
            len(self._instances),
            len(self._annotations),
            len(self._vertices),
        )
