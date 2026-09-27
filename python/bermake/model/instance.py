from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bermake.model.definition import Definition


class Instance:
    """A placement: a 4x4 transform + a reference to a Definition. Non-destructive."""

    __slots__ = ("definition", "hidden", "id", "name", "tag_id", "transform")

    def __init__(
        self, instance_id: int, definition: Definition, transform: np.ndarray | None = None
    ) -> None:
        self.id = int(instance_id)
        self.definition = definition
        if transform is None:
            self.transform = np.eye(4, dtype=np.float64)
        else:
            self.transform = np.asarray(transform, dtype=np.float64).reshape(4, 4).copy()
        self.tag_id = 0  # 0 == TagLibrary.UNTAGGED_ID; set by clone / MakeGroup / TagInstances
        # M7.3: per-INSTANCE name, so renaming one instance of a shared
        # component does not rename its siblings. "" means "fall back to
        # definition.name" -- an un-renamed row still reads "Wall" or "Roof".
        self.name = ""
        # M7.3: per-instance visibility, independent of tag visibility. Hiding
        # an instance hides its whole subtree (see Model._traverse_visible).
        self.hidden = False
