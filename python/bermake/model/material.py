"""Solid-color materials + the per-Model material library (M5b).

Pure Python — no GL, no Qt — so it is fully unit-testable headlessly. A
Material is a named base RGB color plus PBR-shaped shading fields; faces
reference materials by id (see Scene._face_materials). The library owns the
canonical materials and is serialization-ready for M6 file I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class Material:
    """A named material. `base_color` is RGB in 0..1; alpha 1.0 is opaque."""

    id: int
    name: str
    base_color: tuple[float, float, float]
    alpha: float = 1.0
    metallic: float = 0.0
    roughness: float = 0.5
    texture_id: int | None = None
    texture_size: tuple[float, float] = (1.0, 1.0)

    @property
    def is_translucent(self) -> bool:
        return self.alpha < 1.0


# The Default swatch color mirrors the renderer's default diffuse
# (scene_renderer._MATERIAL_DIFFUSE). Duplicated as a literal to avoid a
# viewport -> model import; used only for the dock swatch / hover preview,
# never for face shading (the renderer shades the Default batch with
# _DEFAULT_MATERIAL directly).
_DEFAULT_SWATCH_COLOR = (0.65, 0.65, 0.70)

# Built-in palette seeded into every MaterialLibrary (stable ids 1..N).
_BUILTIN_PALETTE: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("White", (0.92, 0.92, 0.92)),
    ("Warm Gray", (0.66, 0.63, 0.60)),
    ("Concrete", (0.74, 0.73, 0.71)),
    ("Brick Red", (0.70, 0.27, 0.22)),
    ("Wood Tan", (0.78, 0.62, 0.40)),
    ("Slate Blue", (0.36, 0.45, 0.60)),
    ("Forest Green", (0.27, 0.50, 0.31)),
    ("Charcoal", (0.22, 0.22, 0.24)),
)


class MaterialLibrary:
    """Owns the model's Material objects: Default first, then builtins, then customs."""

    DEFAULT_ID = 0

    def __init__(self) -> None:
        self._default = Material(self.DEFAULT_ID, "Default", _DEFAULT_SWATCH_COLOR)
        self._materials: dict[int, Material] = {self.DEFAULT_ID: self._default}
        self._order: list[int] = [self.DEFAULT_ID]
        self._next_id = 1
        for name, color in _BUILTIN_PALETTE:
            self._add(name, color)

    def _add(self, name: str, color: tuple[float, float, float]) -> Material:
        mat = Material(self._next_id, name, (float(color[0]), float(color[1]), float(color[2])))
        self._materials[mat.id] = mat
        self._order.append(mat.id)
        self._next_id += 1
        return mat

    def add_custom(self, name: str, color: tuple[float, float, float]) -> Material:
        """Append a new material with a fresh monotonic id and return it."""
        return self._add(name, color)

    def get(self, mid: int) -> Material:
        """Return the material for `mid`, or the Default material if unknown."""
        return self._materials.get(mid, self._default)

    def materials(self) -> list[Material]:
        """All materials in display order (Default first)."""
        return [self._materials[i] for i in self._order]

    @property
    def next_id(self) -> int:
        return self._next_id

    def edit(self, mid: int, **fields) -> Material:
        """Replace the material with an edited copy. Unknown fields raise TypeError."""
        edited = replace(self._materials[mid], **fields)
        self._materials[mid] = edited
        if mid == self.DEFAULT_ID:
            self._default = edited
        return edited

    def index_of(self, mid: int) -> int:
        """Position of `mid` in display order."""
        return self._order.index(mid)

    def remove(self, mid: int) -> Material:
        """Drop a material and return it. Refuses Default, the unpainted fallback."""
        if mid == self.DEFAULT_ID:
            raise ValueError("the Default material cannot be removed")
        removed = self._materials.pop(mid)
        self._order.remove(mid)
        return removed

    def restore(self, material: Material, index: int) -> None:
        """Put a removed material back at `index` in display order (undo)."""
        self._materials[material.id] = material
        self._order.insert(index, material.id)

    def to_records(self) -> list[dict]:
        """Serialize all materials in display order (Default first)."""
        return [
            {
                "id": m.id,
                "name": m.name,
                "base_color": list(m.base_color),
                "alpha": m.alpha,
                "metallic": m.metallic,
                "roughness": m.roughness,
                "texture_id": m.texture_id,
                "texture_size": list(m.texture_size),
            }
            for m in self.materials()
        ]

    @classmethod
    def from_records(cls, records: list[dict], next_id: int | None = None) -> MaterialLibrary:
        """Rebuild authoritatively from saved records (no auto-seed).

        Accepts the schema <= 4 shape, which wrote "color" and carried no PBR
        fields; those default to an opaque dielectric. Accepts the schema 5
        shape, which carried no texture fields; those default to untextured.
        """
        lib = cls()  # seeds default + builtins, then we overwrite
        lib._materials = {}
        lib._order = []
        for r in records:
            color = r.get("base_color", r.get("color"))
            texture_id = r.get("texture_id")
            texture_size = r.get("texture_size", (1.0, 1.0))
            mat = Material(
                int(r["id"]),
                str(r["name"]),
                (float(color[0]), float(color[1]), float(color[2])),
                alpha=float(r.get("alpha", 1.0)),
                metallic=float(r.get("metallic", 0.0)),
                roughness=float(r.get("roughness", 0.5)),
                texture_id=None if texture_id is None else int(texture_id),
                texture_size=(float(texture_size[0]), float(texture_size[1])),
            )
            lib._materials[mat.id] = mat
            lib._order.append(mat.id)
        lib._default = lib._materials.get(cls.DEFAULT_ID, lib._default)
        if next_id is None:
            next_id = max((m.id for m in lib._materials.values()), default=-1) + 1
        lib._next_id = int(next_id)
        return lib
