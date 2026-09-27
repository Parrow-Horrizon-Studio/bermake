# Bermake — Design Document

**Date:** 2026-05-16
**Status:** Initial design, drafted from brainstorming session
**Author:** Rowee Apor (Parrow Horrizon Studio)
**License:** GPL (matches the project license)

---

## 1. Vision

**Bermake is an open-source, long-horizon, polygonal 3D modeler with CAD-like precision, aimed primarily at architectural 3D modeling. It is intended to be a free, self-hostable alternative to SketchUp Pro, following the Blender model of community-supported free software.**

Bermake is not a hobby project in the soft sense. It is a hobby project in the *no commercial deadline* sense: developed without external pressure, but with the serious architectural and engineering rigor required for a tool that aspires to be globally recognized over the course of years.

### 1.1 Inspirations

- **SketchUp Pro** — the modeling paradigm, push/pull workflow, inferencing/snapping, viewport styles, layout/scenes pattern, and architect-first audience.
- **Blender** — the software development and distribution model: free, GPL, native core (C/C++) with a scripting layer (Python), community-funded.

### 1.2 Long-Horizon Ambition

Bermake's eventual destination is a fully self-sufficient architectural modeler:
- Modeling (Phases 1–4)
- Documentation output (Phase 6 — layout-equivalent, BIM-lite)
- Built-in real-time and production rendering (Phase 5)
- Plugin ecosystem (Phase 3 onward)

We expect this to take years. Blender took roughly 25 years to reach its current standing. Bermake's pace will be honest about the constraints of solo development without forcing artificial deadlines.

---

## 2. Modeling Paradigm

Bermake is a **polygonal modeler with CAD-like precision** — not a B-Rep parametric CAD app.

| Property | Value |
|---|---|
| Geometry primitive | Polygon meshes (vertices, edges, faces) |
| Surfaces | Polygonal approximations — no NURBS |
| Boolean operations | Mesh-based booleans (via CGAL) |
| Parametric history | None — direct manipulation, like SketchUp |
| Precision | Exact numerical input + tolerance-based snapping/inferencing |
| Target use case | Architectural 3D modeling (buildings, interiors, urban) |

This distinction matters: it shapes the kernel choice, the data structures, the interaction model, and the renderer requirements. Bermake is closer in architecture to Blender (polygonal) than to Fusion 360 / FreeCAD (B-Rep / NURBS).

---

## 3. Technical Architecture

### 3.1 Language Stack — Hybrid Python + C++

Bermake uses a hybrid architecture matching Blender's general pattern: a native core for performance-critical work, a Python application layer for everything else.

**Python (≈ 80–90% of the codebase by line count):**
- UI shell (PySide6 / Qt)
- Tools (drawing, push/pull, transform, measure, etc.)
- Scene graph, layers, undo/redo, command pattern
- File I/O orchestration
- Plugin system
- High-level application logic

**C++ (≈ 10–20% of the codebase by line count, but performance-critical):**
- Geometry kernel (mesh data structures, half-edge model)
- Boolean / CSG operations
- BVH and spatial indexing
- Robust geometric predicates
- Per-frame CPU work for the viewport (culling, batching) when Python becomes the bottleneck
- Future production renderer (Phase 5)

### 3.2 Key Libraries and Tooling

| Concern | Choice | Rationale |
|---|---|---|
| Binding library | **nanobind** | Modern successor to pybind11 by the same author; leaner, faster compile, cleaner API. Reference: nanobind.readthedocs.io |
| C++ standard | **C++20** | Modern (concepts, ranges, `<=>`, designated init) with rock-solid compiler support across MSVC/GCC/Clang |
| Geometry library | **CGAL** | Mature computational geometry: mesh ops, booleans, BVH, robust predicates. GPL license matches Bermake's. |
| UI framework | **PySide6** (Qt 6) | Industry-standard cross-platform UI; same family Blender uses for its UI through Python |
| Viewport renderer | **QOpenGLWidget + ModernGL** initially, designed as a swappable abstraction | Get on-screen fast; swap to native C++ renderer in Phase 5 without disturbing the rest of the app |
| C++ dependency manager | **vcpkg** | Microsoft-maintained, one-line CMake integration, manifest mode (`vcpkg.json`), Visual Studio integration. CGAL/Eigen/fmt/spdlog all available. (Qt installed separately via Qt installer.) |
| Build orchestration | **scikit-build-core + CMake + Ninja** | Modern Python/C++ build orchestration; `pip install -e .` builds the whole project end-to-end |
| Python tests | **pytest** | De facto Python testing standard |
| C++ tests | TBD between **GoogleTest** and **Catch2** (deferred to implementation plan) | Either is fine; decided when C++ tests are first written |
| Python tooling | **Ruff** (formatter + linter), **mypy** (type checking) | Modern, fast Python tooling |
| C++ tooling | **clang-format** + **clang-tidy** | Standard formatters and linters |
| Documentation | Deferred — likely **MkDocs Material** + **Doxygen** for C++ API | Decided at Phase 3 (when there's something to document publicly) |

### 3.3 Codebase Structure

```
bermake/
├── cpp/                       # All C++ code
│   ├── include/bermake/       # Public headers (Python-facing API)
│   ├── src/                   # Implementation
│   │   ├── mesh/              # Half-edge mesh, geometry kernel
│   │   ├── boolean/           # CSG / boolean ops via CGAL
│   │   ├── spatial/           # BVH, spatial indexing
│   │   └── render/            # Per-frame CPU work for viewport (Phase 4+)
│   ├── bindings/              # nanobind glue
│   │   └── module.cpp         # The Python-visible module definition
│   ├── tests/                 # C++ unit tests
│   └── CMakeLists.txt
│
├── python/bermake/            # All Python code
│   ├── __init__.py            # Imports the compiled C++ module
│   ├── app.py                 # Entry point
│   ├── ui/                    # PySide6/Qt UI
│   ├── tools/                 # Modeling tools
│   ├── scene/                 # Scene graph, layers, undo/redo
│   ├── viewport/              # Renderer abstraction (ModernGL initially)
│   ├── io/                    # File import/export
│   └── plugins/               # Plugin system (Phase 3)
│
├── tests/                     # Python integration tests
├── docs/                      # Design docs, ADRs, user docs
├── vcpkg.json                 # C++ dependency manifest
├── pyproject.toml             # Python project + scikit-build-core config
├── CMakeLists.txt             # Top-level CMake
├── LICENSE                    # GPL
└── README.md
```

### 3.4 The Geometry Kernel Architecture

The C++ kernel exposes a clean Python-facing API designed for binding from day 1:
- Mesh data structures (half-edge, vertex/edge/face)
- Boolean operations (union, intersection, difference) via CGAL
- Spatial queries (ray-mesh intersection, point-in-mesh, nearest face)
- Geometric primitives (cube, plane, cylinder, etc.)
- Robust predicates for snapping/inferencing

Internal design principle: **the kernel knows about geometry, not the application.** No UI concepts, no scene-graph concepts, no Python objects in the public API. The Python layer wraps and composes kernel operations into application-level features.

---

## 4. Conventions

### 4.1 Coordinate System
- **Z-up** (matches SketchUp, matches architect convention — Z is the vertical axis)
- Right-handed coordinate system

### 4.2 Units
- User-switchable per-document **imperial (feet/inches)** or **metric (mm/cm/m)**
- Setting lives in document preferences (not a global app setting), so opening a project preserves the units it was created with
- Internal representation is always in meters (or unitless) — display layer translates to user's chosen units

### 4.3 License
- **GPL** (specifically GPL-3.0 or later, to match the modern open-source standard and align with Blender)
- All contributions are GPL-licensed

### 4.4 Hosting & Branding
- GitHub: `github.com/parrow-horrizon-studio/bermake`
- Website: `bermake.org`
- Studio org GitHub is calibrated for security (2FA enforcement, conservative member privileges, secure defaults)

---

## 5. Platforms

| Platform | Status | Phase |
|---|---|---|
| Windows | Primary development platform from day 1 | Phase 1+ |
| Linux | First-class supported platform, shipped from v0.1 | Phase 1+ |
| macOS | Added at Phase 3 (M9), when initial installer pipeline matures | Phase 3 |

**Cross-platform discipline from day 1:** all code uses `std::filesystem`, avoids OS-specific extensions, runs CI on both Windows and Linux from M0. Catching portability bugs early is cheap; fixing accumulated portability debt is expensive.

---

## 6. Roadmap

Bermake's development is organized into **6 Phases**. Phases 1–5 form a linear progression toward a feature-complete v1.0. Phase 6 is explicitly an **exploration tier** where priorities emerge from community needs.

Version numbers are directional anchors, not commitments.

A note on `.berm` before the milestones below: the native file extension was
`.pluton` through v0.13.0 and became `.berm` at v0.14.0, when the project
renamed to Bermake. Entries below that describe earlier milestones name the
format's current identity rather than the name it carried at the time.

### Phase 1 — Foundation *(v0.0 → v0.1)*

**End state:** A toy version of SketchUp's core push/pull innovation works. Proof that the technology stack is sound.

- **M0: Hello, Window** — project scaffolding (cpp/python layout, CMake, vcpkg, scikit-build-core, pyproject.toml); Qt window with QOpenGLWidget; draw a triangle on screen; nanobind binding pipeline verified end-to-end; CI green on Windows + Linux.
- **M1: Core viewport** — mesh data structure in C++ exposed via nanobind; orbit/pan/zoom camera; basic Phong shading; Z-up coordinate system established; build a single cube.
- **M2: Basic drawing** — Line tool, Rectangle tool, drawing on the ground plane; mouse input with grid snapping; ability to create geometry interactively.
- **M3: Push/Pull** — the iconic SketchUp interaction: select a face, drag to extrude. Delivered in four sub-milestones: **M3a** (half-edge topology + command-pattern undo/redo), **M3b** (basic push/pull — open-bottom prism, no merge), **M3c** (closed-manifold push/pull via half-edge dissolve — closed bottoms + coplanar seam-merge, no CGAL), **M3d** ✅ *(shipped v0.0.7)* — 3D inferencing (Tier 2): endpoint/midpoint/on-edge/on-face/intersection snaps + 3D axis-lock incl. the vertical Z axis, with a `split_edge` half-edge op so interior snaps build clean (T-junction-free) topology; face-split deferred to Phase 2. True volumetric booleans (push/pull *into* existing geometry, the Hole tool, mesh import + carve) are deferred to Phase 2, where CGAL becomes a meaningful dependency — M3c showed the M3-scope merge cases are pure half-edge operations that don't need it.

### Phase 2 — Modeling App *(v0.1 → v0.3)*

**End state:** An architect could use Bermake for simple real projects.

- **M4: Modeling polish** — Circle, Arc, Polygon tools; Eraser tool; Select tool with multi-select; Move/Rotate/Scale transforms; **Tape Measure / point-to-point measurement tool**; **user-switchable imperial/metric units** in document preferences; **Groups and Components** (SketchUp's most important organizational feature). Delivered in five sub-milestones: **M4a** ✅ *(shipped v0.1.0)* — drawing tools (Circle, Polygon, 2-Point Arc) on ground/face planes, snap-driven; **M4b** ✅ *(shipped v0.1.1)* — selection & eraser (edges+faces, box-select, blue highlight, edge-Eraser with face cascade, Delete); **M4c** ✅ *(shipped v0.1.2)* — Move/Rotate/Scale transforms (`set_vertex_position` kernel op; point-to-point Move, auto-tilt Rotate protractor, full corner/edge/face Scale gizmo); **M4d** ✅ *(shipped v0.1.3)* — units & measurement (metric + architectural imperial; the typed-entry VCB / Measurements box across all 9 tools; measure-only Tape Measure with snapping); **M4e** ✅ *(shipped v0.1.4)* — Groups & Components (a Python scene-graph **Definition + Instance** model replacing the single flat mesh; enter-to-edit full isolation with breadcrumb + dimmed surroundings; Make Group / Make Component / Make Unique / Explode; Move-copy via Ctrl-drag; shared-definition edit-propagation across all instances of a component; every editing tool — draw / push-pull / Move / Rotate / Scale / Eraser / select-hover — made transform-aware so editing works correctly inside a moved or rotated context; root-with-identity stays behaviorally identical to the old flat mesh).
- **M5: Materials and viewport styles** — basic materials (color + simple textures); multiple **viewport styles** (Shaded, Hidden Line, Wireframe, Monochrome, X-Ray, Sketchy edges); Layers/Tags for object organization. Delivered in sub-milestones: **M5a** ✅ *(shipped v0.1.5)* — viewport display styles: four mutually-exclusive face styles (Wireframe / Hidden Line / Monochrome / Shaded) plus an orthogonal X-Ray toggle, switchable from a View menu; renderer-only — a pure descriptor table (`resolve_face_pass`) drives the existing phong shader via a new `u_alpha` uniform that also unifies the dim pass, with the Shaded default kept byte-identical to v0.1.4; **M5b** ✅ *(shipped v0.1.6)* — solid-color materials: a `Material`/`MaterialLibrary` palette, a per-face `face_id→material_id` sidecar on `Scene`, and per-material draw batching (a pure `plan_face_batches` → one `glDrawArrays` per material through M5a's unchanged `resolve_face_pass`, so unpainted models stay byte-identical to v0.1.5 with no shader change), plus a Paint tool (B) with Alt-sample + Default un-paint and a dockable Materials palette (reopenable via View ▸ Materials). **Textures/UV mapping deferred to a later milestone**; **M5c** ✅ *(shipped v0.1.7)* — Layers/Tags: a `Tag`/`TagLibrary` (always-visible "Untagged" sentinel) on the `Model`, a `tag_id` on each group/component `Instance`, a visibility-aware `Model.traverse_visible` that prunes hidden-tag instances and their subtrees (with an active-path bypass, byte-identical render when nothing is hidden), an undoable `TagInstancesCommand`, MakeGroup/MakeComponent inheriting the active tag, and a dockable Tags panel (list with show/hide checkboxes, inline rename, a "Selection: <tag>" indicator, Assign-to-Selection; tabbed with Materials, reopenable via View ▸ Tags). Instances-only tagging; loose-geometry tagging, per-tag color, and rename/delete/reorder deferred. **M5 is complete** (M5a + M5b + M5c all shipped).
- **M6: File I/O** — native `.berm` file format with versioned schema from day 1; import/export OBJ and glTF via Assimp. Delivered in sub-milestones: **M6a** ✅ *(shipped v0.1.8)* — native `.berm` save / open / new: a zip container (`manifest.json` + compact `document.json`, `schema_version` gate from day 1) round-tripping the whole document — scene graph with component **sharing preserved by identity**, per-face materials + material library, tags + visibility, document units, and the camera view. A pure, headlessly-testable `bermake/io/` package (`document_codec` ↔ `bermake_file`, model classes stay format-agnostic): **index-based** geometry encoding (compacts kernel id gaps, rebuild via `add_vertex`/`add_edge`/`add_face_from_loop`) with structural def/instance ids + counters preserved; **atomic** save (temp + `os.replace`) and atomic in-place adopt on open (`Model.load_from` keeps identity so the viewport/tools keep working; open-failure leaves the session untouched). Full File menu (New/Open/Save/Save As) with dirty-state title marker and a Save/Don't-Save/Cancel guard (`Ctrl+N` repurposed to File ▸ New; the old clear-scene moved to Edit ▸ "Clear Active Context"). **M6b** ✅ *(shipped v0.1.9)* — OBJ import/export: a pure `obj_codec` (a world-space polygon-soup IR ↔ `.obj`/`.mtl` text — n-gons preserved, 1 unit = 1 m, material colors round-tripped, `vn`/`vt` ignored on read) plus an `obj_io` model bridge. Export flattens the scene graph to world space (one `o` group per node with geometry, a sibling `.mtl` when materials are painted, atomic temp+`os.replace` write); **adaptive** import routes a grouped OBJ (`o`/`g`) into one group per object in the active context or merges a flat OBJ into the active scene, **best-effort** (skips + counts faces/materials the kernel or a malformed file rejects — a missing/subdir `.mtl` or bad `Kd` degrades gracefully, never hard-failing the import), delivered as an undoable `ImportObjCommand` and wired to File ▸ Import/Export OBJ (generalized path prompts). **M6c** ✅ *(shipped v0.2.0)* — glTF import/export via **Assimp** (Bermake's first C++ format dependency, statically linked — Windows uses the `x64-windows-static-md` vcpkg triplet so `_core` bundles no runtime DLLs). Import goes through a thin Assimp nanobind bridge that decodes `.glb`/`.gltf` — **including Draco mesh compression** (`KHR_draco_mesh_compression`) — into a neutral plain-data IR, then a pure-Python layer maps it onto Bermake's scene graph: the glTF node tree becomes nested groups (each node → one object, single-mesh childless leaves collapsing to a direct instance), meshes shared across nodes become shared **Components** (real instancing reconstructed), PBR `baseColorFactor` becomes Bermake materials, Y-up→Z-up axis conversion, **best-effort** (skips + counts rejected triangles, never hard-failing) and **undoable** (`ImportGltfCommand`). Export is a pure-Python `.glb`/`.gltf` writer (`gltf_codec` + `gltf_export`): Z-up→Y-up, n-gon fan-triangulation grouped into per-material primitives, mesh-level instancing preserved, atomic write. A permanent, non-skippable CI gate keeps Draco decode from silently regressing. Wired to File ▸ Import/Export glTF. **M6 is complete** (M6a + M6b + M6c all shipped). Recent-files, autosave, and launch-with-file remain deferred.
- **M7: Architecture-specific tools** — Wall tool (auto-thickness); Door/Window placement; Roof tools; Dimensions and annotations; Scenes (saved cameras/views). Delivered in sub-milestones: **M7a** ✅ *(shipped v0.2.1)* — the **Wall tool** (`W`): draws a **chaining polyline of baked solid-box walls**. The first click sets the anchor; each later click commits one **centered** wall segment (thickness/2 to each side of the drawn centerline) as its own `"Wall"` group in the active context, then chains (anchor = endpoint) so a whole floor plan traces in one gesture — Esc/Enter ends the chain (committed segments stay). Walls are ordinary baked geometry, not a parametric object: a pure `wall_box` generator (8 verts / 6 outward-wound quad faces, or empty for a degenerate segment) → an undoable `CreateWallCommand` (Ctrl+Z peels back one segment at a time, mirroring the import command's detach-to-unreachable undo) → a **transform-aware** `WallTool` (world→active-local conversion with the base dropped to the context ground plane, VCB typed segment length) → a unit-aware thickness/height **tool-options row** shown only while the Wall tool is active (defaults 100 mm / 2400 mm). Independent boxes (no corner mitering) and centered justification for now. Python-only — no C++/kernel change (ctest stays 79/79). **M7b** ✅ *(shipped v0.2.2)* — the **Door/Window tool** (`D`): pick a wall face and a **framed door or window Component** is placed **flush** on it — auto-oriented to the face (upright, horizontal follows the cursor), **floor-anchored** (door sill 0, window sill default 900 mm), extending inward by a set depth. The wall is **not cut** (the kernel has no boolean/holes) — the opening sits on the still-solid wall (the accepted tradeoff). Identical `(kind, width, height, depth)` openings **share one Component** `Definition` via a runtime dedup registry (real instancing). Pipeline: a pure `opening_frame` generator (door = jambs + head + panel; window = jambs + head + sill + glazing, from closed outward-wound sub-boxes) → a pure `opening_placement_transform` (horizontalized viewer-facing normal → upright flush basis) → a hierarchical `Model.pick_face_local` (world ray → nearest child-face point + normal in active-local) → an undoable `PlaceOpeningCommand` (dedup-or-create Component, detach-undo keeping the shared Definition) → a `DoorWindowTool` with a Door|Window options row (unit-aware width/height/sill/depth). Python-only (ctest stays 79/79). **M7c** ✅ *(shipped v0.2.3)* — the **Roof tool** (`O`): draw a rectangle footprint on the active drawing plane and a parametric **Gable / Hip / Shed** roof is baked as a closed-solid `"Roof"` group over it — ridge auto-aligned to the footprint's longer edge (Up/Down arrows rotate it 90°), form + slope° from an options row, base following the drawing plane (draw on the ground, or hover a wall-top face to land on the walls). Flush (no overhang); each roof is a unique baked group. Pipeline: a pure `roof_solid(kind, width, depth, angle)` generator (Shed = mono-pitch wedge; Gable = full-depth ridge prism; Hip = equal-pitch ridge-setback, collapsing to a pyramid when `depth ≤ width`; all closed outward-wound manifolds) → an undoable `CreateRoofCommand` (baked `"Roof"` group, `CreateInstanceCommand`-style reuse-on-redo undo, no `defn.instances` leak) → a **transform-aware** `RoofTool` (world footprint → `mat_invert(active_world) @ m_world` local placement; live world-space wireframe preview matching the placement even inside an entered group) → a Gable|Hip|Shed options row (unit-aware slope°). Python-only (ctest stays 79/79). **M7d** ✅ *(shipped v0.2.4)* — the **Dimension tool** (`I`) and **Text tool** (`N`): persistent architectural-style annotations that live **per editing context** (`Definition.annotations`, so they ride along when their group/component moves) and draw as **screen-space `QPainter` overlays** over the GL pass — upright and constant pixel size at any zoom, no glyph atlas or texture support needed. The Dimension tool is a **3-click gesture** (snap `p1`, snap `p2`, click to set the perpendicular offset) drawing an **Architectural** dimension (45° ticks, extension-line gap + overshoot, text above an unbroken line); the Text tool is a **2-click + `QInputDialog`** gesture drawing a **Classic callout** (arrowhead at the anchor, dogleg leader into a horizontal landing, text on the landing, side-flipping to read away from the anchor). The spine is a **pure `plan_annotation` draw plan** (numpy, no Qt) that turns an annotation + camera into screen-space segments, texts and hit-boxes — **shared by both rendering and picking**, so what the user can click is by construction exactly what they see. Measurement text is **derived at draw time** from the geometry and the document's `Units` (never stored — switching metric↔imperial re-labels every dimension instantly). Round-trips through `.berm` (per-`Definition` `annotations` block, `schema_version` 1→2, older files still open) — deliberately **not** exported to OBJ/glTF. Full lifecycle: click / Shift-toggle / **hover-highlight** select, Eraser + Delete (undoable, composing into a single undo with any co-deleted geometry), double-click a label to **retype** (`EditLabelTextCommand`), and **move** via the existing Move tool (a Dimension's offset or a Label's text_pos shifts, composing into one undo alongside geometry). Additive `main_window.py` (issue #48 finding count held at 9). Python-only (ctest stays 79/79). **M7e** ✅ *(shipped v0.2.5)* — **Scenes** (saved views): the **Scenes dock** saves the current view — **camera + tag visibility + render style (Face Style / X-Ray)** — as a named Scene, recalled by clicking it with a **SketchUp-style animated camera tween** (orbit-decomposition interpolation, `QVariantAnimation` + `InOutSine`, ~700 ms; interrupted by any manual camera move). Scenes are managed with full undo (create / update / rename / delete / reorder via `CreateViewCommand` … `ReorderViewCommand`) and round-trip through `.berm` (`schema_version` 2→3, new top-level `scenes` + `style` keys, older files still open). The spine is a pure `SavedView` snapshot + `ViewLibrary` on `Model` (mirroring `TagLibrary`) with a pure numpy `interpolate_pose` shared by nothing but tested in isolation; recall applies tags/style instantly and only the camera flies. Also persists the document render style for the first time (previously X-Ray/Wireframe was silently lost on save). Python-only (ctest stays 79/79). **All M7 sub-milestones (M7a–M7e) shipped.**

### Phase 2.5 — Parity & Polish *(v0.3 → v0.12)*

**End state:** Bermake looks and feels like the tool it is — a real toolbar, a
modern side panel, materials, and SketchUp-standard interaction — on a
codebase with no outstanding bug or lint debt.

- **M7.1: Triage & burn-down** ✅ *(shipped v0.3.0)* — verified all 91 open issues against the code and closed 42: 7 that had already shipped, 9 behavior bugs (Scale's world-Z cursor plane, Explode double-ownership, edge-only groups unpickable, selection dropped on undo, Rotate auto-tilt, stale tool options on unit change, no-op Scene reorder dirtying the document, annotations vanishing outside their context, and the kernel's non-tombstoning `HalfEdgeMesh::clear`), the full ruff/clang-format cleanup with a CI lint gate, CI action modernization ahead of the Sept 2026 Node 20 removal, GL buffer eviction, glTF input hardening, and the review roll-ups. The 50 left open are the scheduled feature gaps for M7.2–M7.6, M10 and M12+, plus #17 (command-stack memory profiling, moved to M10) and #99 (the umbrella tracking the sub-items that proved larger than their parent issue) — each assigned to a named milestone.
- **M7.2: UI shell** ✅ *(shipped v0.4.0)* — seven SketchUp-style dockable toolbars driven by a Qt-free action registry, which collapsed the three-places-per-command duplication (menu, toolbar, shortcut) into one declaration each; 29 original monochrome SVG icons loaded through `importlib.resources` and recolored to the active palette; per-tool cursors composited from those same icons, so redrawing an icon updates its cursor and the two cannot drift; five context-sensitive right-click menus reusing the menu bar's own actions; Select All and Zoom Extents; and window, toolbar and dock layout persistence with a guarded restore and a Reset Toolbars escape hatch. Python-only — `ctest` stays 82/82.
- **M7.3: Outliner & Properties panel** ✅ *(shipped v0.5.0)*. The right edge became one Blender-style panel: a model **Outliner** (hierarchy, per-instance visibility toggle, inline rename, two-way selection sync, double-click to enter a context) over an **icon-tabbed Properties editor** whose five tabs absorbed the Materials, Tags and Scenes docks and the three per-tool option bars. `Instance` gained a per-instance `name` (falling back to `definition.name`, so renaming one instance of a shared component does not rename its siblings) and `hidden`, with **Hide / Unhide / Unhide All** declared through M7.2's registry (`H` / `Shift+H`) and an undoable `HideInstancesCommand` / `RenameInstanceCommand`; `.berm` `schema_version` 3 → 4, additive, so v0.4.0 files still open. Row derivation and entity summaries are **pure functions** (`outliner_rows`, `instance_path`, `entity_summary`, `selection_bounds`), so the logic most likely to be wrong is tested with no `QApplication`. Entity Info closes the panel half of [#73](https://github.com/Parrow-Horrizon-Studio/bermake/issues/73). The bottom bar split into a prompt, coordinates, and a labelled Measurements box. Python-only: `ctest` stays 82/82.
- **M7.4: Geometry tools** ✅ *(shipped v0.6.0)*. Two new SketchUp-standard tools, Offset and Follow Me, plus four parametric primitives (box, cylinder, cone, sphere) placed through a shared `PrimitiveTool` base. Push/Pull's sweep logic was pulled out into a shared, Qt-free `sweep_support` layer that Follow Me now also builds on. Tools are addressable by a stable id rather than by shortcut, since five of the six new tools carry none, which the old `ToolManager` registry could not have supported. Also closed two debt items: `#60` (`ScaleTool`'s local-to-world lift) and `#92` (the inverse-transpose face normal under non-uniform scale). `ctest` moved 82 → 104.
- **M7.5a: Materials** ✅ *(shipped v0.7.0)*. A PBR-shaped `Material` (base colour, alpha, metallic, roughness) and a Phong approximation (`phong_material_for`) that makes every editable field visibly change what is on screen, without matching the eventual M12 renderer pixel for pixel. Faces gained independent front and back material slots, with a distinct cool blue-grey back default so an unpainted reversed face is obvious at a glance, and the Paint tool resolves the clicked side from the pick ray. A material with `alpha < 1.0` draws in a sorted second pass: `plan_face_batches` keys batches on the `(front, back)` material pair and stable-sorts translucent faces into a contiguous suffix, the draw loop splits into two traversals so every opaque batch across every definition precedes any translucent one, and the fragment shader gained a second back-facing uniform set plus a normal flip that also fixed a pre-existing back-face lighting bug. Because face buffers are per-definition, the back-to-front sort works within a definition and orders definitions among themselves by world centroid rather than sorting the whole scene per face, a stated limitation carried forward into M12. Material add, edit and delete are undoable and exposed from a Materials page in Properties; deletion is model-wide, so a material painted across two groups is fully reassigned to Default rather than left dangling in one of them. The Paint tool gained drag-to-paint, one stroke composing into a single undo step. `Tag` gained a per-tag colour, seeded from a cycling palette, and a Color-by-Tag view mode that resolves per instance and bypasses material colour: the tag colour is substituted for both sides' material *before* the face style resolves, so diffuse, ambient and specular all derive from the tag and every face on a tag looks the same whatever it was painted, while each face style keeps its own rule (Hidden Line stays a flat background fill, Monochrome stays one grey). Material alpha is deliberately not bypassed, since alpha is not a colour: a translucent face still blends and still draws in the sorted pass. `.berm` schema moved 4 to 5, additive, so v0.6.0 files still open. Also closed two `#106` debt items: the primitive footprint preview and its commit now share one plane convention, and `AddVertexCommand.vertex_id` replaced three private reach-ins. Textures and UV mapping were split out as **M7.5b**, because they cut the same layers again for a different reason and because translucency's draw-order decision deserved undivided attention.
- **M7.5b: Textures** ✅ *(shipped v0.8.0)*. A material can carry an image, projected onto every face painted with it at a real-world size set by the material's `texture_size`, so a brick texture tiles at brick scale instead of stretching to fit each face; the projection uses the face's own plane in definition-local space, so a moved or rotated group carries its textures with it rather than sliding them across the surface. `Texture`/`TextureLibrary` sits beside `MaterialLibrary` on `Model`, storing the original encoded bytes rather than a re-encode (a byte-exact round trip) plus dimensions and a `has_transparency` flag scanned once at import, since `QImage.hasAlphaChannel()` reports the image format rather than whether any texel is actually non-opaque. A face can override the projection with its own offset, uniform scale and rotation, independently per side, because M7.5a's independent front and back materials mean a face can carry two different textures: the face vertex format grew to two UV sets, 10 floats and 40 bytes, front at attribute location 2 and back at location 3, selected by `gl_FrontFacing` exactly as the two-sided material uniforms already are. The sampled colour multiplies `base_color`, so a white material shows the image unchanged and a coloured one tints it, and the sampled alpha multiplies the material's alpha, so a cutout texture draws in M7.5a's sorted translucent pass automatically. UVs are baked into the vertex buffer at upload rather than computed in the shader from uniforms, reusing the buffer-snapshot re-upload mechanism M7.5a built for the translucency-set change; the naive per-face bake made buffer upload 15.7 times slower before it was vectorised across faces. `.berm` schema moved 5 to 6, additive, so v0.7.x files still open; textures persist with a `next_id` in the same shape as the material, tag and view libraries, so deleting and reloading cannot let a new texture reuse a freed id that a material still references. The Materials page gained a texture picker, a clear button and `texture_size` fields, and the Properties panel gained per-face Front/Back placement fields (offset, scale, rotation), enabled only when exactly one face is selected. Texture add, assign, placement and delete are all undoable, including a one-undo-step drag directly on the face; `DeleteTextureCommand` exists and is fully tested but nothing in the UI reaches it, so a document keeps every image ever imported into it, matching M5b's kept-until-purge precedent for materials. Imported per-corner UVs, the OBJ `.mtl`/`vt` round-trip (`#80`) and glTF texture import were deferred to **M7.5c**, since they need UV storage on the mesh that this design deliberately excludes. `ctest` stays 104/104.
- **M7.5c: Imported UVs** ✅ *(shipped v0.9.0)*. Per-corner texture coordinates stored on the mesh, so a textured model imported from OBJ or glTF keeps the UV layout its author gave it instead of being reprojected onto M7.5b's plane projection. Delivered in three stages against one design. **Stage 1, storage**: a `(face_id, Side)`-keyed UV sidecar on `Scene`, beside the face-keyed sidecars M5b through M7.5b already put there, holding one coordinate per *loop corner* rather than per vertex, because the kernel welds coincident vertices and a UV seam needs one welded vertex carrying different coordinates on each face that meets there, which per-vertex storage cannot express; the renderer makes stored UVs the bake's base with M7.5b's per-face placement composing on top, so painting over an imported model still behaves. `.berm` schema moved 6 to 7, additive and sparse, so v0.8.x files still open. Verifying the transfer surfaced a live bug the design had not scoped: `HalfEdgeMesh::split_edge` removes and rebuilds both incident faces, so splitting an edge on a painted face silently un-painted it, and had done since M5b; the repair went in at the shared transfer point rather than as UV maintenance layered on top of a broken transfer. **Stage 2, OBJ**: `vt` and `map_Kd` round trip, closing the coordinate half of `#80`. Export writes a face's UVs only when that face has stored UVs of its own or its front material carries a texture, because measurement showed writing them unconditionally tripled an untextured 40 by 40 grid's file size (263,924 bytes against 85,830) and, worse, froze the model's live plane projection into stored coordinates on re-import, so a later `texture_size` edit silently stopped reprojecting that face. `map_Kd` paths resolve both the document's base directory and the candidate before comparing, so a relative escape, an absolute path or a symlink resolving outside the document's own folder is skipped rather than read into the model and saved into the `.berm`. **Stage 3, glTF**: `TEXCOORD_0` and texture images through the Assimp bridge, embedded for `.glb` and written as siblings for `.gltf`, closing `#80` entire. Export needs a per-primitive vertex pool that OBJ never did, because glTF indexes POSITION and TEXCOORD_0 through one shared index buffer whose accessor counts must be equal, so a UV seam on a shared position has to duplicate that position once per distinct UV, where OBJ's `f v/vt` indexes two pools independently and a seam costs it nothing. The stage's decisive finding was that the design had the V flip on the wrong side: Assimp's own glTF2 importer applies `1 - v` before the bridge ever sees a coordinate, so import performs no flip and export performs one, and `test_assimp_already_flips_v_CI_GATE` pins that against a fixture's raw buffer bytes rather than trusting a comment. Shipped as planned, the second flip would have rendered every imported texture upside down while every count, length and round trip still matched, which is why this milestone's render-and-look gate was also shown to *discriminate*, by monkeypatching the flip back in and watching red move to the bottom of the quad. Also fixed `#114`, which proved to be three call sites rather than one: `MakeGroupCommand`, `ExplodeInstanceCommand` and `Model.clone_definition` each rebuild faces with `add_face_from_loop` and carried no face-keyed sidecar across, so Make Group, Explode and Make Unique all silently un-painted whatever they touched. `ctest` moved 104 to 121 across the milestone's window, 12 of those from the `#110` kernel normal fixes and 5 from stage 3's bridge, the first C++ movement since M7.4. Imported faces are still assigned their material front-only, so an imported model still renders as M7.5a's unpainted grey from behind.
- **M7.6: Interaction parity** — face-split on draw-into-face, richer inferencing, construction guides, on-canvas measurement, smart-select. Split into three sub-milestones during its design, because it spans a kernel change, a schema change and selection semantics, and was wider than M7.5 was before that split. **M7.6a: Face split** ✅ *(shipped v0.10.0)* — drawing a line or arc across a face divides it, and erasing the cut merges it back. `HalfEdgeMesh::split_face` is the inverse of M3c's `dissolve_edge` and the last major topology operation the kernel was missing; it takes both new boundary loops already triangulated, because the triangulator is in Python and `add_face_from_loop` already accepts a caller-supplied triangulation, and it validates the partition as a single directed-edge multiset comparison that carries three guarantees at once: that the two loops partition the parent, that both children preserve its winding, and that the shared chain is traversed in opposite directions, which is what makes the two rebuild calls claim opposite half-edges. Every check runs before the first mutation, so a rejected split leaves the mesh untouched; that atomicity is the reason the operation lives in the kernel rather than in a Python command, since `add_face_from_loop` throws on a missing edge and a throw after `remove_face` would leave a hole. Both children inherit the parent's front and back material and placement, and a chain vertex's stored UV is resolved by barycentric interpolation inside the parent's own triangulation rather than by fitting an affine map, which is exact only for bermake's own projection and wrong for the imported layouts M7.5c added. The cutting path is read from the drawing tool's own gesture state rather than searched for in the mesh, because an edge lying on a face carries no face pointer to find it by; the containment gate collects every parameter where a chain segment touches the face's boundary, counting vertex touches and collinear overlaps rather than only transversal crossings, and requires every resulting sub-interval's midpoint to be strictly inside, a rule validated against dense-sampling ground truth on 10,000 random polygons with zero false accepts and zero false rejects. The Eraser now dissolves a coplanar interior seam instead of deleting both faces, because without that the obvious way to undo a split by hand destroyed the wall; the merged face's sidecars follow an agree-or-drop rule with absence counting as a value, revising M7.5c's D4a, which was correct only while a merge was an implementation detail rather than a user gesture. Two carve-outs are recorded rather than hidden: a chain with an interior vertex leaves the two children sharing two edges, which `dissolve_edge` refuses by design, so **erase-to-merge round-trips a single-segment chord and not a multi-segment cut**; and `ArcTool` lacks the Line tool's mid-edge snap resolution, so **an arc splits only corner to corner, not edge to edge**. `.berm` stays at `schema_version` 7. `ctest` moved 121 to 134, the second movement since M7.4 and the first from a kernel operation. **M7.6b: Inference, construction guides and on-canvas measurement** ✅ *(shipped v0.11.0)*: the interaction half. Its first job turned out to be making the inference bermake already had reachable at all, because two pre-existing defects were measured during its design rather than assumed: `ON_FACE`'s candidate is always within tolerance, since its position is the ray-face intersection and so reprojects onto the cursor exactly, and it outranked `AXIS_LOCK`, so axis inference was invisible in the ordinary case of drawing on a face, confirmed at five of five sampled pixels lying exactly on the red axis over a quad; and `resolve_drawing_plane` keyed off the winning snap kind rather than the face under the cursor, so starting a rectangle at a wall's own corner dropped it onto the ground. `SnapResult.face_id` now rides every result whichever kind wins, which is what makes the precedence demotion safe. On top of that sit the acquisition-driven inferences SketchUp users' hands already know: dwelling on a vertex or edge acquires it, an acquired edge yields Parallel and Perpendicular, an acquired vertex radiates From-Point lines along the three axes, and arrow keys or Shift lock the inference. Perpendicular resolves inside the gesture's drawing plane, because perpendicular to a line in 3D is a plane rather than a direction, and it is skipped rather than guessed where the cross product degenerates. Construction guides ride `Definition.annotations`, the rail `Dimension` and `Label` already use, which gave them context-local placement, persistence, picking, selection, erasing and undo without new machinery; the new work was two dataclasses, two codec kinds, and the screen-space layout of an infinite line, which clips against the camera near plane in 3D before projecting and then Liang-Barsky clips to the viewport, because projecting two far-apart samples produces mirrored or runaway coordinates the moment the line crosses behind the eye. Tape Measure creates guides by default with `Ctrl` for measure-only, and hiding guides makes them unpaintable, unpickable and unsnappable together, because a hidden thing the user can still erase or be pulled toward is a trap. The measurement readout added `Tool.measurement_text` across eleven tools, numeric and distinct from the prose `status_text`, which as a side effect filled a Measurements box that had been blank unless you typed into it. `.berm` goes to `schema_version` 8; `ctest` moved 134 to 135 and pytest 2578 to 2694. The milestone is also a record of its own review process: ten tasks produced twelve fix rounds, only one task passed review first time, subagents refused an instruction in their brief seven times and were right on every occasion, and six of the plan's stated mutants turned out not to discriminate. The whole-branch review then found that the arrow-key axis lock had never worked, pinning the line to an axis through the world origin rather than the gesture anchor, which four per-task reviews had missed because the plan's own test asserted the broken behaviour. Details and all seven corrected claims are in `docs/2026-09-21-M7.6b-inference-design.md`. **M7.6c: Smart-select, selection set operations and vertex selection** ✅ *(shipped v0.12.0)* — the selection-semantics third, and the only one of the three that changed no kernel code and no file format. It closes four M4b carry-overs open since v0.1.1: #37 smart-select, #38 set operations, #36 vertex selection and #39 the dashed crossing rectangle. Double-clicking a face selects it with its bounding edges, an edge with its adjacent faces, and a vertex with its incident edges; triple-clicking floods the connected component. The flood walks shared vertices rather than shared edges, which is what makes its result independent of which entity kind seeded it and therefore what lets the two gestures compose. Qt sends no triple-click event, so a pure run counter with an injected clock recognises the third press, and the work of the milestone turned out to be less in the counting than in what Qt actually delivers: a double-click replaces the second click's `MouseButtonPress` with `MouseButtonDblClick`, so the counter had to be fed from both handlers, and the release that trails every double-click re-ran the ordinary single-click pick and erased the smart selection a frame after it appeared. M4e's `_suppress_next_release` existed for exactly that and had only ever been set by the enter-a-group branch. Vertex selection is a fifth `Selection` kind behind an opt-in `View > Select Vertices` mode, off by default, because extending `pick_selectable`'s thin-target-wins rule to vertices unconditionally would change what every click near a corner selects across every tool that picks, on geometry that is mostly corners; SketchUp exposes no direct vertex selection at all, partly for that reason. One union in `transform_support.selection_vertices` makes a bare vertex draggable in Move, Rotate and Scale at once, and the same chokepoint turned out to be read by Make Group and Make Component too, where a bare vertex rode into a group and duplicated itself. Invert, Grow, Shrink, All with Same Material and All with Same Tag live on the right-click `Select` submenu except Invert, which needs no entity under the cursor and sits on `Edit`; they are pure functions in a new Qt-free `selection_ops` module that both the tool and the menus call, so the adjacency walk behind triple-click and Grow cannot drift into two copies. `.berm` stays at `schema_version` 8 and `ctest` stays at 135, the first milestone since M7.4 to move neither; pytest went 2694 to 2842. The milestone is also a record of what its own process caught: seven defects in the plan's own code and test snippets, found because implementers ran them and reported the failure instead of working around it, and fifteen of forty-seven stated mutants that turned out not to discriminate. The whole-branch review then found that the headline gesture never survived a real mouse release, which eleven task reviews had missed because no test in the branch ever sent one. Details and every corrected claim are in `docs/2026-09-22-M7.6c-smart-select-design.md`. **Phase 2.5 is complete.**

### Phase 2.6 — Beta Readiness *(v0.12 → v0.15)*

**End state:** the application feels like SketchUp, its tools are stable, and
outside testers who use SketchUp daily can run it and report back.

Phase 3 was next on the roadmap and is deliberately held. The reasoning is in
this phase's own name: the plugin system in M8 creates a public API surface,
and committing to one before the modelling tools are stable means other
people's code binds to something still moving. Polish and issue work first,
then the platform.

- **M7.7: Environment presets, the welcome dialog and the light/dark theme** ✅ *(shipped v0.13.0)* — the look-and-first-run milestone, and the first since M7.4 to change the file format without touching the kernel. The viewport's background becomes a per-document `Environment` with five presets: Sky and Ground, Neutral Grey, Landscape, Plain White and Studio. The argument that shapes the whole milestone is that a preset which changes the background without changing what is drawn on top of it is not a complete preset: `_USER_EDGE_COLOR` was `0.85` light grey, tuned for one dark viewport and effectively invisible on white, and the same was true of the grid, the Monochrome fill and, as the visual pass later found, the snap markers. So each preset carries its own edge and grid colours, and a parametrized contrast-floor test requires every preset's ink to differ by at least 0.35 relative luminance from every *enabled* backdrop of that preset. That guarantee is the milestone's real deliverable: when two more environments were added mid-flight at the request of the visual pass, they were held to the same floor without anyone having to remember. `STUDIO` reproduces the pre-M7.7 renderer constant for constant, asserted rather than assumed, so a user who keeps the dark look provably sees v0.12.0. Sky and ground draw as one fullscreen quad after the clear, with the horizon falling out of the view ray's Z component rather than being drawn as geometry, which makes it correct under perspective and under camera roll for free; the pass returns early when both halves are disabled, so Plain White and Studio pay nothing. The grid and the edges re-upload rather than taking a tint uniform, because the grid carries two colours that move in opposite directions between presets and no scalar multiplier maps one pair onto the other; edges get a fourth staleness cause in `_ensure_buffers` beside the three already there for the identical reason. `.berm` goes to `schema_version` 9, defaulting per key to the dark environment so every file written before this milestone reopens exactly as its author left it, while a new document opens in Sky and Ground. The application gains its first QDialog, a welcome window offering six templates over the five environments, with units as a separate axis so four templates and one dropdown cover what SketchUp needs sixteen template names for. Light and dark themes close #101, which turned out to be a refresh rather than a capability: M7.2 had already built icon recolouring and left `clear_icon_cache()` in the tree with zero callers. `ctest` stays at 135 and no C++ changed; pytest went 2842 to 2961. The milestone is also a record of its own process, and the record is uncomfortable: **nine defects were found in the plan's own text**, every one by an implementer who ran the snippet and reported the failure, and **six tests passed for the wrong reason**, five of them originating in plan-authored test code. The worst was a set-membership assertion whose docstring claimed to catch a colour-argument swap that set membership is mathematically incapable of detecting. Details and every corrected claim are in `docs/2026-09-23-M7.7-environment-welcome-theme-design.md`.
- **M7.8: Rename to Bermake** ✅ *(shipped v0.14.0)* — the project's name changed project-wide: the Python package, the C++ headers, the CLI entry point, the native file format (`.pluton` to `.berm`), and every in-tree reference in code, tests and docs. The prior name shared its search terms with Microsoft Pluton, a silicon security processor shipped in Windows PCs; a different class of goods, so a conflict was never certain, but the name was a poor asset regardless, since searching for it surfaces Microsoft's product instead. Bermake is coined, so it is searchable and defensible. Two mechanical commits carried the sweep, recorded in `.git-blame-ignore-revs` so `git blame` on any of the swept files skips past them. No application behavior changed: this milestone renames, it does not add or fix anything, and it is a breaking release for anyone with an existing install, since documents written by v0.13.0 and earlier cannot be opened and stored preferences reset to their defaults. Phase 2.6's anchors are unchanged.
- **M7.9: Windows build** — a build a tester can double-click, so feedback does not require assembling an MSVC, vcpkg and CMake toolchain first. Not signed, not an installer, not cross-platform; those stay with M9. Without it this phase makes the application worth showing people without making it possible to show them.
- **M7.10 onward: issue-driven** — working the open backlog down, roughly half of which traces to Phase 2 milestones. Scope per milestone decided by triage, as M7.1 did for Phase 2.5.

### Phase 3 — Platform *(v0.15 → v0.17)*

**End state:** Public beta. Cross-platform installers. Plugin ecosystem possible.

- **M8: Plugin system** — Python plugin API; plugin manifest format; plugin loader; sample plugins demonstrating common extension patterns.
- **M9: Mac support and installers** — macOS port (code signing + notarization); polished installers for Windows / Linux / macOS; first formal release announcement; documentation site; basic tutorial materials.

### Phase 4 — Scale and Integration *(v0.17 → v1.0)*

**End state:** v1.0. Handles real architectural projects. Integrates with industry rendering tools.

- **M10: Performance and scale** — profile and migrate geometry hot paths from Python to C++; BVH/spatial indexing for large scenes; multi-threading where applicable; goal is comfortable performance on full-building / neighborhood-scale projects.
- **M11: Industry integration** — additional file formats (FBX, USD, DAE); export pipelines tuned for V-Ray, Twinmotion, Lumion, Enscape workflows; potential "live link" plugin for at least one major external renderer.

### Phase 5 — Self-Sufficient *(v1.0+)*

**End state:** Bermake stands alone — full modeling and full rendering, no need to export.

- **M12: Real-time PBR viewport** — Eevee-style real-time renderer; Vulkan (probably) for forward-compatible performance; physically-based materials; real-time shadows, reflections, ambient occlusion.
- **M13: Production renderer** — Cycles-style path tracer for final renders; GPU-accelerated; architectural-grade rendering quality (interior lighting, glass, materials, global illumination).

### Phase 6 — Architectural Maturity & Open Horizons *(exploration tier)*

**End state:** Bermake becomes a complete architect's tool. Priority within this phase emerges from real user demand.

- **Layout-equivalent (integrated)** — 2D documentation from the 3D model: floor plans, elevations, sections, construction documents, title blocks, dimensions, scales. **Integrated into Bermake** (single install), not a separate companion app like SketchUp's Layout.
- **BIM-lite features** — smart architectural components (walls/doors/windows that know what they are); component properties and basic schedules; intentionally *not* full IFC / Revit-grade BIM (that is a multi-year project in its own right).
- **Open horizons** — animation and walkthrough cameras; VR support; AI-assisted modeling features; anything else the community surfaces as valuable.

---

## 7. Time-Shape Estimate

Honest solo-developer pacing at hobby intensity:

| Phase | Estimated duration |
|---|---|
| Phase 1: Foundation | 6–12 months |
| Phase 2: Modeling App | 12–18 months |
| Phase 3: Platform | 6–12 months |
| Phase 4: Scale and Integration | 12–18 months |
| Phase 5: Self-Sufficient | 2–5 years |
| Phase 6: Exploration | open-ended |

**v1.0: 3–6 years from project start. Phase 5 vision: 5–10+ years.** Blender took roughly 25 years to reach its current standing; Bermake is on a similar honest trajectory. The time-shape is part of the project's identity — sustainable pace beats burnout.

---

## 8. Open Questions / Deferred Decisions

These are flagged for resolution during the implementation plan or at the relevant milestone:

- **C++ test framework** — GoogleTest vs Catch2. Deferred to the M0 implementation plan; either is fine.
- **`.berm` file format** — binary (e.g., custom binary or SQLite-based) vs JSON-based vs hybrid. Decided in M6 implementation plan; must support schema versioning from day 1.
- **Internationalization (i18n)** — not a concern for v0.x; revisit for Phase 3 / v0.5.
- **Plugin API surface** — detailed design deferred to M8 implementation plan.
- **Specific Python version target** — most recent stable Python at M0 start (probably 3.12 or 3.13); minimum supported version policy TBD.
- **Specific Qt version** — Qt 6.x latest LTS at M0 start.
- **Vulkan vs OpenGL for the eventual native renderer** — decided in M12 planning.
- **Funding model** — not relevant until at least Phase 3. Likely follows Blender's pattern (donations, optional cloud services, no paid features in core software).

---

## 9. References

- **nanobind**: https://nanobind.readthedocs.io/
- **CGAL**: https://www.cgal.org/
- **PySide6 / Qt for Python**: https://doc.qt.io/qtforpython-6/
- **ModernGL**: https://moderngl.readthedocs.io/
- **scikit-build-core**: https://scikit-build-core.readthedocs.io/
- **vcpkg**: https://vcpkg.io/
- **Blender source code** (architectural reference): https://projects.blender.org/blender/blender
- **FreeCAD source code** (Python + C++ + CGAL reference): https://github.com/FreeCAD/FreeCAD
- **SketchUp documentation** (modeling paradigm reference): https://help.sketchup.com/

---

## 10. Document History

| Date | Author | Change |
|---|---|---|
| 2026-05-16 | Rowee Apor | Initial design from brainstorming session |
| 2026-08-02 | Rowee Apor | Phase 2.5 (Parity & Polish) inserted after M7; per-phase version ranges relabelled. Per-milestone shipped-status annotations are made in place as each milestone lands — `git log --follow` on this file is the authoritative change history. |
| 2026-09-12 | Rowee Apor | The roadmap's single M7.5 entry split into M7.5a and M7.5b, per the M7.5a design's decision to separate materials from textures. The version sequence within Phase 2.5 gains a step; M7.5a shipped v0.7.0. |
| 2026-09-19 | Rowee Apor | M7.5c gains its own roadmap entry. It had existed only as a deferral target inside M7.5b's paragraph, so the milestone that shipped v0.9.0 had nowhere to carry the in-place shipped annotation this document's 2026-08-02 convention requires. The per-phase version anchors move with it: M7.5c took v0.9.0 while M7.6 is still open, so Phase 2.5 now runs to v0.10 rather than v0.8, and Phase 3 and Phase 4 each shift up by the two minor versions Phase 2.5 absorbed. Phase 4 still ends at v1.0 and Phase 5 still opens there. Anchors remain directional, as section 6 says. |
| 2026-09-20 | Rowee Apor | M7.6 splits into three sub-milestones, M7.6a face split, M7.6b inferencing and guides, M7.6c smart-select, because it spanned a kernel change, a schema change and selection semantics and was wider than M7.5 was before its own split. M7.6a shipped v0.10.0. The per-phase anchors move again for the two sub-milestones that split adds: Phase 2.5 now runs to v0.12, and Phases 3 and 4 shift up by two. Phase 4 still ends at v1.0. This is the second anchor shift in two days, which is what section 6 means by calling them directional rather than commitments. |
| 2026-09-21 | Rowee Apor | M7.6b shipped v0.11.0: inference reachability, acquisition-driven Parallel, Perpendicular and From-Point, inference locking, construction guides on the annotation rail at `schema_version` 8, the on-canvas measurement readout, and #31 and #108 closed. Phase 2.5 has one sub-milestone left, M7.6c smart-select, so the anchors set two days ago still hold: Phase 2.5 to v0.12, Phase 3 v0.12 to v0.14, Phase 4 v0.14 to v1.0. |
| 2026-09-23 | Rowee Apor | M7.6c shipped v0.12.0, closing M7.6 and with it **Phase 2.5**. Smart-select on double and triple click, the five selection set operations, vertex selection as an opt-in fifth kind, and the dashed crossing rectangle; #37, #38, #36 and #39 all close. No kernel change, no schema change: `.pluton` stays at 8 and `ctest` stays at 135. The anchors set on 2026-09-20 held exactly, Phase 2.5 ending at v0.12, so Phase 3 now opens at v0.12 and runs to v0.14, and Phase 4 runs v0.14 to v1.0. |
| 2026-09-24 | Rowee Apor | M7.7 shipped v0.13.0, opening **Phase 2.6, Beta Readiness**. Phase 3 is deliberately held: M8's plugin system creates a public API surface, and other people's code should not bind to one while the modelling tools are still moving. The new phase's goal is that outside testers who use SketchUp daily can run the application and report back, which makes **M7.8 a Windows build** rather than more features, since today running pluton means assembling an MSVC, vcpkg and CMake toolchain. M7.9 onward is issue-driven. The anchors move a third time: Phase 2.6 takes v0.12 to v0.15, Phase 3 v0.15 to v0.17, Phase 4 v0.17 to v1.0, and Phase 4 still ends at v1.0. The number of issue-driven milestones inside Phase 2.6 is not yet known, so v0.15 is the most directional anchor this document has carried; section 6 already says anchors are directions rather than commitments. `.pluton` goes to `schema_version` 9 and `ctest` stays at 135. |
| 2026-09-28 | Rowee Apor | The project renamed from Pluton to Bermake, shipping as **M7.8** at v0.14.0. The old name shared its search terms with Microsoft Pluton, a silicon security processor shipped in Windows PCs; a different class of goods, so a conflict was never certain, but the name was hard to search for regardless. The Windows build that this document's 2026-09-24 row still calls M7.8 is renumbered **M7.9**, and the issue-driven milestone after it is renumbered **M7.10 onward**; neither's text changed, only its number. Every document dated before this row, including the rows above this one, keeps the old name deliberately: each is a dated record of the project as it stood on the day it was written, and rewriting a past record to use a name that did not exist yet would falsify it. Sections 1 through 9 of this document describe the project as it is today and were swept to the new name; this section did not. |
