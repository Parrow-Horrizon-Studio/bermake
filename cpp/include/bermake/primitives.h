#pragma once

#include "pluton/halfedge.h"
#include "pluton/mesh.h"

namespace pluton {

/// Axis-aligned cube primitive.
///
/// Bottom face sits on z = 0 (the world ground plane); x and y span
/// [-size/2, +size/2]; z spans [0, size]. Each of the 6 faces has its own
/// outward-pointing normal (flat shading), so corner vertices are duplicated
/// per face — 24 vertices, 36 indices total.
///
/// @param size  Edge length of the cube. Defaults to 1.0.
Mesh make_cube(float size = 1.0f);

/// Axis-aligned box primitive, as a polygonal HalfEdgeMesh.
///
/// Bottom face sits on z = 0 (the world ground plane); x spans
/// [-width/2, +width/2]; y spans [-depth/2, +depth/2]; z spans [0, height].
/// Six quad faces, sharing vertices at the 8 corners.
///
/// @param width   Extent along x. Defaults to 1.0.
/// @param depth   Extent along y. Defaults to 1.0.
/// @param height  Extent along z. Defaults to 1.0.
HalfEdgeMesh make_box(float width = 1.0f, float depth = 1.0f, float height = 1.0f);

/// Cylinder primitive, as a polygonal HalfEdgeMesh.
///
/// Centred on the origin in x and y; base circle sits on z = 0, top circle
/// on z = height. `segments` sides plus a top and bottom cap (each an
/// N-gon fanned from the mesh's triangulation convention).
///
/// @param radius    Radius of the base/top circle. Defaults to 1.0.
/// @param height    Extent along z. Defaults to 1.0.
/// @param segments  Number of sides around the circumference (>= 3). Defaults to 24.
/// @throws std::invalid_argument if segments < 3 (fewer sides collapse the
///         side quads and caps onto a plane instead of enclosing a volume).
HalfEdgeMesh make_cylinder(float radius = 1.0f, float height = 1.0f, int segments = 24);

/// Cone primitive, as a polygonal HalfEdgeMesh.
///
/// Centred on the origin in x and y; base circle sits on z = 0, apex at
/// z = height. `segments` triangular sides plus a bottom cap.
///
/// @param radius    Radius of the base circle. Defaults to 1.0.
/// @param height    Extent along z (apex height above the base). Defaults to 1.0.
/// @param segments  Number of sides around the circumference (>= 3). Defaults to 24.
/// @throws std::invalid_argument if segments < 3 (fewer sides collapse the
///         triangular sides and base onto a plane instead of enclosing a volume).
HalfEdgeMesh make_cone(float radius = 1.0f, float height = 1.0f, int segments = 24);

/// UV-sphere primitive, as a polygonal HalfEdgeMesh.
///
/// Centred on the origin in x and y; bottom pole sits on z = 0, top pole on
/// z = 2 * radius (so the sphere's centre is at (0, 0, radius)). `rings`
/// latitude subdivisions by `segments` longitude subdivisions; the two poles
/// are triangle fans rather than degenerate quads.
///
/// @param radius    Sphere radius. Defaults to 1.0.
/// @param rings     Number of latitude subdivisions (>= 2). Defaults to 12.
/// @param segments  Number of longitude subdivisions (>= 3). Defaults to 24.
/// @throws std::invalid_argument if rings < 2 (fewer than 2 rings leaves the
///         interior latitude ring vectors empty, since only levels 1..rings-1
///         are populated between the two poles) or if segments < 3 (fewer
///         longitude steps collapse each ring onto a degenerate line/point).
HalfEdgeMesh make_sphere(float radius = 1.0f, int rings = 12, int segments = 24);

}  // namespace pluton
