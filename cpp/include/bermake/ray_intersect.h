#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include "pluton/halfedge.h"

namespace pluton {

/// Result of a ray-mesh intersection.
struct RayMeshHit {
    std::uint32_t face_id;
    float t;                     // ray parameter (always > 0)
    std::array<float, 3> point;  // origin + t * direction
};

/// Brute-force ray-mesh intersection over every live face in `mesh`.
///
/// Iterates `mesh.next_live_face(...)`. For each face: walks the face's
/// triangulation (from `face_triangles(face_id)`); runs Möller-Trumbore on
/// each triangle. Returns the closest positive `t` hit across all triangles
/// (or `std::nullopt` if the ray misses everything).
///
/// Hit selection is two-sided: a ray hits a triangle from either face
/// orientation (we're picking, not shading).
///
/// `direction` does NOT need to be normalized; `t` is in `direction`-units.
std::optional<RayMeshHit> ray_intersect_mesh(const HalfEdgeMesh& mesh,
                                             const std::array<float, 3>& origin,
                                             const std::array<float, 3>& direction);

}  // namespace pluton
