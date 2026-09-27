#include "bermake/ray_intersect.h"

#include <cmath>
#include <limits>

namespace bermake {

namespace {

// Möller-Trumbore ray-triangle intersection, two-sided.
//
// Returns the t parameter (always > 0 on hit; std::nullopt on miss).
// Backface culling is intentionally NOT applied: we're picking, not shading.
std::optional<float> ray_triangle(const std::array<float, 3>& origin,
                                  const std::array<float, 3>& dir, const std::array<float, 3>& v0,
                                  const std::array<float, 3>& v1, const std::array<float, 3>& v2) {
    const float e1x = v1[0] - v0[0];
    const float e1y = v1[1] - v0[1];
    const float e1z = v1[2] - v0[2];
    const float e2x = v2[0] - v0[0];
    const float e2y = v2[1] - v0[1];
    const float e2z = v2[2] - v0[2];

    // h = dir × e2
    const float hx = dir[1] * e2z - dir[2] * e2y;
    const float hy = dir[2] * e2x - dir[0] * e2z;
    const float hz = dir[0] * e2y - dir[1] * e2x;

    // a = e1 · h
    const float a = e1x * hx + e1y * hy + e1z * hz;

    // Parallel (or degenerate triangle): skip.
    constexpr float kEpsilon = 1e-8f;
    if (std::fabs(a) < kEpsilon) {
        return std::nullopt;
    }

    const float f = 1.0f / a;
    const float sx = origin[0] - v0[0];
    const float sy = origin[1] - v0[1];
    const float sz = origin[2] - v0[2];
    const float u = f * (sx * hx + sy * hy + sz * hz);
    if (u < 0.0f || u > 1.0f) {
        return std::nullopt;
    }

    // q = s × e1
    const float qx = sy * e1z - sz * e1y;
    const float qy = sz * e1x - sx * e1z;
    const float qz = sx * e1y - sy * e1x;

    const float v = f * (dir[0] * qx + dir[1] * qy + dir[2] * qz);
    if (v < 0.0f || u + v > 1.0f) {
        return std::nullopt;
    }

    const float t = f * (e2x * qx + e2y * qy + e2z * qz);
    if (t <= kEpsilon) {
        return std::nullopt;  // behind origin or on it
    }
    return t;
}

}  // namespace

std::optional<RayMeshHit> ray_intersect_mesh(const HalfEdgeMesh& mesh,
                                             const std::array<float, 3>& origin,
                                             const std::array<float, 3>& direction) {
    std::optional<RayMeshHit> best;
    float best_t = std::numeric_limits<float>::infinity();

    std::uint32_t f = mesh.next_live_face(0);
    while (f != HalfEdgeMesh::INVALID_ID) {
        const auto tris = mesh.face_triangles(f);  // flat: 3*T entries
        for (std::size_t i = 0; i + 2 < tris.size(); i += 3) {
            const auto a_id = static_cast<std::uint32_t>(tris[i]);
            const auto b_id = static_cast<std::uint32_t>(tris[i + 1]);
            const auto c_id = static_cast<std::uint32_t>(tris[i + 2]);
            const auto a = mesh.vertex_position(a_id);
            const auto b = mesh.vertex_position(b_id);
            const auto c = mesh.vertex_position(c_id);

            auto t = ray_triangle(origin, direction, a, b, c);
            if (t && *t < best_t) {
                best_t = *t;
                RayMeshHit hit;
                hit.face_id = f;
                hit.t = *t;
                hit.point = {
                    origin[0] + direction[0] * (*t),
                    origin[1] + direction[1] * (*t),
                    origin[2] + direction[2] * (*t),
                };
                best = hit;
            }
        }
        f = mesh.next_live_face(f + 1);
    }
    return best;
}

}  // namespace bermake
