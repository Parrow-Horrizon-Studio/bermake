#include "bermake/primitives.h"

#include <cmath>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace bermake {

namespace {

// Add a planar (or fan-triangulable) N-gon face to `mesh` from a vertex-id
// loop, wiring up its boundary edges first. Triangulated as a fan from
// loop[0] — valid for the convex loops every generator below produces
// (box faces, cylinder/cone sides and caps, sphere bands and pole fans).
std::uint32_t add_polygon_face(HalfEdgeMesh& mesh, const std::vector<std::uint32_t>& loop) {
    const std::size_t n = loop.size();
    for (std::size_t i = 0; i < n; ++i) {
        mesh.add_halfedge_pair(loop[i], loop[(i + 1) % n]);
    }
    std::vector<std::int32_t> triangles;
    triangles.reserve((n - 2) * 3);
    for (std::size_t i = 1; i + 1 < n; ++i) {
        triangles.push_back(static_cast<std::int32_t>(loop[0]));
        triangles.push_back(static_cast<std::int32_t>(loop[i]));
        triangles.push_back(static_cast<std::int32_t>(loop[i + 1]));
    }
    return mesh.add_face_from_loop(loop, triangles);
}

}  // namespace

Mesh make_cube(float size) {
    constexpr std::size_t kCubeFaces = 6;
    constexpr std::size_t kVertsPerFace = 4;
    constexpr std::size_t kFloatsPerVertex = 3;
    constexpr std::size_t kTrisPerFace = 2;
    constexpr std::size_t kIndicesPerTri = 3;

    const float h = size * 0.5f;

    Mesh mesh;
    mesh.positions.reserve(kCubeFaces * kVertsPerFace * kFloatsPerVertex);
    mesh.normals.reserve(kCubeFaces * kVertsPerFace * kFloatsPerVertex);
    mesh.indices.reserve(kCubeFaces * kTrisPerFace * kIndicesPerTri);

    struct Face {
        float v[4][3];
        float n[3];
    };

    const Face faces[6] = {
        // +X face (right): normal (1, 0, 0)
        {{{+h, -h, 0.f}, {+h, +h, 0.f}, {+h, +h, size}, {+h, -h, size}}, {1.f, 0.f, 0.f}},
        // -X face (left): normal (-1, 0, 0)
        {{{-h, +h, 0.f}, {-h, -h, 0.f}, {-h, -h, size}, {-h, +h, size}}, {-1.f, 0.f, 0.f}},
        // +Y face (back): normal (0, 1, 0)
        {{{+h, +h, 0.f}, {-h, +h, 0.f}, {-h, +h, size}, {+h, +h, size}}, {0.f, 1.f, 0.f}},
        // -Y face (front): normal (0, -1, 0)
        {{{-h, -h, 0.f}, {+h, -h, 0.f}, {+h, -h, size}, {-h, -h, size}}, {0.f, -1.f, 0.f}},
        // +Z face (top): normal (0, 0, 1)
        {{{-h, -h, size}, {+h, -h, size}, {+h, +h, size}, {-h, +h, size}}, {0.f, 0.f, 1.f}},
        // -Z face (bottom): normal (0, 0, -1)
        {{{-h, +h, 0.f}, {+h, +h, 0.f}, {+h, -h, 0.f}, {-h, -h, 0.f}}, {0.f, 0.f, -1.f}},
    };

    for (const auto& face : faces) {
        const std::uint32_t base = static_cast<std::uint32_t>(mesh.vertex_count());
        for (int v = 0; v < 4; ++v) {
            mesh.positions.push_back(face.v[v][0]);
            mesh.positions.push_back(face.v[v][1]);
            mesh.positions.push_back(face.v[v][2]);
            mesh.normals.push_back(face.n[0]);
            mesh.normals.push_back(face.n[1]);
            mesh.normals.push_back(face.n[2]);
        }
        mesh.indices.push_back(base + 0);
        mesh.indices.push_back(base + 1);
        mesh.indices.push_back(base + 2);
        mesh.indices.push_back(base + 0);
        mesh.indices.push_back(base + 2);
        mesh.indices.push_back(base + 3);
    }

    return mesh;
}

HalfEdgeMesh make_box(float width, float depth, float height) {
    const float hw = width * 0.5f;
    const float hd = depth * 0.5f;

    HalfEdgeMesh mesh;

    const std::uint32_t v000 = mesh.add_vertex(-hw, -hd, 0.f);
    const std::uint32_t v100 = mesh.add_vertex(+hw, -hd, 0.f);
    const std::uint32_t v110 = mesh.add_vertex(+hw, +hd, 0.f);
    const std::uint32_t v010 = mesh.add_vertex(-hw, +hd, 0.f);
    const std::uint32_t v001 = mesh.add_vertex(-hw, -hd, height);
    const std::uint32_t v101 = mesh.add_vertex(+hw, -hd, height);
    const std::uint32_t v111 = mesh.add_vertex(+hw, +hd, height);
    const std::uint32_t v011 = mesh.add_vertex(-hw, +hd, height);

    add_polygon_face(mesh, {v100, v110, v111, v101});  // +X
    add_polygon_face(mesh, {v010, v000, v001, v011});  // -X
    add_polygon_face(mesh, {v110, v010, v011, v111});  // +Y
    add_polygon_face(mesh, {v000, v100, v101, v001});  // -Y
    add_polygon_face(mesh, {v001, v101, v111, v011});  // +Z (top)
    add_polygon_face(mesh, {v010, v110, v100, v000});  // -Z (bottom)

    return mesh;
}

HalfEdgeMesh make_cylinder(float radius, float height, int segments) {
    if (segments < 3) {
        throw std::invalid_argument("make_cylinder: segments must be >= 3, got " +
                                    std::to_string(segments));
    }
    const int n = segments;

    HalfEdgeMesh mesh;
    std::vector<std::uint32_t> bottom(static_cast<std::size_t>(n));
    std::vector<std::uint32_t> top(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const float theta =
            2.0f * std::numbers::pi_v<float> * static_cast<float>(i) / static_cast<float>(n);
        const float x = radius * std::cos(theta);
        const float y = radius * std::sin(theta);
        bottom[static_cast<std::size_t>(i)] = mesh.add_vertex(x, y, 0.f);
        top[static_cast<std::size_t>(i)] = mesh.add_vertex(x, y, height);
    }

    // Sides: one outward-facing quad per segment.
    for (int i = 0; i < n; ++i) {
        const int j = (i + 1) % n;
        add_polygon_face(mesh,
                         {bottom[static_cast<std::size_t>(i)], bottom[static_cast<std::size_t>(j)],
                          top[static_cast<std::size_t>(j)], top[static_cast<std::size_t>(i)]});
    }

    // Bottom cap: ring traversed in reverse so the fan's normal points -z.
    std::vector<std::uint32_t> bottom_cap(static_cast<std::size_t>(n));
    bottom_cap[0] = bottom[0];
    for (int i = 1; i < n; ++i) {
        bottom_cap[static_cast<std::size_t>(i)] = bottom[static_cast<std::size_t>(n - i)];
    }
    add_polygon_face(mesh, bottom_cap);

    // Top cap: natural ring order gives a +z-facing fan.
    add_polygon_face(mesh, top);

    return mesh;
}

HalfEdgeMesh make_cone(float radius, float height, int segments) {
    if (segments < 3) {
        throw std::invalid_argument("make_cone: segments must be >= 3, got " +
                                    std::to_string(segments));
    }
    const int n = segments;

    HalfEdgeMesh mesh;
    std::vector<std::uint32_t> base(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const float theta =
            2.0f * std::numbers::pi_v<float> * static_cast<float>(i) / static_cast<float>(n);
        base[static_cast<std::size_t>(i)] =
            mesh.add_vertex(radius * std::cos(theta), radius * std::sin(theta), 0.f);
    }
    const std::uint32_t apex = mesh.add_vertex(0.f, 0.f, height);

    // Sides: one outward-facing triangle per segment, fanned around the apex.
    for (int i = 0; i < n; ++i) {
        const int j = (i + 1) % n;
        add_polygon_face(
            mesh, {base[static_cast<std::size_t>(i)], base[static_cast<std::size_t>(j)], apex});
    }

    // Base cap: ring traversed in reverse so the fan's normal points -z.
    std::vector<std::uint32_t> base_cap(static_cast<std::size_t>(n));
    base_cap[0] = base[0];
    for (int i = 1; i < n; ++i) {
        base_cap[static_cast<std::size_t>(i)] = base[static_cast<std::size_t>(n - i)];
    }
    add_polygon_face(mesh, base_cap);

    return mesh;
}

HalfEdgeMesh make_sphere(float radius, int rings, int segments) {
    if (rings < 2) {
        throw std::invalid_argument("make_sphere: rings must be >= 2, got " +
                                    std::to_string(rings));
    }
    if (segments < 3) {
        throw std::invalid_argument("make_sphere: segments must be >= 3, got " +
                                    std::to_string(segments));
    }
    const int n = segments;
    const int r = rings;

    HalfEdgeMesh mesh;
    const std::uint32_t north_pole = mesh.add_vertex(0.f, 0.f, 2.0f * radius);
    const std::uint32_t south_pole = mesh.add_vertex(0.f, 0.f, 0.f);

    // Interior latitude rings, level j = 1 .. rings - 1 (north pole is
    // level 0, south pole is level rings; neither owns a ring_verts entry).
    std::vector<std::vector<std::uint32_t>> ring_verts(static_cast<std::size_t>(r) + 1);
    for (int j = 1; j < r; ++j) {
        const float phi = std::numbers::pi_v<float> * static_cast<float>(j) / static_cast<float>(r);
        const float rho = radius * std::sin(phi);
        const float z = radius * (1.0f + std::cos(phi));
        auto& ring = ring_verts[static_cast<std::size_t>(j)];
        ring.resize(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            const float theta =
                2.0f * std::numbers::pi_v<float> * static_cast<float>(i) / static_cast<float>(n);
            ring[static_cast<std::size_t>(i)] =
                mesh.add_vertex(rho * std::cos(theta), rho * std::sin(theta), z);
        }
    }

    const auto& ring1 = ring_verts[1];
    const auto& ring_last = ring_verts[static_cast<std::size_t>(r) - 1];

    // North cap: triangle fan from the north pole over the first ring.
    for (int i = 0; i < n; ++i) {
        const int j = (i + 1) % n;
        add_polygon_face(mesh, {ring1[static_cast<std::size_t>(i)],
                                ring1[static_cast<std::size_t>(j)], north_pole});
    }

    // South cap: triangle fan from the south pole over the last ring
    // (reversed pairing, mirroring the north cap's winding).
    for (int i = 0; i < n; ++i) {
        const int j = (i + 1) % n;
        add_polygon_face(mesh, {ring_last[static_cast<std::size_t>(j)],
                                ring_last[static_cast<std::size_t>(i)], south_pole});
    }

    // Interior bands between consecutive latitude rings.
    for (int j = 1; j < r - 1; ++j) {
        const auto& upper = ring_verts[static_cast<std::size_t>(j)];
        const auto& lower = ring_verts[static_cast<std::size_t>(j) + 1];
        for (int i = 0; i < n; ++i) {
            const int k = (i + 1) % n;
            add_polygon_face(
                mesh, {lower[static_cast<std::size_t>(i)], lower[static_cast<std::size_t>(k)],
                       upper[static_cast<std::size_t>(k)], upper[static_cast<std::size_t>(i)]});
        }
    }

    return mesh;
}

}  // namespace bermake
