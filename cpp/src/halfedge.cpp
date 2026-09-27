#include "bermake/halfedge.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <unordered_set>

namespace bermake {

// --- Static helpers ----------------------------------------------------

std::uint64_t HalfEdgeMesh::pack_position(float x, float y, float z) noexcept {
    // We need a stable 64-bit hash key derived from the three float32 bits.
    // FNV-1a over the 12 bytes is good enough for dedup; collisions are
    // tolerable because we compare positions on collision (see add_vertex).
    std::uint32_t bx, by, bz;
    std::memcpy(&bx, &x, 4);
    std::memcpy(&by, &y, 4);
    std::memcpy(&bz, &z, 4);
    std::uint64_t h = 0xcbf29ce484222325ull;
    for (std::uint32_t b : {bx, by, bz}) {
        for (int i = 0; i < 4; ++i) {
            h ^= static_cast<std::uint64_t>((b >> (i * 8)) & 0xFFu);
            h *= 0x100000001b3ull;
        }
    }
    return h;
}

std::uint64_t HalfEdgeMesh::pack_pair(std::uint32_t a, std::uint32_t b) noexcept {
    return (static_cast<std::uint64_t>(a) << 32) | static_cast<std::uint64_t>(b);
}

namespace {

// Degenerate-area threshold, shared by all three call sites that derive a
// face normal from its boundary loop: add_face_from_loop, restore_face and
// compute_face_normal_geometric.
//
// It is compared against the length of the Newell area vector, which is
// 2 * the polygon's own area (see newell_area_vector below) — a property of
// the WHOLE face. Before issue #110 the three sites measured one cross
// product of the first two boundary edges, i.e. twice the area of the first
// corner triangle only, and split into two thresholds:
//
//   - a loose 1e-9f for add_face_from_loop/restore_face, on the reasoning
//     that their normal was "only for the renderer" and any plausible
//     direction would do, and
//   - a tighter 1e-7f for compute_face_normal_geometric, which feeds
//     faces_are_coplanar and recompute_face_normal (reached from
//     set_vertex_position on every interactive vertex drag) and so needs a
//     robust normal rather than a unit vector normalized out of float noise.
//
// Both halves of that split are gone deliberately:
//
//   - The premise that the stored normal is cosmetic is no longer true. All
//     three sites write the same Face::normal field, and since M7.5b the
//     renderer reads it out of face_triangle_buffer to build each face's
//     TEXTURE PROJECTION BASIS (_face_uv_geometry in
//     python/bermake/viewport/scene_renderer.py), on top of lighting, picking
//     and coplanarity. One field with two different definitions of
//     "degenerate" is how a value the geometric path would have rejected
//     reached the renderer anyway. There is now one definition.
//   - The tighter value is the one that survives, because the robustness
//     argument for it applies to every reader of the field. Keeping the
//     looser 1e-9f instead would buy nothing: at float32 precision the
//     direction of an area vector of length 1e-8 on unit-scale geometry is
//     coordinate rounding, not geometry.
//
// Against the OLD quantity this value is both looser and tighter, and the
// looser half is the point of the fix:
//
//   - Looser where it matters. Newell's length is 2 * the total area, which
//     for a convex face is >= twice the first corner triangle's, so faces the
//     first-three estimate called degenerate now pass — including every face
//     whose loop merely STARTS on three collinear vertices, which is exactly
//     what an edge split leaves behind and is the whole of issue #110. Those
//     faces have full area and a perfectly well-defined normal.
//   - Tighter only on microscopic faces, those whose total area vector falls
//     in (1e-9, 1e-7]. A face that small covers no pixels and its normal was
//     float noise under the old guard too; it now gets the honest {0,0,0}
//     sentinel instead of a normalized-noise unit vector.
constexpr float kDegenerateAreaVectorLengthThreshold = 1e-7f;

// The polygon's area vector by Newell's method, summed over the WHOLE
// boundary loop, plus its length. Unnormalized; equal to 2 * A * n_hat with
// the right-hand rule, so the length is twice the polygon's area and the
// direction is the face normal implied by the loop's winding.
//
// This replaces the pre-#110 raw_normal_from_first_three, a cross product of
// the loop's first two edges, at all three of its call sites. Reading one
// corner privileges it: a loop whose first three vertices are COLLINEAR
// yielded a zero vector and fell through to a fallback, and a loop with a
// REFLEX vertex at index 1 yielded the opposite sign, because
// cross(p1 - p0, p2 - p0) is twice the SIGNED area of triangle (p0, p1, p2) —
// the ear test at p1, not the polygon's normal. A sum over every edge cannot
// go wrong either way: the convex corners outweigh the reflex ones by exactly
// the polygon's area.
//
// The sign is unchanged wherever the old estimate was valid, which is
// contractual — lighting, picking, faces_are_coplanar, push/pull, offset,
// follow-me and the M7.5b texture basis all steer by this direction. For a
// triangle the sum IS cross(p1 - p0, p2 - p0), term for term; for any convex
// face the first corner's signed area is positive, so the two agree in
// direction. See FaceNormalNewellMatchesFirstThreeOnATriangle and
// FaceNormalNewellAgreesInSignWithFirstThreeOnRandomConvexFaces.
//
// Written as sum (p_i - p_i+1) x-paired with (p_i + p_i+1), the same form as
// _newell_normal in python/bermake/scene/scene.py so the two implementations
// agree term for term; it is algebraically identical to sum p_i x p_i+1
// because the p_i*p_i products telescope away around a closed loop.
// Accumulated in double: the terms are products of coordinates, so a face far
// from the origin cancels badly in float32.
//
// Assumes a planar loop. For a non-planar one this returns the normal of its
// projection, which is the best single normal such a face has.
struct AreaVector {
    double x, y, z;
    double length;
};

// `position_at(i)` returns loop vertex i's position; `n` is the loop length.
template <typename PositionAt>
AreaVector newell_area_vector(std::size_t n, PositionAt position_at) {
    double ax = 0.0, ay = 0.0, az = 0.0;
    std::array<float, 3> q = position_at(0);
    for (std::size_t i = 0; i < n; ++i) {
        const std::array<float, 3> p = q;
        q = position_at((i + 1) % n);
        const double px = p[0], py = p[1], pz = p[2];
        const double qx = q[0], qy = q[1], qz = q[2];
        ax += (py - qy) * (pz + qz);
        ay += (pz - qz) * (px + qx);
        az += (px - qx) * (py + qy);
    }
    return {ax, ay, az, std::sqrt(ax * ax + ay * ay + az * az)};
}

// The unit face normal for a boundary loop, or the {0,0,0} sentinel when the
// face is genuinely degenerate — zero area, not merely an awkward loop start.
//
// {0,0,0} is the single fallback for all three call sites. The pre-#110
// fallback at add_face_from_loop/restore_face was a hardcoded {0,0,1} "so the
// renderer sees weak lighting", and that is the defect issue #110 was
// reopened for: it is a specific, plausible-looking, SILENTLY WRONG direction.
// A vertical wall whose loop started on a split edge took it and was lit — and
// since M7.5b textured — as if it were a floor. A zero vector cannot be
// mistaken for an answer:
//
//   - It is already this file's convention for "no normal"
//     (compute_face_normal_geometric's callers, faces_are_coplanar and
//     recompute_face_normal, explicitly test for it), so there is now one
//     sentinel rather than two conventions writing one field.
//   - It is the kernel's non-throwing counterpart to Scene.face_normal, which
//     raises on the same condition.
//   - It is not a new exposure. recompute_face_normal already wrote this
//     sentinel into this same field at this same threshold, reached from
//     set_vertex_position on every interactive vertex drag, so every consumer
//     had to cope with it already. This only widens when it first appears,
//     from the first drag to face creation.
//
// Do NOT read this as "a face this small is invisible". It is not: a face
// whose area vector lands in the reject band still triangulates and still
// emits corners, all of them carrying the sentinel. Measured on the installed
// kernel, an XZ wall of side 2e-4 has an area vector of 8e-8 — inside the band
// — and emits 6 corners. It is earcut on a genuinely collinear loop that
// produces no triangles, which is a property of the triangulator and not of
// this function. So the sentinel does reach consumers, and each one owns its
// own guard: plane_bases in python/bermake/viewport/uv_projection.py answers a
// zero-length normal with the world XY basis, and phong.vert falls back to a
// fixed direction for LIGHTING only (see the comment there).
std::array<float, 3> unit_normal_from_area_vector(const AreaVector& a) {
    if (!(a.length > static_cast<double>(kDegenerateAreaVectorLengthThreshold))) {
        return {0.0f, 0.0f, 0.0f};
    }
    return {static_cast<float>(a.x / a.length), static_cast<float>(a.y / a.length),
            static_cast<float>(a.z / a.length)};
}

}  // namespace

// --- Stubs for Task 2+ -------------------------------------------------

std::uint32_t HalfEdgeMesh::add_vertex(float x, float y, float z) {
    // Collapse negative zero so -0.0 and 0.0 hash identically.
    if (x == 0.0f) x = 0.0f;
    if (y == 0.0f) y = 0.0f;
    if (z == 0.0f) z = 0.0f;

    const std::uint64_t key = pack_position(x, y, z);
    auto it = position_index_.find(key);
    if (it != position_index_.end() && vertices_[it->second].alive) {
        const auto& p = vertices_[it->second].pos;
        if (p[0] == x && p[1] == y && p[2] == z) {
            return it->second;
        }
        // Hash collision on a different float triple; fall through to allocate.
    }
    const std::uint32_t vid = static_cast<std::uint32_t>(vertices_.size());
    vertices_.push_back(Vertex{{x, y, z}, INVALID_ID, true});
    position_index_[key] = vid;
    dirty_ = true;
    return vid;
}

std::uint32_t HalfEdgeMesh::add_halfedge_pair(std::uint32_t v1_id, std::uint32_t v2_id) {
    if (v1_id == v2_id) {
        throw std::invalid_argument("HalfEdgeMesh::add_halfedge_pair: self-loop at vertex " +
                                    std::to_string(v1_id));
    }
    if (!vertex_is_live(v1_id)) {
        throw std::out_of_range("HalfEdgeMesh::add_halfedge_pair: v1_id " + std::to_string(v1_id) +
                                " is not live");
    }
    if (!vertex_is_live(v2_id)) {
        throw std::out_of_range("HalfEdgeMesh::add_halfedge_pair: v2_id " + std::to_string(v2_id) +
                                " is not live");
    }
    const std::uint32_t v_min = std::min(v1_id, v2_id);
    const std::uint32_t v_max = std::max(v1_id, v2_id);
    const std::uint64_t key = pack_pair(v_min, v_max);
    auto it = edge_index_.find(key);
    if (it != edge_index_.end() && edge_is_live(it->second)) {
        return it->second;
    }
    const std::uint32_t he_a = static_cast<std::uint32_t>(halfedges_.size());
    const std::uint32_t he_b = he_a + 1;
    const std::uint32_t edge_id = he_a / 2;
    halfedges_.push_back(HalfEdge{v_min, INVALID_ID, he_b, INVALID_ID, true});
    halfedges_.push_back(HalfEdge{v_max, INVALID_ID, he_a, INVALID_ID, true});
    edge_index_[key] = edge_id;
    dirty_ = true;
    return edge_id;
}

std::uint32_t HalfEdgeMesh::add_face_from_loop(const std::vector<std::uint32_t>& loop,
                                               const std::vector<std::int32_t>& triangles) {
    if (loop.size() < 3) {
        throw std::invalid_argument("HalfEdgeMesh::add_face_from_loop: loop has " +
                                    std::to_string(loop.size()) + " vertices; minimum 3");
    }
    for (auto v : loop) {
        if (!vertex_is_live(v)) {
            throw std::out_of_range("HalfEdgeMesh::add_face_from_loop: vertex " +
                                    std::to_string(v) + " is not live");
        }
    }
    const std::uint32_t f_id = static_cast<std::uint32_t>(faces_.size());
    // Geometric normal from the WHOLE boundary loop (Newell), not its first
    // three vertices, so a loop that merely starts on three collinear
    // vertices — what an edge split leaves behind — still gets its own normal
    // instead of a hardcoded default (issue #110). Assumes planar face —
    // M2/M3a only produce planar faces; M4+ will revisit.
    const auto face_normal = unit_normal_from_area_vector(
        newell_area_vector(loop.size(), [&](std::size_t i) { return vertex_position(loop[i]); }));
    Face f{INVALID_ID, {face_normal[0], face_normal[1], face_normal[2]}, triangles, loop, true};

    // Wire each loop[i] → loop[i+1] half-edge to point to loop[i+1] → loop[i+2].
    // The half-edge from v_from to v_to has origin = v_from. Given the canonical
    // convention (he[2*e].origin = min, he[2*e+1].origin = max), pick the index
    // that matches v_from.
    const std::size_t n = loop.size();
    std::vector<std::uint32_t> loop_halfedges(n, INVALID_ID);
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t v_from = loop[i];
        const std::uint32_t v_to = loop[(i + 1) % n];
        const std::uint32_t v_min = std::min(v_from, v_to);
        const std::uint32_t v_max = std::max(v_from, v_to);
        const std::uint64_t key = pack_pair(v_min, v_max);
        auto it = edge_index_.find(key);
        if (it == edge_index_.end() || !edge_is_live(it->second)) {
            throw std::invalid_argument("HalfEdgeMesh::add_face_from_loop: edge (" +
                                        std::to_string(v_from) + ", " + std::to_string(v_to) +
                                        ") is missing");
        }
        const std::uint32_t edge_id = it->second;
        loop_halfedges[i] = (v_from < v_to) ? (edge_id * 2) : (edge_id * 2 + 1);
    }
    // Wire next pointers + face pointers.
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t he = loop_halfedges[i];
        const std::uint32_t he_next = loop_halfedges[(i + 1) % n];
        halfedges_[he].next = he_next;
        halfedges_[he].face = f_id;
    }
    f.boundary_he = loop_halfedges[0];
    // outgoing_he on each loop vertex points to one of its outgoing half-edges
    // (any will do for now; M3b's adjacency walks pick a starting half-edge).
    for (std::size_t i = 0; i < n; ++i) {
        if (vertices_[loop[i]].outgoing_he == INVALID_ID) {
            vertices_[loop[i]].outgoing_he = loop_halfedges[i];
        }
    }

    faces_.push_back(std::move(f));
    dirty_ = true;
    return f_id;
}

void HalfEdgeMesh::remove_edge(std::uint32_t e_id) {
    if (!edge_is_live(e_id)) {
        throw std::out_of_range("HalfEdgeMesh::remove_edge: edge " + std::to_string(e_id) +
                                " is not live");
    }
    const std::uint32_t he_a = e_id * 2;
    const std::uint32_t he_b = he_a + 1;
    if (halfedges_[he_a].face != INVALID_ID || halfedges_[he_b].face != INVALID_ID) {
        throw std::invalid_argument("HalfEdgeMesh::remove_edge: edge " + std::to_string(e_id) +
                                    " still bordered by a face");
    }
    const std::uint32_t v_min = halfedges_[he_a].origin;
    const std::uint32_t v_max = halfedges_[he_b].origin;
    edge_index_.erase(pack_pair(v_min, v_max));
    halfedges_[he_a].alive = false;
    halfedges_[he_b].alive = false;
    dirty_ = true;
}

void HalfEdgeMesh::remove_vertex(std::uint32_t v_id) {
    if (!vertex_is_live(v_id)) {
        throw std::out_of_range("HalfEdgeMesh::remove_vertex: vertex " + std::to_string(v_id) +
                                " is not live");
    }
    // Scan live half-edges; reject if any has origin == v_id.
    for (const auto& he : halfedges_) {
        if (he.alive && he.origin == v_id) {
            throw std::invalid_argument("HalfEdgeMesh::remove_vertex: vertex " +
                                        std::to_string(v_id) + " still has incident edges");
        }
    }
    const auto& v = vertices_[v_id];
    position_index_.erase(pack_position(v.pos[0], v.pos[1], v.pos[2]));
    vertices_[v_id].alive = false;
    dirty_ = true;
}
void HalfEdgeMesh::remove_face(std::uint32_t f_id) {
    if (!face_is_live(f_id)) {
        throw std::out_of_range("HalfEdgeMesh::remove_face: face " + std::to_string(f_id) +
                                " is not live");
    }
    // Walk the boundary half-edge cycle and clear face pointers.
    Face& f = faces_[f_id];
    std::uint32_t he = f.boundary_he;
    if (he != INVALID_ID) {
        const std::uint32_t start = he;
        do {
            halfedges_[he].face = INVALID_ID;
            he = halfedges_[he].next;
            if (he == INVALID_ID) break;  // defensive: malformed cycle
        } while (he != start);
    }
    f.alive = false;
    f.boundary_he = INVALID_ID;
    dirty_ = true;
}

void HalfEdgeMesh::restore_vertex(std::uint32_t v_id, float x, float y, float z) {
    if (v_id >= vertices_.size()) {
        throw std::out_of_range("HalfEdgeMesh::restore_vertex: v_id " + std::to_string(v_id) +
                                " out of range");
    }
    if (vertices_[v_id].alive) {
        throw std::logic_error("HalfEdgeMesh::restore_vertex: slot " + std::to_string(v_id) +
                               " is already live");
    }
    if (x == 0.0f) x = 0.0f;
    if (y == 0.0f) y = 0.0f;
    if (z == 0.0f) z = 0.0f;
    vertices_[v_id].pos[0] = x;
    vertices_[v_id].pos[1] = y;
    vertices_[v_id].pos[2] = z;
    vertices_[v_id].alive = true;
    position_index_[pack_position(x, y, z)] = v_id;
    dirty_ = true;
}

void HalfEdgeMesh::restore_edge(std::uint32_t e_id, std::uint32_t v1_id, std::uint32_t v2_id) {
    const std::uint32_t he_a = e_id * 2;
    const std::uint32_t he_b = he_a + 1;
    if (he_b >= halfedges_.size()) {
        throw std::out_of_range("HalfEdgeMesh::restore_edge: e_id " + std::to_string(e_id) +
                                " out of range");
    }
    if (halfedges_[he_a].alive || halfedges_[he_b].alive) {
        throw std::logic_error("HalfEdgeMesh::restore_edge: slot " + std::to_string(e_id) +
                               " is already live");
    }
    const std::uint32_t v_min = std::min(v1_id, v2_id);
    const std::uint32_t v_max = std::max(v1_id, v2_id);
    halfedges_[he_a].origin = v_min;
    halfedges_[he_a].face = INVALID_ID;
    halfedges_[he_a].next = INVALID_ID;
    halfedges_[he_a].alive = true;
    halfedges_[he_b].origin = v_max;
    halfedges_[he_b].face = INVALID_ID;
    halfedges_[he_b].next = INVALID_ID;
    halfedges_[he_b].alive = true;
    edge_index_[pack_pair(v_min, v_max)] = e_id;
    dirty_ = true;
}

void HalfEdgeMesh::restore_face(std::uint32_t f_id, const std::vector<std::uint32_t>& loop,
                                const std::vector<std::int32_t>& triangles) {
    if (f_id >= faces_.size()) {
        throw std::out_of_range("HalfEdgeMesh::restore_face: f_id " + std::to_string(f_id) +
                                " out of range");
    }
    if (faces_[f_id].alive) {
        throw std::logic_error("HalfEdgeMesh::restore_face: slot " + std::to_string(f_id) +
                               " is already live");
    }
    // Same wiring as add_face_from_loop but writes into the existing slot.
    const std::size_t n = loop.size();
    std::vector<std::uint32_t> loop_halfedges(n, INVALID_ID);
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t v_from = loop[i];
        const std::uint32_t v_to = loop[(i + 1) % n];
        const std::uint32_t v_min = std::min(v_from, v_to);
        const std::uint32_t v_max = std::max(v_from, v_to);
        auto it = edge_index_.find(pack_pair(v_min, v_max));
        if (it == edge_index_.end() || !edge_is_live(it->second)) {
            throw std::invalid_argument("HalfEdgeMesh::restore_face: edge (" +
                                        std::to_string(v_from) + ", " + std::to_string(v_to) +
                                        ") is missing");
        }
        loop_halfedges[i] = (v_from < v_to) ? (it->second * 2) : (it->second * 2 + 1);
    }
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t he = loop_halfedges[i];
        halfedges_[he].next = loop_halfedges[(i + 1) % n];
        halfedges_[he].face = f_id;
    }
    Face& f = faces_[f_id];
    f.boundary_he = loop_halfedges[0];
    f.tris = triangles;
    f.loop = loop;
    f.alive = true;
    // Recompute geometric normal (same shared helpers as add_face_from_loop)
    // so that undo→redo round-trips produce the correct normal rather than
    // preserving a stale value from before the fix.
    {
        const auto face_normal = unit_normal_from_area_vector(newell_area_vector(
            loop.size(), [&](std::size_t i) { return vertex_position(loop[i]); }));
        f.normal[0] = face_normal[0];
        f.normal[1] = face_normal[1];
        f.normal[2] = face_normal[2];
    }
    dirty_ = true;
}

void HalfEdgeMesh::clear() noexcept {
    // Tombstone rather than shrink: keep the slabs at size so ids stay
    // addressable and restore_* can revive them (the invariant every other
    // removal follows). Memory reclamation is out of scope — see #16 (M10).
    for (auto& v : vertices_) v.alive = false;
    for (auto& h : halfedges_) h.alive = false;
    for (auto& f : faces_) f.alive = false;
    position_index_.clear();
    edge_index_.clear();
    dirty_ = true;
}

bool HalfEdgeMesh::vertex_is_live(std::uint32_t v_id) const noexcept {
    return v_id < vertices_.size() && vertices_[v_id].alive;
}
bool HalfEdgeMesh::edge_is_live(std::uint32_t e_id) const noexcept {
    const std::uint32_t he = e_id * 2;
    return he < halfedges_.size() && halfedges_[he].alive;
}
std::uint32_t HalfEdgeMesh::edge_between(std::uint32_t v1_id, std::uint32_t v2_id) const noexcept {
    const std::uint32_t v_min = std::min(v1_id, v2_id);
    const std::uint32_t v_max = std::max(v1_id, v2_id);
    const auto it = edge_index_.find(pack_pair(v_min, v_max));
    if (it == edge_index_.end() || !edge_is_live(it->second)) {
        return INVALID_ID;
    }
    return it->second;
}
bool HalfEdgeMesh::face_is_live(std::uint32_t f_id) const noexcept {
    return f_id < faces_.size() && faces_[f_id].alive;
}

std::array<float, 3> HalfEdgeMesh::vertex_position(std::uint32_t v_id) const {
    if (!vertex_is_live(v_id)) {
        throw std::out_of_range("HalfEdgeMesh::vertex_position: vertex " + std::to_string(v_id) +
                                " is not live");
    }
    const auto& v = vertices_[v_id];
    return {v.pos[0], v.pos[1], v.pos[2]};
}

std::uint32_t HalfEdgeMesh::vertex_outgoing_halfedge(std::uint32_t v_id) const {
    if (!vertex_is_live(v_id)) {
        throw std::out_of_range("HalfEdgeMesh::vertex_outgoing_halfedge: vertex " +
                                std::to_string(v_id) + " is not live");
    }
    return vertices_[v_id].outgoing_he;
}

std::array<std::uint32_t, 2> HalfEdgeMesh::edge_vertices(std::uint32_t e_id) const {
    if (!edge_is_live(e_id)) {
        throw std::out_of_range("HalfEdgeMesh::edge_vertices: edge " + std::to_string(e_id) +
                                " is not live");
    }
    const std::uint32_t he_a = e_id * 2;
    const std::uint32_t he_b = he_a + 1;
    return {halfedges_[he_a].origin, halfedges_[he_b].origin};
}

std::vector<std::uint32_t> HalfEdgeMesh::face_loop_vertices(std::uint32_t f_id) const {
    if (!face_is_live(f_id)) {
        throw std::out_of_range("HalfEdgeMesh::face_loop_vertices: face " + std::to_string(f_id) +
                                " is not live");
    }
    return faces_[f_id].loop;
}

std::vector<std::int32_t> HalfEdgeMesh::face_triangles(std::uint32_t f_id) const {
    if (!face_is_live(f_id)) {
        throw std::out_of_range("HalfEdgeMesh::face_triangles: face " + std::to_string(f_id) +
                                " is not live");
    }
    return faces_[f_id].tris;
}

namespace {

inline float dot3(std::array<float, 3> a, std::array<float, 3> b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
inline float len3(std::array<float, 3> a) {
    return std::sqrt(dot3(a, a));
}

// Geometric face normal from the WHOLE boundary loop (Newell), or the
// {0,0,0} sentinel for a genuinely degenerate — zero-area — face. This is the
// third of the three call sites the M3c review flagged as duplicated; since
// issue #110 all three share the same math (newell_area_vector), the same
// threshold (kDegenerateAreaVectorLengthThreshold) and the same sentinel
// (unit_normal_from_area_vector), because all three write the same
// Face::normal field and had no business disagreeing about what degenerate
// means. This function feeds faces_are_coplanar and recompute_face_normal
// (reached from set_vertex_position on every interactive vertex drag), which
// is where the robustness argument for the surviving threshold comes from.
std::array<float, 3> compute_face_normal_geometric(const bermake::HalfEdgeMesh& m,
                                                   std::uint32_t f_id) {
    auto loop = m.face_loop_vertices(f_id);
    if (loop.size() < 3) return {0, 0, 0};
    return unit_normal_from_area_vector(
        newell_area_vector(loop.size(), [&](std::size_t i) { return m.vertex_position(loop[i]); }));
}

// Insert vertex w into `loop` between the adjacent pair (va, vb) (either order),
// returning the new loop. Caller guarantees va,vb are consecutive in loop.
std::vector<std::uint32_t> loop_with_inserted(const std::vector<std::uint32_t>& loop,
                                              std::uint32_t va, std::uint32_t vb, std::uint32_t w) {
    const std::size_t n = loop.size();
    std::vector<std::uint32_t> out;
    out.reserve(n + 1);
    for (std::size_t i = 0; i < n; ++i) {
        out.push_back(loop[i]);
        const std::uint32_t cur = loop[i];
        const std::uint32_t nxt = loop[(i + 1) % n];
        if ((cur == va && nxt == vb) || (cur == vb && nxt == va)) {
            out.push_back(w);
        }
    }
    return out;
}

}  // namespace

void bermake::HalfEdgeMesh::recompute_face_normal(std::uint32_t f_id) {
    if (!face_is_live(f_id)) return;
    auto n = compute_face_normal_geometric(*this, f_id);  // {0,0,0} if degenerate
    faces_[f_id].normal[0] = n[0];
    faces_[f_id].normal[1] = n[1];
    faces_[f_id].normal[2] = n[2];
}

void bermake::HalfEdgeMesh::set_vertex_position(std::uint32_t v_id, float x, float y, float z) {
    if (v_id >= vertices_.size() || !vertices_[v_id].alive) {
        throw std::out_of_range("HalfEdgeMesh::set_vertex_position: v_id " + std::to_string(v_id) +
                                " is not live");
    }
    // Collapse negative zero so -0.0 and 0.0 hash identically (matches add_vertex).
    if (x == 0.0f) x = 0.0f;
    if (y == 0.0f) y = 0.0f;
    if (z == 0.0f) z = 0.0f;

    Vertex& v = vertices_[v_id];
    // Dedup-index upkeep: drop the old packed key, install the new one.
    position_index_.erase(pack_position(v.pos[0], v.pos[1], v.pos[2]));
    v.pos[0] = x;
    v.pos[1] = y;
    v.pos[2] = z;
    position_index_[pack_position(x, y, z)] = v_id;

    // Recompute cached normals on every incident face. Each incident face has
    // exactly one boundary half-edge originating at v_id; recompute is
    // idempotent, so no dedup is needed.
    for (const auto& he : halfedges_) {
        if (he.alive && he.origin == v_id && he.face != INVALID_ID) {
            recompute_face_normal(he.face);
        }
    }
    dirty_ = true;
}

bool bermake::HalfEdgeMesh::faces_are_coplanar(std::uint32_t f1_id, std::uint32_t f2_id,
                                               float angle_tol_cos, float dist_tol) const {
    if (!face_is_live(f1_id) || !face_is_live(f2_id)) return false;
    auto n1 = compute_face_normal_geometric(*this, f1_id);
    auto n2 = compute_face_normal_geometric(*this, f2_id);
    // Degenerate normal → refuse. compute_face_normal_geometric returns either
    // a unit vector or the exact {0,0,0} sentinel and nothing in between, so
    // this is a test for the sentinel, not a magnitude test. Written as an
    // exact zero check rather than against kDegenerateAreaVectorLengthThreshold:
    // that constant measures an AREA VECTOR in world units, an unrelated
    // quantity to the length of an already-normalized vector, and borrowing it
    // here would couple the two under one name for no gain.
    if (len3(n1) == 0.0f || len3(n2) == 0.0f) return false;

    // Angle test: |dot(n1, n2)| > tolerance — accept either winding direction.
    float ang = std::abs(dot3(n1, n2));
    if (ang < angle_tol_cos) return false;

    // Distance test: every vertex of f2 within `dist_tol` of f1's plane, and vv.
    auto check_side = [&](std::array<float, 3> n, std::uint32_t plane_face,
                          std::uint32_t other_face) -> bool {
        auto plane_loop = face_loop_vertices(plane_face);
        auto p_anchor = vertex_position(plane_loop[0]);
        float d_anchor = dot3(n, p_anchor);
        for (auto v : face_loop_vertices(other_face)) {
            auto p = vertex_position(v);
            float signed_d = dot3(n, p) - d_anchor;
            if (std::abs(signed_d) > dist_tol) return false;
        }
        return true;
    };
    return check_side(n1, f1_id, f2_id) && check_side(n2, f2_id, f1_id);
}

std::uint32_t bermake::HalfEdgeMesh::dissolve_edge(std::uint32_t e_id) {
    // The two half-edges of edge e are at slab indices 2e and 2e+1.
    std::uint32_t he_a = 2u * e_id;
    std::uint32_t he_b = 2u * e_id + 1u;
    if (he_b >= halfedges_.size()) return INVALID_ID;
    if (!halfedges_[he_a].alive || !halfedges_[he_b].alive) return INVALID_ID;

    std::uint32_t f1 = halfedges_[he_a].face;
    std::uint32_t f2 = halfedges_[he_b].face;
    if (f1 == INVALID_ID || f2 == INVALID_ID) return INVALID_ID;  // boundary edge
    if (f1 == f2) return INVALID_ID;                              // same face on both sides

    // Reject multi-shared-edge: count how many edges f1 and f2 share. Walk f1's
    // boundary half-edges; for each, see if its twin is on f2. If more than one
    // such half-edge exists, refuse.
    {
        std::uint32_t shared_count = 0;
        std::uint32_t start = faces_[f1].boundary_he;
        std::uint32_t cur = start;
        do {
            std::uint32_t twin = halfedges_[cur].twin;
            if (twin != INVALID_ID && halfedges_[twin].face == f2) ++shared_count;
            cur = halfedges_[cur].next;
        } while (cur != start);
        if (shared_count > 1) return INVALID_ID;
    }

    // Splice the two boundary loops at the shared edge.
    //   Loop of f1: ... -> A -> he_a -> B -> ...   (B = he_a.next, A's next was he_a)
    //   Loop of f2: ... -> C -> he_b -> D -> ...
    // After dissolve the merged loop becomes:
    //   ... -> A -> D -> ... -> C -> B -> ...
    // (skipping he_a and he_b, splicing across the gap.)

    // Find the predecessor of `target` within its own next-cycle. Used for
    // both A (predecessor of he_a in f1's loop) and C (predecessor of he_b in
    // f2's loop) — previously copy-pasted as two near-identical blocks.
    // Bounded by the live half-edge count so a malformed boundary cycle
    // fails a debug-build assertion instead of looping forever.
    auto find_predecessor = [this](std::uint32_t target) {
        std::uint32_t cur = halfedges_[target].next;
        std::size_t steps = 0;
        while (halfedges_[cur].next != target) {
            cur = halfedges_[cur].next;
            ++steps;
            assert(steps <= halfedges_.size() &&
                   "dissolve_edge: boundary cycle did not close within halfedge count");
        }
        return cur;
    };
    const std::uint32_t A = find_predecessor(he_a);
    const std::uint32_t C = find_predecessor(he_b);
    std::uint32_t B = halfedges_[he_a].next;
    std::uint32_t D = halfedges_[he_b].next;

    // Splice next-pointers across the gap.
    halfedges_[A].next = D;
    halfedges_[C].next = B;

    // Walk the new merged loop, collecting vertex IDs for the new face's
    // `loop` cache. Face pointers are reassigned later inside add_face_from_loop().
    // Bounded the same way as find_predecessor above, for the same reason.
    std::vector<std::uint32_t> merged_loop;
    std::uint32_t walk_start = D;
    std::uint32_t walk_cur = walk_start;
    {
        std::size_t walk_steps = 0;
        do {
            merged_loop.push_back(halfedges_[walk_cur].origin);
            walk_cur = halfedges_[walk_cur].next;
            ++walk_steps;
            assert(walk_steps <= halfedges_.size() &&
                   "dissolve_edge: merged-loop walk did not close within halfedge count");
        } while (walk_cur != walk_start);
    }

    // Refuse a merge that would build a vertex-revisiting (pinched) loop --
    // e.g. two coplanar faces sharing exactly one edge PLUS an extra vertex
    // present in both loops elsewhere. The only other guard above is "more
    // than one shared edge", which this shape passes: it shares exactly one
    // edge and still produces a loop that touches itself. split_face gained
    // an equivalent loop_is_simple guard for exactly the same reason -- a
    // self-touching loop corrupts add_face_from_loop's wiring, and here it
    // additionally corrupts vertex-keyed UV merging downstream in
    // Scene._merge_corner_uvs. Must run BEFORE anything is tombstoned; the
    // splice above only rewrote two `next` pointers to make the loop
    // walkable, and is undone here on rejection so the mesh comes back
    // exactly as it was (EraserTool._try_dissolve reads INVALID_ID back as
    // "cascade instead" and needs both source faces still intact for that).
    {
        std::unordered_set<std::uint32_t> seen;
        seen.reserve(merged_loop.size());
        for (auto v : merged_loop) {
            if (!seen.insert(v).second) {
                halfedges_[A].next = he_a;
                halfedges_[C].next = he_b;
                return INVALID_ID;
            }
        }
    }

    // Retriangulate the merged loop with a simple fan (works for convex; the
    // merged shape from coplanar dissolves is convex by construction in M3c's
    // Case 2). For now use fan from vertex 0.
    assert(merged_loop.size() >= 3 &&
           "dissolve_edge: merged loop must have at least 3 vertices (two faces of >=3 sides "
           "sharing exactly one edge cannot produce fewer)");
    std::vector<std::int32_t> tris;
    tris.reserve((merged_loop.size() - 2) * 3);
    for (std::size_t i = 1; i + 1 < merged_loop.size(); ++i) {
        tris.push_back((std::int32_t)merged_loop[0]);
        tris.push_back((std::int32_t)merged_loop[i]);
        tris.push_back((std::int32_t)merged_loop[i + 1]);
    }

    // Tombstone the two source faces and the dissolved edge's two half-edges.
    faces_[f1].alive = false;
    faces_[f2].alive = false;
    halfedges_[he_a].alive = false;
    halfedges_[he_b].alive = false;

    // Also tombstone the edge's entry in edge_index_ so add_halfedge_pair can't
    // resurrect it as live (the slot stays dead — IDs are never reused).
    {
        // Origins were the (v_min, v_max) pair before tombstoning above; they're
        // still readable in the struct (we only flipped `alive`).
        const std::uint32_t v_min = halfedges_[he_a].origin;
        const std::uint32_t v_max = halfedges_[he_b].origin;
        edge_index_.erase(pack_pair(v_min, v_max));
    }

    // Repoint each endpoint vertex's cached outgoing_he if it pointed at the
    // half-edge we just tombstoned above. Latent-only today (nothing reads
    // outgoing_he yet), but leaving it dangling would silently corrupt a
    // future vertex-incident walk.
    {
        auto repoint_if_stale = [this](std::uint32_t v_id, std::uint32_t dead_he) {
            if (vertices_[v_id].outgoing_he != dead_he) return;
            std::uint32_t replacement = INVALID_ID;
            for (std::uint32_t h = 0; h < halfedges_.size(); ++h) {
                if (halfedges_[h].alive && halfedges_[h].origin == v_id) {
                    replacement = h;
                    break;
                }
            }
            vertices_[v_id].outgoing_he = replacement;
        };
        repoint_if_stale(halfedges_[he_a].origin, he_a);
        repoint_if_stale(halfedges_[he_b].origin, he_b);
    }

    // Allocate the new face on the merged loop. Note: this calls
    // add_face_from_loop, which re-walks half-edges — they must already point
    // to a consistent next-chain. Splicing above ensured this.
    auto new_face = add_face_from_loop(merged_loop, tris);

    // After add_face_from_loop, the merged_loop's half-edges now have face = new_face.
    // (add_face_from_loop sets this internally.)

    dirty_ = true;
    return new_face;
}

std::array<std::uint32_t, 2> bermake::HalfEdgeMesh::split_face(
    std::uint32_t f_id, const std::vector<std::uint32_t>& loop_a,
    const std::vector<std::int32_t>& tris_a, const std::vector<std::uint32_t>& loop_b,
    const std::vector<std::int32_t>& tris_b) {
    const std::array<std::uint32_t, 2> fail{INVALID_ID, INVALID_ID};

    if (!face_is_live(f_id)) return fail;
    if (loop_a.size() < 3 || loop_b.size() < 3) return fail;
    if (tris_a.size() % 3 != 0 || tris_b.size() % 3 != 0) return fail;

    // Every loop vertex live, every loop edge present and live. add_face_from_loop
    // throws on both, and it would throw after remove_face had run.
    //
    // Beyond that: the directed half-edge each step will CLAIM must be
    // unowned or already owned by f_id itself. add_face_from_loop
    // unconditionally overwrites halfedges_[he].face with no ownership check
    // of its own, so without this a chain edge that happens to already be a
    // live edge of some OTHER, unrelated face -- e.g. a fin standing on the
    // parent's own diagonal -- gets silently stolen: split_face would report
    // success while quietly detaching that face from its own boundary.
    // Refusal is correct here, not a fallback: the edge would need three
    // incident faces (the fin plus both new children), and a half-edge mesh
    // has exactly two slots per edge. A legitimate split's own non-chain
    // steps are always owned by f_id (they retrace the parent's own directed
    // loop) and its chain steps are always unowned (a chord freshly drawn
    // across the face), so this rejects nothing that a real split needs.
    auto loop_is_buildable = [this, f_id](const std::vector<std::uint32_t>& loop) {
        const std::size_t n = loop.size();
        for (std::size_t i = 0; i < n; ++i) {
            const std::uint32_t v_from = loop[i];
            const std::uint32_t v_to = loop[(i + 1) % n];
            if (!vertex_is_live(v_from)) return false;
            const std::uint32_t e = edge_between(v_from, v_to);
            if (e == INVALID_ID || !edge_is_live(e)) return false;
            // Same directed-half-edge convention as add_face_from_loop:
            // he[2*e].origin == min(v_from, v_to).
            const std::uint32_t he = (v_from < v_to) ? (e * 2) : (e * 2 + 1);
            const std::uint32_t owner = halfedges_[he].face;
            if (owner != INVALID_ID && owner != f_id) return false;
        }
        return true;
    };
    if (!loop_is_buildable(loop_a) || !loop_is_buildable(loop_b)) return fail;

    // Every triangle index names a vertex of its own loop.
    auto tris_index_loop = [](const std::vector<std::uint32_t>& loop,
                              const std::vector<std::int32_t>& tris) {
        for (auto t : tris) {
            if (t < 0) return false;
            if (std::find(loop.begin(), loop.end(), static_cast<std::uint32_t>(t)) == loop.end())
                return false;
        }
        return true;
    };
    if (!tris_index_loop(loop_a, tris_a) || !tris_index_loop(loop_b, tris_b)) return fail;

    // Reject a loop that revisits a vertex. Without this, a loop like
    // {v0, x, y, x} can self-cancel in the directed-edge check below: its
    // (x,y)/(y,x) pair and its (v0,x)/(x,v0) pair both cancel entirely WITHIN
    // that one loop, before the other loop is even considered, so the loop's
    // net contribution to the directed multiset is zero and it can ride
    // along on whatever the other loop alone supplies. A vertex-revisiting,
    // self-intersecting loop must be refused before the cancellation logic
    // even sees it, because add_face_from_loop does not check simplicity
    // either and would happily build the corrupted face.
    auto loop_is_simple = [](const std::vector<std::uint32_t>& loop) {
        std::unordered_set<std::uint32_t> seen;
        seen.reserve(loop.size());
        for (auto v : loop) {
            if (!seen.insert(v).second) return false;
        }
        return true;
    };
    if (!loop_is_simple(loop_a) || !loop_is_simple(loop_b)) return fail;

    // The partition check, as directed edges. Walking both sub-loops gives every
    // directed boundary step of both children. Both loops are already known
    // simple AND at least 3 vertices long (both checked above), so a directed
    // edge can only be reversed by an edge in the OTHER loop, never by another
    // edge in the same loop: for a loop of length >= 3, reversing step x->y to
    // y->x within that same loop would require x to reappear in it, which
    // simplicity forbids. (Length 2 is the one case where a loop's own two
    // steps are unconditionally reverses of each other regardless of
    // repetition -- e.g. {v0,v2} walks v0->v2 then v2->v0 -- which is exactly
    // why the length check above runs before this one ever sees such a loop.)
    // So a chain edge walked once in each direction is always a genuine
    // shared-chain edge between loop_a and loop_b, and what survives
    // cancellation must be exactly the parent's own
    // directed loop. This one test carries three separate claims: the two
    // loops really do partition f, the winding is preserved in both children
    // (a reversed child would leave uncancelled reversed edges), and the chain
    // is traversed in opposite directions, which is what makes
    // add_face_from_loop claim opposite half-edges of each chain edge.
    std::unordered_map<std::uint64_t, int> directed;
    auto add_directed = [&directed](const std::vector<std::uint32_t>& loop) {
        const std::size_t n = loop.size();
        for (std::size_t i = 0; i < n; ++i) {
            ++directed[pack_pair(loop[i], loop[(i + 1) % n])];
        }
    };
    add_directed(loop_a);
    add_directed(loop_b);

    std::uint32_t cancelled = 0;
    for (auto& [key, count] : directed) {
        if (count <= 0) continue;
        const std::uint32_t u = static_cast<std::uint32_t>(key >> 32);
        const std::uint32_t v = static_cast<std::uint32_t>(key & 0xFFFFFFFFull);
        auto it = directed.find(pack_pair(v, u));
        if (it == directed.end()) continue;
        const int pairs = std::min(count, it->second);
        if (pairs <= 0) continue;
        count -= pairs;
        it->second -= pairs;
        cancelled += static_cast<std::uint32_t>(pairs);
    }
    if (cancelled < 1) return fail;  // the two loops do not touch

    const std::vector<std::uint32_t>& parent = faces_[f_id].loop;
    std::unordered_map<std::uint64_t, int> want;
    for (std::size_t i = 0; i < parent.size(); ++i) {
        ++want[pack_pair(parent[i], parent[(i + 1) % parent.size()])];
    }
    for (const auto& [key, count] : directed) {
        if (count == 0) continue;
        auto it = want.find(key);
        if (it == want.end() || it->second != count) return fail;
    }
    for (const auto& [key, count] : want) {
        auto it = directed.find(key);
        if (it == directed.end() || it->second != count) return fail;
    }

    // Validated. From here nothing can fail.
    remove_face(f_id);
    const std::uint32_t a = add_face_from_loop(loop_a, tris_a);
    const std::uint32_t b = add_face_from_loop(loop_b, tris_b);
    dirty_ = true;
    return {a, b};
}

std::optional<bermake::SplitEdgeResult> bermake::HalfEdgeMesh::split_edge(std::uint32_t e_id,
                                                                          float t) {
    if (!edge_is_live(e_id)) return std::nullopt;
    if (!(t > 0.0f && t < 1.0f)) return std::nullopt;

    const std::uint32_t he_a = 2u * e_id;
    const std::uint32_t he_b = 2u * e_id + 1u;
    const std::uint32_t va = halfedges_[he_a].origin;  // v_min
    const std::uint32_t vb = halfedges_[he_b].origin;  // v_max

    const auto pa = vertices_[va].pos;
    const auto pb = vertices_[vb].pos;
    const float wx = pa[0] + t * (pb[0] - pa[0]);
    const float wy = pa[1] + t * (pb[1] - pa[1]);
    const float wz = pa[2] + t * (pb[2] - pa[2]);
    const std::size_t n_before = vertices_.size();
    const std::uint32_t w = add_vertex(wx, wy, wz);
    if (vertices_.size() == n_before) {
        // add_vertex is position-idempotent: an unchanged slab size means w
        // coincides with an EXISTING vertex (an endpoint or any other vertex).
        // Splitting onto an existing vertex would create degenerate/broken
        // topology, so reject. (This subsumes the old w == va || w == vb check.)
        return std::nullopt;
    }

    const std::uint32_t fa = halfedges_[he_a].face;
    const std::uint32_t fb = halfedges_[he_b].face;
    std::vector<std::uint32_t> loopA, loopB;
    if (fa != INVALID_ID) loopA = faces_[fa].loop;
    if (fb != INVALID_ID) loopB = faces_[fb].loop;

    if (fa != INVALID_ID) remove_face(fa);
    if (fb != INVALID_ID) remove_face(fb);
    remove_edge(e_id);

    const std::uint32_t edge_a = add_halfedge_pair(va, w);
    const std::uint32_t edge_b = add_halfedge_pair(w, vb);

    auto rebuild = [&](const std::vector<std::uint32_t>& loop) -> std::uint32_t {
        if (loop.empty()) return INVALID_ID;
        std::vector<std::uint32_t> nl = loop_with_inserted(loop, va, vb, w);
        std::vector<std::int32_t> tris;
        tris.reserve((nl.size() - 2) * 3);
        for (std::size_t i = 1; i + 1 < nl.size(); ++i) {
            tris.push_back((std::int32_t)nl[0]);
            tris.push_back((std::int32_t)nl[i]);
            tris.push_back((std::int32_t)nl[i + 1]);
        }
        return add_face_from_loop(nl, tris);
    };
    const std::uint32_t new_fa = rebuild(loopA);
    const std::uint32_t new_fb = rebuild(loopB);

    dirty_ = true;
    return SplitEdgeResult{w, edge_a, edge_b, new_fa, new_fb};
}

std::uint32_t HalfEdgeMesh::halfedge_origin(std::uint32_t he_id) const noexcept {
    return he_id < halfedges_.size() && halfedges_[he_id].alive ? halfedges_[he_id].origin
                                                                : INVALID_ID;
}
std::uint32_t HalfEdgeMesh::halfedge_next(std::uint32_t he_id) const noexcept {
    return he_id < halfedges_.size() && halfedges_[he_id].alive ? halfedges_[he_id].next
                                                                : INVALID_ID;
}
std::uint32_t HalfEdgeMesh::halfedge_twin(std::uint32_t he_id) const noexcept {
    return he_id < halfedges_.size() && halfedges_[he_id].alive ? halfedges_[he_id].twin
                                                                : INVALID_ID;
}
std::uint32_t HalfEdgeMesh::halfedge_face(std::uint32_t he_id) const noexcept {
    return he_id < halfedges_.size() && halfedges_[he_id].alive ? halfedges_[he_id].face
                                                                : INVALID_ID;
}

std::uint32_t HalfEdgeMesh::next_live_vertex(std::uint32_t start) const noexcept {
    for (std::uint32_t i = start; i < vertices_.size(); ++i) {
        if (vertices_[i].alive) return i;
    }
    return INVALID_ID;
}
std::uint32_t HalfEdgeMesh::next_live_edge(std::uint32_t start) const noexcept {
    for (std::uint32_t i = start; (i * 2) < halfedges_.size(); ++i) {
        if (halfedges_[i * 2].alive) return i;
    }
    return INVALID_ID;
}
std::uint32_t HalfEdgeMesh::next_live_face(std::uint32_t start) const noexcept {
    for (std::uint32_t i = start; i < faces_.size(); ++i) {
        if (faces_[i].alive) return i;
    }
    return INVALID_ID;
}

std::vector<float> HalfEdgeMesh::edge_line_buffer() const {
    std::vector<float> out;
    for (std::uint32_t e = next_live_edge(0); e != INVALID_ID; e = next_live_edge(e + 1)) {
        const std::uint32_t he_a = e * 2;
        const std::uint32_t va = halfedges_[he_a].origin;
        const std::uint32_t vb = halfedges_[he_a + 1].origin;
        const auto& pa = vertices_[va].pos;
        const auto& pb = vertices_[vb].pos;
        out.insert(out.end(), {pa[0], pa[1], pa[2], pb[0], pb[1], pb[2]});
    }
    return out;
}

std::pair<std::vector<float>, std::vector<float>> HalfEdgeMesh::face_triangle_buffer() const {
    std::vector<float> positions;
    std::vector<float> normals;
    for (std::uint32_t f = next_live_face(0); f != INVALID_ID; f = next_live_face(f + 1)) {
        const auto& face = faces_[f];
        for (std::size_t i = 0; i + 2 < face.tris.size(); i += 3) {
            for (std::size_t k = 0; k < 3; ++k) {
                const std::uint32_t v = static_cast<std::uint32_t>(face.tris[i + k]);
                const auto& p = vertices_[v].pos;
                positions.insert(positions.end(), {p[0], p[1], p[2]});
                normals.insert(normals.end(), {face.normal[0], face.normal[1], face.normal[2]});
            }
        }
    }
    return {std::move(positions), std::move(normals)};
}

}  // namespace bermake
