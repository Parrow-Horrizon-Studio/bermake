#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pluton {

/// Result of HalfEdgeMesh::split_edge — the ids of the entities it created.
/// face_a / face_b are INVALID_ID for a boundary edge's empty side.
struct SplitEdgeResult {
    std::uint32_t vertex;  // the new vertex w inserted on the edge
    std::uint32_t edge_a;  // new edge (v_min — w)
    std::uint32_t edge_b;  // new edge (w — v_max)
    std::uint32_t face_a;  // rebuilt face on he(2e) side, or INVALID_ID
    std::uint32_t face_b;  // rebuilt face on he(2e+1) side, or INVALID_ID
};

/// Half-edge mesh — the topology source of truth for Pluton's M3+ kernel.
///
/// Storage layout:
///   - Vertices, half-edges, and faces live in std::vector slabs.
///   - Each entity has a stable uint32_t ID = its index in the slab.
///   - Removal sets `alive = false` (tombstone). Slots are never reused;
///     IDs stay valid for resurrection by restore_*.
///   - Edges are implicit: every "edge" is a pair of twin half-edges at
///     adjacent vector indices (2*edge_id and 2*edge_id+1). The half-edge
///     at the even index has origin = min(v1, v2); the odd one has
///     origin = max(v1, v2).
class HalfEdgeMesh {
public:
    static constexpr std::uint32_t INVALID_ID = 0xFFFFFFFFu;

    // ---- Mutators ----------------------------------------------------
    std::uint32_t add_vertex(float x, float y, float z);
    std::uint32_t add_halfedge_pair(std::uint32_t v1_id, std::uint32_t v2_id);
    std::uint32_t add_face_from_loop(const std::vector<std::uint32_t>& loop,
                                     const std::vector<std::int32_t>& triangles);

    void remove_vertex(std::uint32_t v_id);
    void remove_edge(std::uint32_t e_id);
    void remove_face(std::uint32_t f_id);

    void restore_vertex(std::uint32_t v_id, float x, float y, float z);
    void restore_edge(std::uint32_t e_id, std::uint32_t v1_id, std::uint32_t v2_id);
    void restore_face(std::uint32_t f_id, const std::vector<std::uint32_t>& loop,
                      const std::vector<std::int32_t>& triangles);

    /// Move an existing live vertex to (x, y, z) in place. Updates the
    /// position dedup index (last-writer-wins on a coincident collision) and
    /// recomputes the cached normal of every incident face. Throws
    /// std::out_of_range if v_id is dead/out of range. No re-triangulation,
    /// no topological merge.
    void set_vertex_position(std::uint32_t v_id, float x, float y, float z);

    void clear() noexcept;

    // ---- Queries -----------------------------------------------------
    bool vertex_is_live(std::uint32_t v_id) const noexcept;
    bool edge_is_live(std::uint32_t e_id) const noexcept;
    bool face_is_live(std::uint32_t f_id) const noexcept;

    /// Look up the edge id between v1 and v2 without creating one.
    /// Returns INVALID_ID if no live edge connects them. Unlike
    /// add_halfedge_pair, this never mutates edge_index_ or the slabs.
    std::uint32_t edge_between(std::uint32_t v1_id, std::uint32_t v2_id) const noexcept;

    std::array<float, 3> vertex_position(std::uint32_t v_id) const;

    /// One outgoing half-edge slab index for `v_id` (INVALID_ID if the
    /// vertex has none, e.g. freshly added with no incident edges yet).
    /// Exposed for testing internal bookkeeping (e.g. that dissolve_edge
    /// repoints a stale reference); no adjacency walk reads it yet.
    std::uint32_t vertex_outgoing_halfedge(std::uint32_t v_id) const;

    std::array<std::uint32_t, 2> edge_vertices(std::uint32_t e_id) const;
    std::vector<std::uint32_t> face_loop_vertices(std::uint32_t f_id) const;
    std::vector<std::int32_t> face_triangles(std::uint32_t f_id) const;

    /// Robust planar-coplanarity test for two faces.
    /// Returns true iff both:
    ///   - the angle between unit normals satisfies dot(n1, n2) > angle_tol_cos, AND
    ///   - every vertex of either face lies within `dist_tol` of the other face's plane.
    /// Returns false (without crashing) for a face whose geometric normal is the
    /// {0,0,0} degenerate sentinel — one whose Newell area vector falls at or
    /// below the kernel's degenerate-area threshold.
    /// Both tolerances are required (no defaults at this layer); Scene.faces_are_coplanar
    /// applies the project's recommended values: angle_tol_cos = cos(0.5°) ≈ 0.9999619f,
    /// dist_tol = 1e-4f.
    bool faces_are_coplanar(std::uint32_t f1_id, std::uint32_t f2_id, float angle_tol_cos,
                            float dist_tol) const;

    /// Dissolve an edge between two adjacent faces — merges the faces into one.
    /// Returns the new (surviving) face id on success.
    /// Returns INVALID_ID if the edge is on the mesh boundary (only 1 incident
    /// face), already tombstoned, or if the two adjacent faces share more than
    /// one edge (would create a degenerate result).
    /// The dissolved edge id is tombstoned (never reused).
    std::uint32_t dissolve_edge(std::uint32_t e_id);

    /// Replace face f_id with two faces that share a chain of vertices: the
    /// inverse of dissolve_edge. loop_a and loop_b are the two new boundary
    /// loops, each already triangulated by the caller with GLOBAL vertex ids,
    /// exactly as add_face_from_loop expects. Each loop must be simple (no
    /// repeated vertex); a loop that revisits a vertex is rejected.
    ///
    /// Returns {id_a, id_b}, or {INVALID_ID, INVALID_ID} if any precondition
    /// fails, in which case the mesh is untouched. Every check runs before the
    /// first mutation on purpose: add_face_from_loop throws on a missing edge,
    /// and a throw after remove_face would leave a hole with no way back.
    std::array<std::uint32_t, 2> split_face(std::uint32_t f_id,
                                            const std::vector<std::uint32_t>& loop_a,
                                            const std::vector<std::int32_t>& tris_a,
                                            const std::vector<std::uint32_t>& loop_b,
                                            const std::vector<std::int32_t>& tris_b);

    /// Split an edge at parameter t ∈ (0,1), inserting a new vertex w at
    /// p(v_min) + t*(p(v_max) - p(v_min)) where (v_min, v_max) = edge_vertices(e_id).
    /// The edge is replaced by two collinear edges and w is inserted into the
    /// boundary loop of each incident face (which is re-triangulated). Manifold
    /// structure is preserved. Returns the created ids, or std::nullopt if e_id
    /// is dead, t ∉ (0,1), or w coincides with an existing endpoint.
    std::optional<SplitEdgeResult> split_edge(std::uint32_t e_id, float t);

    std::uint32_t halfedge_origin(std::uint32_t he_id) const noexcept;
    std::uint32_t halfedge_next(std::uint32_t he_id) const noexcept;
    std::uint32_t halfedge_twin(std::uint32_t he_id) const noexcept;
    std::uint32_t halfedge_face(std::uint32_t he_id) const noexcept;

    std::uint32_t next_live_vertex(std::uint32_t start = 0) const noexcept;
    std::uint32_t next_live_edge(std::uint32_t start = 0) const noexcept;
    std::uint32_t next_live_face(std::uint32_t start = 0) const noexcept;

    std::vector<float> edge_line_buffer() const;
    std::pair<std::vector<float>, std::vector<float>> face_triangle_buffer() const;

    bool is_dirty() const noexcept { return dirty_; }
    void mark_clean() noexcept { dirty_ = false; }

    std::size_t vertex_slab_size() const noexcept { return vertices_.size(); }
    std::size_t halfedge_slab_size() const noexcept { return halfedges_.size(); }
    std::size_t face_slab_size() const noexcept { return faces_.size(); }

private:
    struct Vertex {
        float pos[3];
        std::uint32_t outgoing_he;
        bool alive;
    };
    struct HalfEdge {
        std::uint32_t origin;
        std::uint32_t next;
        std::uint32_t twin;
        std::uint32_t face;
        bool alive;
    };
    struct Face {
        std::uint32_t boundary_he;
        /// Cached unit geometric normal, by Newell's method over the whole
        /// boundary loop, or {0,0,0} for a genuinely degenerate (zero-area)
        /// face — never a guessed direction (issue #110). Read by the
        /// renderer through face_triangle_buffer, for lighting and for each
        /// face's texture projection basis.
        float normal[3];
        std::vector<std::int32_t> tris;
        std::vector<std::uint32_t> loop;
        bool alive;
    };

    std::vector<Vertex> vertices_;
    std::vector<HalfEdge> halfedges_;
    std::vector<Face> faces_;

    // Packed position → live vertex id, for idempotent add_vertex.
    std::unordered_map<std::uint64_t, std::uint32_t> position_index_;
    // (min, max) vertex pair → edge id, for idempotent add_halfedge_pair.
    std::unordered_map<std::uint64_t, std::uint32_t> edge_index_;

    bool dirty_ = false;

    // Helpers
    static std::uint64_t pack_position(float x, float y, float z) noexcept;
    static std::uint64_t pack_pair(std::uint32_t a, std::uint32_t b) noexcept;
    void recompute_face_normal(std::uint32_t f_id);
};

}  // namespace pluton
