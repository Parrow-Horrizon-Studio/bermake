#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "bermake/halfedge.h"

TEST(HalfEdgeMeshTest, DefaultConstructedIsEmpty) {
    bermake::HalfEdgeMesh m;
    EXPECT_EQ(m.vertex_slab_size(), 0u);
    EXPECT_EQ(m.halfedge_slab_size(), 0u);
    EXPECT_EQ(m.face_slab_size(), 0u);
    EXPECT_FALSE(m.is_dirty());
}

TEST(HalfEdgeMeshTest, ClearSetsDirty) {
    bermake::HalfEdgeMesh m;
    m.clear();
    EXPECT_TRUE(m.is_dirty());
    EXPECT_EQ(m.vertex_slab_size(), 0u);
}

TEST(HalfEdgeMeshTest, ClearTombstonesSoRestoreStillWorks) {
    bermake::HalfEdgeMesh m;
    auto v0 = m.add_vertex(0.0f, 0.0f, 0.0f);
    auto v1 = m.add_vertex(1.0f, 0.0f, 0.0f);
    m.add_halfedge_pair(v0, v1);

    m.clear();

    // Everything is dead...
    EXPECT_FALSE(m.vertex_is_live(v0));
    EXPECT_FALSE(m.vertex_is_live(v1));
    // ...but the ids are still addressable, so undo can revive them.
    EXPECT_NO_THROW(m.restore_vertex(v0, 0.0f, 0.0f, 0.0f));
    EXPECT_TRUE(m.vertex_is_live(v0));
}

TEST(HalfEdgeMeshTest, InvalidIdConstant) {
    EXPECT_EQ(bermake::HalfEdgeMesh::INVALID_ID, 0xFFFFFFFFu);
}

TEST(HalfEdgeMeshTest, AddVertexReturnsNewIds) {
    bermake::HalfEdgeMesh m;
    auto v0 = m.add_vertex(0.0f, 0.0f, 0.0f);
    auto v1 = m.add_vertex(1.0f, 0.0f, 0.0f);
    EXPECT_NE(v0, v1);
    EXPECT_TRUE(m.vertex_is_live(v0));
    EXPECT_TRUE(m.vertex_is_live(v1));
    EXPECT_TRUE(m.is_dirty());
}

TEST(HalfEdgeMeshTest, AddVertexIsIdempotentOnExactMatch) {
    bermake::HalfEdgeMesh m;
    auto v0 = m.add_vertex(2.0f, 3.0f, 0.0f);
    auto v1 = m.add_vertex(2.0f, 3.0f, 0.0f);
    EXPECT_EQ(v0, v1);
    EXPECT_EQ(m.vertex_slab_size(), 1u);
}

TEST(HalfEdgeMeshTest, AddVertexCollapsesNegativeZero) {
    bermake::HalfEdgeMesh m;
    auto v0 = m.add_vertex(0.0f, 0.0f, 0.0f);
    auto v1 = m.add_vertex(-0.0f, 0.0f, 0.0f);
    EXPECT_EQ(v0, v1);
}

TEST(HalfEdgeMeshTest, AddVertexStoresPosition) {
    bermake::HalfEdgeMesh m;
    auto v = m.add_vertex(5.0f, 6.0f, 7.0f);
    auto p = m.vertex_position(v);
    EXPECT_FLOAT_EQ(p[0], 5.0f);
    EXPECT_FLOAT_EQ(p[1], 6.0f);
    EXPECT_FLOAT_EQ(p[2], 7.0f);
}

TEST(HalfEdgeMeshTest, AddHalfedgePairReturnsEdgeId) {
    bermake::HalfEdgeMesh m;
    auto v0 = m.add_vertex(0.0f, 0.0f, 0.0f);
    auto v1 = m.add_vertex(1.0f, 0.0f, 0.0f);
    auto e = m.add_halfedge_pair(v0, v1);
    EXPECT_EQ(e, 0u);
    EXPECT_TRUE(m.edge_is_live(e));
    EXPECT_EQ(m.halfedge_slab_size(), 2u);  // exactly one pair allocated
}

TEST(HalfEdgeMeshTest, AddHalfedgePairIsIdempotentUnordered) {
    bermake::HalfEdgeMesh m;
    auto v0 = m.add_vertex(0.0f, 0.0f, 0.0f);
    auto v1 = m.add_vertex(1.0f, 0.0f, 0.0f);
    auto a = m.add_halfedge_pair(v0, v1);
    auto b = m.add_halfedge_pair(v1, v0);
    EXPECT_EQ(a, b);
    EXPECT_EQ(m.halfedge_slab_size(), 2u);
}

TEST(HalfEdgeMeshTest, AddHalfedgePairRejectsSelfLoop) {
    bermake::HalfEdgeMesh m;
    auto v0 = m.add_vertex(0.0f, 0.0f, 0.0f);
    EXPECT_THROW(m.add_halfedge_pair(v0, v0), std::invalid_argument);
}

TEST(HalfEdgeMeshTest, AddHalfedgePairWiresTwinsAndOrigins) {
    bermake::HalfEdgeMesh m;
    auto v0 = m.add_vertex(0.0f, 0.0f, 0.0f);
    auto v1 = m.add_vertex(1.0f, 0.0f, 0.0f);
    auto e = m.add_halfedge_pair(v1, v0);  // swapped order on input

    const auto verts = m.edge_vertices(e);
    EXPECT_EQ(verts[0], std::min(v0, v1));  // canonical: v1 < v2
    EXPECT_EQ(verts[1], std::max(v0, v1));

    const std::uint32_t he_a = e * 2;
    const std::uint32_t he_b = he_a + 1;
    EXPECT_EQ(m.halfedge_twin(he_a), he_b);
    EXPECT_EQ(m.halfedge_twin(he_b), he_a);
    EXPECT_EQ(m.halfedge_origin(he_a), std::min(v0, v1));
    EXPECT_EQ(m.halfedge_origin(he_b), std::max(v0, v1));
    EXPECT_EQ(m.halfedge_face(he_a), bermake::HalfEdgeMesh::INVALID_ID);
    EXPECT_EQ(m.halfedge_face(he_b), bermake::HalfEdgeMesh::INVALID_ID);
}

TEST(HalfEdgeMeshTest, EdgeBetweenFindsAnExistingEdge) {
    bermake::HalfEdgeMesh m;
    const auto a = m.add_vertex(0.0f, 0.0f, 0.0f);
    const auto b = m.add_vertex(1.0f, 0.0f, 0.0f);
    const auto e = m.add_halfedge_pair(a, b);
    EXPECT_EQ(m.edge_between(a, b), e);
}

TEST(HalfEdgeMeshTest, EdgeBetweenIsOrderIndependent) {
    bermake::HalfEdgeMesh m;
    const auto a = m.add_vertex(0.0f, 0.0f, 0.0f);
    const auto b = m.add_vertex(1.0f, 0.0f, 0.0f);
    const auto e = m.add_halfedge_pair(a, b);
    EXPECT_EQ(m.edge_between(b, a), e);
}

TEST(HalfEdgeMeshTest, EdgeBetweenReportsAbsence) {
    bermake::HalfEdgeMesh m;
    const auto a = m.add_vertex(0.0f, 0.0f, 0.0f);
    const auto b = m.add_vertex(1.0f, 0.0f, 0.0f);
    // No edge added between them.
    EXPECT_EQ(m.edge_between(a, b), bermake::HalfEdgeMesh::INVALID_ID);
}

TEST(HalfEdgeMeshTest, EdgeBetweenDoesNotMutate) {
    bermake::HalfEdgeMesh m;
    const auto a = m.add_vertex(0.0f, 0.0f, 0.0f);
    const auto b = m.add_vertex(1.0f, 0.0f, 0.0f);
    const auto slab_before = m.halfedge_slab_size();
    (void)m.edge_between(a, b);
    EXPECT_EQ(m.halfedge_slab_size(), slab_before);
}

TEST(HalfEdgeMeshTest, AddFaceFromLoopWiresBoundaryCycle) {
    bermake::HalfEdgeMesh m;
    auto v0 = m.add_vertex(0.0f, 0.0f, 0.0f);
    auto v1 = m.add_vertex(1.0f, 0.0f, 0.0f);
    auto v2 = m.add_vertex(1.0f, 1.0f, 0.0f);
    auto v3 = m.add_vertex(0.0f, 1.0f, 0.0f);
    m.add_halfedge_pair(v0, v1);
    m.add_halfedge_pair(v1, v2);
    m.add_halfedge_pair(v2, v3);
    m.add_halfedge_pair(v3, v0);

    const std::vector<std::uint32_t> loop = {v0, v1, v2, v3};
    const std::vector<std::int32_t> tris = {
        static_cast<std::int32_t>(v0), static_cast<std::int32_t>(v1),
        static_cast<std::int32_t>(v2), static_cast<std::int32_t>(v0),
        static_cast<std::int32_t>(v2), static_cast<std::int32_t>(v3)};
    auto f = m.add_face_from_loop(loop, tris);
    EXPECT_EQ(f, 0u);
    EXPECT_TRUE(m.face_is_live(f));
    EXPECT_EQ(m.face_loop_vertices(f), loop);
    EXPECT_EQ(m.face_triangles(f), tris);
}

TEST(HalfEdgeMeshTest, AddFaceFromLoopRejectsShortLoop) {
    bermake::HalfEdgeMesh m;
    auto v0 = m.add_vertex(0.0f, 0.0f, 0.0f);
    auto v1 = m.add_vertex(1.0f, 0.0f, 0.0f);
    EXPECT_THROW(m.add_face_from_loop({v0, v1}, {}), std::invalid_argument);
}

TEST(HalfEdgeMeshTest, AddFaceFromLoopSetsHalfedgeFacePointers) {
    bermake::HalfEdgeMesh m;
    auto v0 = m.add_vertex(0.0f, 0.0f, 0.0f);
    auto v1 = m.add_vertex(1.0f, 0.0f, 0.0f);
    auto v2 = m.add_vertex(0.0f, 1.0f, 0.0f);
    m.add_halfedge_pair(v0, v1);
    m.add_halfedge_pair(v1, v2);
    m.add_halfedge_pair(v2, v0);
    const std::vector<std::int32_t> tris = {static_cast<std::int32_t>(v0),
                                            static_cast<std::int32_t>(v1),
                                            static_cast<std::int32_t>(v2)};
    auto f = m.add_face_from_loop({v0, v1, v2}, tris);

    // For each edge in the loop, exactly ONE of the two halfedges should have
    // its face set to f (the one walking the loop in order). The twin should
    // stay at INVALID_ID (it's a boundary edge).
    for (std::uint32_t e = 0; e < 3; ++e) {
        const std::uint32_t he_a = e * 2;
        const std::uint32_t he_b = he_a + 1;
        EXPECT_TRUE(m.halfedge_face(he_a) == f || m.halfedge_face(he_b) == f);
        EXPECT_TRUE(m.halfedge_face(he_a) == bermake::HalfEdgeMesh::INVALID_ID ||
                    m.halfedge_face(he_b) == bermake::HalfEdgeMesh::INVALID_ID);
    }
}

TEST(HalfEdgeMeshTest, RemoveFaceTombstonesSlot) {
    bermake::HalfEdgeMesh m;
    auto v0 = m.add_vertex(0.0f, 0.0f, 0.0f);
    auto v1 = m.add_vertex(1.0f, 0.0f, 0.0f);
    auto v2 = m.add_vertex(0.0f, 1.0f, 0.0f);
    m.add_halfedge_pair(v0, v1);
    m.add_halfedge_pair(v1, v2);
    m.add_halfedge_pair(v2, v0);
    auto f = m.add_face_from_loop({v0, v1, v2},
                                  {static_cast<std::int32_t>(v0), static_cast<std::int32_t>(v1),
                                   static_cast<std::int32_t>(v2)});
    EXPECT_TRUE(m.face_is_live(f));

    m.remove_face(f);
    EXPECT_FALSE(m.face_is_live(f));

    // Vertices and edges stay alive.
    EXPECT_TRUE(m.vertex_is_live(v0));
    EXPECT_TRUE(m.edge_is_live(0u));
}

TEST(HalfEdgeMeshTest, RemoveFaceClearsHalfedgeFacePointers) {
    bermake::HalfEdgeMesh m;
    auto v0 = m.add_vertex(0.0f, 0.0f, 0.0f);
    auto v1 = m.add_vertex(1.0f, 0.0f, 0.0f);
    auto v2 = m.add_vertex(0.0f, 1.0f, 0.0f);
    m.add_halfedge_pair(v0, v1);
    m.add_halfedge_pair(v1, v2);
    m.add_halfedge_pair(v2, v0);
    auto f = m.add_face_from_loop({v0, v1, v2}, {0, 1, 2});

    m.remove_face(f);
    for (std::uint32_t he = 0; he < m.halfedge_slab_size(); ++he) {
        EXPECT_EQ(m.halfedge_face(he), bermake::HalfEdgeMesh::INVALID_ID);
    }
}

TEST(HalfEdgeMeshTest, RemoveFaceAlreadyDeadThrows) {
    bermake::HalfEdgeMesh m;
    auto v0 = m.add_vertex(0.0f, 0.0f, 0.0f);
    auto v1 = m.add_vertex(1.0f, 0.0f, 0.0f);
    auto v2 = m.add_vertex(0.0f, 1.0f, 0.0f);
    m.add_halfedge_pair(v0, v1);
    m.add_halfedge_pair(v1, v2);
    m.add_halfedge_pair(v2, v0);
    auto f = m.add_face_from_loop({v0, v1, v2}, {0, 1, 2});
    m.remove_face(f);
    EXPECT_THROW(m.remove_face(f), std::out_of_range);
}

TEST(HalfEdgeMeshTest, RemoveEdgeRejectsIfFaceUsesIt) {
    bermake::HalfEdgeMesh m;
    auto v0 = m.add_vertex(0.0f, 0.0f, 0.0f);
    auto v1 = m.add_vertex(1.0f, 0.0f, 0.0f);
    auto v2 = m.add_vertex(0.0f, 1.0f, 0.0f);
    m.add_halfedge_pair(v0, v1);
    m.add_halfedge_pair(v1, v2);
    m.add_halfedge_pair(v2, v0);
    m.add_face_from_loop({v0, v1, v2}, {0, 1, 2});
    EXPECT_THROW(m.remove_edge(0u), std::invalid_argument);
}

TEST(HalfEdgeMeshTest, RemoveEdgeAfterFaceWorks) {
    bermake::HalfEdgeMesh m;
    auto v0 = m.add_vertex(0.0f, 0.0f, 0.0f);
    auto v1 = m.add_vertex(1.0f, 0.0f, 0.0f);
    m.add_halfedge_pair(v0, v1);
    m.remove_edge(0u);
    EXPECT_FALSE(m.edge_is_live(0u));
}

TEST(HalfEdgeMeshTest, RemoveEdgeAlreadyDeadThrows) {
    bermake::HalfEdgeMesh m;
    auto v0 = m.add_vertex(0.0f, 0.0f, 0.0f);
    auto v1 = m.add_vertex(1.0f, 0.0f, 0.0f);
    m.add_halfedge_pair(v0, v1);
    m.remove_edge(0u);
    EXPECT_THROW(m.remove_edge(0u), std::out_of_range);
}

TEST(HalfEdgeMeshTest, RemoveVertexRejectsIfEdgeUsesIt) {
    bermake::HalfEdgeMesh m;
    auto v0 = m.add_vertex(0.0f, 0.0f, 0.0f);
    auto v1 = m.add_vertex(1.0f, 0.0f, 0.0f);
    m.add_halfedge_pair(v0, v1);
    EXPECT_THROW(m.remove_vertex(v0), std::invalid_argument);
}

TEST(HalfEdgeMeshTest, RemoveVertexAfterEdgeWorks) {
    bermake::HalfEdgeMesh m;
    auto v0 = m.add_vertex(0.0f, 0.0f, 0.0f);
    auto v1 = m.add_vertex(1.0f, 0.0f, 0.0f);
    m.add_halfedge_pair(v0, v1);
    m.remove_edge(0u);
    m.remove_vertex(v0);
    EXPECT_FALSE(m.vertex_is_live(v0));
    EXPECT_TRUE(m.vertex_is_live(v1));
}

TEST(HalfEdgeMeshTest, RestoreVertexRoundTrips) {
    bermake::HalfEdgeMesh m;
    auto v = m.add_vertex(1.0f, 2.0f, 3.0f);
    m.remove_vertex(v);
    EXPECT_FALSE(m.vertex_is_live(v));

    m.restore_vertex(v, 1.0f, 2.0f, 3.0f);
    EXPECT_TRUE(m.vertex_is_live(v));
    auto p = m.vertex_position(v);
    EXPECT_FLOAT_EQ(p[0], 1.0f);
    EXPECT_FLOAT_EQ(p[1], 2.0f);
    EXPECT_FLOAT_EQ(p[2], 3.0f);
}

TEST(HalfEdgeMeshTest, RestoreVertexLiveSlotThrows) {
    bermake::HalfEdgeMesh m;
    auto v = m.add_vertex(1.0f, 2.0f, 3.0f);
    EXPECT_THROW(m.restore_vertex(v, 0.0f, 0.0f, 0.0f), std::logic_error);
}

TEST(HalfEdgeMeshTest, RestoreEdgeRoundTrips) {
    bermake::HalfEdgeMesh m;
    auto v0 = m.add_vertex(0.0f, 0.0f, 0.0f);
    auto v1 = m.add_vertex(1.0f, 0.0f, 0.0f);
    auto e = m.add_halfedge_pair(v0, v1);
    m.remove_edge(e);
    EXPECT_FALSE(m.edge_is_live(e));

    m.restore_edge(e, v0, v1);
    EXPECT_TRUE(m.edge_is_live(e));
    auto verts = m.edge_vertices(e);
    EXPECT_EQ(verts[0], std::min(v0, v1));
    EXPECT_EQ(verts[1], std::max(v0, v1));
}

TEST(HalfEdgeMeshTest, RestoreFaceRoundTrips) {
    bermake::HalfEdgeMesh m;
    auto v0 = m.add_vertex(0.0f, 0.0f, 0.0f);
    auto v1 = m.add_vertex(1.0f, 0.0f, 0.0f);
    auto v2 = m.add_vertex(0.0f, 1.0f, 0.0f);
    m.add_halfedge_pair(v0, v1);
    m.add_halfedge_pair(v1, v2);
    m.add_halfedge_pair(v2, v0);
    const std::vector<std::int32_t> tris = {0, 1, 2};
    auto f = m.add_face_from_loop({v0, v1, v2}, tris);
    m.remove_face(f);
    EXPECT_FALSE(m.face_is_live(f));

    m.restore_face(f, {v0, v1, v2}, tris);
    EXPECT_TRUE(m.face_is_live(f));
    EXPECT_EQ(m.face_loop_vertices(f), std::vector<std::uint32_t>({v0, v1, v2}));
}

TEST(HalfEdgeMeshTest, NextLiveVertexSkipsTombstones) {
    bermake::HalfEdgeMesh m;
    auto v0 = m.add_vertex(0.0f, 0.0f, 0.0f);
    auto v1 = m.add_vertex(1.0f, 0.0f, 0.0f);
    auto v2 = m.add_vertex(2.0f, 0.0f, 0.0f);
    EXPECT_EQ(m.next_live_vertex(0), v0);
    EXPECT_EQ(m.next_live_vertex(v0 + 1), v1);
    EXPECT_EQ(m.next_live_vertex(v1 + 1), v2);

    m.remove_vertex(v1);
    EXPECT_EQ(m.next_live_vertex(0), v0);
    EXPECT_EQ(m.next_live_vertex(v0 + 1), v2);  // skipped v1
    EXPECT_EQ(m.next_live_vertex(v2 + 1), bermake::HalfEdgeMesh::INVALID_ID);
}

TEST(HalfEdgeMeshTest, ClearEmptiesEverythingAndMarksDirty) {
    bermake::HalfEdgeMesh m;
    m.add_vertex(0.0f, 0.0f, 0.0f);
    m.add_vertex(1.0f, 0.0f, 0.0f);
    m.mark_clean();
    EXPECT_FALSE(m.is_dirty());

    m.clear();
    EXPECT_TRUE(m.is_dirty());
    // clear() tombstones rather than shrinks (so ids stay addressable for
    // restore_*, see ClearTombstonesSoRestoreStillWorks) -- the slab keeps
    // its size, but nothing in it is live anymore.
    EXPECT_EQ(m.vertex_slab_size(), 2u);
    EXPECT_EQ(m.next_live_vertex(0), bermake::HalfEdgeMesh::INVALID_ID);
}

TEST(HalfEdgeMeshTest, MarkCleanClearsDirty) {
    bermake::HalfEdgeMesh m;
    m.add_vertex(0.0f, 0.0f, 0.0f);
    EXPECT_TRUE(m.is_dirty());
    m.mark_clean();
    EXPECT_FALSE(m.is_dirty());
}

TEST(HalfEdgeMeshTest, EdgeLineBufferShape) {
    bermake::HalfEdgeMesh m;
    auto v0 = m.add_vertex(0.0f, 0.0f, 0.0f);
    auto v1 = m.add_vertex(1.0f, 0.0f, 0.0f);
    m.add_halfedge_pair(v0, v1);

    const auto buf = m.edge_line_buffer();
    ASSERT_EQ(buf.size(), 6u);  // 2 endpoints × 3 floats
    EXPECT_FLOAT_EQ(buf[0], 0.0f);
    EXPECT_FLOAT_EQ(buf[1], 0.0f);
    EXPECT_FLOAT_EQ(buf[2], 0.0f);
    EXPECT_FLOAT_EQ(buf[3], 1.0f);
    EXPECT_FLOAT_EQ(buf[4], 0.0f);
    EXPECT_FLOAT_EQ(buf[5], 0.0f);
}

TEST(HalfEdgeMeshTest, EdgeLineBufferSkipsTombstones) {
    bermake::HalfEdgeMesh m;
    auto v0 = m.add_vertex(0.0f, 0.0f, 0.0f);
    auto v1 = m.add_vertex(1.0f, 0.0f, 0.0f);
    auto v2 = m.add_vertex(2.0f, 0.0f, 0.0f);
    m.add_halfedge_pair(v0, v1);
    m.add_halfedge_pair(v1, v2);
    m.remove_edge(0u);

    const auto buf = m.edge_line_buffer();
    EXPECT_EQ(buf.size(), 6u);  // only the live edge contributes
}

TEST(HalfEdgeMeshTest, FaceTriangleBufferShape) {
    bermake::HalfEdgeMesh m;
    auto v0 = m.add_vertex(0.0f, 0.0f, 0.0f);
    auto v1 = m.add_vertex(1.0f, 0.0f, 0.0f);
    auto v2 = m.add_vertex(0.0f, 1.0f, 0.0f);
    m.add_halfedge_pair(v0, v1);
    m.add_halfedge_pair(v1, v2);
    m.add_halfedge_pair(v2, v0);
    m.add_face_from_loop({v0, v1, v2}, {0, 1, 2});

    auto [positions, normals] = m.face_triangle_buffer();
    EXPECT_EQ(positions.size(), 9u);  // 1 triangle × 3 verts × 3 floats
    EXPECT_EQ(normals.size(), 9u);
    // Normal of every vertex is the face's +Z normal.
    // Loop (0,0,0)→(1,0,0)→(0,1,0) is CCW when viewed from +Z, so cross
    // product e1=(1,0,0) × e2=(0,1,0) = (0,0,1).
    for (std::size_t i = 0; i + 2 < normals.size(); i += 3) {
        EXPECT_FLOAT_EQ(normals[i + 0], 0.0f);
        EXPECT_FLOAT_EQ(normals[i + 1], 0.0f);
        EXPECT_FLOAT_EQ(normals[i + 2], 1.0f);
    }
}

// Regression test: add_face_from_loop must compute the geometric normal from
// the cross product of the first two boundary edges — NOT hardcode (0,0,1).
// A face on the YZ-plane (x=0) with CCW winding (viewed from +X) should have
// normal (+1, 0, 0).
//
// Vertices: A=(0,0,0), B=(0,1,0), C=(0,1,1), D=(0,0,1)
// e1 = B - A = (0,1,0)
// e2 = C - A = (0,1,1)
// n = e1 × e2 = (1*1 - 0*1, 0*0 - 0*1, 0*1 - 1*0) = (1, 0, 0)  → normalised: (+1, 0, 0)
TEST(HalfEdgeMeshTest, FaceNormalComputedGeometricallyYZPlane) {
    bermake::HalfEdgeMesh m;
    // YZ-plane quad: x=0, CCW from +X
    auto vA = m.add_vertex(0.0f, 0.0f, 0.0f);
    auto vB = m.add_vertex(0.0f, 1.0f, 0.0f);
    auto vC = m.add_vertex(0.0f, 1.0f, 1.0f);
    auto vD = m.add_vertex(0.0f, 0.0f, 1.0f);
    m.add_halfedge_pair(vA, vB);
    m.add_halfedge_pair(vB, vC);
    m.add_halfedge_pair(vC, vD);
    m.add_halfedge_pair(vD, vA);
    const std::vector<std::int32_t> tris = {
        static_cast<std::int32_t>(vA), static_cast<std::int32_t>(vB), static_cast<std::int32_t>(vC),
        static_cast<std::int32_t>(vA), static_cast<std::int32_t>(vC), static_cast<std::int32_t>(vD),
    };
    m.add_face_from_loop({vA, vB, vC, vD}, tris);

    auto [positions, normals] = m.face_triangle_buffer();
    ASSERT_EQ(normals.size(), 18u);  // 2 triangles × 3 verts × 3 floats
    // Every vertex should share the face normal: (+1, 0, 0)
    for (std::size_t i = 0; i + 2 < normals.size(); i += 3) {
        EXPECT_NEAR(normals[i + 0], +1.0f, 1e-6f);
        EXPECT_NEAR(normals[i + 1], 0.0f, 1e-6f);
        EXPECT_NEAR(normals[i + 2], 0.0f, 1e-6f);
    }
}

// ====================================================================
// M3c: faces_are_coplanar
// ====================================================================

namespace {

// Helper: build a triangle face from 3 explicit positions, return face id.
std::uint32_t add_triangle(bermake::HalfEdgeMesh& m, std::array<float, 3> p0,
                           std::array<float, 3> p1, std::array<float, 3> p2) {
    auto v0 = m.add_vertex(p0[0], p0[1], p0[2]);
    auto v1 = m.add_vertex(p1[0], p1[1], p1[2]);
    auto v2 = m.add_vertex(p2[0], p2[1], p2[2]);
    m.add_halfedge_pair(v0, v1);
    m.add_halfedge_pair(v1, v2);
    m.add_halfedge_pair(v2, v0);
    return m.add_face_from_loop({v0, v1, v2}, {(int)v0, (int)v1, (int)v2});
}

constexpr float kCos05Deg = 0.99996192306f;  // cos(0.5°)
constexpr float kDistTol = 1.0e-4f;

}  // namespace

TEST(HalfEdgeMeshTest, FacesAreCoplanar_TrueForIdenticalPlanes) {
    bermake::HalfEdgeMesh m;
    auto f1 = add_triangle(m, {0, 0, 0}, {1, 0, 0}, {0, 1, 0});  // XY plane
    auto f2 = add_triangle(m, {2, 2, 0}, {3, 2, 0}, {2, 3, 0});  // also XY plane
    EXPECT_TRUE(m.faces_are_coplanar(f1, f2, kCos05Deg, kDistTol));
    EXPECT_TRUE(m.faces_are_coplanar(f2, f1, kCos05Deg, kDistTol));  // symmetric
}

TEST(HalfEdgeMeshTest, FacesAreCoplanar_TrueWithinAngleTolerance) {
    // Two faces on planes whose normals differ by 0.3° — under the 0.5° tolerance.
    bermake::HalfEdgeMesh m;
    auto f1 = add_triangle(m, {0, 0, 0}, {1, 0, 0}, {0, 1, 0});  // normal (0,0,1)
    // Rotate the second face by 0.3° about X: normal becomes (0, -sin(0.3°), cos(0.3°))
    float c = std::cos(0.3f * 3.14159265f / 180.0f);
    float s = std::sin(0.3f * 3.14159265f / 180.0f);
    auto f2 = add_triangle(m, {2, 2, 0}, {3, 2, 0}, {2, 2 + c, s});
    // Loosened dist_tol: 0.3° tilt on a face anchored 2 units from origin gives
    // a ~1.05e-2 worst-case plane offset in the symmetric distance check, so
    // the project default 1e-4 would fail this geometry. The angle test is
    // what's being exercised here.
    EXPECT_TRUE(m.faces_are_coplanar(f1, f2, kCos05Deg, 2e-2f));  // looser dist
}

TEST(HalfEdgeMeshTest, FacesAreCoplanar_FalseBeyondAngleTolerance) {
    // 1.0° apart — over the 0.5° tolerance.
    bermake::HalfEdgeMesh m;
    auto f1 = add_triangle(m, {0, 0, 0}, {1, 0, 0}, {0, 1, 0});
    float c = std::cos(1.0f * 3.14159265f / 180.0f);
    float s = std::sin(1.0f * 3.14159265f / 180.0f);
    auto f2 = add_triangle(m, {2, 2, 0}, {3, 2, 0}, {2, 2 + c, s});
    EXPECT_FALSE(m.faces_are_coplanar(f1, f2, kCos05Deg, 1.0f));
}

TEST(HalfEdgeMeshTest, FacesAreCoplanar_FalseBeyondDistanceTolerance) {
    // Two parallel XY planes offset by 1e-3 (over the 1e-4 dist tolerance).
    bermake::HalfEdgeMesh m;
    auto f1 = add_triangle(m, {0, 0, 0}, {1, 0, 0}, {0, 1, 0});              // z = 0
    auto f2 = add_triangle(m, {2, 2, 1e-3f}, {3, 2, 1e-3f}, {2, 3, 1e-3f});  // z = 0.001
    EXPECT_FALSE(m.faces_are_coplanar(f1, f2, kCos05Deg, kDistTol));
}

TEST(HalfEdgeMeshTest, FacesAreCoplanar_FalseForDegenerateNormal) {
    // f1 has zero area (all 3 vertices collinear). Must not crash; must return false.
    bermake::HalfEdgeMesh m;
    auto v0 = m.add_vertex(0, 0, 0);
    auto v1 = m.add_vertex(1, 0, 0);
    auto v2 = m.add_vertex(2, 0, 0);
    m.add_halfedge_pair(v0, v1);
    m.add_halfedge_pair(v1, v2);
    m.add_halfedge_pair(v2, v0);
    auto f_degen = m.add_face_from_loop({v0, v1, v2}, {(int)v0, (int)v1, (int)v2});

    auto f_good = add_triangle(m, {5, 0, 0}, {6, 0, 0}, {5, 1, 0});
    EXPECT_FALSE(m.faces_are_coplanar(f_degen, f_good, kCos05Deg, kDistTol));
    EXPECT_FALSE(m.faces_are_coplanar(f_good, f_degen, kCos05Deg, kDistTol));
}

// Regression test for the #26 "threshold unification" review fix: the
// geometric-normal path (compute_face_normal_geometric, reached here via
// recompute_face_normal — set_vertex_position's real-world trigger, and also
// underlying faces_are_coplanar) must use its own TIGHTER 1e-7f degenerate
// guard, not the looser 1e-9f guard shared by add_face_from_loop/restore_face's
// render-fallback path.
//
// v0=(0,0,0), v1=(1,0,0), v2=(0.5, 1e-8, 0): e1=(1,0,0), e2=(0.5,1e-8,0),
// cross = e1 x e2 = (0, 0, 1e-8), so |cross| = 1e-8 — strictly between the
// two thresholds (1e-9, 1e-7). Under the (buggy) unified 1e-9f threshold this
// is "not degenerate" and gets normalized into the unit vector (0,0,1) —
// numerically fine here, but arbitrary-direction noise in general. Under the
// correct 1e-7f threshold it must be treated as degenerate and recompute to
// the {0,0,0} sentinel.
//
// add_face_from_loop computes its own initial normal via its own 1e-9f path,
// so the geometric path must be triggered explicitly. recompute_face_normal
// is private; set_vertex_position is its public real-world caller (also
// exercised interactively whenever a vertex is dragged), so we call it with
// the vertex's unchanged position purely to force the recompute, then read
// the result back via face_triangle_buffer (there is no public face_normal
// accessor).
TEST(HalfEdgeMeshTest, RecomputeFaceNormalGeometricDegenerateSentinel) {
    bermake::HalfEdgeMesh m;
    auto v0 = m.add_vertex(0.0f, 0.0f, 0.0f);
    auto v1 = m.add_vertex(1.0f, 0.0f, 0.0f);
    auto v2 = m.add_vertex(0.5f, 1e-8f, 0.0f);
    m.add_halfedge_pair(v0, v1);
    m.add_halfedge_pair(v1, v2);
    m.add_halfedge_pair(v2, v0);
    m.add_face_from_loop({v0, v1, v2}, {(int)v0, (int)v1, (int)v2});

    // Trigger recompute_face_normal (-> compute_face_normal_geometric) on the
    // face incident to v0 without moving anything.
    m.set_vertex_position(v0, 0.0f, 0.0f, 0.0f);

    auto [positions, normals] = m.face_triangle_buffer();
    ASSERT_EQ(normals.size(), 9u);  // 1 triangle x 3 verts x 3 floats
    EXPECT_FLOAT_EQ(normals[0], 0.0f);
    EXPECT_FLOAT_EQ(normals[1], 0.0f);
    EXPECT_FLOAT_EQ(normals[2], 0.0f);  // sentinel, NOT a unit (0,0,1) from normalized noise
}

// ====================================================================
// M3c: dissolve_edge — happy path
// ====================================================================

TEST(HalfEdgeMeshTest, DissolveEdge_TwoTrianglesIntoQuad) {
    // Build two triangles sharing edge v1—v2:
    //   T1 = (v0, v1, v2)   T2 = (v1, v3, v2)   shared edge: v1—v2
    bermake::HalfEdgeMesh m;
    auto v0 = m.add_vertex(0, 0, 0);
    auto v1 = m.add_vertex(1, 0, 0);
    auto v2 = m.add_vertex(1, 1, 0);
    auto v3 = m.add_vertex(2, 1, 0);
    m.add_halfedge_pair(v0, v1);
    m.add_halfedge_pair(v1, v2);  // shared
    m.add_halfedge_pair(v2, v0);
    m.add_halfedge_pair(v1, v3);
    m.add_halfedge_pair(v3, v2);

    auto f1 = m.add_face_from_loop({v0, v1, v2}, {(int)v0, (int)v1, (int)v2});
    auto f2 = m.add_face_from_loop({v1, v3, v2}, {(int)v1, (int)v3, (int)v2});

    // Find the shared edge id (the v1—v2 pair).
    std::uint32_t shared_edge = bermake::HalfEdgeMesh::INVALID_ID;
    for (std::uint32_t e = 0; e < m.halfedge_slab_size() / 2; ++e) {
        auto verts = m.edge_vertices(e);
        if ((verts[0] == v1 && verts[1] == v2) || (verts[0] == v2 && verts[1] == v1)) {
            shared_edge = e;
            break;
        }
    }
    ASSERT_NE(shared_edge, bermake::HalfEdgeMesh::INVALID_ID);

    auto merged = m.dissolve_edge(shared_edge);

    EXPECT_NE(merged, bermake::HalfEdgeMesh::INVALID_ID);
    EXPECT_FALSE(m.face_is_live(f1));
    EXPECT_FALSE(m.face_is_live(f2));
    EXPECT_FALSE(m.edge_is_live(shared_edge));
    EXPECT_TRUE(m.face_is_live(merged));

    // The merged face is a quad with 4 vertices.
    auto loop = m.face_loop_vertices(merged);
    EXPECT_EQ(loop.size(), 4u);

    // Verify the merged loop contains exactly {v0, v1, v2, v3} (set equality).
    std::set<std::uint32_t> loop_set(loop.begin(), loop.end());
    std::set<std::uint32_t> expected{v0, v1, v2, v3};
    EXPECT_EQ(loop_set, expected);

    // Verify each consecutive pair in the merged loop is connected by a live
    // half-edge pair (i.e. the spliced loop is a valid traversable cycle, not
    // four disconnected vertices).
    for (std::size_t i = 0; i < loop.size(); ++i) {
        std::uint32_t a = loop[i];
        std::uint32_t b = loop[(i + 1) % loop.size()];
        // Find the edge slot for (a, b); both halves must be live and at least
        // one half must belong to the merged face.
        bool found_live_edge_to_merged = false;
        for (std::uint32_t e = 0; e < m.halfedge_slab_size() / 2u; ++e) {
            if (!m.edge_is_live(e)) continue;
            auto verts = m.edge_vertices(e);
            if ((verts[0] == a && verts[1] == b) || (verts[0] == b && verts[1] == a)) {
                std::uint32_t he_lo = 2u * e;
                std::uint32_t he_hi = 2u * e + 1u;
                if (m.halfedge_face(he_lo) == merged || m.halfedge_face(he_hi) == merged) {
                    found_live_edge_to_merged = true;
                }
                break;
            }
        }
        EXPECT_TRUE(found_live_edge_to_merged)
            << "Consecutive merged-loop pair (" << a << ", " << b
            << ") has no live edge belonging to merged face " << merged;
    }
}

TEST(HalfEdgeMeshTest, DissolveEdge_TombstonesEdgeId) {
    // After dissolve, the edge slot should be tombstoned (not compacted).
    // Querying the now-dead edge returns invalid; slab size unchanged.
    bermake::HalfEdgeMesh m;
    auto v0 = m.add_vertex(0, 0, 0);
    auto v1 = m.add_vertex(1, 0, 0);
    auto v2 = m.add_vertex(1, 1, 0);
    auto v3 = m.add_vertex(2, 1, 0);
    m.add_halfedge_pair(v0, v1);
    m.add_halfedge_pair(v1, v2);
    m.add_halfedge_pair(v2, v0);
    m.add_halfedge_pair(v1, v3);
    m.add_halfedge_pair(v3, v2);
    m.add_face_from_loop({v0, v1, v2}, {(int)v0, (int)v1, (int)v2});
    m.add_face_from_loop({v1, v3, v2}, {(int)v1, (int)v3, (int)v2});

    // Find the shared edge again.
    std::uint32_t shared_edge = bermake::HalfEdgeMesh::INVALID_ID;
    for (std::uint32_t e = 0; e < m.halfedge_slab_size() / 2; ++e) {
        auto verts = m.edge_vertices(e);
        if ((verts[0] == v1 && verts[1] == v2) || (verts[0] == v2 && verts[1] == v1)) {
            shared_edge = e;
            break;
        }
    }

    auto slab_before = m.halfedge_slab_size();
    m.dissolve_edge(shared_edge);
    EXPECT_EQ(m.halfedge_slab_size(), slab_before);  // no compaction
    EXPECT_FALSE(m.edge_is_live(shared_edge));
}

TEST(HalfEdgeMeshTest, DissolveEdge_TwoQuadsIntoHexagon) {
    // Two quads sharing an edge — dissolve produces a 6-vertex face.
    //   Q1 = (v0, v1, v2, v3)  Q2 = (v1, v4, v5, v2)  shared: v1—v2
    bermake::HalfEdgeMesh m;
    auto v0 = m.add_vertex(0, 0, 0);
    auto v1 = m.add_vertex(1, 0, 0);
    auto v2 = m.add_vertex(1, 1, 0);
    auto v3 = m.add_vertex(0, 1, 0);
    auto v4 = m.add_vertex(2, 0, 0);
    auto v5 = m.add_vertex(2, 1, 0);
    m.add_halfedge_pair(v0, v1);
    m.add_halfedge_pair(v1, v2);
    m.add_halfedge_pair(v2, v3);
    m.add_halfedge_pair(v3, v0);
    m.add_halfedge_pair(v1, v4);
    m.add_halfedge_pair(v4, v5);
    m.add_halfedge_pair(v5, v2);
    m.add_face_from_loop({v0, v1, v2, v3}, {(int)v0, (int)v1, (int)v2, (int)v0, (int)v2, (int)v3});
    m.add_face_from_loop({v1, v4, v5, v2}, {(int)v1, (int)v4, (int)v5, (int)v1, (int)v5, (int)v2});

    std::uint32_t shared_edge = bermake::HalfEdgeMesh::INVALID_ID;
    for (std::uint32_t e = 0; e < m.halfedge_slab_size() / 2; ++e) {
        auto verts = m.edge_vertices(e);
        if ((verts[0] == v1 && verts[1] == v2) || (verts[0] == v2 && verts[1] == v1)) {
            shared_edge = e;
            break;
        }
    }

    auto merged = m.dissolve_edge(shared_edge);
    EXPECT_NE(merged, bermake::HalfEdgeMesh::INVALID_ID);
    auto loop = m.face_loop_vertices(merged);
    EXPECT_EQ(loop.size(), 6u);
}

TEST(HalfEdgeMeshTest, DissolveEdge_RejectsBoundaryEdge) {
    // Single triangle — all three edges are boundary (only one half-edge each).
    bermake::HalfEdgeMesh m;
    auto v0 = m.add_vertex(0, 0, 0);
    auto v1 = m.add_vertex(1, 0, 0);
    auto v2 = m.add_vertex(0, 1, 0);
    // add_halfedge_pair already returns the edge id — no /2u needed (see
    // DissolveEdge_RejectsAlreadyTombstonedEdge for the same fix).
    auto e01 = m.add_halfedge_pair(v0, v1);
    m.add_halfedge_pair(v1, v2);
    m.add_halfedge_pair(v2, v0);
    m.add_face_from_loop({v0, v1, v2}, {(int)v0, (int)v1, (int)v2});

    EXPECT_EQ(m.dissolve_edge(e01), bermake::HalfEdgeMesh::INVALID_ID);
    EXPECT_TRUE(m.edge_is_live(e01));  // unchanged
}

TEST(HalfEdgeMeshTest, DissolveEdge_RepointsStaleOutgoingHalfedge) {
    // T1 = (v0, v1, v2)   T2 = (v1, v3, v2)   shared edge: v1—v2
    // v1's cached outgoing_he is set (by T1's construction, first-touch-wins)
    // to the v1->v2 half-edge — exactly the one dissolve_edge tombstones.
    // Verify it gets repointed to a still-live outgoing half-edge instead of
    // dangling at a now-dead slab index.
    bermake::HalfEdgeMesh m;
    auto v0 = m.add_vertex(0, 0, 0);
    auto v1 = m.add_vertex(1, 0, 0);
    auto v2 = m.add_vertex(1, 1, 0);
    auto v3 = m.add_vertex(2, 1, 0);
    m.add_halfedge_pair(v0, v1);
    m.add_halfedge_pair(v1, v2);  // shared
    m.add_halfedge_pair(v2, v0);
    m.add_halfedge_pair(v1, v3);
    m.add_halfedge_pair(v3, v2);
    m.add_face_from_loop({v0, v1, v2}, {(int)v0, (int)v1, (int)v2});
    m.add_face_from_loop({v1, v3, v2}, {(int)v1, (int)v3, (int)v2});

    const std::uint32_t stale_he = m.vertex_outgoing_halfedge(v1);
    ASSERT_EQ(m.halfedge_origin(stale_he), v1);

    std::uint32_t shared_edge = bermake::HalfEdgeMesh::INVALID_ID;
    for (std::uint32_t e = 0; e < m.halfedge_slab_size() / 2; ++e) {
        auto verts = m.edge_vertices(e);
        if ((verts[0] == v1 && verts[1] == v2) || (verts[0] == v2 && verts[1] == v1)) {
            shared_edge = e;
            break;
        }
    }
    ASSERT_NE(shared_edge, bermake::HalfEdgeMesh::INVALID_ID);
    ASSERT_EQ(stale_he, 2u * shared_edge)
        << "sanity check: v1's outgoing_he must be the exact half-edge dissolve_edge tombstones "
        << "for this test to exercise the repoint path";

    ASSERT_NE(m.dissolve_edge(shared_edge), bermake::HalfEdgeMesh::INVALID_ID);

    const std::uint32_t new_he = m.vertex_outgoing_halfedge(v1);
    EXPECT_NE(new_he, stale_he) << "outgoing_he must not still point at the tombstoned half-edge";
    EXPECT_NE(new_he, bermake::HalfEdgeMesh::INVALID_ID);
    EXPECT_EQ(m.halfedge_origin(new_he), v1);
}

TEST(HalfEdgeMeshTest, DissolveEdge_RejectsAlreadyTombstonedEdge) {
    bermake::HalfEdgeMesh m;
    auto v0 = m.add_vertex(0, 0, 0);
    auto v1 = m.add_vertex(1, 0, 0);
    // add_halfedge_pair already returns the edge id (not a half-edge slab
    // index) — no /2u needed. Both happened to be edge 0 here, so the stray
    // /2u previously passed by coincidence rather than correctness.
    auto e = m.add_halfedge_pair(v0, v1);
    m.remove_edge(e);

    EXPECT_EQ(m.dissolve_edge(e), bermake::HalfEdgeMesh::INVALID_ID);
}

TEST(HalfEdgeMeshTest, DissolveEdge_RejectsMultiSharedEdges) {
    // Folded-bigon topology: a triangle and a quad that share TWO of the
    // triangle's three edges (v0-v1 and v1-v2), not just one.
    //
    //   T1 = (v0, v1, v2)       edges: v0-v1, v1-v2, v2-v0
    //   T2 = (v2, v1, v0, v3)   edges: v2-v1, v1-v0, v0-v3, v3-v2
    //
    // T2 walks v0-v1 and v1-v2 in the opposite direction from T1 (grabbing
    // their twin half-edges), so f1 and f2 share both v0-v1 and v1-v2 while
    // v2-v0 stays a boundary edge of f1 alone. Dissolving either shared edge
    // would have to splice two faces that already touch along a second,
    // unrelated edge — the guard must refuse rather than produce a
    // self-intersecting merged loop.
    bermake::HalfEdgeMesh m;
    auto v0 = m.add_vertex(0, 0, 0);
    auto v1 = m.add_vertex(1, 0, 0);
    auto v2 = m.add_vertex(1, 1, 0);
    auto v3 = m.add_vertex(0, 1, 0);
    m.add_halfedge_pair(v0, v1);  // shared edge #1
    m.add_halfedge_pair(v1, v2);  // shared edge #2
    m.add_halfedge_pair(v2, v0);
    m.add_halfedge_pair(v0, v3);
    m.add_halfedge_pair(v3, v2);

    auto f1 = m.add_face_from_loop({v0, v1, v2}, {(int)v0, (int)v1, (int)v2});
    auto f2 = m.add_face_from_loop({v2, v1, v0, v3},
                                   {(int)v2, (int)v1, (int)v0, (int)v2, (int)v0, (int)v3});

    // Find the v0—v1 edge id (one of the two multiply-shared edges).
    std::uint32_t shared_edge = bermake::HalfEdgeMesh::INVALID_ID;
    for (std::uint32_t e = 0; e < m.halfedge_slab_size() / 2; ++e) {
        auto verts = m.edge_vertices(e);
        if ((verts[0] == v0 && verts[1] == v1) || (verts[0] == v1 && verts[1] == v0)) {
            shared_edge = e;
            break;
        }
    }
    ASSERT_NE(shared_edge, bermake::HalfEdgeMesh::INVALID_ID);

    EXPECT_EQ(m.dissolve_edge(shared_edge), bermake::HalfEdgeMesh::INVALID_ID);
    // Mesh left unchanged: both faces and the edge remain live.
    EXPECT_TRUE(m.face_is_live(f1));
    EXPECT_TRUE(m.face_is_live(f2));
    EXPECT_TRUE(m.edge_is_live(shared_edge));
}

// Whole-branch review Finding 2 (IMPORTANT): two coplanar faces sharing
// EXACTLY one edge can still merge into a self-touching loop when an extra
// vertex is present in both loops away from the shared edge. The kernel's
// only pre-existing guard is "more than one shared edge", which this shape
// passes cleanly -- Task 1's loop_is_simple guard on split_face has no
// inverse-operation counterpart until this fix.
TEST(HalfEdgeMeshTest, DissolveEdge_RejectsAPinchedMergedLoop) {
    //   A = (v0, v1, v2)              edges: v0-v1, v1-v2, v2-v0
    //   B = (v1, v0, v3, v2, v4)      edges: v1-v0, v0-v3, v3-v2, v2-v4, v4-v1
    // A and B share only v0-v1, but vertex v2 appears in BOTH loops at a
    // position other than the shared edge, so the naive splice produces
    // merged loop (v0, v3, v2, v4, v1, v2) -- v2 twice, exactly the pinch
    // measured in the branch review.
    bermake::HalfEdgeMesh m;
    auto v0 = m.add_vertex(0, 0, 0);
    auto v1 = m.add_vertex(1, 0, 0);
    auto v2 = m.add_vertex(1, 1, 0);
    auto v3 = m.add_vertex(0, 2, 0);
    auto v4 = m.add_vertex(2, 2, 0);
    m.add_halfedge_pair(v0, v1);  // shared edge
    m.add_halfedge_pair(v1, v2);
    m.add_halfedge_pair(v2, v0);
    m.add_halfedge_pair(v0, v3);
    m.add_halfedge_pair(v3, v2);
    m.add_halfedge_pair(v2, v4);
    m.add_halfedge_pair(v4, v1);

    auto f1 = m.add_face_from_loop({v0, v1, v2}, {(int)v0, (int)v1, (int)v2});
    auto f2 = m.add_face_from_loop(
        {v1, v0, v3, v2, v4},
        {(int)v1, (int)v0, (int)v3, (int)v1, (int)v3, (int)v2, (int)v1, (int)v2, (int)v4});

    std::uint32_t shared_edge = m.edge_between(v0, v1);
    ASSERT_NE(shared_edge, bermake::HalfEdgeMesh::INVALID_ID);

    EXPECT_EQ(m.dissolve_edge(shared_edge), bermake::HalfEdgeMesh::INVALID_ID);
    // Mesh left unchanged: both faces and the edge remain live, and the
    // splice used to walk the candidate merged loop must have been undone.
    EXPECT_TRUE(m.face_is_live(f1));
    EXPECT_TRUE(m.face_is_live(f2));
    EXPECT_TRUE(m.edge_is_live(shared_edge));
    EXPECT_EQ(m.face_loop_vertices(f1).size(), 3u);
    EXPECT_EQ(m.face_loop_vertices(f2).size(), 5u);
}

// ---- split_edge -------------------------------------------------------------

namespace {
// Build two quads sharing edge (v1,v2): f1=[v0,v1,v2,v3], f2=[v1,v4,v5,v2].
bermake::HalfEdgeMesh make_two_quads(std::uint32_t& shared_edge_out) {
    using bermake::HalfEdgeMesh;
    HalfEdgeMesh m;
    auto v0 = m.add_vertex(0, 0, 0);
    auto v1 = m.add_vertex(1, 0, 0);
    auto v2 = m.add_vertex(1, 1, 0);
    auto v3 = m.add_vertex(0, 1, 0);
    auto v4 = m.add_vertex(2, 0, 0);
    auto v5 = m.add_vertex(2, 1, 0);
    m.add_halfedge_pair(v0, v1);
    shared_edge_out = m.add_halfedge_pair(v1, v2);
    m.add_halfedge_pair(v2, v3);
    m.add_halfedge_pair(v3, v0);
    m.add_halfedge_pair(v1, v4);
    m.add_halfedge_pair(v4, v5);
    m.add_halfedge_pair(v5, v2);
    m.add_face_from_loop({v0, v1, v2, v3}, {(int)v0, (int)v1, (int)v2, (int)v0, (int)v2, (int)v3});
    m.add_face_from_loop({v1, v4, v5, v2}, {(int)v1, (int)v4, (int)v5, (int)v1, (int)v5, (int)v2});
    return m;
}
}  // namespace

TEST(SplitEdge, InteriorEdgeInsertsVertexAndRebuildsBothFaces) {
    std::uint32_t e_shared = 0;
    auto m = make_two_quads(e_shared);

    auto res = m.split_edge(e_shared, 0.5f);
    ASSERT_TRUE(res.has_value());

    auto wp = m.vertex_position(res->vertex);
    EXPECT_FLOAT_EQ(wp[0], 1.0f);
    EXPECT_FLOAT_EQ(wp[1], 0.5f);
    EXPECT_FLOAT_EQ(wp[2], 0.0f);

    EXPECT_FALSE(m.edge_is_live(e_shared));
    EXPECT_TRUE(m.edge_is_live(res->edge_a));
    EXPECT_TRUE(m.edge_is_live(res->edge_b));

    EXPECT_NE(res->face_a, bermake::HalfEdgeMesh::INVALID_ID);
    EXPECT_NE(res->face_b, bermake::HalfEdgeMesh::INVALID_ID);
    EXPECT_TRUE(m.face_is_live(res->face_a));
    EXPECT_TRUE(m.face_is_live(res->face_b));
    EXPECT_EQ(m.face_loop_vertices(res->face_a).size(), 5u);
    EXPECT_EQ(m.face_loop_vertices(res->face_b).size(), 5u);

    std::uint32_t live = 0;
    for (auto f = m.next_live_face(0); f != bermake::HalfEdgeMesh::INVALID_ID;
         f = m.next_live_face(f + 1))
        ++live;
    EXPECT_EQ(live, 2u);
}

TEST(SplitEdge, BoundaryEdgeSplitsTheSingleIncidentFace) {
    using bermake::HalfEdgeMesh;
    HalfEdgeMesh m;
    auto v0 = m.add_vertex(0, 0, 0);
    auto v1 = m.add_vertex(2, 0, 0);
    auto v2 = m.add_vertex(0, 2, 0);
    m.add_halfedge_pair(v0, v1);
    auto e01 = (m.add_halfedge_pair(v0, v1));  // idempotent → same edge id
    m.add_halfedge_pair(v1, v2);
    m.add_halfedge_pair(v2, v0);
    m.add_face_from_loop({v0, v1, v2}, {(int)v0, (int)v1, (int)v2});

    auto res = m.split_edge(e01, 0.5f);
    ASSERT_TRUE(res.has_value());
    const bool one_face =
        (res->face_a != HalfEdgeMesh::INVALID_ID) != (res->face_b != HalfEdgeMesh::INVALID_ID);
    EXPECT_TRUE(one_face);
    const std::uint32_t live_face =
        res->face_a != HalfEdgeMesh::INVALID_ID ? res->face_a : res->face_b;
    EXPECT_EQ(m.face_loop_vertices(live_face).size(), 4u);
}

TEST(SplitEdge, RejectsParameterOutOfRange) {
    std::uint32_t e = 0;
    auto m = make_two_quads(e);
    EXPECT_FALSE(m.split_edge(e, 0.0f).has_value());
    EXPECT_FALSE(m.split_edge(e, 1.0f).has_value());
    EXPECT_FALSE(m.split_edge(e, -0.2f).has_value());
    EXPECT_FALSE(m.split_edge(e, 1.5f).has_value());
}

TEST(SplitEdge, RejectsDeadEdge) {
    std::uint32_t e = 0;
    auto m = make_two_quads(e);
    auto first = m.split_edge(e, 0.5f);
    ASSERT_TRUE(first.has_value());
    EXPECT_FALSE(m.split_edge(e, 0.5f).has_value());  // e is now dead
}

TEST(SplitEdge, PreservesManifoldTwinsOnNewEdges) {
    using bermake::HalfEdgeMesh;
    std::uint32_t e = 0;
    auto m = make_two_quads(e);
    auto res = m.split_edge(e, 0.5f);
    ASSERT_TRUE(res.has_value());
    for (std::uint32_t ne : {res->edge_a, res->edge_b}) {
        std::uint32_t ha = 2u * ne, hb = 2u * ne + 1u;
        EXPECT_EQ(m.halfedge_twin(ha), hb);
        EXPECT_EQ(m.halfedge_twin(hb), ha);
        EXPECT_NE(m.halfedge_face(ha), HalfEdgeMesh::INVALID_ID);
        EXPECT_NE(m.halfedge_face(hb), HalfEdgeMesh::INVALID_ID);
    }
}

TEST(SplitEdge, RejectsSplitLandingOnExistingVertex) {
    std::uint32_t e = 0;
    auto m = make_two_quads(e);
    // The shared edge is (1,0,0)-(1,1,0); its midpoint is (1,0.5,0). Pre-create
    // a vertex exactly there — splitting at t=0.5 would land on it, which would
    // create degenerate topology, so split_edge must reject (return nullopt) and
    // leave the mesh unchanged.
    m.add_vertex(1.0f, 0.5f, 0.0f);
    const std::size_t faces_before = 0u + [&] {
        std::uint32_t c = 0;
        for (auto f = m.next_live_face(0); f != bermake::HalfEdgeMesh::INVALID_ID;
             f = m.next_live_face(f + 1))
            ++c;
        return c;
    }();
    EXPECT_FALSE(m.split_edge(e, 0.5f).has_value());
    // Mesh unchanged: still the two original faces.
    std::uint32_t faces_after = 0;
    for (auto f = m.next_live_face(0); f != bermake::HalfEdgeMesh::INVALID_ID;
         f = m.next_live_face(f + 1))
        ++faces_after;
    EXPECT_EQ(faces_after, faces_before);
}

TEST(SplitEdge, SequentialSplitsOnSameFaceKeepLoopConsistent) {
    // #31 remainder: split_edge is only ever exercised once per test elsewhere
    // in this file. Splitting a second time, on one of the edges the FIRST
    // split just created, is the case that actually exercises loop_with_inserted
    // against a freshly rebuilt (not original) loop and a freshly rebuilt face id.
    using bermake::HalfEdgeMesh;
    HalfEdgeMesh m;
    auto v0 = m.add_vertex(0.0f, 0.0f, 0.0f);
    auto v1 = m.add_vertex(2.0f, 0.0f, 0.0f);
    auto v2 = m.add_vertex(2.0f, 2.0f, 0.0f);
    auto v3 = m.add_vertex(0.0f, 2.0f, 0.0f);
    auto e01 = m.add_halfedge_pair(v0, v1);
    m.add_halfedge_pair(v1, v2);
    m.add_halfedge_pair(v2, v3);
    m.add_halfedge_pair(v3, v0);
    auto f = m.add_face_from_loop({v0, v1, v2, v3},
                                  {(int)v0, (int)v1, (int)v2, (int)v0, (int)v2, (int)v3});

    const auto assert_live_loop = [&](std::uint32_t face_id) {
        EXPECT_TRUE(m.face_is_live(face_id));
        auto loop = m.face_loop_vertices(face_id);
        for (auto vid : loop) {
            EXPECT_TRUE(m.vertex_is_live(vid));
        }
        return loop;
    };

    auto loop0 = assert_live_loop(f);
    ASSERT_EQ(loop0.size(), 4u);

    auto first = m.split_edge(e01, 0.5f);
    ASSERT_TRUE(first.has_value());
    const std::uint32_t face_after_first =
        first->face_a != HalfEdgeMesh::INVALID_ID ? first->face_a : first->face_b;
    auto loop1 = assert_live_loop(face_after_first);
    EXPECT_EQ(loop1.size(), loop0.size() + 1u);

    auto second = m.split_edge(first->edge_a, 0.5f);
    ASSERT_TRUE(second.has_value());
    const std::uint32_t face_after_second =
        second->face_a != HalfEdgeMesh::INVALID_ID ? second->face_a : second->face_b;
    auto loop2 = assert_live_loop(face_after_second);
    EXPECT_EQ(loop2.size(), loop1.size() + 1u);
}

TEST(HalfEdgeSetVertexPosition, MovesVertexAndUpdatesIndex) {
    bermake::HalfEdgeMesh m;
    auto a = m.add_vertex(0.0f, 0.0f, 0.0f);
    m.set_vertex_position(a, 5.0f, 6.0f, 7.0f);
    auto p = m.vertex_position(a);
    EXPECT_FLOAT_EQ(p[0], 5.0f);
    EXPECT_FLOAT_EQ(p[1], 6.0f);
    EXPECT_FLOAT_EQ(p[2], 7.0f);
    // Old key freed → re-adding the old position allocates a NEW vertex.
    auto a_old = m.add_vertex(0.0f, 0.0f, 0.0f);
    EXPECT_NE(a_old, a);
    // New position is idempotent → returns the moved vertex.
    auto a_new = m.add_vertex(5.0f, 6.0f, 7.0f);
    EXPECT_EQ(a_new, a);
}

TEST(HalfEdgeSetVertexPosition, RecomputesIncidentFaceNormal) {
    bermake::HalfEdgeMesh m;
    auto a = m.add_vertex(0.0f, 0.0f, 0.0f);
    auto b = m.add_vertex(1.0f, 0.0f, 0.0f);
    auto c = m.add_vertex(0.0f, 1.0f, 0.0f);
    m.add_halfedge_pair(a, b);
    m.add_halfedge_pair(b, c);
    m.add_halfedge_pair(c, a);
    m.add_face_from_loop({a, b, c}, {static_cast<std::int32_t>(a), static_cast<std::int32_t>(b),
                                     static_cast<std::int32_t>(c)});
    auto buf0 = m.face_triangle_buffer();  // (positions, normals)
    ASSERT_GE(buf0.second.size(), 3u);
    EXPECT_NEAR(std::abs(buf0.second[2]), 1.0f, 1e-4f);  // flat in XY → |nz| ≈ 1
    // Tilt the face: lift c in +Z.
    m.set_vertex_position(c, 0.0f, 1.0f, 1.0f);
    auto buf1 = m.face_triangle_buffer();
    ASSERT_GE(buf1.second.size(), 3u);
    float nx = buf1.second[0], ny = buf1.second[1];
    EXPECT_GT(std::abs(nx) + std::abs(ny), 0.1f);  // normal now has a horizontal component
}

TEST(HalfEdgeSetVertexPosition, ThrowsOnDeadVertex) {
    bermake::HalfEdgeMesh m;
    EXPECT_THROW(m.set_vertex_position(999u, 1.0f, 2.0f, 3.0f), std::out_of_range);
}

TEST(HalfEdgeSetVertexPosition, RecomputesAllIncidentFacesOfAFanVertexAndLeavesOthers) {
    bermake::HalfEdgeMesh m;
    auto o = m.add_vertex(0.0f, 0.0f, 0.0f);
    auto p1 = m.add_vertex(1.0f, 0.0f, 0.0f);
    auto p2 = m.add_vertex(1.0f, 1.0f, 0.0f);
    auto p3 = m.add_vertex(0.0f, 1.0f, 0.0f);
    // Two coplanar (XY) triangles sharing vertex o and edge o-p2.
    m.add_halfedge_pair(o, p1);
    m.add_halfedge_pair(p1, p2);
    m.add_halfedge_pair(p2, o);
    m.add_face_from_loop({o, p1, p2}, {static_cast<std::int32_t>(o), static_cast<std::int32_t>(p1),
                                       static_cast<std::int32_t>(p2)});
    m.add_halfedge_pair(o, p2);
    m.add_halfedge_pair(p2, p3);
    m.add_halfedge_pair(p3, o);
    m.add_face_from_loop({o, p2, p3}, {static_cast<std::int32_t>(o), static_cast<std::int32_t>(p2),
                                       static_cast<std::int32_t>(p3)});
    // A separate, non-incident triangle far away (also flat in XY).
    auto q0 = m.add_vertex(5.0f, 5.0f, 0.0f);
    auto q1 = m.add_vertex(6.0f, 5.0f, 0.0f);
    auto q2 = m.add_vertex(5.0f, 6.0f, 0.0f);
    m.add_halfedge_pair(q0, q1);
    m.add_halfedge_pair(q1, q2);
    m.add_halfedge_pair(q2, q0);
    m.add_face_from_loop({q0, q1, q2},
                         {static_cast<std::int32_t>(q0), static_cast<std::int32_t>(q1),
                          static_cast<std::int32_t>(q2)});

    // Faces are emitted in id (creation) order; each is one triangle → 9 normal floats.
    auto before = m.face_triangle_buffer().second;
    ASSERT_GE(before.size(), 27u);
    for (int f = 0; f < 3; ++f) {
        EXPECT_NEAR(std::abs(before[f * 9 + 2]), 1.0f, 1e-4f);  // all flat → |nz| ≈ 1
    }

    // Move the shared fan vertex up: both incident faces must tilt, the far one must not.
    m.set_vertex_position(o, 0.0f, 0.0f, 1.0f);
    auto after = m.face_triangle_buffer().second;
    ASSERT_GE(after.size(), 27u);
    EXPECT_GT(std::abs(after[0 * 9 + 0]) + std::abs(after[0 * 9 + 1]), 0.1f);  // face 0 incident
    EXPECT_GT(std::abs(after[1 * 9 + 0]) + std::abs(after[1 * 9 + 1]), 0.1f);  // face 1 incident
    EXPECT_NEAR(std::abs(after[2 * 9 + 2]), 1.0f, 1e-4f);                      // face 2 untouched
    EXPECT_LT(std::abs(after[2 * 9 + 0]) + std::abs(after[2 * 9 + 1]), 1e-3f);
}

// ====================================================================
// #110: Face::normal by Newell's method over the whole boundary loop
// ====================================================================
//
// The kernel used to estimate Face::normal from cross(p1 - p0, p2 - p0), the
// loop's first two edges, and to substitute a hardcoded (0, 0, 1) when that
// came out near zero. A loop whose first three vertices are COLLINEAR — the
// exact shape an edge split leaves behind — hit that fallback, so a vertical
// wall with a split edge was handed the normal of a floor. That is not merely
// wrong lighting: since M7.5b the renderer reads this very field out of
// face_triangle_buffer to build each face's TEXTURE PROJECTION BASIS
// (_face_uv_geometry in python/bermake/viewport/scene_renderer.py), and
// picking, faces_are_coplanar, push/pull, offset and follow-me steer by it
// too.
//
// The Python half of the same defect was fixed first (Scene.face_normal, via
// _newell_normal in python/bermake/scene/scene.py); the two implementations are
// cross-checked against each other in tests/test_kernel_face_normal.py.
//
// Note the axis-aligned cases below cover all six orientations, not one per
// axis: three of the six were the cases the v0.7.1 earcut winding bug
// mirrored, so a test that only checked the axis and not the sign would have
// passed straight through it.

namespace {

// Build a single face from an ordered loop of world points in `m` and return
// its cached normal, read back the only way the kernel exposes it — through
// face_triangle_buffer, which is also exactly what the renderer and the M7.5b
// texture basis read. `m` must be empty.
std::array<float, 3> face_normal_of_loop(bermake::HalfEdgeMesh& m,
                                         const std::vector<std::array<float, 3>>& pts) {
    std::vector<std::uint32_t> vids;
    vids.reserve(pts.size());
    for (const auto& p : pts) vids.push_back(m.add_vertex(p[0], p[1], p[2]));
    const std::size_t n = vids.size();
    for (std::size_t i = 0; i < n; ++i) m.add_halfedge_pair(vids[i], vids[(i + 1) % n]);
    // A fan from loop[0] is enough to make the face emit corners; these tests
    // read the per-face normal, not the tessellation.
    std::vector<std::int32_t> tris;
    for (std::size_t i = 1; i + 1 < n; ++i) {
        tris.push_back(static_cast<std::int32_t>(vids[0]));
        tris.push_back(static_cast<std::int32_t>(vids[i]));
        tris.push_back(static_cast<std::int32_t>(vids[i + 1]));
    }
    m.add_face_from_loop(vids, tris);
    auto normals = m.face_triangle_buffer().second;
    EXPECT_GE(normals.size(), 3u);
    if (normals.size() < 3) return {0.0f, 0.0f, 0.0f};
    return {normals[0], normals[1], normals[2]};
}

std::array<float, 3> face_normal_of_loop(const std::vector<std::array<float, 3>>& pts) {
    bermake::HalfEdgeMesh m;
    return face_normal_of_loop(m, pts);
}

// The pre-#110 estimate, written out here rather than kept as a copy of the
// implementation, so the equivalence tests state the PROPERTY ("Newell agrees
// with a first-three cross product wherever that estimate is valid") instead
// of restating code that could drift with it. Returns false when the estimate
// was degenerate and there is nothing to compare against.
bool first_three_unit_normal(const std::vector<std::array<float, 3>>& pts,
                             std::array<double, 3>& out) {
    const auto& p0 = pts[0];
    const auto& p1 = pts[1];
    const auto& p2 = pts[2];
    const double e1[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
    const double e2[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
    const double n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2],
                         e1[0] * e2[1] - e1[1] * e2[0]};
    const double len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (len < 1e-9) return false;
    out = {n[0] / len, n[1] / len, n[2] / len};
    return true;
}

// A unit square wound so its normal points along `orientation`. Mirrors
// _square_in_plane in tests/test_face_normal_newell.py.
std::vector<std::array<float, 3>> square_in_plane(const std::string& orientation) {
    const std::array<std::array<float, 2>, 4> base = {
        {{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}}};
    std::vector<std::array<float, 3>> out;
    for (const auto& b : base) {
        const float x = b[0], y = b[1];
        if (orientation == "+Z") {
            out.push_back({x, y, 0.0f});
        } else if (orientation == "-Z") {
            out.push_back({y, x, 0.0f});
        } else if (orientation == "+X") {
            out.push_back({0.0f, x, y});
        } else if (orientation == "-X") {
            out.push_back({0.0f, y, x});
        } else if (orientation == "+Y") {
            out.push_back({y, 0.0f, x});
        } else if (orientation == "-Y") {
            out.push_back({x, 0.0f, y});
        } else {
            // Fail HERE with the bad string. Falling through would return an
            // empty loop and add_face_from_loop would throw "loop has 0
            // vertices", which names neither this helper nor the typo.
            ADD_FAILURE() << "square_in_plane: unknown orientation \"" << orientation << "\"";
            return {};
        }
    }
    return out;
}

std::array<float, 3> axis_vector(const std::string& orientation) {
    const float s = (orientation[0] == '+') ? 1.0f : -1.0f;
    if (orientation[1] == 'X') return {s, 0.0f, 0.0f};
    if (orientation[1] == 'Y') return {0.0f, s, 0.0f};
    return {0.0f, 0.0f, s};
}

// Split the loop's FIRST edge by inserting its midpoint at index 1. The first
// three vertices are then collinear — the #110 shape — while the face, its
// area and its winding are otherwise untouched.
std::vector<std::array<float, 3>> with_split_first_edge(
    const std::vector<std::array<float, 3>>& pts) {
    std::vector<std::array<float, 3>> out;
    out.push_back(pts[0]);
    out.push_back({0.5f * (pts[0][0] + pts[1][0]), 0.5f * (pts[0][1] + pts[1][1]),
                   0.5f * (pts[0][2] + pts[1][2])});
    for (std::size_t i = 1; i < pts.size(); ++i) out.push_back(pts[i]);
    return out;
}

const std::array<const char*, 6> kOrientations = {"+X", "-X", "+Y", "-Y", "+Z", "-Z"};

// A concave L wound CCW in XY, so its normal is +Z. Its reflex corner is
// (1,1), at index 3 — a convex vertex sits at index 1, which is what keeps the
// OLD estimate valid here and so makes this face usable as an equivalence case.
const std::vector<std::array<float, 3>> kLShape = {{{0.0f, 0.0f, 0.0f},
                                                    {2.0f, 0.0f, 0.0f},
                                                    {2.0f, 1.0f, 0.0f},
                                                    {1.0f, 1.0f, 0.0f},
                                                    {1.0f, 2.0f, 0.0f},
                                                    {0.0f, 2.0f, 0.0f}}};

}  // namespace

// The reopening evidence on issue #110, verbatim: a vertical wall in the XZ
// plane whose loop starts on a split edge. The old kernel answered (0, 0, 1),
// the hardcoded fallback, which happens to be a plausible FLOOR normal — so
// the wall was lit, and since M7.5b textured, as a floor. Its true normal is
// (0, -1, 0).
TEST(HalfEdgeMeshTest, FaceNormalCollinearLoopStartOnAVerticalWall) {
    const std::vector<std::array<float, 3>> wall = {{{0.0f, 0.0f, 0.0f},
                                                     {0.5f, 0.0f, 0.0f},
                                                     {1.0f, 0.0f, 0.0f},
                                                     {1.0f, 0.0f, 1.0f},
                                                     {0.0f, 0.0f, 1.0f}}};
    std::array<double, 3> old_estimate{};
    ASSERT_FALSE(first_three_unit_normal(wall, old_estimate))
        << "this loop must be one the first-three estimate could not handle, or it proves nothing";

    const auto n = face_normal_of_loop(wall);
    EXPECT_NEAR(n[0], 0.0f, 1e-6f);
    EXPECT_NEAR(n[1], -1.0f, 1e-6f);
    EXPECT_NEAR(n[2], 0.0f, 1e-6f);
}

// The same wall without the split vertex must give the same answer — the
// normal must not notice an extra collinear vertex, including its sign.
TEST(HalfEdgeMeshTest, FaceNormalOfASplitEdgeWallMatchesItsUnsplitWall) {
    const auto plain = square_in_plane("-Y");
    const auto split = with_split_first_edge(plain);
    ASSERT_EQ(split.size(), plain.size() + 1);

    const auto n_plain = face_normal_of_loop(plain);
    const auto n_split = face_normal_of_loop(split);
    for (int i = 0; i < 3; ++i) EXPECT_NEAR(n_split[i], n_plain[i], 1e-6f);
}

// All six axis-aligned orientations, plain loops. Three of the six were the
// cases the v0.7.1 earcut winding bug mirrored, so the SIGN is what is being
// pinned, not the axis.
TEST(HalfEdgeMeshTest, FaceNormalAllSixAxisAlignedOrientations) {
    for (const char* o : kOrientations) {
        SCOPED_TRACE(o);
        const auto expected = axis_vector(o);
        const auto n = face_normal_of_loop(square_in_plane(o));
        for (int i = 0; i < 3; ++i) EXPECT_NEAR(n[i], expected[i], 1e-6f);
    }
}

// The same six with a split first edge, which is the combination the old
// kernel got wrong: it answered (0, 0, 1) for all six — right by accident on
// "+Z" and wrong on the other five.
TEST(HalfEdgeMeshTest, FaceNormalAllSixAxisAlignedOrientationsWithACollinearLoopStart) {
    for (const char* o : kOrientations) {
        SCOPED_TRACE(o);
        const auto split = with_split_first_edge(square_in_plane(o));
        std::array<double, 3> old_estimate{};
        ASSERT_FALSE(first_three_unit_normal(split, old_estimate))
            << "the split loop must defeat the first-three estimate, or this proves nothing";

        const auto expected = axis_vector(o);
        const auto n = face_normal_of_loop(split);
        for (int i = 0; i < 3; ++i) EXPECT_NEAR(n[i], expected[i], 1e-6f);
    }
}

// For a TRIANGLE the Newell sum is cross(p1 - p0, p2 - p0) term for term —
// the polygon IS its first corner triangle — so the two must agree exactly,
// not merely in direction. An oblique triangle on small integer coordinates,
// so neither answer is protected by an axis-aligned zero.
TEST(HalfEdgeMeshTest, FaceNormalNewellMatchesFirstThreeOnATriangle) {
    const std::vector<std::array<float, 3>> tri = {
        {{1.0f, 2.0f, 3.0f}, {4.0f, 0.0f, -2.0f}, {-3.0f, 5.0f, 1.0f}}};
    std::array<double, 3> old_estimate{};
    ASSERT_TRUE(first_three_unit_normal(tri, old_estimate));

    const auto n = face_normal_of_loop(tri);
    for (int i = 0; i < 3; ++i) EXPECT_NEAR(n[i], static_cast<float>(old_estimate[i]), 1e-6f);

    // And reversed winding gives exactly the opposite, not some other vector.
    std::vector<std::array<float, 3>> reversed(tri.rbegin(), tri.rend());
    const auto n_rev = face_normal_of_loop(reversed);
    for (int i = 0; i < 3; ++i) EXPECT_NEAR(n_rev[i], -n[i], 1e-6f);
}

// A concave face: the area-weighted sum must not be dragged off by the reflex
// corner. Both windings, so the sign is pinned in each direction.
TEST(HalfEdgeMeshTest, FaceNormalOfAConcavePolygon) {
    const auto n = face_normal_of_loop(kLShape);
    EXPECT_NEAR(n[0], 0.0f, 1e-6f);
    EXPECT_NEAR(n[1], 0.0f, 1e-6f);
    EXPECT_NEAR(n[2], 1.0f, 1e-6f);

    std::vector<std::array<float, 3>> reversed(kLShape.rbegin(), kLShape.rend());
    const auto n_rev = face_normal_of_loop(reversed);
    EXPECT_NEAR(n_rev[2], -1.0f, 1e-6f);
}

// The improvement case, and the one that separates "stopped falling back" from
// "computes the right thing". cross(p1 - p0, p2 - p0) is twice the SIGNED area
// of triangle (p0, p1, p2) — the ear test at p1 — so a REFLEX vertex at INDEX 1
// makes the old estimate point the opposite way to the polygon's own normal.
// Rotating kLShape by two positions lands its reflex corner (1,1) there; the
// winding, and therefore the true normal (+Z), is unchanged.
TEST(HalfEdgeMeshTest, FaceNormalConcaveWithAReflexVertexAtIndexOne) {
    std::vector<std::array<float, 3>> rolled(kLShape.begin() + 2, kLShape.end());
    rolled.insert(rolled.end(), kLShape.begin(), kLShape.begin() + 2);
    ASSERT_FLOAT_EQ(rolled[1][0], 1.0f);
    ASSERT_FLOAT_EQ(rolled[1][1], 1.0f);  // the reflex vertex, now at index 1

    std::array<double, 3> old_estimate{};
    ASSERT_TRUE(first_three_unit_normal(rolled, old_estimate));
    EXPECT_NEAR(old_estimate[2], -1.0, 1e-9) << "the old estimate must be wrong here";

    const auto n = face_normal_of_loop(rolled);
    EXPECT_NEAR(n[2], 1.0f, 1e-6f);  // ...and the new one right
}

// The sign is contractual — lighting, picking, faces_are_coplanar, push/pull,
// offset, follow-me and the M7.5b texture basis all steer by this direction,
// and a flip would be a far worse bug than the one being fixed while staying
// invisible to any test that only checks the axis. Newell can only differ in
// direction from the first-three estimate where that estimate is itself wrong
// (a reflex vertex at index 1), which a CONVEX face cannot have. This measures
// that claim over random convex faces in random planes rather than asserting
// it: the dot product must be +1, never -1.
TEST(HalfEdgeMeshTest, FaceNormalNewellAgreesInSignWithFirstThreeOnRandomConvexFaces) {
    std::mt19937 rng(20250913u);  // fixed seed: a failure here must be reproducible
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    std::uniform_real_distribution<double> gap(0.15, 0.9);
    std::uniform_int_distribution<int> sides(3, 9);

    int checked = 0;
    for (int trial = 0; trial < 600; ++trial) {
        // A random plane, via a random unit normal and any basis orthogonal to it.
        double nx = unit(rng), ny = unit(rng), nz = unit(rng);
        const double nl = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (nl < 1e-3) continue;
        nx /= nl;
        ny /= nl;
        nz /= nl;
        const bool seed_z = std::abs(nz) < 0.9;
        const double sx = seed_z ? 0.0 : 1.0;
        const double sy = 0.0;
        const double sz = seed_z ? 1.0 : 0.0;
        double ux = ny * sz - nz * sy, uy = nz * sx - nx * sz, uz = nx * sy - ny * sx;
        const double ul = std::sqrt(ux * ux + uy * uy + uz * uz);
        if (ul < 1e-6) continue;
        ux /= ul;
        uy /= ul;
        uz /= ul;
        const double vx = ny * uz - nz * uy, vy = nz * ux - nx * uz, vz = nx * uy - ny * ux;

        // Points at strictly increasing angles on a circle in that plane are
        // convex by construction, and wound CCW about (nx, ny, nz).
        const int k = sides(rng);
        std::vector<std::array<float, 3>> pts;
        double theta = 0.0;
        for (int i = 0; i < k; ++i) {
            const double c = std::cos(theta), s = std::sin(theta);
            pts.push_back({static_cast<float>(ux * c + vx * s), static_cast<float>(uy * c + vy * s),
                           static_cast<float>(uz * c + vz * s)});
            theta += gap(rng);
        }
        if (theta >= 2.0 * 3.14159265358979) continue;  // wrapped past a full turn

        std::array<double, 3> old_estimate{};
        if (!first_three_unit_normal(pts, old_estimate)) continue;

        const auto got = face_normal_of_loop(pts);
        const double d =
            got[0] * old_estimate[0] + got[1] * old_estimate[1] + got[2] * old_estimate[2];
        EXPECT_GT(d, 0.999) << "trial " << trial << ": Newell normal (" << got[0] << ", " << got[1]
                            << ", " << got[2] << ") disagrees with the first-three estimate ("
                            << old_estimate[0] << ", " << old_estimate[1] << ", " << old_estimate[2]
                            << ")";
        ++checked;
    }
    EXPECT_GT(checked, 300) << "too few usable trials to call this evidence";
}

// A genuinely zero-area face has no normal, and must say so with the {0,0,0}
// sentinel rather than the old hardcoded (0, 0, 1). A plausible-looking but
// silently wrong direction is precisely what issue #110 was: it is why a
// split-edge wall was textured as a floor. The zero vector cannot be mistaken
// for an answer, it is already this file's convention for "no normal", and it
// costs nothing visible — a face of no area covers no pixels.
TEST(HalfEdgeMeshTest, FaceNormalOfAZeroAreaFaceIsTheSentinelNotAnUpwardGuess) {
    const std::vector<std::array<float, 3>> collinear = {
        {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {2.0f, 0.0f, 0.0f}, {3.0f, 0.0f, 0.0f}}};
    const auto n = face_normal_of_loop(collinear);
    EXPECT_FLOAT_EQ(n[0], 0.0f);
    EXPECT_FLOAT_EQ(n[1], 0.0f);
    EXPECT_FLOAT_EQ(n[2], 0.0f);
}

// restore_face is the undo→redo path and the second of the three call sites.
// It must reach the same answer as add_face_from_loop on the #110 shape, or a
// wall's texture basis would change under an undo.
TEST(HalfEdgeMeshTest, RestoreFaceRecomputesTheNewellNormalOfACollinearStartWall) {
    bermake::HalfEdgeMesh m;
    const auto wall = with_split_first_edge(square_in_plane("-Y"));
    const auto n_before = face_normal_of_loop(m, wall);
    ASSERT_NEAR(n_before[1], -1.0f, 1e-6f);

    const std::uint32_t f_id = 0;
    ASSERT_TRUE(m.face_is_live(f_id));
    const auto loop = m.face_loop_vertices(f_id);
    const auto tris = m.face_triangles(f_id);
    m.remove_face(f_id);
    ASSERT_FALSE(m.face_is_live(f_id));
    m.restore_face(f_id, loop, tris);

    auto normals = m.face_triangle_buffer().second;
    ASSERT_GE(normals.size(), 3u);
    EXPECT_NEAR(normals[0], 0.0f, 1e-6f);
    EXPECT_NEAR(normals[1], -1.0f, 1e-6f);
    EXPECT_NEAR(normals[2], 0.0f, 1e-6f);
}

// set_vertex_position → recompute_face_normal → compute_face_normal_geometric,
// the third call site and the one reached on every interactive vertex drag.
// Before #110 it shared only the cross-product math with the other two; now it
// shares their threshold and sentinel as well, and it must agree with them on
// the #110 shape rather than answering {0,0,0} for a face that has area.
TEST(HalfEdgeMeshTest, RecomputeFaceNormalUsesNewellOnACollinearStartWall) {
    bermake::HalfEdgeMesh m;
    const auto wall = with_split_first_edge(square_in_plane("-Y"));
    (void)face_normal_of_loop(m, wall);

    const auto loop = m.face_loop_vertices(0);
    const auto p = m.vertex_position(loop[0]);
    m.set_vertex_position(loop[0], p[0], p[1], p[2]);  // force the recompute, move nothing

    auto normals = m.face_triangle_buffer().second;
    ASSERT_GE(normals.size(), 3u);
    EXPECT_NEAR(normals[0], 0.0f, 1e-6f);
    EXPECT_NEAR(normals[1], -1.0f, 1e-6f);
    EXPECT_NEAR(normals[2], 0.0f, 1e-6f);
}

// faces_are_coplanar reads compute_face_normal_geometric, so it inherits the
// fix. Be clear about what this one is worth: it is a CHARACTERIZATION test,
// not a regression test. It does not discriminate the old implementation from
// the new — it is absent from this change's red-probe failure list, because
// under the probe the DISTANCE half of the test still separates a wall from a
// floor even when the angle half is fed a fallback normal. It is here to pin
// the behaviour of the consumer at the end of the chain, so a later change to
// the sentinel or the threshold cannot silently make a wall coplanar with a
// floor. Every other test in this block fails without the fix; this one does
// not, and that is deliberate.
TEST(HalfEdgeMeshTest, FacesAreCoplanar_CollinearStartWallIsNotCoplanarWithAFloor) {
    bermake::HalfEdgeMesh m;
    // Wall in the XZ plane (y = 0) with a split first edge.
    const auto wall = with_split_first_edge(square_in_plane("-Y"));
    (void)face_normal_of_loop(m, wall);
    const std::uint32_t f_wall = 0;

    // Floor in the XY plane (z = 0), well away from the wall's vertices.
    auto q0 = m.add_vertex(5.0f, 5.0f, 0.0f);
    auto q1 = m.add_vertex(6.0f, 5.0f, 0.0f);
    auto q2 = m.add_vertex(5.0f, 6.0f, 0.0f);
    m.add_halfedge_pair(q0, q1);
    m.add_halfedge_pair(q1, q2);
    m.add_halfedge_pair(q2, q0);
    const auto f_floor = m.add_face_from_loop(
        {q0, q1, q2}, {static_cast<std::int32_t>(q0), static_cast<std::int32_t>(q1),
                       static_cast<std::int32_t>(q2)});

    EXPECT_FALSE(m.faces_are_coplanar(f_wall, f_floor, kCos05Deg, kDistTol));
    EXPECT_FALSE(m.faces_are_coplanar(f_floor, f_wall, kCos05Deg, kDistTol));
}

// ---- split_face -------------------------------------------------------------

namespace {
// One quad f = [v0,v1,v2,v3] on z=0, plus the chord edge (v0,v2) already
// present but belonging to no face. That is the state the Line tool leaves
// behind today: the edge lands, the face stays whole.
bermake::HalfEdgeMesh make_quad_with_chord(std::uint32_t& f_out, std::uint32_t v_out[4]) {
    using bermake::HalfEdgeMesh;
    HalfEdgeMesh m;
    auto v0 = m.add_vertex(0, 0, 0);
    auto v1 = m.add_vertex(1, 0, 0);
    auto v2 = m.add_vertex(1, 1, 0);
    auto v3 = m.add_vertex(0, 1, 0);
    m.add_halfedge_pair(v0, v1);
    m.add_halfedge_pair(v1, v2);
    m.add_halfedge_pair(v2, v3);
    m.add_halfedge_pair(v3, v0);
    m.add_halfedge_pair(v0, v2);  // the chord
    f_out = m.add_face_from_loop({v0, v1, v2, v3},
                                 {(int)v0, (int)v1, (int)v2, (int)v0, (int)v2, (int)v3});
    v_out[0] = v0;
    v_out[1] = v1;
    v_out[2] = v2;
    v_out[3] = v3;
    return m;
}
}  // namespace

TEST(SplitFace, ChordSplitsOneQuadIntoTwoTriangles) {
    std::uint32_t f = 0;
    std::uint32_t v[4];
    auto m = make_quad_with_chord(f, v);

    auto out = m.split_face(f, {v[0], v[1], v[2]}, {(int)v[0], (int)v[1], (int)v[2]},
                            {v[2], v[3], v[0]}, {(int)v[2], (int)v[3], (int)v[0]});

    ASSERT_NE(out[0], bermake::HalfEdgeMesh::INVALID_ID);
    ASSERT_NE(out[1], bermake::HalfEdgeMesh::INVALID_ID);
    EXPECT_FALSE(m.face_is_live(f));
    EXPECT_TRUE(m.face_is_live(out[0]));
    EXPECT_TRUE(m.face_is_live(out[1]));
    EXPECT_EQ(m.face_loop_vertices(out[0]).size(), 3u);
    EXPECT_EQ(m.face_loop_vertices(out[1]).size(), 3u);

    std::uint32_t live = 0;
    for (auto g = m.next_live_face(0); g != bermake::HalfEdgeMesh::INVALID_ID;
         g = m.next_live_face(g + 1))
        ++live;
    EXPECT_EQ(live, 2u);
}

// Every rejection asserts the mesh is UNTOUCHED, not merely that the call
// returned INVALID_ID. A version that removed the face and then bailed would
// pass a return-value-only test while leaving a hole.
namespace {
// loop_before is the parent face's boundary loop, snapshotted BEFORE the
// (expected-to-be-rejected) split_face call. Comparing it against the loop
// read back afterward catches boundary-cycle corruption (stale `next`
// pointers, a rewritten loop cache) that a live/dead flag and a face count
// alone would miss -- a rejection path that mangled the half-edge chain but
// left the tombstone bits untouched would still pass the checks below it.
// expected_live_faces defaults to 1 (every pre-existing caller has only the
// one parent face in the mesh); the fin-theft repro below passes 2, since its
// mesh also carries an unrelated live face that must survive untouched.
void expect_rejected_and_untouched(bermake::HalfEdgeMesh& m, std::uint32_t f,
                                   const std::vector<std::uint32_t>& loop_before,
                                   const std::array<std::uint32_t, 2>& out,
                                   std::uint32_t expected_live_faces = 1) {
    EXPECT_EQ(out[0], bermake::HalfEdgeMesh::INVALID_ID);
    EXPECT_EQ(out[1], bermake::HalfEdgeMesh::INVALID_ID);
    EXPECT_TRUE(m.face_is_live(f)) << "a rejected split must not remove the face";
    EXPECT_EQ(m.face_loop_vertices(f), loop_before)
        << "a rejected split must not corrupt the face's boundary loop";
    std::uint32_t live = 0;
    for (auto g = m.next_live_face(0); g != bermake::HalfEdgeMesh::INVALID_ID;
         g = m.next_live_face(g + 1))
        ++live;
    EXPECT_EQ(live, expected_live_faces);
}
}  // namespace

TEST(SplitFace, RejectsADeadFace) {
    std::uint32_t f = 0;
    std::uint32_t v[4];
    auto m = make_quad_with_chord(f, v);
    m.remove_face(f);
    auto out = m.split_face(f, {v[0], v[1], v[2]}, {(int)v[0], (int)v[1], (int)v[2]},
                            {v[2], v[3], v[0]}, {(int)v[2], (int)v[3], (int)v[0]});
    EXPECT_EQ(out[0], bermake::HalfEdgeMesh::INVALID_ID);
    EXPECT_EQ(out[1], bermake::HalfEdgeMesh::INVALID_ID);
}

TEST(SplitFace, RejectsALoopWithFewerThanThreeVertices) {
    // Pairing this 2-vertex loop_a = {v0, v2} with loop_b equal to the
    // parent's own FULL unmodified loop isolates the size guard from the
    // partition check: loop_a's own two steps (v0->v2, v2->v0) cancel
    // against EACH OTHER (cancelled becomes 1, so `cancelled < 1` does not
    // fire), and the residual directed set is exactly loop_b's four edges,
    // which equals `want` exactly. Every other check in split_face passes;
    // only `loop_a.size() < 3` rejects this input, so the guard is not
    // redundant with the partition check. Without it, remove_face(f_id)
    // would run and then add_face_from_loop(loop_a, tris_a) would throw
    // std::invalid_argument for a loop of size 2 -- after the face is
    // already gone. That is the exact atomicity violation this whole
    // kernel op exists to prevent.
    std::uint32_t f = 0;
    std::uint32_t v[4];
    auto m = make_quad_with_chord(f, v);
    const auto loop_before = m.face_loop_vertices(f);
    auto out =
        m.split_face(f, {v[0], v[2]}, {(int)v[0], (int)v[2], (int)v[0]}, {v[0], v[1], v[2], v[3]},
                     {(int)v[0], (int)v[1], (int)v[2], (int)v[0], (int)v[2], (int)v[3]});
    expect_rejected_and_untouched(m, f, loop_before, out);
}

TEST(SplitFace, RejectsAMissingLoopEdge) {
    // v1 to v3 is not an edge of this mesh, so add_face_from_loop would throw.
    std::uint32_t f = 0;
    std::uint32_t v[4];
    auto m = make_quad_with_chord(f, v);
    const auto loop_before = m.face_loop_vertices(f);
    auto out = m.split_face(f, {v[0], v[1], v[3]}, {(int)v[0], (int)v[1], (int)v[3]},
                            {v[1], v[2], v[3]}, {(int)v[1], (int)v[2], (int)v[3]});
    expect_rejected_and_untouched(m, f, loop_before, out);
}

TEST(SplitFace, RejectsATriangleIndexNotInItsOwnLoop) {
    std::uint32_t f = 0;
    std::uint32_t v[4];
    auto m = make_quad_with_chord(f, v);
    const auto loop_before = m.face_loop_vertices(f);
    auto out = m.split_face(f, {v[0], v[1], v[2]}, {(int)v[0], (int)v[1], (int)v[3]},
                            {v[2], v[3], v[0]}, {(int)v[2], (int)v[3], (int)v[0]});
    expect_rejected_and_untouched(m, f, loop_before, out);
}

TEST(SplitFace, RejectsALoopThatRevisitsAVertex) {
    // loop_a = {v0, v1, v2, v1} revisits v1. Its directed edges are
    // (v0,v1), (v1,v2), (v2,v1), (v1,v0): the (v1,v2)/(v2,v1) pair and the
    // (v0,v1)/(v1,v0) pair both cancel entirely WITHIN loop_a, before loop_b
    // ever enters the picture, leaving zero net contribution from loop_a.
    // Pairing it with loop_b equal to the parent's own unmodified loop then
    // lets the final directed-multiset match succeed, because loop_b alone
    // already equals `want` exactly. Without a simplicity guard this
    // self-cancelling, vertex-revisiting loop sails through validation.
    std::uint32_t f = 0;
    std::uint32_t v[4];
    auto m = make_quad_with_chord(f, v);
    const auto loop_before = m.face_loop_vertices(f);
    auto out = m.split_face(f, {v[0], v[1], v[2], v[1]}, {(int)v[0], (int)v[1], (int)v[2]},
                            {v[0], v[1], v[2], v[3]},
                            {(int)v[0], (int)v[1], (int)v[2], (int)v[0], (int)v[2], (int)v[3]});
    expect_rejected_and_untouched(m, f, loop_before, out);
}

TEST(SplitFace, RejectsTwoLoopsThatDoNotTouch) {
    // Both loops are the parent's own loop walked the same direction, so no
    // directed edge has a reverse anywhere and `cancelled` stays 0: this is
    // what is actually rejected on today. But this input does not isolate
    // `cancelled < 1` from the later count-mismatch check either: delete
    // that guard and every parent edge is still contributed twice (once by
    // each identical loop) against `want`'s count of 1, so the mismatch
    // check catches it next. This test documents that overlap rather than
    // claiming to isolate one cause.
    std::uint32_t f = 0;
    std::uint32_t v[4];
    auto m = make_quad_with_chord(f, v);
    const auto loop_before = m.face_loop_vertices(f);
    auto out = m.split_face(f, {v[0], v[1], v[2], v[3]},
                            {(int)v[0], (int)v[1], (int)v[2], (int)v[0], (int)v[2], (int)v[3]},
                            {v[0], v[1], v[2], v[3]},
                            {(int)v[0], (int)v[1], (int)v[2], (int)v[0], (int)v[2], (int)v[3]});
    expect_rejected_and_untouched(m, f, loop_before, out);
}

TEST(SplitFace, RejectsAChildWoundBackwards) {
    // loop_b reversed. Correct loop_b walks the chord v0 -> v2, opposite to
    // loop_a's v2 -> v0, and that pair cancels. Reversed, loop_b walks
    // v2 -> v0 as well, so nothing cancels and the check fails on
    // `cancelled < 1`. That is the mechanism, and without it the two
    // children's normals would face opposite ways.
    std::uint32_t f = 0;
    std::uint32_t v[4];
    auto m = make_quad_with_chord(f, v);
    const auto loop_before = m.face_loop_vertices(f);
    auto out = m.split_face(f, {v[0], v[1], v[2]}, {(int)v[0], (int)v[1], (int)v[2]},
                            {v[0], v[3], v[2]}, {(int)v[0], (int)v[3], (int)v[2]});
    expect_rejected_and_untouched(m, f, loop_before, out);
}

TEST(SplitFace, BothChildrenKeepTheParentsNormal) {
    // There is no public per-face normal accessor on HalfEdgeMesh; the cached
    // normal reaches the outside only through face_triangle_buffer, which is
    // how test_halfedge.cpp:447 already reads one. Every corner of every
    // triangle of both children must carry the parent's +Z.
    std::uint32_t f = 0;
    std::uint32_t v[4];
    auto m = make_quad_with_chord(f, v);
    auto out = m.split_face(f, {v[0], v[1], v[2]}, {(int)v[0], (int)v[1], (int)v[2]},
                            {v[2], v[3], v[0]}, {(int)v[2], (int)v[3], (int)v[0]});
    ASSERT_NE(out[0], bermake::HalfEdgeMesh::INVALID_ID);

    auto [positions, normals] = m.face_triangle_buffer();
    ASSERT_EQ(normals.size(), 18u) << "two triangles, three corners each, three floats each";
    for (std::size_t i = 0; i + 2 < normals.size(); i += 3) {
        EXPECT_FLOAT_EQ(normals[i + 0], 0.0f);
        EXPECT_FLOAT_EQ(normals[i + 1], 0.0f);
        EXPECT_FLOAT_EQ(normals[i + 2], 1.0f)
            << "a child wound backwards renders as an unpainted back face and pushes "
               "the wrong way";
    }
}

TEST(SplitFace, SplittingTheSameFaceTwiceInSequence) {
    // The second split targets one of the first split's children, which is the
    // case an id-reuse or stale-loop bug shows up in. This also covers #31's
    // outstanding sequential-splits GoogleTest item.
    std::uint32_t f = 0;
    std::uint32_t v[4];
    auto m = make_quad_with_chord(f, v);
    auto first = m.split_face(f, {v[0], v[1], v[2]}, {(int)v[0], (int)v[1], (int)v[2]},
                              {v[2], v[3], v[0]}, {(int)v[2], (int)v[3], (int)v[0]});
    ASSERT_NE(first[0], bermake::HalfEdgeMesh::INVALID_ID);

    // Split child [v0,v1,v2] by inserting a vertex on (v0,v1) and chording to v2.
    auto se = m.split_edge(m.edge_between(v[0], v[1]), 0.5f);
    ASSERT_TRUE(se.has_value());
    const std::uint32_t w = se->vertex;
    const std::uint32_t child =
        (se->face_a != bermake::HalfEdgeMesh::INVALID_ID) ? se->face_a : se->face_b;
    ASSERT_TRUE(m.face_is_live(child));
    // face_loop_vertices(child) is [w, v1, v2, v0] or a rotation of it; read it
    // rather than assuming the rotation, which earcut's version can change.
    m.add_halfedge_pair(w, v[2]);
    auto second = m.split_face(child, {w, v[1], v[2]}, {(int)w, (int)v[1], (int)v[2]},
                               {v[2], v[0], w}, {(int)v[2], (int)v[0], (int)w});
    ASSERT_NE(second[0], bermake::HalfEdgeMesh::INVALID_ID);

    std::uint32_t live = 0;
    for (auto g = m.next_live_face(0); g != bermake::HalfEdgeMesh::INVALID_ID;
         g = m.next_live_face(g + 1))
        ++live;
    EXPECT_EQ(live, 3u);
}

TEST(SplitFace, DissolvingTheChordReturnsTheParentsLoop) {
    std::uint32_t f = 0;
    std::uint32_t v[4];
    auto m = make_quad_with_chord(f, v);
    const auto parent_loop = m.face_loop_vertices(f);
    auto out = m.split_face(f, {v[0], v[1], v[2]}, {(int)v[0], (int)v[1], (int)v[2]},
                            {v[2], v[3], v[0]}, {(int)v[2], (int)v[3], (int)v[0]});
    ASSERT_NE(out[0], bermake::HalfEdgeMesh::INVALID_ID);

    const std::uint32_t merged = m.dissolve_edge(m.edge_between(v[0], v[2]));
    ASSERT_NE(merged, bermake::HalfEdgeMesh::INVALID_ID);
    EXPECT_EQ(m.face_loop_vertices(merged).size(), parent_loop.size());
}

// Whole-branch review Finding 1 (CRITICAL): loop_is_buildable checked that
// every chain edge exists and is live, but never that the directed
// half-edge each child loop would CLAIM is unowned. add_face_from_loop does
// not check either -- it unconditionally overwrites halfedges_[he].face --
// so a chain edge that is already a live edge of some unrelated face got
// silently stolen out from under it.
TEST(SplitFace, RejectsStealingAHalfEdgeFromAnUnrelatedLiveFace) {
    // Floor quad [v0,v1,v2,v3] on z=0. A 'fin' triangle [v0,vt,v2] stands on
    // the floor's diagonal (v0,v2), which is NOT one of the floor's four
    // boundary edges: it is a free chord until the fin's own
    // add_face_from_loop call auto-creates and claims it. This is exactly
    // the state two ordinary Line-tool gestures leave behind: close a loop
    // from v0 up to a point and back to v2 (creating the fin, which
    // deliberately does not split anything), then draw v0 to v2 and press
    // Enter -- chain_cuts_face returns the floor and split_face runs.
    bermake::HalfEdgeMesh m;
    auto v0 = m.add_vertex(0, 0, 0);
    auto v1 = m.add_vertex(1, 0, 0);
    auto v2 = m.add_vertex(1, 1, 0);
    auto v3 = m.add_vertex(0, 1, 0);
    auto vt = m.add_vertex(2, 2, 1);
    m.add_halfedge_pair(v0, v1);
    m.add_halfedge_pair(v1, v2);
    m.add_halfedge_pair(v2, v3);
    m.add_halfedge_pair(v3, v0);
    const std::uint32_t floor = m.add_face_from_loop(
        {v0, v1, v2, v3}, {(int)v0, (int)v1, (int)v2, (int)v0, (int)v2, (int)v3});

    // The fin's loop is [v0, vt, v2]; its own add_face_from_loop call
    // auto-creates (v0,vt), (vt,v2) and (v2,v0) -- the last of which is the
    // floor's diagonal, previously a non-edge, now a live edge owned by the
    // fin alone.
    m.add_halfedge_pair(v0, vt);
    m.add_halfedge_pair(vt, v2);
    m.add_halfedge_pair(v2, v0);
    const std::uint32_t fin = m.add_face_from_loop({v0, vt, v2}, {(int)v0, (int)vt, (int)v2});

    const std::uint32_t diagonal = m.edge_between(v0, v2);
    ASSERT_NE(diagonal, bermake::HalfEdgeMesh::INVALID_ID);
    ASSERT_TRUE(m.face_is_live(fin));
    const auto floor_loop_before = m.face_loop_vertices(floor);

    auto out = m.split_face(floor, {v0, v1, v2}, {(int)v0, (int)v1, (int)v2}, {v2, v3, v0},
                            {(int)v2, (int)v3, (int)v0});

    expect_rejected_and_untouched(m, floor, floor_loop_before, out, /*expected_live_faces=*/2u);
    EXPECT_TRUE(m.face_is_live(fin))
        << "the fin must not be silently detached from its own boundary edge";
}
