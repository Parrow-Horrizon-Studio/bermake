#include <gtest/gtest.h>

#include "bermake/halfedge.h"
#include "bermake/ray_intersect.h"

using bermake::HalfEdgeMesh;
using bermake::ray_intersect_mesh;
using bermake::RayMeshHit;

namespace {

// Make a unit-square rectangle face on the XY plane at z=0.
// Returns (mesh, face_id).
std::pair<HalfEdgeMesh, std::uint32_t> make_ground_rect() {
    HalfEdgeMesh m;
    const auto v0 = m.add_vertex(0.0f, 0.0f, 0.0f);
    const auto v1 = m.add_vertex(1.0f, 0.0f, 0.0f);
    const auto v2 = m.add_vertex(1.0f, 1.0f, 0.0f);
    const auto v3 = m.add_vertex(0.0f, 1.0f, 0.0f);
    m.add_halfedge_pair(v0, v1);
    m.add_halfedge_pair(v1, v2);
    m.add_halfedge_pair(v2, v3);
    m.add_halfedge_pair(v3, v0);
    std::vector<std::int32_t> tris = {
        static_cast<int>(v0), static_cast<int>(v1), static_cast<int>(v2),
        static_cast<int>(v0), static_cast<int>(v2), static_cast<int>(v3),
    };
    const auto f = m.add_face_from_loop({v0, v1, v2, v3}, tris);
    return {std::move(m), f};
}

}  // namespace

TEST(RayIntersectMesh, EmptyMeshReturnsNullopt) {
    HalfEdgeMesh m;
    auto hit = ray_intersect_mesh(m, {0.0f, 0.0f, 5.0f}, {0.0f, 0.0f, -1.0f});
    EXPECT_FALSE(hit.has_value());
}

TEST(RayIntersectMesh, RayFromAboveHitsGroundRectangle) {
    auto [m, f] = make_ground_rect();
    auto hit = ray_intersect_mesh(m, {0.5f, 0.5f, 5.0f}, {0.0f, 0.0f, -1.0f});
    ASSERT_TRUE(hit.has_value());
    EXPECT_EQ(hit->face_id, f);
    EXPECT_NEAR(hit->t, 5.0f, 1e-5f);
    EXPECT_NEAR(hit->point[0], 0.5f, 1e-5f);
    EXPECT_NEAR(hit->point[1], 0.5f, 1e-5f);
    EXPECT_NEAR(hit->point[2], 0.0f, 1e-5f);
}

TEST(RayIntersectMesh, RayMissesRectangleSideways) {
    auto [m, f] = make_ground_rect();
    auto hit = ray_intersect_mesh(m, {5.0f, 5.0f, 5.0f}, {0.0f, 0.0f, -1.0f});
    EXPECT_FALSE(hit.has_value());
}

TEST(RayIntersectMesh, RayBehindOriginDoesNotHit) {
    auto [m, f] = make_ground_rect();
    // Origin BELOW the rectangle, looking DOWN — ray never crosses z=0 in t>0.
    auto hit = ray_intersect_mesh(m, {0.5f, 0.5f, -1.0f}, {0.0f, 0.0f, -1.0f});
    EXPECT_FALSE(hit.has_value());
}

TEST(RayIntersectMesh, TwoSidedHitFromBelow) {
    auto [m, f] = make_ground_rect();
    // Origin below, looking up — should still pick the face (two-sided).
    auto hit = ray_intersect_mesh(m, {0.5f, 0.5f, -3.0f}, {0.0f, 0.0f, 1.0f});
    ASSERT_TRUE(hit.has_value());
    EXPECT_EQ(hit->face_id, f);
    EXPECT_NEAR(hit->t, 3.0f, 1e-5f);
}

TEST(RayIntersectMesh, ClosestFaceWinsWhenTwoFacesAlongRay) {
    HalfEdgeMesh m;
    // Lower face at z=0
    {
        const auto v0 = m.add_vertex(0.0f, 0.0f, 0.0f);
        const auto v1 = m.add_vertex(1.0f, 0.0f, 0.0f);
        const auto v2 = m.add_vertex(1.0f, 1.0f, 0.0f);
        const auto v3 = m.add_vertex(0.0f, 1.0f, 0.0f);
        m.add_halfedge_pair(v0, v1);
        m.add_halfedge_pair(v1, v2);
        m.add_halfedge_pair(v2, v3);
        m.add_halfedge_pair(v3, v0);
        m.add_face_from_loop({v0, v1, v2, v3},
                             {static_cast<int>(v0), static_cast<int>(v1), static_cast<int>(v2),
                              static_cast<int>(v0), static_cast<int>(v2), static_cast<int>(v3)});
    }
    // Upper face at z=2 (will be hit FIRST from a ray coming from above)
    std::uint32_t upper_face;
    {
        const auto u0 = m.add_vertex(0.0f, 0.0f, 2.0f);
        const auto u1 = m.add_vertex(1.0f, 0.0f, 2.0f);
        const auto u2 = m.add_vertex(1.0f, 1.0f, 2.0f);
        const auto u3 = m.add_vertex(0.0f, 1.0f, 2.0f);
        m.add_halfedge_pair(u0, u1);
        m.add_halfedge_pair(u1, u2);
        m.add_halfedge_pair(u2, u3);
        m.add_halfedge_pair(u3, u0);
        upper_face = m.add_face_from_loop(
            {u0, u1, u2, u3}, {static_cast<int>(u0), static_cast<int>(u1), static_cast<int>(u2),
                               static_cast<int>(u0), static_cast<int>(u2), static_cast<int>(u3)});
    }

    auto hit = ray_intersect_mesh(m, {0.5f, 0.5f, 5.0f}, {0.0f, 0.0f, -1.0f});
    ASSERT_TRUE(hit.has_value());
    EXPECT_EQ(hit->face_id, upper_face);
    EXPECT_NEAR(hit->t, 3.0f, 1e-5f);  // 5 - 2 = 3
}

TEST(RayIntersectMesh, TombstonedFaceIsSkipped) {
    auto [m, f] = make_ground_rect();
    m.remove_face(f);
    auto hit = ray_intersect_mesh(m, {0.5f, 0.5f, 5.0f}, {0.0f, 0.0f, -1.0f});
    EXPECT_FALSE(hit.has_value());
}

TEST(RayIntersectMesh, NormalizedAndUnnormalizedDirectionsAgreeOnFaceId) {
    auto [m, f] = make_ground_rect();
    auto a = ray_intersect_mesh(m, {0.5f, 0.5f, 5.0f}, {0.0f, 0.0f, -1.0f});
    auto b = ray_intersect_mesh(m, {0.5f, 0.5f, 5.0f},
                                {0.0f, 0.0f, -7.5f});  // same direction, different magnitude
    ASSERT_TRUE(a.has_value());
    ASSERT_TRUE(b.has_value());
    EXPECT_EQ(a->face_id, b->face_id);
    EXPECT_EQ(a->face_id, f);
    // The t parameters differ because direction magnitudes differ.
    // direction_b = 7.5 * direction_a  =>  t_b = t_a / 7.5
    EXPECT_NEAR(a->t, b->t * 7.5f, 1e-4f);
}
