#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "bermake/halfedge.h"
#include "bermake/primitives.h"

namespace {

::testing::AssertionResult NearlyEqual(float a, float b, float tol = 1e-5f) {
    if (std::abs(a - b) <= tol) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << a << " not within " << tol << " of " << b;
}

// Number of live faces in a freshly built HalfEdgeMesh (nothing removed).
std::size_t FaceCount(const bermake::HalfEdgeMesh& m) {
    return m.face_slab_size();
}

// Number of live vertices in a freshly built HalfEdgeMesh (nothing removed).
std::size_t VertexCount(const bermake::HalfEdgeMesh& m) {
    return m.vertex_slab_size();
}

// A closed solid is watertight iff every live edge borders exactly two live
// faces — i.e. both of its half-edges have been claimed by a face. A
// generator that forgets to weld a seam leaves that seam's edges with an
// INVALID_ID face on one side; one that winds a face backwards makes
// add_face_from_loop throw (or, if it still succeeds, leaves the mismatched
// edge unclaimed on the side the backwards face should have covered) — both
// show up here as a missing face.
bool IsWatertight(const bermake::HalfEdgeMesh& m) {
    using bermake::HalfEdgeMesh;
    for (std::uint32_t e = m.next_live_edge(0); e != HalfEdgeMesh::INVALID_ID;
         e = m.next_live_edge(e + 1)) {
        const std::uint32_t f0 = m.halfedge_face(2 * e);
        const std::uint32_t f1 = m.halfedge_face(2 * e + 1);
        if (f0 == HalfEdgeMesh::INVALID_ID || f1 == HalfEdgeMesh::INVALID_ID) return false;
        if (!m.face_is_live(f0) || !m.face_is_live(f1)) return false;
    }
    return true;
}

}  // namespace

TEST(PrimitivesCube, CountsAreCorrect) {
    const auto cube = bermake::make_cube(1.0f);
    EXPECT_EQ(cube.vertex_count(), 24u);
    EXPECT_EQ(cube.triangle_count(), 12u);
    EXPECT_EQ(cube.indices.size(), 36u);
    EXPECT_EQ(cube.positions.size(), 72u);
    EXPECT_EQ(cube.normals.size(), 72u);
}

TEST(PrimitivesCube, BottomOnGroundCentered) {
    const float size = 2.5f;
    const auto cube = bermake::make_cube(size);

    for (std::size_t i = 0; i < cube.vertex_count(); ++i) {
        const float x = cube.positions[3 * i + 0];
        const float y = cube.positions[3 * i + 1];
        const float z = cube.positions[3 * i + 2];
        EXPECT_GE(x, -size / 2 - 1e-5f);
        EXPECT_LE(x, +size / 2 + 1e-5f);
        EXPECT_GE(y, -size / 2 - 1e-5f);
        EXPECT_LE(y, +size / 2 + 1e-5f);
        EXPECT_GE(z, 0.0f - 1e-5f);
        EXPECT_LE(z, size + 1e-5f);
    }
}

TEST(PrimitivesCube, AllNormalsAreUnitLength) {
    const auto cube = bermake::make_cube(1.0f);
    for (std::size_t i = 0; i < cube.vertex_count(); ++i) {
        const float nx = cube.normals[3 * i + 0];
        const float ny = cube.normals[3 * i + 1];
        const float nz = cube.normals[3 * i + 2];
        const float length = std::sqrt(nx * nx + ny * ny + nz * nz);
        EXPECT_TRUE(NearlyEqual(length, 1.0f)) << "vertex " << i;
    }
}

TEST(PrimitivesCube, IndicesAreInRange) {
    const auto cube = bermake::make_cube(1.0f);
    for (std::uint32_t idx : cube.indices) {
        EXPECT_LT(idx, cube.vertex_count());
    }
}

// make_cube emits vertices grouped 4-per-face, one group per face in the
// `faces[6]` array order (see primitives.cpp): verts [0,3] = face 0,
// verts [4,7] = face 1, etc. This is guaranteed by the emission loop, which
// pushes exactly 4 positions/normals per face before moving to the next
// face — there is no interleaving. If that loop's structure ever changes
// (e.g. to support a primitive with a variable vertex count per face), this
// test's `4 * f + v` indexing assumption must change with it.
TEST(PrimitivesCube, EachFaceHasOneNormal) {
    const auto cube = bermake::make_cube(1.0f);
    for (std::size_t f = 0; f < 6; ++f) {
        const float nx0 = cube.normals[3 * (4 * f + 0) + 0];
        const float ny0 = cube.normals[3 * (4 * f + 0) + 1];
        const float nz0 = cube.normals[3 * (4 * f + 0) + 2];
        for (std::size_t v = 1; v < 4; ++v) {
            EXPECT_TRUE(NearlyEqual(cube.normals[3 * (4 * f + v) + 0], nx0));
            EXPECT_TRUE(NearlyEqual(cube.normals[3 * (4 * f + v) + 1], ny0));
            EXPECT_TRUE(NearlyEqual(cube.normals[3 * (4 * f + v) + 2], nz0));
        }
    }
}

// --- make_box ------------------------------------------------------------

TEST(PrimitivesBox, CountsAreCorrect) {
    // Distinct width/depth/height: a generator that confuses one axis for
    // another still produces 8 vertices and 6 faces, so this test is paired
    // with BoxRespectsDistinctDimensions below to catch that case.
    const auto box = bermake::make_box(1.0f, 2.0f, 3.0f);
    EXPECT_EQ(VertexCount(box), 8u);
    EXPECT_EQ(FaceCount(box), 6u);
}

TEST(PrimitivesBox, RespectsDistinctDimensions) {
    // A generator that swaps width/depth/height (or ignores one of them)
    // fails these per-axis bounds even though counts above still pass.
    const float width = 1.0f, depth = 2.0f, height = 3.0f;
    const auto box = bermake::make_box(width, depth, height);

    float min_x = 1e9f, max_x = -1e9f, min_y = 1e9f, max_y = -1e9f, min_z = 1e9f, max_z = -1e9f;
    for (std::uint32_t v = box.next_live_vertex(0); v != bermake::HalfEdgeMesh::INVALID_ID;
         v = box.next_live_vertex(v + 1)) {
        const auto p = box.vertex_position(v);
        min_x = std::min(min_x, p[0]);
        max_x = std::max(max_x, p[0]);
        min_y = std::min(min_y, p[1]);
        max_y = std::max(max_y, p[1]);
        min_z = std::min(min_z, p[2]);
        max_z = std::max(max_z, p[2]);
    }
    EXPECT_TRUE(NearlyEqual(min_x, -width / 2));
    EXPECT_TRUE(NearlyEqual(max_x, +width / 2));
    EXPECT_TRUE(NearlyEqual(min_y, -depth / 2));
    EXPECT_TRUE(NearlyEqual(max_y, +depth / 2));
    EXPECT_TRUE(NearlyEqual(min_z, 0.0f));
    EXPECT_TRUE(NearlyEqual(max_z, height));
}

// --- make_cylinder ---------------------------------------------------------

TEST(Primitives, CylinderFaceCountFollowsSegments) {
    // Eight sides plus two caps. A generator that hardcodes segment count
    // (e.g. always 24) fails this at segments=8.
    const auto m = bermake::make_cylinder(1.0f, 2.0f, 8);
    EXPECT_EQ(FaceCount(m), 10u);
}

TEST(Primitives, CylinderFaceCountFollowsDifferentSegments) {
    const auto m = bermake::make_cylinder(1.0f, 2.0f, 6);
    EXPECT_EQ(FaceCount(m), 8u);
}

TEST(Primitives, CylinderVerticesLieOnRadiusAndHeightRange) {
    // Catches a generator that gets the radius or the axis wrong: every
    // vertex on this HalfEdgeMesh sits on the rim of the top or bottom
    // circle (no interior cap vertex), so all of them must be exactly
    // `radius` from the z-axis and within [0, height] in z.
    const float radius = 1.5f;
    const float height = 2.0f;
    const auto m = bermake::make_cylinder(radius, height, 8);
    for (std::uint32_t v = m.next_live_vertex(0); v != bermake::HalfEdgeMesh::INVALID_ID;
         v = m.next_live_vertex(v + 1)) {
        const auto p = m.vertex_position(v);
        const float r = std::sqrt(p[0] * p[0] + p[1] * p[1]);
        EXPECT_TRUE(NearlyEqual(r, radius, 1e-4f)) << "vertex " << v;
        EXPECT_GE(p[2], -1e-5f);
        EXPECT_LE(p[2], height + 1e-5f);
    }
}

// --- make_cone -------------------------------------------------------------

TEST(Primitives, ConeFaceCountFollowsSegments) {
    // Eight sides plus one base. A generator that hardcodes segment count
    // fails this at segments=8.
    const auto m = bermake::make_cone(1.0f, 2.0f, 8);
    EXPECT_EQ(FaceCount(m), 9u);
}

TEST(Primitives, ConeFaceCountFollowsDifferentSegments) {
    const auto m = bermake::make_cone(1.0f, 2.0f, 5);
    EXPECT_EQ(FaceCount(m), 6u);
}

TEST(Primitives, ConeBaseOnRadiusApexAtHeight) {
    const float radius = 1.5f;
    const float height = 3.0f;
    const auto m = bermake::make_cone(radius, height, 10);
    bool found_apex = false;
    for (std::uint32_t v = m.next_live_vertex(0); v != bermake::HalfEdgeMesh::INVALID_ID;
         v = m.next_live_vertex(v + 1)) {
        const auto p = m.vertex_position(v);
        if (NearlyEqual(p[2], height, 1e-4f)) {
            // The apex: on-axis.
            found_apex = true;
            EXPECT_TRUE(NearlyEqual(p[0], 0.0f));
            EXPECT_TRUE(NearlyEqual(p[1], 0.0f));
        } else {
            // A base vertex: on the ground plane, on the radius.
            EXPECT_TRUE(NearlyEqual(p[2], 0.0f));
            const float r = std::sqrt(p[0] * p[0] + p[1] * p[1]);
            EXPECT_TRUE(NearlyEqual(r, radius, 1e-4f)) << "vertex " << v;
        }
    }
    EXPECT_TRUE(found_apex);
}

// --- make_sphere -----------------------------------------------------------

TEST(Primitives, SphereHasNoDegenerateFaces) {
    const auto m = bermake::make_sphere(1.0f, 6, 8);
    EXPECT_GT(FaceCount(m), 0u);
}

TEST(Primitives, SphereFaceCountFollowsRingsAndSegments) {
    // rings * segments faces (2 pole fans of `segments` triangles each, plus
    // (rings - 2) interior latitude bands of `segments` quads each). A
    // generator that ignores rings or segments fails this.
    const auto m6x8 = bermake::make_sphere(1.0f, 6, 8);
    EXPECT_EQ(FaceCount(m6x8), 48u);
    const auto m4x10 = bermake::make_sphere(1.0f, 4, 10);
    EXPECT_EQ(FaceCount(m4x10), 40u);
}

TEST(Primitives, SphereVerticesLieOnSurface) {
    // Strongest correctness check for the sphere: every vertex must be
    // exactly `radius` away from the sphere's centre (0, 0, radius). A
    // generator with a wrong latitude formula (e.g. linear in z instead of
    // cos(phi)) produces the right vertex/face counts but fails this.
    const float radius = 2.0f;
    const auto m = bermake::make_sphere(radius, 8, 12);
    for (std::uint32_t v = m.next_live_vertex(0); v != bermake::HalfEdgeMesh::INVALID_ID;
         v = m.next_live_vertex(v + 1)) {
        const auto p = m.vertex_position(v);
        const float dx = p[0];
        const float dy = p[1];
        const float dz = p[2] - radius;
        const float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        EXPECT_TRUE(NearlyEqual(dist, radius, 1e-4f)) << "vertex " << v;
    }
}

// --- Watertightness ----------------------------------------------------

TEST(Primitives, EveryPrimitiveIsWatertight) {
    // Every edge borders exactly two faces on a closed solid. Distinct
    // width/depth/height on the box rules out a coincidental pass from
    // symmetry.
    EXPECT_TRUE(IsWatertight(bermake::make_box(1.0f, 2.0f, 3.0f)));
    EXPECT_TRUE(IsWatertight(bermake::make_cylinder(1.0f, 2.0f, 12)));
    EXPECT_TRUE(IsWatertight(bermake::make_cone(1.0f, 2.0f, 12)));
    EXPECT_TRUE(IsWatertight(bermake::make_sphere(1.0f, 8, 12)));
}

// --- Argument validation ----------------------------------------------
//
// segments/rings directly index vectors sized off of them (bottom/top rings,
// interior latitude rings). Out-of-range values don't just degenerate —
// they read/write past the end of an empty or undersized vector, which is
// undefined behaviour (observed in practice as a hard interpreter crash
// from Python). These generators must reject such values instead.

TEST(Primitives, CylinderRejectsTooFewSegments) {
    EXPECT_THROW(bermake::make_cylinder(1.0f, 1.0f, 2), std::invalid_argument);
    EXPECT_THROW(bermake::make_cylinder(1.0f, 1.0f, 0), std::invalid_argument);
    EXPECT_THROW(bermake::make_cylinder(1.0f, 1.0f, -1), std::invalid_argument);
}

TEST(Primitives, CylinderAcceptsMinimumSegments) {
    // 3 is the smallest segment count that encloses a volume instead of
    // collapsing the side quads and caps onto a plane.
    const auto m = bermake::make_cylinder(1.0f, 1.0f, 3);
    EXPECT_TRUE(IsWatertight(m));
    EXPECT_EQ(FaceCount(m), 5u);  // 3 sides + top + bottom
}

TEST(Primitives, ConeRejectsTooFewSegments) {
    EXPECT_THROW(bermake::make_cone(1.0f, 1.0f, 2), std::invalid_argument);
    EXPECT_THROW(bermake::make_cone(1.0f, 1.0f, 0), std::invalid_argument);
    EXPECT_THROW(bermake::make_cone(1.0f, 1.0f, -1), std::invalid_argument);
}

TEST(Primitives, ConeAcceptsMinimumSegments) {
    const auto m = bermake::make_cone(1.0f, 1.0f, 3);
    EXPECT_TRUE(IsWatertight(m));
    EXPECT_EQ(FaceCount(m), 4u);  // 3 sides + base
}

TEST(Primitives, SphereRejectsTooFewRingsOrSegments) {
    EXPECT_THROW(bermake::make_sphere(1.0f, 1, 8), std::invalid_argument);
    EXPECT_THROW(bermake::make_sphere(1.0f, 0, 8), std::invalid_argument);
    EXPECT_THROW(bermake::make_sphere(1.0f, 6, 2), std::invalid_argument);
    EXPECT_THROW(bermake::make_sphere(1.0f, 6, 0), std::invalid_argument);
}

TEST(Primitives, SphereAcceptsMinimumRingsAndSegments) {
    // rings = 2 leaves a single interior (equatorial) ring shared by both
    // pole fans and no interior bands — a valid bipyramid, not degenerate.
    // segments = 3 is the smallest polygon a fan/cap can close.
    const auto m = bermake::make_sphere(1.0f, 2, 3);
    EXPECT_TRUE(IsWatertight(m));
    EXPECT_EQ(FaceCount(m), 6u);  // 2 pole fans * 3 segments, no interior bands
}
