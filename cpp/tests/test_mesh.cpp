#include <gtest/gtest.h>

#include "bermake/mesh.h"

TEST(MeshTest, DefaultConstructedIsEmpty) {
    bermake::Mesh m;
    EXPECT_EQ(m.vertex_count(), 0u);
    EXPECT_EQ(m.triangle_count(), 0u);
    EXPECT_TRUE(m.positions.empty());
    EXPECT_TRUE(m.normals.empty());
    EXPECT_TRUE(m.indices.empty());
}

TEST(MeshTest, CountsMatchArrayLengths) {
    bermake::Mesh m;
    // 3 vertices, 1 triangle
    m.positions = {0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f};
    m.normals = {0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f};
    m.indices = {0u, 1u, 2u};

    EXPECT_EQ(m.vertex_count(), 3u);
    EXPECT_EQ(m.triangle_count(), 1u);
}

TEST(MeshTest, IsValidOnDefaultConstructed) {
    bermake::Mesh m;
    EXPECT_TRUE(m.is_valid());
}

TEST(MeshTest, IsValidOnWellFormedMesh) {
    bermake::Mesh m;
    m.positions = {0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f};
    m.normals = {0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f};
    m.indices = {0u, 1u, 2u};
    EXPECT_TRUE(m.is_valid());
}

TEST(MeshTest, IsInvalidWhenNormalsLengthMismatchesPositions) {
    bermake::Mesh m;
    m.positions = {0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f};  // 3 vertices
    m.normals = {0.f, 0.f, 1.f};                                  // 1 normal
    m.indices = {0u, 1u, 2u};
    EXPECT_FALSE(m.is_valid());
}

TEST(MeshTest, IsInvalidWhenPositionsNotMultipleOfThree) {
    bermake::Mesh m;
    m.positions = {0.f, 0.f, 0.f, 1.f, 0.f};  // 5 floats — not a whole number of XYZ triples
    m.normals = {0.f, 0.f, 1.f, 0.f, 0.f};
    EXPECT_FALSE(m.is_valid());
}

TEST(MeshTest, IsInvalidWhenIndicesNotMultipleOfThree) {
    bermake::Mesh m;
    m.positions = {0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f};
    m.normals = {0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f};
    m.indices = {0u, 1u};  // 2 indices — not a whole triangle
    EXPECT_FALSE(m.is_valid());
}
