#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <string>

#include "bermake/gltf_import.h"

namespace {

std::string sample(const char* name) {
    // Tests run from the build dir; BERMAKE_TEST_DATA is set by CMake.
    return std::string(BERMAKE_TEST_DATA) + "/gltf/" + name;
}

TEST(GltfImport, PlainBoxHasGeometryAndNodes) {
    const bermake::ImportedScene s = bermake::import_gltf(sample("plain_box.glb"));
    ASSERT_FALSE(s.meshes.empty());
    EXPECT_GT(s.meshes[0].positions.size(), 0u);
    EXPECT_GT(s.meshes[0].triangles.size(), 0u);
    EXPECT_FALSE(s.nodes.empty());
    EXPECT_EQ(s.nodes[0].parent, -1);  // root first
}

TEST(GltfImport, DracoBoxDecodes) {
    const bermake::ImportedScene s = bermake::import_gltf(sample("draco_box.glb"));
    ASSERT_FALSE(s.meshes.empty());
    EXPECT_GT(s.meshes[0].triangles.size(), 0u);  // Draco actually decoded
}

TEST(GltfImport, MissingFileThrows) {
    EXPECT_THROW(bermake::import_gltf(sample("does_not_exist.glb")), std::runtime_error);
}

TEST(GltfImport, TexturedBoxCarriesUvsParallelToPositions) {
    const bermake::ImportedScene s = bermake::import_gltf(sample("textured_box.glb"));
    ASSERT_FALSE(s.meshes.empty());
    const bermake::ImportedMesh& m = s.meshes[0];
    ASSERT_EQ(m.uvs.size(), m.positions.size());
    // The fixture's own v values run 0.25..1.0, but Assimp's glTF2 importer
    // applies 1 - v before the bridge sees them, so what arrives here runs
    // 0.0..0.75 and is ALREADY in Bermake's convention. The bridge passes it
    // through untouched and nothing downstream flips again (D14, revised
    // after measurement). tests/test_gltf_integration.py holds the permanent
    // gate that pins this relationship to the raw file bytes.
    float min_v = 1.0f;
    float max_v = 0.0f;
    for (const auto& uv : m.uvs) {
        min_v = std::min(min_v, uv[1]);
        max_v = std::max(max_v, uv[1]);
    }
    EXPECT_NEAR(min_v, 0.0f, 1e-5f);
    EXPECT_NEAR(max_v, 0.75f, 1e-5f);
}

TEST(GltfImport, MeshWithoutTexcoordsHasEmptyUvs) {
    // Discriminates against filling uvs with zeros, which would make every
    // untextured import look like it had a legitimate UV map at the origin.
    const bermake::ImportedScene s = bermake::import_gltf(sample("plain_box.glb"));
    ASSERT_FALSE(s.meshes.empty());
    EXPECT_TRUE(s.meshes[0].uvs.empty());
}

TEST(GltfImport, EmbeddedImageArrivesAsEncodedBytes) {
    const bermake::ImportedScene s = bermake::import_gltf(sample("textured_box.glb"));
    ASSERT_EQ(s.images.size(), 1u);
    const bermake::ImportedImage& img = s.images[0];
    ASSERT_GE(img.data.size(), 8u);
    const unsigned char png_magic[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    EXPECT_EQ(std::memcmp(img.data.data(), png_magic, 8), 0) << "not the encoded PNG";
    ASSERT_FALSE(s.materials.empty());
    EXPECT_EQ(s.materials[0].texture_index, 0);
    EXPECT_TRUE(s.materials[0].texture_uri.empty());
}

TEST(GltfImport, ExternalImageArrivesAsAnUnresolvedUri) {
    // D15: the bridge reports the filename, Python resolves it with the
    // containment rule. Discriminates against C++ silently reading the file.
    const bermake::ImportedScene s = bermake::import_gltf(sample("textured_box.gltf"));
    ASSERT_FALSE(s.materials.empty());
    EXPECT_EQ(s.materials[0].texture_index, -1);
    EXPECT_EQ(s.materials[0].texture_uri, "uvgrid.png");
}

TEST(GltfImport, DracoBoxUvsDecode_CI_GATE) {
    // PERMANENT GATE. draco_box.glb has no UVs, so it cannot show that a
    // vcpkg assimp bump broke Draco's TEXCOORD_0 path. This one can.
    const bermake::ImportedScene s = bermake::import_gltf(sample("avocado_draco.glb"));
    ASSERT_FALSE(s.meshes.empty());
    EXPECT_GT(s.meshes[0].triangles.size(), 0u) << "Draco decode produced no geometry";
    EXPECT_EQ(s.meshes[0].uvs.size(), s.meshes[0].positions.size())
        << "Draco decoded geometry but not TEXCOORD_0";
}

}  // namespace
