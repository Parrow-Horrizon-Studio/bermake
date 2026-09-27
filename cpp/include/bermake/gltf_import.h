#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace pluton {

struct ImportedImage {
    std::string name;
    // The image file's own ENCODED bytes (PNG/JPEG). EMPTY when Assimp gave
    // raw texels instead (aiTexture::mHeight != 0), which Pluton's Texture
    // library cannot store: it holds encoded bytes only, and pluton/io may
    // not reach Qt to encode them. glTF 2.0 mandates PNG or JPEG, so this
    // should never happen for glTF; the Python layer counts and skips it.
    std::vector<std::uint8_t> data;
    std::string format_hint;  // aiTexture::achFormatHint, e.g. "png", "jpg"
};

struct ImportedMaterial {
    std::string name;
    std::array<float, 4> base_color;  // RGBA
    int texture_index;                // -1 = none; else index into ImportedScene::images
    std::string texture_uri;          // non-empty = an EXTERNAL file, unresolved
};

struct ImportedMesh {
    std::vector<std::array<float, 3>> positions;
    // Empty, or exactly positions.size() entries. Exactly what Assimp
    // returns from mTextureCoords, passed through untouched: Assimp's glTF2
    // importer already applies v' = 1 - v itself, so these values are
    // ALREADY in Pluton's convention (v = 0 at the image's bottom), not
    // glTF's (v = 0 at the top). No later stage flips them again. Export is
    // asymmetric and does flip, because it writes through Pluton's own
    // codec instead of relying on a third-party importer's behaviour (D14,
    // revised after measurement; see the permanent CI gate in
    // tests/test_gltf_integration.py that pins this to the raw file bytes).
    std::vector<std::array<float, 2>> uvs;
    std::vector<std::array<std::uint32_t, 3>> triangles;
    int material_index;  // -1 = none
};

struct ImportedNode {
    std::string name;
    int parent;                       // -1 = root
    std::array<float, 16> transform;  // row-major (aiMatrix4x4 order)
    std::vector<int> mesh_indices;
};

struct ImportedScene {
    std::vector<ImportedNode> nodes;
    std::vector<ImportedMesh> meshes;
    std::vector<ImportedMaterial> materials;
    std::vector<ImportedImage> images;
};

// Load a glTF/GLB and flatten to neutral data. Throws std::runtime_error on a
// whole-file load failure (missing/undecodable).
ImportedScene import_gltf(const std::string& path);

}  // namespace pluton
