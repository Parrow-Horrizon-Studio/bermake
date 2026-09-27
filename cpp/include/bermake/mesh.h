#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace pluton {

/// A polygonal mesh stored as three flat, GPU-ready arrays.
///
/// Layout matches what OpenGL VBOs want: tightly-packed floats for positions
/// and normals, and a 32-bit index buffer for triangles. This avoids any
/// reshape or copy when the data is pulled to Python via nanobind.
///
/// Invariant: `is_valid()` must hold whenever the mesh is read by code that
/// consumes geometry (renderers, M3 push/pull, M11 export). In debug builds,
/// the count accessors assert that `positions` and `normals` have equal
/// lengths.
class Mesh {
public:
    /// Vertex positions in XYZ order: [x0,y0,z0, x1,y1,z1, ...].
    std::vector<float> positions;

    /// Vertex normals in XYZ order, parallel to `positions` (same length).
    std::vector<float> normals;

    /// Triangle indices into `positions` / `normals` (3 indices per triangle).
    std::vector<std::uint32_t> indices;

    /// True iff the three arrays are internally consistent: positions and
    /// normals have equal lengths, positions is a whole number of XYZ
    /// triples, and indices is a whole number of triangles.
    bool is_valid() const noexcept {
        return positions.size() == normals.size() && positions.size() % 3 == 0 &&
               indices.size() % 3 == 0;
    }

    /// Number of vertices (positions.size() / 3).
    std::size_t vertex_count() const {
        assert(positions.size() == normals.size() &&
               "Mesh invariant: positions and normals must have equal length");
        return positions.size() / 3;
    }

    /// Number of triangles (indices.size() / 3).
    std::size_t triangle_count() const {
        assert(indices.size() % 3 == 0 && "Mesh invariant: indices must be a multiple of 3");
        return indices.size() / 3;
    }
};

}  // namespace pluton
