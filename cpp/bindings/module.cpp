#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>

#include "bermake/gltf_import.h"
#include "bermake/halfedge.h"
#include "bermake/mesh.h"
#include "bermake/primitives.h"
#include "bermake/ray_intersect.h"
#include "bermake/version.h"

namespace nb = nanobind;
using bermake::HalfEdgeMesh;
using bermake::Mesh;
using bermake::ray_intersect_mesh;
using bermake::RayMeshHit;
using bermake::SplitEdgeResult;

namespace {

// Expose std::vector<float> as a read-only (N, 3) numpy view of the data.
// The `const float` dtype is what makes the resulting numpy array
// non-writable from Python. The `nb::rv_policy::reference_internal` policy
// on the def_prop_ro call (below) keeps the owning Mesh alive as long as
// the returned ndarray is alive.
nb::ndarray<const float, nb::numpy, nb::shape<-1, 3>> as_vec3_array(const std::vector<float>& v) {
    const std::size_t n = v.size() / 3;
    return nb::ndarray<const float, nb::numpy, nb::shape<-1, 3>>(const_cast<float*>(v.data()),
                                                                 {n, static_cast<std::size_t>(3)});
}

// Expose std::vector<std::uint32_t> as a read-only (M,) numpy view of the
// data. The `const std::uint32_t` dtype is what makes the resulting numpy
// array non-writable from Python; the `const_cast` below is required
// because nanobind's `ndarray` constructor takes a non-const pointer even
// though the read-only marker is the `const` template parameter, not
// runtime mutability of the buffer.
nb::ndarray<const std::uint32_t, nb::numpy, nb::shape<-1>> as_index_array(
    const std::vector<std::uint32_t>& v) {
    return nb::ndarray<const std::uint32_t, nb::numpy, nb::shape<-1>>(
        const_cast<std::uint32_t*>(v.data()), {v.size()});
}

}  // namespace

// Issue #20 — the pytest run occasionally prints "nanobind: leaked N
// instances/types/functions!" on stderr at interpreter shutdown. Investigated
// (M7.1 T18) and confirmed BENIGN, not a binding-lifetime bug:
//   - A single HalfEdgeMesh instance surviving to shutdown does not reproduce
//     from direct construct/destroy cycles in isolation, even at 5000x, so
//     it is not caused by a missing/incorrect ownership policy here.
//   - It appears only when many Qt-hosted tests run in the same process
//     (e.g. tests/test_main_window_*.py constructing MainWindow() — and by
//     extension a Scene/HalfEdgeMesh — without qtbot teardown). Running a
//     partial suite left dozens of instances pending; running the full
//     suite left just one. That pattern is consistent with Qt's C++-side
//     event/timer/widget bookkeeping holding the last few Python objects
//     alive until the QApplication itself tears down — not something
//     Python's own cyclic GC can reach (forcing extra gc.collect() at
//     session end did not clear it), and unrelated to reference-counting
//     in this module's bindings.
//   - The reported "leaked types" / "leaked functions" are a knock-on effect
//     of that one live instance keeping its type object (and therefore the
//     type's whole method table) referenced; they are not independent leaks.
// nanobind (2.12.0, pinned) only exposes a single process-wide toggle for
// this diagnostic (nb::set_leak_warnings), which would also hide a genuine
// future instance leak — so we deliberately do NOT suppress it here. Accept
// the shutdown-time message as expected noise. See:
// https://nanobind.readthedocs.io/en/latest/refleaks.html
NB_MODULE(_core, m) {
    m.doc() = "Bermake C++ core module";

    m.def("version", &bermake::version, "Returns the Bermake library version as a string.");

    nb::class_<Mesh>(m, "Mesh", "Polygonal mesh: positions, normals, indices.")
        .def(nb::init<>())
        .def_prop_ro(
            "positions", [](Mesh& self) { return as_vec3_array(self.positions); },
            nb::rv_policy::reference_internal,
            "Vertex positions as a read-only (N, 3) float32 numpy view.")
        .def_prop_ro(
            "normals", [](Mesh& self) { return as_vec3_array(self.normals); },
            nb::rv_policy::reference_internal,
            "Vertex normals as a read-only (N, 3) float32 numpy view.")
        .def_prop_ro(
            "indices", [](Mesh& self) { return as_index_array(self.indices); },
            nb::rv_policy::reference_internal,
            "Triangle indices as a read-only (M,) uint32 numpy view.")
        .def_prop_ro("vertex_count", &Mesh::vertex_count, "Number of vertices.")
        .def_prop_ro("triangle_count", &Mesh::triangle_count, "Number of triangles.");

    m.def("make_cube", &bermake::make_cube, nb::arg("size") = 1.0f,
          "Create an axis-aligned cube of the given edge length, "
          "with its bottom face on the ground plane (z = 0).");

    nb::class_<HalfEdgeMesh>(m, "HalfEdgeMesh",
                             "Half-edge topology mesh — geometric source of truth")
        .def(nb::init<>())

        // Mutators
        .def("add_vertex", &HalfEdgeMesh::add_vertex)
        .def("add_halfedge_pair", &HalfEdgeMesh::add_halfedge_pair)
        .def("add_face_from_loop", &HalfEdgeMesh::add_face_from_loop)
        .def("remove_vertex", &HalfEdgeMesh::remove_vertex)
        .def("remove_edge", &HalfEdgeMesh::remove_edge)
        .def("remove_face", &HalfEdgeMesh::remove_face)
        .def("restore_vertex", &HalfEdgeMesh::restore_vertex)
        .def("restore_edge", &HalfEdgeMesh::restore_edge)
        .def("restore_face", &HalfEdgeMesh::restore_face)
        .def("set_vertex_position", &HalfEdgeMesh::set_vertex_position)
        .def("clear", &HalfEdgeMesh::clear)

        // Queries
        .def("vertex_is_live", &HalfEdgeMesh::vertex_is_live)
        .def("edge_is_live", &HalfEdgeMesh::edge_is_live)
        .def("face_is_live", &HalfEdgeMesh::face_is_live)
        .def(
            "edge_between",
            [](const HalfEdgeMesh& m, std::uint32_t v1_id,
               std::uint32_t v2_id) -> std::optional<std::uint32_t> {
                const auto e = m.edge_between(v1_id, v2_id);
                if (e == HalfEdgeMesh::INVALID_ID) {
                    return std::nullopt;
                }
                return e;
            },
            nb::arg("v1_id"), nb::arg("v2_id"),
            "Look up the edge id between v1_id and v2_id without creating one. "
            "Returns None if no live edge connects them. Never mutates the mesh.")
        .def("vertex_position", &HalfEdgeMesh::vertex_position)
        .def("edge_vertices", &HalfEdgeMesh::edge_vertices)
        .def("face_loop_vertices", &HalfEdgeMesh::face_loop_vertices)
        .def("face_triangles", &HalfEdgeMesh::face_triangles)
        .def("set_face_triangles", &HalfEdgeMesh::set_face_triangles, nb::arg("face_id"),
             nb::arg("flat_vertex_ids"),
             "Replace a live face's triangulation with flat_vertex_ids, three global "
             "vertex ids per triangle. Raises ValueError, leaving the face unchanged, "
             "if the face is not live, the length is not a multiple of 3, or an id is "
             "not on the face's boundary loop.")

        // M3c: topology editing + coplanarity
        .def("dissolve_edge", &HalfEdgeMesh::dissolve_edge, nb::arg("edge_id"),
             "Collapse two adjacent faces sharing this edge into one merged face. "
             "Returns the new face id, or INVALID_ID if the edge is boundary/dead "
             "or the two faces share more than one edge.")
        .def("faces_are_coplanar", &HalfEdgeMesh::faces_are_coplanar, nb::arg("f1_id"),
             nb::arg("f2_id"), nb::arg("angle_tol_cos"), nb::arg("dist_tol"),
             "True iff |dot(n1, n2)| > angle_tol_cos AND every vertex of either "
             "face lies within dist_tol of the other face's plane. Recommended values "
             "(applied by Scene.faces_are_coplanar): cos(0.5°) ≈ 0.9999619, 1e-4.")
        .def("split_edge", &HalfEdgeMesh::split_edge, nb::arg("edge_id"), nb::arg("t"),
             "Split an edge at parameter t in (0,1), inserting a vertex and "
             "rebuilding incident faces. Returns a SplitEdgeResult, or None if "
             "the edge is dead, t is out of range, or w coincides with an existing vertex.")
        .def("split_face", &HalfEdgeMesh::split_face, nb::arg("f_id"), nb::arg("loop_a"),
             nb::arg("tris_a"), nb::arg("loop_b"), nb::arg("tris_b"),
             "Replace a face with two that share a chain of vertices: the inverse "
             "of dissolve_edge. Both loops must be triangulated by the caller with "
             "GLOBAL vertex ids. Returns (id_a, id_b), or (INVALID_ID, INVALID_ID) "
             "if any precondition fails, in which case the mesh is untouched.")

        // Half-edge adjacency (used by M3b push/pull)
        .def("halfedge_origin", &HalfEdgeMesh::halfedge_origin)
        .def("halfedge_next", &HalfEdgeMesh::halfedge_next)
        .def("halfedge_twin", &HalfEdgeMesh::halfedge_twin)
        .def("halfedge_face", &HalfEdgeMesh::halfedge_face)

        // Iteration
        .def("next_live_vertex", &HalfEdgeMesh::next_live_vertex, nb::arg("start") = 0u)
        .def("next_live_edge", &HalfEdgeMesh::next_live_edge, nb::arg("start") = 0u)
        .def("next_live_face", &HalfEdgeMesh::next_live_face, nb::arg("start") = 0u)

        // Buffer projection
        .def("edge_line_buffer", &HalfEdgeMesh::edge_line_buffer)
        .def("face_triangle_buffer", &HalfEdgeMesh::face_triangle_buffer)

        // Dirty flag
        .def("is_dirty", &HalfEdgeMesh::is_dirty)
        .def("mark_clean", &HalfEdgeMesh::mark_clean)

        // Slab introspection (mostly for tests)
        .def("vertex_slab_size", &HalfEdgeMesh::vertex_slab_size)
        .def("halfedge_slab_size", &HalfEdgeMesh::halfedge_slab_size)
        .def("face_slab_size", &HalfEdgeMesh::face_slab_size)

        .def_ro_static("INVALID_ID", &HalfEdgeMesh::INVALID_ID);

    m.def("make_box", &bermake::make_box, nb::arg("width") = 1.0f, nb::arg("depth") = 1.0f,
          nb::arg("height") = 1.0f,
          "Create an axis-aligned box as a HalfEdgeMesh, centred on the origin "
          "in x and y with its base on the ground plane (z = 0).");
    m.def("make_cylinder", &bermake::make_cylinder, nb::arg("radius") = 1.0f,
          nb::arg("height") = 1.0f, nb::arg("segments") = 24,
          "Create a cylinder as a HalfEdgeMesh, centred on the origin in x and "
          "y with its base on the ground plane (z = 0).");
    m.def("make_cone", &bermake::make_cone, nb::arg("radius") = 1.0f, nb::arg("height") = 1.0f,
          nb::arg("segments") = 24,
          "Create a cone as a HalfEdgeMesh, centred on the origin in x and y "
          "with its base on the ground plane (z = 0) and apex at z = height.");
    m.def("make_sphere", &bermake::make_sphere, nb::arg("radius") = 1.0f, nb::arg("rings") = 12,
          nb::arg("segments") = 24,
          "Create a UV-sphere as a HalfEdgeMesh, centred on the origin in x "
          "and y with its bottom pole on the ground plane (z = 0).");

    nb::class_<SplitEdgeResult>(m, "SplitEdgeResult", "Result of HalfEdgeMesh.split_edge")
        .def_ro("vertex", &SplitEdgeResult::vertex)
        .def_ro("edge_a", &SplitEdgeResult::edge_a)
        .def_ro("edge_b", &SplitEdgeResult::edge_b)
        .def_ro("face_a", &SplitEdgeResult::face_a)
        .def_ro("face_b", &SplitEdgeResult::face_b);

    nb::class_<RayMeshHit>(m, "RayMeshHit", "Result of bermake::ray_intersect_mesh")
        .def_ro("face_id", &RayMeshHit::face_id)
        .def_ro("t", &RayMeshHit::t)
        .def_ro("point", &RayMeshHit::point, "Hit point in world coordinates (3-tuple).");

    m.def("ray_intersect_mesh", &ray_intersect_mesh, nb::arg("mesh"), nb::arg("origin"),
          nb::arg("direction"), "Brute-force ray-mesh face picking. Returns RayMeshHit or None.");

    // M6c: glTF/GLB import bridge (Assimp) — neutral plain-data structs.
    nb::class_<bermake::ImportedImage>(m, "ImportedImage")
        .def_ro("name", &bermake::ImportedImage::name)
        .def_ro("data", &bermake::ImportedImage::data)
        .def_ro("format_hint", &bermake::ImportedImage::format_hint);
    nb::class_<bermake::ImportedMaterial>(m, "ImportedMaterial")
        .def_ro("name", &bermake::ImportedMaterial::name)
        .def_ro("base_color", &bermake::ImportedMaterial::base_color)
        .def_ro("texture_index", &bermake::ImportedMaterial::texture_index)
        .def_ro("texture_uri", &bermake::ImportedMaterial::texture_uri);
    nb::class_<bermake::ImportedMesh>(m, "ImportedMesh")
        .def_ro("positions", &bermake::ImportedMesh::positions)
        .def_ro("uvs", &bermake::ImportedMesh::uvs)
        .def_ro("triangles", &bermake::ImportedMesh::triangles)
        .def_ro("material_index", &bermake::ImportedMesh::material_index);
    nb::class_<bermake::ImportedNode>(m, "ImportedNode")
        .def_ro("name", &bermake::ImportedNode::name)
        .def_ro("parent", &bermake::ImportedNode::parent)
        .def_ro("transform", &bermake::ImportedNode::transform)
        .def_ro("mesh_indices", &bermake::ImportedNode::mesh_indices);
    nb::class_<bermake::ImportedScene>(m, "ImportedScene")
        .def_ro("nodes", &bermake::ImportedScene::nodes)
        .def_ro("meshes", &bermake::ImportedScene::meshes)
        .def_ro("materials", &bermake::ImportedScene::materials)
        .def_ro("images", &bermake::ImportedScene::images);
    m.def("import_gltf", &bermake::import_gltf, nb::arg("path"),
          "Load a glTF/GLB file into a neutral ImportedScene (M6c import bridge).");
}
