#include "bermake/mesh.h"

// Mesh is currently header-only (data + inline accessors). This translation
// unit exists to keep CMake symmetry with primitives.cpp and to give us a
// place to put non-inline methods in future milestones (M2: add face/edge
// methods, M3: half-edge adjacency).

namespace bermake {

// Intentionally empty for M1.

}  // namespace bermake
