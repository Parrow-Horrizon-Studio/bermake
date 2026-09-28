# Core library — the C++ implementation
add_library(bermake_core STATIC
    src/version.cpp
    src/mesh.cpp
    src/primitives.cpp
    src/halfedge.cpp
    src/ray_intersect.cpp
    src/gltf_import.cpp
)

target_include_directories(bermake_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Assimp — glTF/GLB import (M6c). PRIVATE: the public header exposes only
# neutral structs, so consumers of bermake_core don't need Assimp headers.
target_link_libraries(bermake_core PRIVATE assimp::assimp)

# Position-independent code required when linking into a Python module
set_target_properties(bermake_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# The version comes from pyproject.toml through the root CMakeLists.txt
# (M7.9, #130), so version.cpp carries no literal of its own.
target_compile_definitions(bermake_core PRIVATE
    BERMAKE_VERSION_STRING="${PROJECT_VERSION}"
)

# The nanobind module — exposed to Python as `bermake._core`
nanobind_add_module(_core
    bindings/module.cpp
)

target_link_libraries(_core PRIVATE bermake_core)

# Install the compiled module into the Python package directory
install(TARGETS _core LIBRARY DESTINATION bermake)
