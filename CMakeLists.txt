cmake_minimum_required(VERSION 3.27)

# On Windows, use a STATIC vcpkg triplet so Assimp (M6c) and its transitive
# deps (Draco, zlib, ...) link into the _core module — no runtime DLLs to
# bundle next to the installed .pyd (which broke editable installs + CI).
# static-md = static libs with the dynamic CRT (/MD), matching CPython.
# Set before project() so the vcpkg toolchain picks it up. Linux's default
# x64-linux triplet is already static, so this is Windows-only.
if(CMAKE_HOST_WIN32 AND NOT DEFINED VCPKG_TARGET_TRIPLET)
    set(VCPKG_TARGET_TRIPLET "x64-windows-static-md" CACHE STRING "vcpkg triplet")
endif()

# Project metadata
project(bermake
    VERSION 0.14.0
    DESCRIPTION "Polygonal 3D modeler for architecture"
    LANGUAGES CXX
)

# Require C++20 globally
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Generate compile_commands.json for IDE support (clangd, etc.)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler warning baseline
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find Python (driven by scikit-build-core at build time)
find_package(Python 3.13 COMPONENTS Interpreter Development.Module REQUIRED)

# Find nanobind (installed via pip as a build dep — scikit-build-core handles this)
find_package(nanobind CONFIG REQUIRED)

# Find Assimp (glTF/GLB import — vcpkg, M6c)
find_package(assimp CONFIG REQUIRED)

# Add C++ source tree
add_subdirectory(cpp)

# Enable tests if this is the top-level project (not when consumed as a subproject)
# and the cpp/tests directory exists (it's added in M0 Task 10).
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cpp/tests/CMakeLists.txt")
    enable_testing()
    add_subdirectory(cpp/tests)
endif()
