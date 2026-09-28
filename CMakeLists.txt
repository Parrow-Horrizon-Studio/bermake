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

# pyproject.toml is the single version source (M7.9, #130). Read it here so
# the standalone build/tests configure, which scikit-build-core does not drive,
# gets the same number as the wheel. Anchored at the start of the line so
# `cmake.version` and `minimum-version` under [tool.scikit-build] cannot match,
# and not anchored at the end, so a CRLF checkout's trailing \r cannot break it.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" _bermake_version_line
    REGEX "^version = \"[0-9]+\\.[0-9]+\\.[0-9]+\"" LIMIT_COUNT 1)
string(REGEX REPLACE "^version = \"([0-9]+\\.[0-9]+\\.[0-9]+)\".*$" "\\1"
    BERMAKE_VERSION "${_bermake_version_line}")
if(NOT BERMAKE_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
    message(FATAL_ERROR "Could not read the project version from pyproject.toml")
endif()

# Project metadata
project(bermake
    VERSION ${BERMAKE_VERSION}
    DESCRIPTION "Polygonal 3D modeler for architecture"
    LANGUAGES CXX
)

# Re-run configure when pyproject.toml changes, so a version bump reaches an
# existing build/tests tree without deleting its cache.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml")

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
