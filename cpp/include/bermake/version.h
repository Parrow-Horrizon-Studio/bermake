#pragma once

#include <string>

namespace bermake {

/// Returns the Bermake library version, e.g. "X.Y.Z". The value is set at
/// build time from pyproject.toml (see the root CMakeLists.txt).
std::string version();

}  // namespace bermake
