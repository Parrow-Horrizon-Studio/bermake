#pragma once

#include <string>

namespace bermake {

/// Returns the Bermake library version, e.g. "0.14.0". The value is set at
/// build time from pyproject.toml (see the root CMakeLists.txt).
std::string version();

}  // namespace bermake
