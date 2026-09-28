#include "bermake/version.h"

#ifndef BERMAKE_VERSION_STRING
#error "BERMAKE_VERSION_STRING must be defined by the build (see cpp/CMakeLists.txt)"
#endif

namespace bermake {

std::string version() {
    return BERMAKE_VERSION_STRING;
}

}  // namespace bermake
