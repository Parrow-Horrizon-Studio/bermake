find_package(GTest CONFIG REQUIRED)

add_executable(bermake_tests
    test_version.cpp
    test_mesh.cpp
    test_primitives.cpp
    test_halfedge.cpp
    test_ray_intersect.cpp
    test_gltf_import.cpp
)

target_compile_definitions(bermake_tests PRIVATE
    BERMAKE_TEST_DATA="${CMAKE_SOURCE_DIR}/tests/data")

target_link_libraries(bermake_tests PRIVATE
    bermake_core
    GTest::gtest
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(bermake_tests)
