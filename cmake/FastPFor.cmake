include_guard(GLOBAL)
include(FetchContent)
get_filename_component(GEF_FASTPFOR_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
# Official FastPFor library (Lemire & Boytsov), pinned; built unmodified.
set(FASTPFOR_WITH_TEST OFF CACHE BOOL "" FORCE)
FetchContent_Declare(gef_fastpfor_upstream
    URL https://codeload.github.com/lemire/FastPFor/tar.gz/ae4e62db4de1ce6904e278f94afb56a87a36b32c
    URL_HASH SHA256=24b28f763ffc39deae2a750d3a7fb5c4492ac59df8ef3ab83b2c85ccb40999b7)
FetchContent_MakeAvailable(gef_fastpfor_upstream)
# Only the FastPFOR library is needed (built as a dependency of gef_fastpfor). Upstream's
# tools stay out of `all`: at this commit ztimer.h opens its namespace only under _WIN32,
# so tools that include it (e.g. entropy) do not compile on Linux.
set_property(DIRECTORY "${gef_fastpfor_upstream_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL TRUE)
add_library(gef_fastpfor STATIC "${GEF_FASTPFOR_ROOT}/benchmark/FastPFor/FastPFor.cpp")
target_compile_features(gef_fastpfor PUBLIC cxx_std_17)
target_include_directories(gef_fastpfor PUBLIC "${GEF_FASTPFOR_ROOT}/benchmark")
target_include_directories(gef_fastpfor PRIVATE "${gef_fastpfor_upstream_SOURCE_DIR}/headers")
target_link_libraries(gef_fastpfor PRIVATE FastPFOR)
