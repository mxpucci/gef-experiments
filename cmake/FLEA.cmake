include_guard(GLOBAL)
include(FetchContent)
get_filename_component(GEF_FLEA_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
FetchContent_Declare(gef_flea_upstream
    URL https://codeload.github.com/Mr-Spade/FLEA/tar.gz/47a9a480b34f5c82b6602e2d7aa217ae52aeca05
    URL_HASH SHA256=0bc7e17d8e4fd065fbd73a95a7f5c9c5cd469e98c875844144f8e7965299a983
    SOURCE_SUBDIR algorithms/flea)
FetchContent_MakeAvailable(gef_flea_upstream)
# Upstream has scalar implementations but includes x86 intrinsics unconditionally.
# Guard only that include in the downloaded tree; no codec is copied or replaced.
set(GEF_FLEA_SIMD "${gef_flea_upstream_SOURCE_DIR}/algorithms/flea/include/flea/simd_utils.hpp")
file(READ "${GEF_FLEA_SIMD}" GEF_FLEA_SIMD_TEXT)
if(NOT GEF_FLEA_SIMD_TEXT MATCHES "GEF_PORTABLE_INTRINSICS")
    string(REPLACE "#include <immintrin.h>"
        "// GEF_PORTABLE_INTRINSICS\n#if defined(__x86_64__) || defined(__i386__)\n#include <immintrin.h>\n#endif"
        GEF_FLEA_SIMD_TEXT "${GEF_FLEA_SIMD_TEXT}")
    file(WRITE "${GEF_FLEA_SIMD}" "${GEF_FLEA_SIMD_TEXT}")
endif()
# A pinned dependency makes Linux builds independent of system FFTW availability.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ENABLE_THREADS OFF CACHE BOOL "" FORCE)
set(ENABLE_OPENMP OFF CACHE BOOL "" FORCE)
set(DISABLE_FORTRAN ON CACHE BOOL "" FORCE)
set(ENABLE_FLOAT OFF CACHE BOOL "" FORCE)
set(ENABLE_LONG_DOUBLE OFF CACHE BOOL "" FORCE)
set(ENABLE_QUAD_PRECISION OFF CACHE BOOL "" FORCE)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(ENABLE_SSE2 ON CACHE BOOL "" FORCE)
    set(ENABLE_AVX2 ON CACHE BOOL "" FORCE)
endif()
FetchContent_Declare(gef_fftw
    URL https://www.fftw.org/fftw-3.3.10.tar.gz
    URL_HASH SHA256=56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467)
FetchContent_MakeAvailable(gef_fftw)
add_library(gef_flea STATIC "${GEF_FLEA_ROOT}/benchmark/FLEA/FLEA.cpp")
target_compile_features(gef_flea PUBLIC cxx_std_20)
target_compile_options(gef_flea PRIVATE -fno-fast-math -ffp-contract=off)
target_include_directories(gef_flea PUBLIC "${GEF_FLEA_ROOT}/benchmark")
target_include_directories(gef_flea PRIVATE
    "${gef_flea_upstream_SOURCE_DIR}/algorithms/flea/include" "${gef_fftw_SOURCE_DIR}/api")
target_link_libraries(gef_flea PUBLIC fftw3)
