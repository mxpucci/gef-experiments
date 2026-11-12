include_guard(GLOBAL)
get_filename_component(GEF_BTRBLOCKS_ADAPTER_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
# Official BtrBlocks (maxi-k/btrblocks) from a source tree built with its own CMake
# (target btrblocks); GEF_BTRBLOCKS_ROOT points at that tree, its build in ./build.
set(B "${GEF_BTRBLOCKS_ROOT}/build")
file(GLOB GEF_BTR_BOOST_INCLUDES "${B}/vendor/boost/boost_src-src/libs/*/include")
add_library(gef_btrblocks STATIC "${GEF_BTRBLOCKS_ADAPTER_ROOT}/benchmark/BtrBlocks/BtrBlocks.cpp")
target_compile_features(gef_btrblocks PUBLIC cxx_std_17)
target_compile_options(gef_btrblocks PRIVATE -march=native)
target_include_directories(gef_btrblocks PUBLIC "${GEF_BTRBLOCKS_ADAPTER_ROOT}/benchmark")
target_include_directories(gef_btrblocks SYSTEM PRIVATE "${GEF_BTRBLOCKS_ROOT}/btrblocks"
    "${B}/vendor/cwida/fsst/src/fsst_src" "${B}/vendor/lemire/fastpfor/src/fastpfor_src"
    "${B}/vendor/croaring/include" ${GEF_BTR_BOOST_INCLUDES})
target_link_libraries(gef_btrblocks PRIVATE "${B}/libbtrblocks.a" "${B}/vendor/cwida/fsst/src/fsst_src-build/libfsst.a"
    "${B}/vendor/lemire/fastpfor/src/fastpfor_src-build/libFastPFOR.a" "${B}/vendor/croaring/lib/libroaring.so")
