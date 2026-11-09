# Include from the main project; link gef_pcodec to any target using Pcodec.hpp.
# Official pco_c C bindings from the pcodec v1.0.2 release commit, built unmodified.
include_guard(GLOBAL)
include(FetchContent)
FetchContent_Declare(gef_pcodec_upstream
    URL https://codeload.github.com/pcodec/pcodec/tar.gz/2d8555888b21bbaa19326580b740fa24b7da6bd3
    URL_HASH SHA256=892942842b558ffc25097b2c02d761b37adfb06dc87c39386266d407f7c1c2d0)
FetchContent_MakeAvailable(gef_pcodec_upstream)
find_program(GEF_PCODEC_CARGO cargo REQUIRED)
get_filename_component(GEF_PCODEC_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(GEF_PCODEC_MANIFEST "${gef_pcodec_upstream_SOURCE_DIR}/Cargo.toml")
set(GEF_PCODEC_TARGET_DIR "${CMAKE_BINARY_DIR}/pcodec-cargo")
set(GEF_PCODEC_LIBRARY "${GEF_PCODEC_TARGET_DIR}/release/${CMAKE_STATIC_LIBRARY_PREFIX}cpcodec${CMAKE_STATIC_LIBRARY_SUFFIX}")
set(GEF_PCODEC_RUSTFLAGS "" CACHE STRING "Optional Rust flags for Pcodec, e.g. -C target-cpu=native")
# pco_c declares a Rust dylib; --crate-type requests a C-linkable static library
# without editing upstream files. The upstream Cargo.lock is used via --locked.
add_custom_command(OUTPUT "${GEF_PCODEC_LIBRARY}"
    COMMAND ${CMAKE_COMMAND} -E env "CARGO_TARGET_DIR=${GEF_PCODEC_TARGET_DIR}"
        "RUSTFLAGS=${GEF_PCODEC_RUSTFLAGS}"
        "${GEF_PCODEC_CARGO}" rustc --release --locked --manifest-path "${GEF_PCODEC_MANIFEST}"
        -p cpcodec --crate-type staticlib
    DEPENDS "${GEF_PCODEC_MANIFEST}"
    COMMENT "Building official Pcodec 1.0.2 pco_c C bindings"
    VERBATIM)
add_custom_target(gef_pcodec_build DEPENDS "${GEF_PCODEC_LIBRARY}")
add_library(gef_pcodec STATIC IMPORTED GLOBAL)
set_target_properties(gef_pcodec PROPERTIES IMPORTED_LOCATION "${GEF_PCODEC_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${GEF_PCODEC_ROOT}/benchmark;${gef_pcodec_upstream_SOURCE_DIR}/pco_c/include")
add_dependencies(gef_pcodec gef_pcodec_build)
if(UNIX AND NOT APPLE)
    set_property(TARGET gef_pcodec PROPERTY INTERFACE_LINK_LIBRARIES "pthread;dl;m")
endif()
