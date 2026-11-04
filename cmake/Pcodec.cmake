# Include from the main project; link gef_pcodec to any target using Pcodec.hpp.
include_guard(GLOBAL)
find_program(GEF_PCODEC_CARGO cargo REQUIRED)
get_filename_component(GEF_PCODEC_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(GEF_PCODEC_MANIFEST "${GEF_PCODEC_ROOT}/lib/pcodec-ffi/Cargo.toml")
set(GEF_PCODEC_TARGET_DIR "${CMAKE_BINARY_DIR}/pcodec-cargo")
set(GEF_PCODEC_LIBRARY "${GEF_PCODEC_TARGET_DIR}/release/${CMAKE_STATIC_LIBRARY_PREFIX}gef_pcodec_ffi${CMAKE_STATIC_LIBRARY_SUFFIX}")
set(GEF_PCODEC_RUSTFLAGS "" CACHE STRING "Optional Rust flags for Pcodec, e.g. -C target-cpu=native")
add_custom_command(OUTPUT "${GEF_PCODEC_LIBRARY}"
    COMMAND ${CMAKE_COMMAND} -E env "CARGO_TARGET_DIR=${GEF_PCODEC_TARGET_DIR}"
        "RUSTFLAGS=${GEF_PCODEC_RUSTFLAGS}"
        "${GEF_PCODEC_CARGO}" build --release --locked --manifest-path "${GEF_PCODEC_MANIFEST}"
    DEPENDS "${GEF_PCODEC_MANIFEST}"
        "${GEF_PCODEC_ROOT}/lib/pcodec-ffi/Cargo.lock"
        "${GEF_PCODEC_ROOT}/lib/pcodec-ffi/src/lib.rs"
        "${GEF_PCODEC_ROOT}/lib/pcodec-ffi/src/indexed.rs"
    COMMENT "Building official Pcodec 1.0.2 Rust dependency and C ABI adapter"
    VERBATIM)
add_custom_target(gef_pcodec_build DEPENDS "${GEF_PCODEC_LIBRARY}")
add_library(gef_pcodec STATIC IMPORTED GLOBAL)
set_target_properties(gef_pcodec PROPERTIES IMPORTED_LOCATION "${GEF_PCODEC_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${GEF_PCODEC_ROOT}/benchmark")
add_dependencies(gef_pcodec gef_pcodec_build)
if(UNIX AND NOT APPLE)
    set_property(TARGET gef_pcodec PROPERTY INTERFACE_LINK_LIBRARIES "pthread;dl;m")
endif()
