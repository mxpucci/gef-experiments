# Author's Rust source, compiled directly without copying or rewriting codecs.
include(FetchContent)
FetchContent_Declare(gef_fauna_upstream
    URL https://codeload.github.com/aabduvakhobov/PlatypusMacaque/tar.gz/26ea89142f0395927d0922a9b29b2524c08849e0
    URL_HASH SHA256=5ead943fa7a4d16afad71637be22a3d4e03a6fb025912cd9009a488f94d3eb8a)
FetchContent_MakeAvailable(gef_fauna_upstream)
find_program(CARGO_EXECUTABLE cargo REQUIRED)
get_filename_component(GEF_FAUNA_REPO "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(GEF_FAUNA_FFI_SOURCE "${GEF_FAUNA_REPO}/lib/fauna-ffi")
set(GEF_FAUNA_UPSTREAM "${gef_fauna_upstream_SOURCE_DIR}/ModelarDB-versions/ModelarDB-Macaque")
set(GEF_FAUNA_CRATE "${CMAKE_CURRENT_BINARY_DIR}/fauna-ffi")
file(MAKE_DIRECTORY "${GEF_FAUNA_CRATE}")
configure_file("${GEF_FAUNA_FFI_SOURCE}/Cargo.toml.in" "${GEF_FAUNA_CRATE}/Cargo.toml" @ONLY)
configure_file("${GEF_FAUNA_FFI_SOURCE}/build.rs" "${GEF_FAUNA_CRATE}/build.rs" COPYONLY)
if(EXISTS "${GEF_FAUNA_FFI_SOURCE}/Cargo.lock")
    configure_file("${GEF_FAUNA_FFI_SOURCE}/Cargo.lock" "${GEF_FAUNA_CRATE}/Cargo.lock" COPYONLY)
endif()
set(GEF_FAUNA_LIBRARY "${GEF_FAUNA_CRATE}/target/release/libgef_fauna_ffi.a")
add_custom_command(OUTPUT "${GEF_FAUNA_LIBRARY}"
    COMMAND "${CMAKE_COMMAND}" -E env "GEF_FAUNA_UPSTREAM=${GEF_FAUNA_UPSTREAM}"
        "${CARGO_EXECUTABLE}" build --release --locked --manifest-path "${GEF_FAUNA_CRATE}/Cargo.toml"
    DEPENDS "${GEF_FAUNA_FFI_SOURCE}/src/lib.rs" "${GEF_FAUNA_CRATE}/build.rs"
        "${GEF_FAUNA_CRATE}/Cargo.toml" "${GEF_FAUNA_CRATE}/Cargo.lock"
    COMMENT "Building official Fauna/Macaque Rust implementation" VERBATIM)
add_custom_target(gef_fauna_build DEPENDS "${GEF_FAUNA_LIBRARY}")
add_library(gef_fauna STATIC IMPORTED GLOBAL)
set_target_properties(gef_fauna PROPERTIES IMPORTED_LOCATION "${GEF_FAUNA_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${GEF_FAUNA_REPO}/benchmark")
add_dependencies(gef_fauna gef_fauna_build)
if(UNIX AND NOT APPLE)
    set_property(TARGET gef_fauna PROPERTY INTERFACE_LINK_LIBRARIES "pthread;dl;m")
elseif(APPLE)
    set_property(TARGET gef_fauna PROPERTY INTERFACE_LINK_LIBRARIES "-framework CoreFoundation")
endif()
