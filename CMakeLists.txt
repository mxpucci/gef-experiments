cmake_minimum_required(VERSION 3.22)
project(NeaTS)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_POLICY_DEFAULT_CMP0048 NEW)

# Build type - default to Release for maximum optimization
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Common optimization flags for both architectures
set(COMMON_OPT_FLAGS "-O3 -DNDEBUG -fno-math-errno -funroll-loops -ftree-vectorize")

# Detect architecture and set appropriate flags
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(DEFAULT_ARCH_FLAGS "-mcpu=native")
else()
    # Default to native for local development, generic for portable builds
    if(BUILD_PORTABLE)
        set(DEFAULT_ARCH_FLAGS "-march=x86-64 -mtune=generic -mfpmath=sse")
    else()
        set(DEFAULT_ARCH_FLAGS "-march=native -mtune=native -mfpmath=sse")
    endif()
endif()

# Allow CI to override ARCH_FLAGS (e.g., -DARCH_FLAGS="-march=skylake-avx512")
if(NOT DEFINED ARCH_FLAGS)
    set(ARCH_FLAGS "${DEFAULT_ARCH_FLAGS}")
endif()

message(STATUS "Target Architecture Flags: ${ARCH_FLAGS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_FLAGS} ${COMMON_OPT_FLAGS}")
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${ARCH_FLAGS} ${COMMON_OPT_FLAGS} -Wall -D_FILE_OFFSET_BITS=64")

# Statically link standard libraries for Linux portability
if(UNIX AND NOT APPLE)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++")
endif()

# Link-time optimization for Release builds (significant performance improvement)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    include(CheckIPOSupported)
    check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)
    if(IPO_SUPPORTED)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
        message(STATUS "LTO enabled")
    else()
        message(STATUS "LTO not supported: ${IPO_ERROR}")
    endif()
endif()

# Enable OpenMP for parallel processing (used by GEF)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found - parallel processing enabled")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_library(sux INTERFACE)
target_include_directories(sux INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/lib/sux)

add_library(sdsl INTERFACE)
target_include_directories(sdsl INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/lib/sdsl-lite/include)

# --- Pasta (rank/select bitvectors) ---
include(FetchContent)
set(TLX_INSTALL OFF CACHE BOOL "" FORCE)
set(TLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TLX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    pasta
    GIT_REPOSITORY https://github.com/pasta-toolbox/bit_vector.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(pasta)
set(PASTA_INCLUDE_DIR ${pasta_SOURCE_DIR}/include)

# --- GEF Library (Manual integration to avoid submodule CMake issues) ---
add_library(gef STATIC lib/generalized-elias-fano/src/gef.cpp)
target_include_directories(gef PUBLIC 
    lib/generalized-elias-fano/include
    ${PASTA_INCLUDE_DIR}
)
target_link_libraries(gef PUBLIC sdsl sux pasta_bit_vector)
# Add required flags for GEF
target_compile_options(gef PUBLIC -ffast-math -funroll-loops)

# --- ALP Library (vendored build, avoids ALP's standalone CMake restrictions) ---
add_library(neats_alp STATIC
    lib/ALP/src/falp.cpp
    lib/ALP/src/fastlanes_ffor.cpp
    lib/ALP/src/fastlanes_generated_ffor.cpp
    lib/ALP/src/fastlanes_generated_unffor.cpp
    lib/ALP/src/fastlanes_unffor.cpp
)
target_include_directories(neats_alp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/lib/ALP/include)
target_compile_options(neats_alp PUBLIC -fPIC)

# --- ALP benchmark TU (optional clang++ hybrid build) ---
set(NEATS_ALP_BENCH_OBJ "${CMAKE_CURRENT_BINARY_DIR}/alp_benchmark_clang.o")
set(NEATS_ALP_BENCH_SRC "${CMAKE_CURRENT_SOURCE_DIR}/benchmark/alp_benchmark.cpp")

# Comprehensive lossless compressor benchmark (with squash - optional)
option(NEATS_WITH_SQUASH "Build LosslessBenchmarkFull with Squash" ON)
option(NEATS_WITH_LECO "Build LeCo support (requires Boost headers + Eigen; compiled with GCC 11)" OFF)
option(NEATS_ALP_HYBRID_CLANG "Compile ALP benchmark TU with clang++ for AVX-512 while keeping GCC for the rest" ON)
set(NEATS_ALP_CLANGXX "" CACHE FILEPATH "Path to clang++ used for NEATS_ALP_HYBRID_CLANG (if empty, auto-detect)")
set(NEATS_LECO_EXTRA_INCLUDES "" CACHE STRING "Semicolon-separated extra include paths for LeCo clang build (e.g., for Boost/Eigen)")

if(NEATS_ALP_HYBRID_CLANG)
    # On macOS with GCC as main compiler, hybrid clang build is not possible because
    # Apple clang only supports libc++ while GCC uses libstdc++ (ABI incompatible).
    # Auto-disable in this case.
    set(NEATS_ALP_HYBRID_CLANG_EFFECTIVE ON)
    if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message(STATUS "NEATS_ALP_HYBRID_CLANG: Auto-disabled on macOS with GCC (Apple clang doesn't support libstdc++)")
        set(NEATS_ALP_HYBRID_CLANG_EFFECTIVE OFF)
    endif()

    if(NEATS_ALP_HYBRID_CLANG_EFFECTIVE)
        if(NEATS_ALP_CLANGXX)
            set(ALP_CLANGXX "${NEATS_ALP_CLANGXX}")
        else()
            find_program(ALP_CLANGXX NAMES clang++ clang++-18 clang++-17 clang++-16)
        endif()

        if(ALP_CLANGXX)
            # ARCH_FLAGS / COMMON_OPT_FLAGS are space-separated strings. Convert them into proper argument lists
            # so clang++ receives each flag as a separate argv entry.
            string(REPLACE " " ";" NEATS_ALP_ARCH_FLAGS_LIST "${ARCH_FLAGS}")
            string(REPLACE " " ";" NEATS_ALP_COMMON_OPT_FLAGS_LIST "${COMMON_OPT_FLAGS}")

            # On non-Apple platforms, when mixing clang with GCC, we need to use libstdc++ to match ABI.
            set(NEATS_ALP_STDLIB_FLAG "")
            if(NOT APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
                # Linux with GCC: force clang to use libstdc++ for ABI compatibility
                set(NEATS_ALP_STDLIB_FLAG "-stdlib=libstdc++")
            endif()

            add_custom_command(
                OUTPUT "${NEATS_ALP_BENCH_OBJ}"
                COMMAND "${ALP_CLANGXX}"
                    -c "${NEATS_ALP_BENCH_SRC}"
                    -o "${NEATS_ALP_BENCH_OBJ}"
                    "-I${CMAKE_CURRENT_SOURCE_DIR}/benchmark"
                    "-I${CMAKE_CURRENT_SOURCE_DIR}/lib/ALP/include"
                    ${NEATS_ALP_ARCH_FLAGS_LIST}
                    ${NEATS_ALP_COMMON_OPT_FLAGS_LIST}
                    -std=gnu++20
                    -fPIC
                    ${NEATS_ALP_STDLIB_FLAG}
                DEPENDS "${NEATS_ALP_BENCH_SRC}" "${CMAKE_CURRENT_SOURCE_DIR}/benchmark/benchmark_common.hpp"
                VERBATIM
            )
            add_custom_target(neats_alp_bench_obj DEPENDS "${NEATS_ALP_BENCH_OBJ}")
            set_source_files_properties("${NEATS_ALP_BENCH_OBJ}" PROPERTIES GENERATED TRUE)
        else()
            message(WARNING "NEATS_ALP_HYBRID_CLANG=ON but clang++ was not found; ALP benchmark will be compiled with the main compiler (AVX-512 may be disabled on GCC).")
        endif()
    endif()
endif()

# --- LeCo benchmark TU (compiled with Clang + libc++ to avoid libstdc++ type_traits conflicts) ---
set(NEATS_LECO_BENCH_OBJ "${CMAKE_CURRENT_BINARY_DIR}/leco_benchmark_clang.o")
set(NEATS_LECO_BENCH_SRC "${CMAKE_CURRENT_SOURCE_DIR}/benchmark/leco_benchmark.cpp")

if(NEATS_WITH_LECO)
    # Reuse the same clang++ as ALP (or find one)
    if(ALP_CLANGXX)
        set(LECO_CLANGXX "${ALP_CLANGXX}")
    elseif(NEATS_ALP_CLANGXX)
        set(LECO_CLANGXX "${NEATS_ALP_CLANGXX}")
    else()
        find_program(LECO_CLANGXX NAMES clang++ clang++-18 clang++-17 clang++-16)
    endif()

    if(LECO_CLANGXX)
        message(STATUS "LeCo will be compiled with: ${LECO_CLANGXX}")
        
        string(REPLACE " " ";" NEATS_LECO_ARCH_FLAGS_LIST "${ARCH_FLAGS}")
        string(REPLACE " " ";" NEATS_LECO_COMMON_OPT_FLAGS_LIST "${COMMON_OPT_FLAGS}")
        
        # Build include flags list
        set(NEATS_LECO_INCLUDE_FLAGS
            "-I${CMAKE_CURRENT_SOURCE_DIR}/benchmark"
            "-I${CMAKE_CURRENT_SOURCE_DIR}/lib/Learn-to-Compress/headers"
        )
        if(NEATS_LECO_EXTRA_INCLUDES)
            foreach(inc_path IN LISTS NEATS_LECO_EXTRA_INCLUDES)
                list(APPEND NEATS_LECO_INCLUDE_FLAGS "-isystem${inc_path}")
            endforeach()
        endif()

        add_custom_command(
            OUTPUT "${NEATS_LECO_BENCH_OBJ}"
            COMMAND "${LECO_CLANGXX}"
                -c "${NEATS_LECO_BENCH_SRC}"
                -o "${NEATS_LECO_BENCH_OBJ}"
                ${NEATS_LECO_INCLUDE_FLAGS}
                ${NEATS_LECO_ARCH_FLAGS_LIST}
                ${NEATS_LECO_COMMON_OPT_FLAGS_LIST}
                -std=gnu++17
                -fPIC
            DEPENDS "${NEATS_LECO_BENCH_SRC}" "${CMAKE_CURRENT_SOURCE_DIR}/benchmark/benchmark_common.hpp"
            VERBATIM
        )
        add_custom_target(neats_leco_bench_obj DEPENDS "${NEATS_LECO_BENCH_OBJ}")
        set_source_files_properties("${NEATS_LECO_BENCH_OBJ}" PROPERTIES GENERATED TRUE)
    else()
        message(FATAL_ERROR "NEATS_WITH_LECO=ON but clang++ was not found. Install clang or disable NEATS_WITH_LECO.")
    endif()
endif()

# Comprehensive lossless compressor benchmark (without squash)
add_executable(LosslessBenchmark benchmark/lossless_benchmark.cpp
        benchmark/NeaTS/NeaTS.hpp
        benchmark/NeaTS/algorithms.hpp)
target_link_libraries(LosslessBenchmark sdsl sux gef neats_alp)
if(TARGET neats_alp_bench_obj)
    add_dependencies(LosslessBenchmark neats_alp_bench_obj)
    target_sources(LosslessBenchmark PRIVATE "${NEATS_ALP_BENCH_OBJ}")
else()
    target_sources(LosslessBenchmark PRIVATE benchmark/alp_benchmark.cpp)
endif()

if(NEATS_WITH_SQUASH)
    set(SQUASH_ROOT "/usr/local")

    # 1. Find Headers (look for squash-0.7 folder)
    find_path(SQUASH_INCLUDE_DIR 
        NAMES squash/squash.h 
        PATHS "${SQUASH_ROOT}/include"
        PATH_SUFFIXES squash-0.8 squash-0.7 squash
        NO_DEFAULT_PATH
    )

    if(SQUASH_INCLUDE_DIR)
        message(STATUS "Found Squash headers: ${SQUASH_INCLUDE_DIR}")
        
        # ... (PkgConfig checks remain the same) ...
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(LZ4    REQUIRED IMPORTED_TARGET liblz4)
        pkg_check_modules(ZSTD   REQUIRED IMPORTED_TARGET libzstd)
        pkg_check_modules(SNAPPY REQUIRED IMPORTED_TARGET snappy)
        pkg_check_modules(BROTLI REQUIRED IMPORTED_TARGET libbrotlienc libbrotlidec libbrotlicommon)
        find_package(ZLIB  REQUIRED)
        find_package(BZip2 REQUIRED)
        find_package(LibLZMA REQUIRED)

        # 2. Find Library (Look for 0.7 or 0.8, in lib OR lib64)
        # We search specifically for the library name 'squash0.7' which 0.7.0 produces
        find_library(SQUASH_SHARED_LIB 
            NAMES squash0.8 squash0.7 squash
            PATHS "${SQUASH_ROOT}"
            PATH_SUFFIXES lib lib64
            NO_DEFAULT_PATH
        )

        if(NOT SQUASH_SHARED_LIB)
            message(FATAL_ERROR "Squash headers found but libsquash not found in ${SQUASH_ROOT}/lib or lib64")
        endif()

        # Import the found library
        add_library(Squash::Lib SHARED IMPORTED GLOBAL)
        set_target_properties(Squash::Lib PROPERTIES
            IMPORTED_LOCATION "${SQUASH_SHARED_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${SQUASH_INCLUDE_DIR}"
        )
        
        add_executable(LosslessBenchmarkFull
            benchmark/lossless_benchmark.cpp
            benchmark/NeaTS/NeaTS.hpp
            benchmark/NeaTS/algorithms.hpp
        )
        target_compile_definitions(LosslessBenchmarkFull PRIVATE USE_SQUASH GEF_DISABLE_OPENMP=1)

        target_link_libraries(LosslessBenchmarkFull
            PRIVATE
                sdsl
                sux
                gef
                neats_alp
                Squash::Lib
                PkgConfig::LZ4
                PkgConfig::ZSTD
                PkgConfig::SNAPPY
                PkgConfig::BROTLI
                ZLIB::ZLIB
                BZip2::BZip2
                LibLZMA::LibLZMA
                dl
        )
        if(TARGET neats_alp_bench_obj)
            add_dependencies(LosslessBenchmarkFull neats_alp_bench_obj)
            target_sources(LosslessBenchmarkFull PRIVATE "${NEATS_ALP_BENCH_OBJ}")
        else()
            target_sources(LosslessBenchmarkFull PRIVATE benchmark/alp_benchmark.cpp)
        endif()
        
        # --- Optional LeCo integration (compiled with Clang in separate TU) ---
        if(NEATS_WITH_LECO)
            find_library(GMP_LIBRARY gmp REQUIRED)
            target_compile_definitions(LosslessBenchmarkFull PRIVATE NEATS_ENABLE_LECO=1)
            target_link_libraries(LosslessBenchmarkFull PRIVATE ${GMP_LIBRARY})
            if(TARGET neats_leco_bench_obj)
                add_dependencies(LosslessBenchmarkFull neats_leco_bench_obj)
                target_sources(LosslessBenchmarkFull PRIVATE "${NEATS_LECO_BENCH_OBJ}")
            endif()
        endif()
        # Add RPATH so it looks in ./lib relative to binary
        target_link_options(LosslessBenchmarkFull PRIVATE "-Wl,-rpath,\$ORIGIN/lib")
        
        # --- LosslessBenchmarkFullNoSIMD: Same as Full but with SIMD and OpenMP disabled ---
        # This is a fallback for systems where AVX-512/OpenMP causes issues with GEF
        add_executable(LosslessBenchmarkFullNoSIMD
            benchmark/lossless_benchmark.cpp
            benchmark/NeaTS/NeaTS.hpp
            benchmark/NeaTS/algorithms.hpp
        )
        target_compile_definitions(LosslessBenchmarkFullNoSIMD PRIVATE 
            USE_SQUASH 
            GEF_DISABLE_SIMD=1 
            GEF_DISABLE_OPENMP=1
        )
        # Disable compiler auto-vectorization to avoid AVX-512 issues even when targeting skylake-avx512
        # This ensures truly scalar execution for debugging memory corruption issues
        target_compile_options(LosslessBenchmarkFullNoSIMD PRIVATE
            -fno-tree-vectorize      # Disable GCC tree vectorizer
            -fno-tree-slp-vectorize  # Disable SLP vectorizer  
        )
        target_link_libraries(LosslessBenchmarkFullNoSIMD
            PRIVATE
                sdsl
                sux
                gef
                neats_alp
                Squash::Lib
                PkgConfig::LZ4
                PkgConfig::ZSTD
                PkgConfig::SNAPPY
                PkgConfig::BROTLI
                ZLIB::ZLIB
                BZip2::BZip2
                LibLZMA::LibLZMA
                dl
        )
        if(TARGET neats_alp_bench_obj)
            add_dependencies(LosslessBenchmarkFullNoSIMD neats_alp_bench_obj)
            target_sources(LosslessBenchmarkFullNoSIMD PRIVATE "${NEATS_ALP_BENCH_OBJ}")
        else()
            target_sources(LosslessBenchmarkFullNoSIMD PRIVATE benchmark/alp_benchmark.cpp)
        endif()
        target_link_options(LosslessBenchmarkFullNoSIMD PRIVATE "-Wl,-rpath,\$ORIGIN/lib")

        if(NEATS_WITH_LECO)
            target_compile_definitions(LosslessBenchmarkFullNoSIMD PRIVATE NEATS_ENABLE_LECO=1)
            target_link_libraries(LosslessBenchmarkFullNoSIMD PRIVATE ${GMP_LIBRARY})
            if(TARGET neats_leco_bench_obj)
                add_dependencies(LosslessBenchmarkFullNoSIMD neats_leco_bench_obj)
                target_sources(LosslessBenchmarkFullNoSIMD PRIVATE "${NEATS_LECO_BENCH_OBJ}")
            endif()
        endif()
        
    endif()
endif()
