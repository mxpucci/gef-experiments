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

add_library(sux INTERFACE)
target_include_directories(sux INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/lib/sux)

add_library(sdsl INTERFACE)
target_include_directories(sdsl INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/lib/sdsl-lite/include)

# Comprehensive lossless compressor benchmark (without squash)
add_executable(LosslessBenchmark benchmark/lossless_benchmark.cpp
        benchmark/NeaTS/NeaTS.hpp
        benchmark/NeaTS/algorithms.hpp)
target_link_libraries(LosslessBenchmark sdsl sux)

# Comprehensive lossless compressor benchmark (with squash - optional)
option(NEATS_WITH_SQUASH "Build LosslessBenchmarkFull with Squash" ON)

if(NEATS_WITH_SQUASH)
    # Prefer Homebrew prefix on macOS; fall back to /usr/local elsewhere.
    if(APPLE)
        set(SQUASH_ROOT "/opt/homebrew" CACHE PATH "Squash install prefix")
    else()
        set(SQUASH_ROOT "/usr/local" CACHE PATH "Squash install prefix")
    endif()

    # 1. Find Headers (look for squash-0.7 folder)
    find_path(SQUASH_INCLUDE_DIR 
        NAMES squash/squash.h 
        PATHS "${SQUASH_ROOT}/include"
        PATH_SUFFIXES squash-0.8 squash-0.7 squash
        NO_DEFAULT_PATH
    )

    if(NOT SQUASH_INCLUDE_DIR)
        message(STATUS "Squash headers not found under ${SQUASH_ROOT}/include. Skipping LosslessBenchmarkFull. (Set -DSQUASH_ROOT=<prefix> to enable)")
    else()
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
    find_library(SQUASH_SHARED_LIB 
        NAMES squash0.8 squash0.7 squash
        PATHS "${SQUASH_ROOT}"
        PATH_SUFFIXES lib lib64
        NO_DEFAULT_PATH
    )

        if(NOT SQUASH_SHARED_LIB)
            message(STATUS "Squash headers found but libsquash not found under ${SQUASH_ROOT}/lib or ${SQUASH_ROOT}/lib64. Skipping LosslessBenchmarkFull. (Set -DSQUASH_ROOT=<prefix> to enable)")
        else()

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
            target_compile_definitions(LosslessBenchmarkFull PRIVATE USE_SQUASH)

            target_link_libraries(LosslessBenchmarkFull
                PRIVATE
                    sdsl
                    sux
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
            # Add RPATH so it looks in ./lib relative to binary (Linux); macOS uses @executable_path.
            if(APPLE)
                target_link_options(LosslessBenchmarkFull PRIVATE "-Wl,-rpath,@executable_path/lib")
            else()
                target_link_options(LosslessBenchmarkFull PRIVATE "-Wl,-rpath,\$ORIGIN/lib")
            endif()
        endif()
    endif()
endif()
