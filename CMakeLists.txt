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
    # Apple Silicon / ARM64
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=native ${COMMON_OPT_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcpu=native ${COMMON_OPT_FLAGS} -Wall -D_FILE_OFFSET_BITS=64")
else()
    # x86_64 - enable AVX-512 and all SIMD extensions available on native CPU
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native ${COMMON_OPT_FLAGS} -mfpmath=sse")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -mtune=native ${COMMON_OPT_FLAGS} -mfpmath=sse -Wall -D_FILE_OFFSET_BITS=64 -m64")
endif()

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
    set(SQUASH_ROOT "/usr/local")
    set(SQUASH_INCLUDE_DIR "${SQUASH_ROOT}/include/squash-0.8")

    if(EXISTS "${SQUASH_INCLUDE_DIR}/squash/squash.h")
        find_package(PkgConfig REQUIRED)

        # If you ALSO want these codec deps to be static, set this ON and ensure .a exist:
        # set(PKG_CONFIG_USE_STATIC_LIBS ON)

        pkg_check_modules(LZ4    REQUIRED IMPORTED_TARGET liblz4)
        pkg_check_modules(ZSTD   REQUIRED IMPORTED_TARGET libzstd)
        pkg_check_modules(SNAPPY REQUIRED IMPORTED_TARGET snappy)
        pkg_check_modules(BROTLI REQUIRED IMPORTED_TARGET libbrotlienc libbrotlidec libbrotlicommon)

        find_package(ZLIB  REQUIRED)
        find_package(BZip2 REQUIRED)
        find_package(LibLZMA REQUIRED)

        # ---- Force Squash static (.a) if available ----
        set(_old_suffixes "${CMAKE_FIND_LIBRARY_SUFFIXES}")
        set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
        find_library(SQUASH_STATIC_LIB NAMES squash0.8 PATHS "${SQUASH_ROOT}/lib" NO_DEFAULT_PATH)
        set(CMAKE_FIND_LIBRARY_SUFFIXES "${_old_suffixes}")

        if(SQUASH_STATIC_LIB)
            add_library(Squash::squash0_8 STATIC IMPORTED GLOBAL)
            set_target_properties(Squash::squash0_8 PROPERTIES
                IMPORTED_LOCATION "${SQUASH_STATIC_LIB}"
                INTERFACE_INCLUDE_DIRECTORIES "${SQUASH_INCLUDE_DIR}"
            )
            message(STATUS "Using static Squash: ${SQUASH_STATIC_LIB}")
            set(SQUASH_TARGET Squash::squash0_8)
        else()
            # Fallback to shared if static is not present
            find_library(SQUASH_SHARED_LIB NAMES squash0.8 PATHS "${SQUASH_ROOT}/lib" NO_DEFAULT_PATH)
            if(NOT SQUASH_SHARED_LIB)
                message(FATAL_ERROR "Squash headers found but no libsquash0.8 found in ${SQUASH_ROOT}/lib")
            endif()

            add_library(Squash::squash0_8 SHARED IMPORTED GLOBAL)
            set_target_properties(Squash::squash0_8 PROPERTIES
                IMPORTED_LOCATION "${SQUASH_SHARED_LIB}"
                INTERFACE_INCLUDE_DIRECTORIES "${SQUASH_INCLUDE_DIR}"
            )
            message(WARNING "Static Squash not found; using shared: ${SQUASH_SHARED_LIB}")
            set(SQUASH_TARGET Squash::squash0_8)
        endif()

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
                ${SQUASH_TARGET}
                PkgConfig::LZ4
                PkgConfig::ZSTD
                PkgConfig::SNAPPY
                PkgConfig::BROTLI
                ZLIB::ZLIB
                BZip2::BZip2
                LibLZMA::LibLZMA
                dl
        )

        # If (and only if) you ended up on shared Squash, keep your RPATH trick:
        if(NOT SQUASH_STATIC_LIB)
            target_link_options(LosslessBenchmarkFull PRIVATE "-Wl,-rpath,\$ORIGIN/lib")
        endif()
    endif()
endif()
