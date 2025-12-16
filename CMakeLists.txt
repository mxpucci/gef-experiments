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
# Requires squash library to be installed
include_directories(/usr/local/include/squash-0.8)
link_directories(/usr/local/lib)
if(EXISTS "/usr/local/include/squash-0.8")
    add_executable(LosslessBenchmarkFull benchmark/lossless_benchmark.cpp
            benchmark/NeaTS/NeaTS.hpp
            benchmark/NeaTS/algorithms.hpp)
    target_compile_definitions(LosslessBenchmarkFull PRIVATE USE_SQUASH)
    target_link_libraries(LosslessBenchmarkFull sdsl squash0.8 sux)
endif()
