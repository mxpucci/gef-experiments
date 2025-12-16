cmake_minimum_required(VERSION 3.22)
project(NeaTS)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_POLICY_DEFAULT_CMP0048 NEW)

# Detect architecture and set appropriate flags
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    # Apple Silicon / ARM64 - use mcpu instead of march for GCC
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=native -fno-math-errno -O3")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -D_FILE_OFFSET_BITS=64 -O3 -funroll-loops -mcpu=native")
else()
    # x86_64
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native -mfpmath=sse -mavx512f -fno-math-errno -ldl")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -D_FILE_OFFSET_BITS=64 -m64 -O3 -funroll-loops -lm -lrt -ldl")
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
include_directories(/usr/local/include/squash-0.7)
link_directories(/usr/local/lib)
if(EXISTS "/usr/local/include/squash-0.7")
    add_executable(LosslessBenchmarkFull benchmark/lossless_benchmark.cpp
            benchmark/NeaTS/NeaTS.hpp
            benchmark/NeaTS/algorithms.hpp)
    target_compile_definitions(LosslessBenchmarkFull PRIVATE USE_SQUASH)
    target_link_libraries(LosslessBenchmarkFull sdsl squash0.7 sux)
endif()
