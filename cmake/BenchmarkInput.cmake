include_guard(GLOBAL)
add_library(gef_benchmark_input STATIC
    "${CMAKE_CURRENT_LIST_DIR}/../benchmark/benchmark_input.cpp")
target_compile_features(gef_benchmark_input PUBLIC cxx_std_17)
target_include_directories(gef_benchmark_input PUBLIC "${CMAKE_CURRENT_LIST_DIR}/../benchmark")
# Preserve division even when the benchmark caller uses fast-math and LTO.
set_property(TARGET gef_benchmark_input PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
target_compile_options(gef_benchmark_input PRIVATE -fno-fast-math -ffp-contract=off -fno-lto)
