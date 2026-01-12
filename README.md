# Generalized Elias Fano Benchmarks

This repository contains benchmarking code and scripts for evaluating **Generalized Elias Fano (GEF)**.

The core implementation of Generalized Elias Fano is available here:  
👉 **[https://github.com/mxpucci/generalized-elias-fano](https://github.com/mxpucci/generalized-elias-fano)**

## 📋 Prerequisites

*   **CMake** (3.22 or higher)
*   **C++ Compiler** with C++20/23 support (GCC 11+, Clang 14+)
*   **Python 3** (for generating plots and tables)
*   **Squash** (optional, for comparing against a wide range of compressors)

## 🛠️ Building the Benchmarks

To build the project, run the following commands:

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

This will produce the benchmark executables, primarily:
*   `LosslessBenchmark`: Minimal benchmark binary.
*   `LosslessBenchmarkFull`: Full benchmark suite including comparisons with other compressors (requires Squash).

## 🚀 Running Benchmarks

You can run the benchmarks using the provided helper script. It requires a directory containing binary data files (`.bin`).

```bash
./run_benchmarks.sh <directory_with_bin_files>
```

The results will be saved in the `benchmark_results/` directory as text files.

## 💾 Dataset Format

The benchmark expects binary files (`.bin`) containing 64-bit integers. Two formats are supported:

**1. Standard Format (with Decimals parameter):**
*   **Header (16 bytes):**
    *   `uint64_t` : Number of values ($N$)
    *   `uint64_t` : Decimals parameter (used by some float compressors like Camel/Falcon)
*   **Body:**
    *   $N \times$ `int64_t` : The data values

**2. Simple Format:**
*   **Header (8 bytes):**
    *   `uint64_t` : Number of values ($N$)
*   **Body:**
    *   $N \times$ `int64_t` : The data values

The benchmark automatically detects the format based on the file size.

## 📊 Generating Plots and Tables

After running the benchmarks, you can generate visualizations and LaTeX tables using the provided scripts.

### Plots
Generates Pareto charts and throughput comparisons.

```bash
./generate_plots.sh [benchmark_results.csv] [output_dir]
```
*   Default input: `benchmark_results/results.csv`
*   Default output: `plots/`

### Tables
Generates LaTeX tables for compression ratios and speeds.

```bash
./generate_tables.sh [benchmark_results.csv] [output_dir]
```
*   Default input: `benchmark_results/results.csv`
*   Default output: `tables/`

## 📂 Project Structure

*   `benchmark/` - Benchmark source code and compressor implementations.
*   `lib/` - Third-party libraries and dependencies (including GEF).
*   `scripts/` - Python scripts for data analysis and plotting.
*   `NeaTS/` - Header files for the benchmarking framework.

## 🔗 References

*   **Generalized Elias Fano Implementation:** [mxpucci/generalized-elias-fano](https://github.com/mxpucci/generalized-elias-fano)
*   **Original Benchmark Suite:** This repository is a follow-up to the benchmark suite originally implemented in [NeaTS](https://github.com/and-gue/NeaTS).
