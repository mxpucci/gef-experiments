# Generalized Elias Fano Benchmarks

This repository contains benchmarking code and scripts for evaluating **Generalized Elias Fano (GEF)**.

The core implementation of Generalized Elias Fano is available here:  
👉 **[https://github.com/mxpucci/generalized-elias-fano](https://github.com/mxpucci/generalized-elias-fano)**

## 📋 Prerequisites

*   **CMake** (3.22 or higher)
*   **C++ Compiler** with C++20/23 support (GCC 11+, Clang 14+)
*   **Rust/Cargo 1.87+** (Pcodec, Fauna/Macaque; dependencies are locked)
*   **Python 3** (for smoke tests, plots and tables)
*   **Squash** (optional, for comparing against a wide range of compressors)

## 📥 Cloning the Repository

To clone this repository, make sure to include the `--recursive` flag to initialize all submodules:

```bash
git clone --recursive https://github.com/mxpucci/gef-experiments.git
cd gef-experiments
```

If you have already cloned the repository without the recursive flag, you can initialize the submodules manually:

```bash
git submodule update --init --recursive
```

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

## Competitors

We use each compressor's official implementation whenever one exists in C, C++, Rust or Go, imported at a pinned revision. Only when the authors published Java alone do we port their code to C++. The only differences from upstream are those needed to run the code here: build and binding glue, 64-bit inputs, and fixes for upstream bugs that would crash, produce undecodable output, or distort compressed size or throughput. Upstream bugs whose only effect is that some decoded values are wrong are left unpatched. The harness reports them as validation warnings (see *Validation* below).

| CLI name | Implementation | Access layout |
| --- | --- | --- |
| `alp` | Official C++ (cwida/ALP) | Native 1,024-value vectors |
| `buff` | Official Rust (Tranway1/buff), plain variant | Native, whole column |
| `camel` | C++ translation of the official Java (yoyo185644/camel) | 1,000-value blocks |
| `chimp`, `chimp128` | C++ implementations from the NeaTS authors' benchmark (and-gue/NeaTS) | 1,000-value blocks |
| `dac` | sdsl-lite `dac_vector_dp` | Native |
| `dexor` | C++ port of the official Java | 1,000-value blocks |
| `elf` | C++ port of the official Java (Spatio-Temporal-Lab/elf) | 1,000-value blocks |
| `elfstar`, `selfstar` | C++ ports of the official Java | 1,000-value blocks |
| `fauna`, `macaque` | Official Rust (PlatypusMacaque) | Fauna: native segments in 65,536-value batches; Macaque: 1,000-value blocks |
| `flea` | Official C++ | 1,000-value blocks |
| `gorilla` | Facebook Beringei codec, extracted verbatim | 1,000-value blocks |
| `leco` | Official C++ LeCo-fix (Learn-to-Compress) | Native partitions |
| `neats` | Official C++ (and-gue/NeaTS) | Native |
| `pcodec`, `pcodec_i64` | Official `pco_c` C bindings, v1.0.2 | 1,000-value blocks |
| `tsxor` | C++ implementation from the NeaTS authors' benchmark (and-gue/NeaTS) | 1,000-value blocks |
| `lz4`, `zstd`, `snappy`, `xz`, `brotli` | Squash 0.7 | 1,000-value blocks |

Codecs without native random access are measured on independent 1,000-value blocks (`-b`). A point query decodes the block that holds the value; a range query decodes the blocks it covers. All block directories, headers and metadata count toward compressed size.

### Deviations from upstream

- **ALP:** upstream kernels handle only complete 1,024-value vectors. As in the ALP authors' DuckDB integration, the last partial vector is padded with its last value, and only the real values are encoded and stored. The codec is compiled without fast-math, because fast-math folds away ALP's rounding step.
- **BUFF:** the authors' plain scalar code (Tranway1/buff @ d6550b2), extracted verbatim, one column per dataset; the BUFF-major variant is not used, because its selective decoder is unimplemented upstream. Known upstream encoder bugs do not trigger on our datasets. A bug in the selective decoder's trailing bits makes some point/range results wrong on 14 of 16 datasets without affecting size or throughput, so it is left unpatched and reported as a warning. BUFF stores values to the dataset's declared decimal precision.
- **Camel:** the authors published only Java, so we use a statement-by-statement C++ translation. With the authors' 4-decimal setting its output is byte-identical to theirs on all 16 datasets. Upstream's format stores at most 4 decimals; for datasets with more, the decimal-count field is widened and the remainder is encoded with the paper's own rule (Algorithm 2). Upstream's 16-bit integer-difference field is kept, so one LAT value decodes incorrectly, exactly as in upstream, and is reported as a warning.
- **DeXOR:** upstream is Java-only, so we use a C++ port that is byte-identical to the official code (SuDIS-ZJU/DeXOR@06128b3) on sampled blocks from all 16 datasets. That includes upstream's tolerance-based acceptance, so some values decode inexactly. On LAT, LON and AP, upstream's own decoding errors exceed 1e-9 and are reported as validation warnings. The only deviations cover inputs on which upstream crashes or writes a stream it cannot decode.
- **Elf\* / SElf\*:** C++ ports of the official Java (SElfStar@457ceb0); the Elf* output is byte-identical to Java on sampled blocks. The authors' only C++ Elf* (Serf/Crush) implements the paper's no-Huffman variant, so we do not use it. Our SElf* adds per-block restart checkpoints so that blocks decode independently. These are counted in compressed size.
- **Fauna / Macaque:** upstream supports float32 only. We build the authors' code with its `Value` type widened to f64 by a small patch applied at build time: 64-bit bit casts, Beringei's 64-bit Gorilla field layout, and value-size constants. The algorithms and defaults are unchanged. Fauna's Swing models are evaluated in f64, so some values (at most 0.03%) decode a few ulp off, within 1e-11. MacaqueV is bit-exact.
- **Falcon:** excluded, because its upstream repository is no longer available.
- **Gorilla:** the value codec is taken verbatim from Facebook's Beringei, the production implementation from the Gorilla authors' team. Only the containers and bit casts are adapted to compile without folly.
- **LeCo:** the authors' unmodified LeCo-fix codec (`Leco_int<uint64_t>`, @28bf8bf). Partition size is chosen by their released 1% sampling search (`search_hyper.h`, candidates 200 to 10,000), run once per dataset. The changes are the 64-bit adaptation needed to run it, and reseeding `rand()` before each search to reproduce upstream's one-dataset-per-process runs. Point reads use LeCo's per-element access, and the search time counts toward compression time.
- **Pcodec:** the official `pco_c` bindings from pcodec v1.0.2, built unmodified as a static library at the default level (8). Each block is a complete standalone `.pco` file.
- **Kangaroo, EAS:** not included; no implementation is available (see [docs/kangaroo.md](docs/kangaroo.md), [docs/eas.md](docs/eas.md)).

### Validation and floating-point input

Datasets store scaled integers. Codecs that take doubles receive `value / 10^decimals`, computed in a translation unit compiled with strict IEEE arithmetic, so that every double is the correctly rounded decimal. Competitor code is compiled without fast-math. After each run, decoded values are compared with the input: a value counts as a mismatch only if it differs by more than 1e-9. A mismatch logs a warning, and the metrics are still reported. `validation_status` and `roundtrip_mismatches` hold the result, and `bitwise_mismatches` gives the bit-level count for information.

### Building the full suite on Linux

`LosslessBenchmarkFull` adds the Squash codecs (LZ4, Zstd, Snappy, XZ, Brotli); LeCo needs GMP, Eigen and Boost. If these dependencies live outside `/usr/local`, pass their locations explicitly:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=$DEPS -DNEATS_WITH_SQUASH=ON -DNEATS_WITH_LECO=ON \
  -DSQUASH_INCLUDE_DIR=$SQUASH/include/squash-0.7 \
  -DSQUASH_SHARED_LIB=$SQUASH/lib64/libsquash0.7.so \
  "-DNEATS_LECO_EXTRA_INCLUDES=$DEPS/include;$DEPS/include/eigen3"
cmake --build build --target LosslessBenchmark LosslessBenchmarkFull -j
```

At run time, set `SQUASH_PLUGINS` to Squash's plugin directory. `scripts/build_remote.sh` builds the minimal configuration (no Squash, no LeCo) with a user-owned GCC 13.4 and Rust 1.87 toolchain.

## 🚀 Running Benchmarks

You can run the benchmarks using the provided helper script. It requires a directory containing binary data files (`.bin`).

```bash
./run_benchmarks.sh <directory_with_bin_files>
```

The results will be saved in the `benchmark_results/` directory as text files.

## 📂 Datasets

The datasets used for benchmarking are stored as a **split zip archive** to accommodate file size limits. They are managed via **Git LFS** and excluded from the default clone.

### Downloading and Extracting
To get the datasets, you can use the provided helper script:

```bash
./install_datasets.sh
```

Alternatively, you can manually pull and extract them:

1.  **Download the archive:**
    ```bash
    git lfs pull --include="datasets/datasets_archive.*"
    ```

2.  **Extract the archive:**
    You need to combine the split files. Use `7z` or `unzip`:
    ```bash
    # Using 7zip (Recommended)
    7z x datasets/datasets_archive.zip

    # OR using unzip (if your version supports split archives)
    unzip datasets/datasets_archive.zip
    ```
    This will extract the binary files into the `datasets/` directory.

### Dataset Credits
We gratefully acknowledge the sources of the datasets used in this benchmark:

*   **IT (IR Sensor Temp)**: [NEON (DP1.00005.001)](https://doi.org/10.48443/7RS6-FF56)
*   **US / UK / GE (Stock Exchanges)**: [INFORE Project](https://doi.org/10.5281/zenodo.3886895)
*   **ECG**: [Zheng et al.](https://doi.org/10.13026/wgex-er52)
*   **WD (Wind Direction)**: [NEON (DP1.00001.001)](https://doi.org/10.48443/77N6-EH42)
*   **AP (Barometric Pressure)**: [NEON (DP1.00004.001)](https://doi.org/10.48443/ZR37-0238)
*   **LAT / LON (GPS)**: [Microsoft Research (GeoLife)](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/)
*   **DP (Dew Point)**: [NEON (DP1.20271.001)](https://doi.org/10.48443/1W06-WM51)
*   **CT (City Temp)**: [University of Dayton (via Kaggle)](https://www.kaggle.com/datasets)
*   **DU (PM10 Dust)**: [NEON (DP1.00017.001)](https://doi.org/10.48443/RDZ9-XR84)
*   **BW (Basel Wind) / BT (Basel Temp)**: [Meteoblue](https://www.meteoblue.com/)
*   **BM (Bird Migration) / BP (Bitcoin)**: [InfluxData](https://github.com/influxdata/influxdb2-sample-data)

## 💾 Dataset Format

The benchmark expects binary files (`.bin`) containing 64-bit integers. Two formats are supported:

**1. Standard Format (with Decimals parameter):**
*   **Header (16 bytes):**
    *   `uint64_t` : Number of values ($N$)
    *   `uint64_t` : Decimals parameter (used to reconstruct the decimal values passed to floating-point compressors)
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
