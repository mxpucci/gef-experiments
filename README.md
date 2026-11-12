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

## Recent compressor integrations

The new entries are available in `LosslessBenchmark` and the optional Full
variants. Native dependencies are imported at pinned revisions; the Java-only
algorithms have documented C++ ports.

| CLI name | Implementation | Input |
| --- | --- | --- |
| `fauna`, `macaque` | Official Rust Fauna and standalone MacaqueV | Float32 conversion; changed values produce warnings and flagged metrics |
| `dexor` | C++ port of official DeXOR | Float64 |
| `pcodec` | Official Rust Pcodec 1.0.2 | Float64 |
| `pcodec_i64` | Same official Pcodec | Original int64 values |
| `elfstar`, `selfstar` | C++ ports of official Elf*/SElf* | Float64 |
| `flea` | Official C++ FLEA with FFTW | Original scaled int64 values |

**Kangaroo and EAS are deferred at the user’s request** and are outside this
integration’s current scope. Their source searches and the details missing
from the papers are recorded in
[docs/kangaroo.md](docs/kangaroo.md) and [docs/eas.md](docs/eas.md).

For example:

```sh
./build/LosslessBenchmark -c dexor,pcodec,pcodec_i64,elfstar,selfstar,flea -b 1000 data.bin
ctest --test-dir build --output-on-failure
```

The Fauna/Macaque benchmark explicitly narrows doubles to float32 and continues
with metrics when values change. It logs a warning and records `input_conversion`
and `input_changed_values` in CSV. Validation checks the actual converted input;
passing that check does not imply preservation of the original float64 values.
All recent codecs report `validation_status` and `roundtrip_mismatches`; value
mismatches warn and still emit measured metrics. The standalone codec APIs remain
strict. FLEA rejects blocks outside its safe numeric envelope. Hard API errors,
malformed decoded lengths and unknown codecs fail the affected case and give a
nonzero exit status while the CLI continues subsequent cases. Unmeasurable
operations do not receive invented metrics.

Full Fauna and both Pcodec entries now use native segment/page access. `-b`
does not split their compression units: Fauna defaults to 65,536-value fitting
batches; Pcodec defaults to 262,144-value chunks/pages. Use
`--fauna-batch-size`, `--pcodec-chunk-size` and `--pcodec-page-size` to configure
these separately. CSV records the layout and native limits. All stored
directories and metadata count toward compressed size.

SElf* carries adaptation across external blocks and stores counted checkpoints
for independent decoding. Sequential codecs retain the external `-b` wrapper. See [docs/recent-compressors.md](docs/recent-compressors.md)
for measurement conventions and links to each implementation's provenance,
input restrictions and tests.

On the Linux x86-64 server, the following builds with a user-owned GCC 13.4 and
Rust 1.87 toolchain (no root access or GitHub Actions required):

```sh
./scripts/build_remote.sh
```

This helper builds the main benchmark with all new codecs and BUFF. Optional
Squash and LeCo are disabled because they require additional system libraries;
their existing build options remain available. Native CPU flags target the
build machine. CMake and network access are needed initially. To build without
individual fetched codecs, set `GEF_WITH_PCODEC`, `GEF_WITH_FAUNA`, or
`GEF_WITH_FLEA` to `OFF`. Full test builds default to `BUILD_TESTING=ON`.

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
