# FLEA integration

This adapter imports the authors' native C++ implementation from
[Mr-Spade/FLEA](https://github.com/Mr-Spade/FLEA), pinned to commit
`47a9a480b34f5c82b6602e2d7aa217ae52aeca05`. The code is fetched at configure
time with a SHA-256-verified archive; no compression implementation is copied
into this repository or rewritten. The paper is Xia, Xiao and Song,
*FLEA: Frequency-based Lossless Encoding Algorithm for Periodic Time Series*,
SIGMOD 2026, [DOI 10.1145/3802063](https://doi.org/10.1145/3802063).

The imported `algorithms/flea/include/flea/encoder.hpp` supplies the original
DFT, quantization search, dense/sparse laminar coefficient partitioning,
and hybrid residual encoding. The default search is the implementation's
`MIN_BETA=11` through `MAX_BETA=20` (the paper describes a broader possible
search space). All implementation defaults remain unchanged.

## API and numeric representation

Include `FLEA/FLEA.hpp` and link `gef_flea` after `include(cmake/FLEA.cmake)`:

```cpp
auto bytes = gef_compressors::flea::compress_i64(std::vector<int64_t>{125, -250});
auto values = gef_compressors::flea::decompress_i64(bytes, 2);
```

Use GEF's **raw, already scaled int64 sequence**, before converting to doubles
or subtracting the minimum. Paper §5.1.2 applies native-decimal scaling before
compression. This adapter does no additional scaling, rounding, or narrowing;
the integer residual preserves the input values. Arbitrary IEEE floating-point
bit patterns are not this codec's input representation.

The returned bytes are the complete original FLEA bitstream, including its
30-bit count, FFT padding indicator/length, beta, partitions, frequency data,
residual metadata, and padding bits. Charge all bytes plus the enclosing
benchmark's block directory. There is no native point-access API: a point query
requires decoding its enclosing block.

The upstream implementation performs unchecked double-to-int64 conversions
and signed arithmetic. To avoid accepting values for which those operations
can overflow, the adapter explicitly limits each block to at most 1,048,576
values and `sum(abs(value)) <= 2^50`. It rejects inputs outside this conservative
safe envelope with an explanatory exception; it never switches to another
compressor. These limits do not claim the authors designed FLEA for only this
range. The safe envelope bounds spectral coefficients and reconstruction
magnitudes well below int64 overflow. Normal fixed-decimal time-series values
are tested, along with the accepted envelope boundary and rejected int64
extremes. The main benchmark additionally validates exact decoded integers
before emitting metrics.

FLEA's decoder assumes its own well-formed compressed stream. The adapter
checks count, truncated headers and FFT padding metadata, but does not turn
upstream into a hardened parser for arbitrary attacker-controlled bitstreams.

## Build and reproducibility

`cmake/FLEA.cmake` fetches and statically builds official **FFTW 3.3.10**, pinned
by SHA-256, because the author's current FFT header requires FFTW even though
its build script mentions pocketfft. On x86-64, FFTW SSE2/AVX2 kernels are enabled;
its planner selects supported kernels. FLEA itself follows the compiler's AVX2
setting. Both encode and decode must use the same SIMD/FFT implementation:
upstream uses ties-to-even rounding in AVX2 lanes and `std::round` in its scalar
branches, and numerical FFT implementations can differ at integer rounding
boundaries. This is a benchmark for a fixed executable/architecture, not a
promise of cross-build bitstream interchange.

The sole source patch guards upstream's unconditional `<immintrin.h>` include
on x86 platforms. Existing scalar branches then compile on ARM. This patch is
applied to the fetched build tree, not a vendored copy. The adapter compiles as
C++20 with `-fno-fast-math -ffp-contract=off`, independently of GEF's optimization
flags. A mutex protects the original FFTW plan cache's shared scratch buffers;
the benchmark remains single-threaded.

The FLEA repository supplies no project-level license for its own codec at
this pinned commit; no license is invented here. FFTW is distributed under its
upstream GPL license, retained in the fetched source tree.

```sh
cmake -S tests/flea -B build-flea -DCMAKE_BUILD_TYPE=Release
cmake --build build-flea -j4
ctest --test-dir build-flea --output-on-failure
```

Tests cover exact integer round trips on empty/singleton blocks, constants,
signed values, periodic data with residual noise, random data, FFT padding and
multiple lengths. They also compare the adapter's bytes to direct calls to the
authors' encoder and decode them using that original API. Checks stay active
in release builds.
