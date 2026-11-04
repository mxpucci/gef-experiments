# Pcodec integration

The adapter imports the authors' **`pco = 1.0.2` Rust crate** from crates.io;
no compressor code is copied or rewritten. `Cargo.lock` pins all dependencies
and their registry checksums. The published crate records source commit
`2d8555888b21bbaa19326580b740fa24b7da6bd3` (`pco/`). Upstream is <https://github.com/pcodec/pcodec>;
the crate is Apache-2.0 licensed. This is a 2026 implementation of the algorithm
in Loncaric, Jeppesen, and Zinberg, *Pcodec: Better Compression for Numerical
Sequences*, arXiv:2502.06112v2 (25 May 2025). The paper's modes, delta encoding,
and binning are provided by upstream unchanged. The 1.0.2 release is pinned
rather than an evolving main branch.

The upstream repository also has experimental C bindings (`pco_c`), but those
are an unpublished 0.0.0 package and described upstream as incomplete. The
small local ABI adapter instead imports the published Rust API directly. It
handles C++ buffers, exact-length checks, and panic containment; it implements
no compression algorithm. It requires **Rust/Cargo 1.87 or newer**. The remote
machine's original Rust 1.72 is insufficient. C++11 suffices for this adapter
(the enclosing GEF benchmark has its own newer compiler requirements).

## API and representation

Include `Pcodec/Pcodec.hpp` and link the CMake target `gef_pcodec` after
`include(cmake/Pcodec.cmake)`.

```cpp
using namespace gef_compressors::pcodec;
auto bytes = compress(std::vector<double>{1.25, -2.5});
auto restored = decompress(bytes, 2);
auto integer_bytes = compress_i64(std::vector<std::int64_t>{1, -2});
auto restored_integers = decompress_i64(integer_bytes, 2);
```

Both functions store a complete official standalone `.pco` stream, including
header, type, chunk metadata, page state, and footer. `bytes.size()` counts
all encoded bytes. No original values or uncounted metadata are retained.
The externally supplied decode count bounds allocation and is validated;
the stream remains independently decodable by upstream tools. All f64 bits,
including signed zeros, infinities, subnormals, and NaN payloads, are preserved.
The integer API avoids loss from converting int64 values to double.

## Native page access in the benchmark

The benchmark uses `Pcodec/Indexed.hpp` and the original **wrapped** API,
for both `pcodec` (f64) and `pcodec_i64`. The standalone functions above remain
available, but are no longer used to implement benchmark point/range queries.
The compression layout matches the pinned authors' `pco_cli/src/bench/codecs/pcopage.rs`:
level 8, automatic mode/delta selection, balanced chunks and balanced pages
using `PagingSpec::EqualPagesUpTo`. Both native size limits default to 262,144.
`--pcodec-chunk-size` controls statistical chunks and `--pcodec-page-size`
controls independently decodable pages within them. External `-b` does not
change either size. Smaller native pages can be evaluated without forcing
1,000-value statistical chunks; the chosen settings must accompany results.

A counted directory locates the requested page and its shared chunk metadata.
The original `PageDecompressor` starts at that page and consumes only the
necessary prefix, in its required 256-value batches. It cannot seek directly
to an arbitrary value inside a page. Full pages and contiguous requested spans
are decoded directly into the output buffer. Each query includes native
metadata/decoder initialization; chunk metadata is reused across pages within
a range query. No expanded decoder tables or decoded values persist between
queries. Allocation, prefix decoding and result copying are timed.

The self-contained indexed frame contains a 64-byte header, 48 bytes per
chunk, 40 bytes per page, the original wrapped file header, original chunk
metadata and original page bytes. All of these bytes count toward compressed
size. This container is new integration code, not the standalone `.pco` file
format; the compression and decompression algorithms are unchanged upstream.
`access_layout=native_pages`, `compression_unit_values` and `native_page_values`
record the layout and configured limits in CSV (actual balanced sizes may be
smaller). Older benchmark runs using external 1,000-value standalone streams
are superseded for this comparison.

## Build and checks

```sh
cmake -S tests/pcodec -B build-pcodec -DCMAKE_BUILD_TYPE=Release
cmake --build build-pcodec -j2
ctest --test-dir build-pcodec --output-on-failure
```

`GEF_PCODEC_CARGO` can select a newer Cargo executable. For a machine-specific
benchmark build, set `-DGEF_PCODEC_RUSTFLAGS="-C target-cpu=native"`; the default
is portable Rust code generation. Do not move a native build to a CPU lacking
those instruction sets. Cargo requires access to crates.io on the first build
and uses `--locked` thereafter. Building on the destination Linux system
avoids relying on the local ARM machine's binaries or a newer glibc ABI.

The standalone C++ test checks empty and singleton inputs, constants, extreme
integers, all floating-point bit classes, random raw 64-bit patterns, decimal
sequences, and data crossing the default chunk boundary. It also rejects
wrong types/counts, every truncated prefix of a valid small file, and trailing
bytes. It uses explicit checks that remain active in release builds.

The main/recent build also runs `native_access`: out-of-order point/range reads,
multiple pages per chunk, default-size boundaries, f64 bit patterns, int64
extremes, reopening from serialized bytes, and reading a later page after an
earlier page's payload is damaged. The CLI test verifies that changing `-b`
does not change native compressed size.
