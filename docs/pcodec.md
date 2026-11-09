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

## Benchmark layout

Pcodec does not support random access to individual values, so both `pcodec`
(f64) and `pcodec_i64` use the harness's independent external blocks, like the
other codecs without native random access (DeXOR, Elf*, Macaque, FLEA). Each
`-b` block (default 1,000 values) is one complete standalone `.pco` file
produced by the official `simple_compress_into` with the default
`ChunkConfig` (level 8, automatic mode/delta selection). A point query
decodes its whole block with the official decompressor and selects the value; a
range query decodes the blocks it covers. Compressed size counts every `.pco`
byte plus the shared block directory (16 bytes and 8 bytes per block offset);
see [recent-compressors.md](recent-compressors.md). CSV rows report
`access_layout=external_blocks` and `compression_unit_values` equal to `-b`.

Very small standalone files carry fixed header/chunk metadata, so 1,000-value
blocks compress worse than Pcodec's large default chunks. This is the stated
fairness convention for codecs without random access, not an adapter change to
the algorithm. An interim layout with 262,144-value native chunks/pages and a
custom page directory has been removed; its results are superseded.

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

The main/recent build also runs a CLI smoke test checking that both Pcodec
entries are validated on `-b` blocks.
