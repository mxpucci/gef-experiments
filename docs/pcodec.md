# Pcodec integration

The adapter calls the authors' official **`pco_c` C bindings** from
<https://github.com/pcodec/pcodec> at the `v1.0.2` release commit
`2d8555888b21bbaa19326580b740fa24b7da6bd3`. That tag is the `pco` 1.0.2
release, so `pco_c` and the `pco` crate it wraps are the same source revision
(no later commit is used). CMake downloads that tarball (SHA-256 pinned) and
builds it unmodified with upstream's own `Cargo.lock` (`--locked`). The one
build-time adjustment is `cargo rustc -p cpcodec --crate-type staticlib`:
upstream declares a Rust `dylib`, and a static library links more simply into
the C++ benchmark. No source file is changed. No compressor code is copied or
rewritten. The project is Apache-2.0 licensed. This is a 2026 implementation of
the algorithm in Loncaric, Jeppesen, and Zinberg, *Pcodec: Better Compression
for Numerical Sequences*, arXiv:2502.06112v2 (25 May 2025).

Upstream labels `pco_c` an unpublished 0.0.0 package with incomplete
bindings. The three standalone functions used here are complete:
`pco_standalone_guarantee_file_size`, `pco_standalone_simple_compress_into` and
`pco_standalone_simple_decompress_into`. The benchmark passes a null config,
which selects pco_c's defaults: compression level 8, default paging and
automatic mode/delta. Those defaults match `ChunkConfig::default()` for 64-bit
types. pco_c also sets `enable_8_bit`, but that affects only 8-bit types.
Output is byte-identical to the earlier direct-crate adapter on all 16
benchmark datasets. `Pcodec/Pcodec.hpp` is a header-only C++ wrapper. It
manages buffers, checks the decoded count and passes non-null pointers for
empty inputs. Building requires **Rust/Cargo 1.87 or newer** (the `pco`
`rust-version`). C++11 suffices for the wrapper.

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
The externally supplied decode count bounds the output and is validated. Like
upstream `simple_decompress`, bytes after the standalone footer are ignored.
The benchmark always decodes exactly the bytes that compression returned. All f64 bits,
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

`GEF_PCODEC_CARGO` can select a newer Cargo executable. The Rust build uses
upstream's default release profile. For a machine-specific
benchmark build, set `-DGEF_PCODEC_RUSTFLAGS="-C target-cpu=native"`; the default
is portable Rust code generation. Do not move a native build to a CPU lacking
those instruction sets. Cargo requires access to crates.io on the first build
and uses `--locked` thereafter. Building on the destination Linux system
avoids relying on the local ARM machine's binaries or a newer glibc ABI.

The standalone C++ test checks empty and singleton inputs, constants, extreme
integers, all floating-point bit classes, random raw 64-bit patterns, decimal
sequences, and data crossing the default chunk boundary. It also rejects
wrong types/counts and every truncated prefix of a valid small file. It uses explicit checks that remain active in release builds.

The main/recent build also runs a CLI smoke test checking that both Pcodec
entries are validated on `-b` blocks.
