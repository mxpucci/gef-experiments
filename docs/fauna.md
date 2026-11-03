# Fauna and MacaqueV

The integration compiles the author's Rust implementation from
[aabduvakhobov/PlatypusMacaque](https://github.com/aabduvakhobov/PlatypusMacaque),
commit `26ea89142f0395927d0922a9b29b2524c08849e0`, under the upstream Apache-2.0
license. The implementation corresponds to **Compressing High-Frequency Time
Series Through Multiple Models and Stealing From Residuals**, ICDE 2026,
DOI [10.1109/ICDE65706.2026.00025](https://doi.org/10.1109/ICDE65706.2026.00025).

`cmake/Fauna.cmake` fetches the archive with a SHA-256 checksum. The Rust build
uses original files from `ModelarDB-versions/ModelarDB-Macaque`; algorithms are
neither copied into this repository nor rewritten. `build.rs` imports their
modules by source path because upstream keeps the standalone MacaqueV module
private. Only FFI, input checks and a compact serialization envelope are new.
The original source headers and repository license remain in the fetched tree.

Two separate benchmark entries are available:

- `gef_compressors::fauna`: the full Platypus fitting pipeline, PMC/Swing models,
  MacaqueV residuals and MacaqueTS segment timestamps, at absolute error zero.
- `gef_compressors::macaque`: standalone MacaqueV, including the author's swapped
  Gorilla flags, at absolute error zero. At zero error the author's code disables
  value and bit rewriting. This entry does not pretend to be the full Fauna.

The paper and implementation use **float32**. Both namespaces provide native
`compress_f32(vector<float>)` / `decompress_f32(bytes, count)` functions. The
`vector<double>` adapters accept only values that survive a float32 round trip
bit for bit. A value such as double `0.1`, or integer `16777217`, is rejected;
no rounding, quantization or fallback compressor is used. The benchmark should
report an unsupported input rather than produce a misleading result when a
dataset fails eligibility. Fauna's model fitting treats signed zero and NaNs
numerically, so its adapter additionally rejects negative zero and nonfinite
values. Standalone MacaqueV's native float32 API preserves their bit patterns.

The value-series adapter supplies synthetic regular timestamps `0..n-1` because
the GEF files contain values, not original timestamps. Fauna still encodes its
timestamp information using the author's implementation. No claim about
compression of a dataset's real timestamps is made. Neither adapter claims
native random access; an individual access needs block decompression.

## Byte accounting

MacaqueV uses a 16-byte magic/version/count header followed by the exact upstream
compressed bytes. Fauna uses a 24-byte magic/version/count/segment-count header.
Every segment stores its model id (1 byte), start/end timestamps (16 bytes),
minimum/maximum (8 bytes), and three byte vectors (timestamps, model values,
residuals), each with an 8-byte length. Thus segment overhead is 49 bytes plus
the three payloads. All bytes returned by `compress` are counted. IDs and error
fields are not serialized: the adapter always uses series id zero and error
zero, and these fields are not needed to decode values. This is a self-contained
codec frame, not an Arrow or Parquet file; no Arrow allocation overhead is
misreported as encoded storage.

## Building and tests

Include `cmake/Fauna.cmake` and link the target `gef_fauna`. Rust 1.87 is the
integration's selected toolchain minimum; upstream uses `LazyLock`, which
requires at least Rust 1.80. Arrow and transitive dependency versions are pinned
in the committed `lib/fauna-ffi/Cargo.lock`, and Cargo runs with `--locked`.
Chrono is pinned to 0.4.39 to avoid its later `quarter()` conflict with Arrow 53.
No server, Java, Python, or GPU runtime is required.

`tests/fauna_roundtrip.cpp` exercises both pipelines at empty/singleton and
segment/block boundary lengths, repeated/linear/periodic/random inputs, bitwise
MacaqueV random float patterns including nonfinite values, float64 eligibility
and wrong-count rejection. It uses explicit checks, so Release builds retain
validation. Round trips should also be checked on every benchmark dataset.

The FFI catches upstream Rust panics and returns an error; it does not silently
substitute another compressor. As with the research implementation, its frame
decoder is intended for trusted benchmark output, not hostile files.
