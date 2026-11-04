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
no rounding, quantization or fallback compressor is used by that strict API.
The benchmark explicitly converts its input to float32 before calling
the API. Inexact conversions produce one warning per dataset/codec and metrics
are still emitted, with `input_conversion=float32` and `input_changed_values`
recording how many original float64 values changed. This preprocessing is timed.
Validation compares decoded values against the converted float32 input, expanded
to the benchmark's double representation. A passing validation therefore does
not claim lossless preservation of the original float64 or fixed-decimal input.
The conversion can lose decimal digits as well as changing binary precision.
Fauna's model fitting treats signed zero and NaNs
numerically, so its adapter additionally rejects negative zero and nonfinite
values. Standalone MacaqueV's native float32 API preserves their bit patterns.

The value-series adapter supplies synthetic regular timestamps `0..n-1` because
the GEF files contain values, not original timestamps. Fauna still encodes its
timestamp information using the author's implementation. No claim about
compression of a dataset's real timestamps is made. Full Fauna now uses its native segments for point/range access. Its default
fitting batch is 65,536 values, matching the original ModelarDB ingestion
batch. `--fauna-batch-size` changes that batch; external `-b` does not.
A counted positional directory selects only overlapping segments. Constant
(PMC) and linear (Swing) models evaluate only the requested synthetic timestamps
through the authors' original model functions. Macaque/Gorilla segments decode
the necessary prefix of that segment. Residual tails decode their necessary
prefix using the model's last value as the original predictor seed. A whole
batch may be a Gorilla fallback, so this is native segment access, not uniform
constant-time access to every value. No decoded values are retained between
queries. Decoder allocation, prefix work and output expansion are timed.

Standalone MacaqueV remains sequential and uses external `-b` blocks.

## Byte accounting

The standalone serialization APIs retain their original frames. MacaqueV uses a 16-byte magic/version/count header followed by the exact upstream
compressed bytes. Fauna uses a 24-byte magic/version/count/segment-count header.
Every segment stores its model id (1 byte), start/end timestamps (16 bytes),
minimum/maximum (8 bytes), and three byte vectors (timestamps, model values,
residuals), each with an 8-byte length. Thus segment overhead is 49 bytes plus
the three payloads. All bytes returned by `compress` are counted. IDs and error
fields are not serialized: the adapter always uses series id zero and error
zero, and these fields are not needed to decode values. This is a self-contained
codec frame, not an Arrow or Parquet file; no Arrow allocation overhead is
misreported as encoded storage.


The benchmark's indexed Fauna container instead has a 32-byte header and a
32-byte directory entry per native segment (first global position, batch base,
payload offset and payload length), followed by the unchanged 49-byte-plus-payload
segment representations. All bytes count; there is no additional external
1,000-value block wrapper. CSV records `access_layout=native_segments`, the
configured batch limit in `compression_unit_values`, and `native_page_values=0`.
Older full-block Fauna measurements are superseded for this comparison.

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

`tests/recent/native_access.cpp` compares indexed queries with original full
Fauna decoding on the same fitting batches, including explicit PMC/Swing
residual-tail fixtures, segment/batch boundaries and serialized reopening.
The CLI test checks that external `-b` does not affect Fauna's native layout.
