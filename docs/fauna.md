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
private. Only FFI, input checks, a compact serialization envelope and the
float64 patch below are new. The original source headers and repository license
remain in the fetched tree.

Two separate benchmark entries are available:

- `gef_compressors::fauna`: the full Platypus fitting pipeline, PMC/Swing models,
  MacaqueV residuals and MacaqueTS segment timestamps, at absolute error zero.
- `gef_compressors::macaque`: standalone MacaqueV, including the author's swapped
  Gorilla flags, at absolute error zero. At zero error the author's code disables
  value and bit rewriting. This entry does not pretend to be the full Fauna.

## Float64 values

The paper and the upstream implementation use **float32**: upstream defines
`pub type Value = f32` (`modelardb_types/src/types.rs`). The benchmark datasets
are float64, and converting them to float32 changes most values, so the build
widens that alias to f64. `lib/fauna-ffi/upstream-f64.patch` is applied by
`PATCH_COMMAND` to the pristine checksummed archive; configuration fails if the
fetched tree is not patched. The patch (17 hunks, 4 files) only changes what
hard-codes the 32-bit representation:

- `Value`/`ArrowValue` become `f64`/`Float64Type`; value byte sizes follow
  from `size_of::<Value>()` upstream, except the swing parameter decoder's
  literal lengths 5 and 8, which become 9 and 16 (flag + value, two values).
- Gorilla/MacaqueV: `u32`/`f32` bit casts become `u64`/`f64`. The control fields
  keep the Gorilla paper's 5-bit leading-zero and 6-bit length widths, using the
  standard 64-bit layout of Facebook's Beringei (the same layout used by this
  repository's Gorilla codec): leading zeros are capped at 31 and the
  meaningful-bit length is stored minus one, so 64 fits in 6 bits. Upstream's
  flag order (`10` for a repeated value), first-value handling and residual
  seeding are unchanged.
- The lossy bit-rewriting helpers (unused at error bound zero) use binary64
  constants: 52 mantissa bits, 11 exponent bits, bias 1023.
- Two type mismatches caused by the alias (`f32::NAN` and a relative-bound
  comparison) are cast to `Value`.

The algorithms and defaults are unchanged: error bound zero, Swing/PMC/Gorilla
model selection, MacaqueV residuals and the 65,536-value fitting batch. The
model selection still compares bytes per value against `size_of::<Value>()`,
which is now 8 bytes. Values cross the FFI as `double`; no float32 conversion
is performed, `input_conversion=none` and `input_changed_values=0`. Validation
compares decoded values with the original float64 input. Compression ratios use
64 bits per value like every other codec.

Upstream fits float32 values in f64 arithmetic, which leaves headroom that the
float64 version does not have. Two consequences are documented, not patched,
because they do not prevent meaningful aggregate metrics; any affected values
are reported by validation as mismatches:

- A Swing line is decoded as `slope * t + intercept` in f64. At error bound zero
  the fitted line can reproduce a value only to within f64 rounding, so a decoded
  Swing value may differ from the input in its last bits (e.g. 0.1 decoded as
  0.10000000000000002 for the series `0, 0.1, ..., 0.6`). PMC values and MacaqueV
  residuals are exact. Validation records such values; tests require 1e-9.
- Magnitudes near `f64::MAX` (above about 2^1000) can overflow the fitting
  arithmetic (`inf - inf`) and decode as NaN; no benchmark dataset is near that
  range. As before, the Fauna adapter rejects nonfinite values and negative
  zero, which its model fitting identifies numerically. Standalone MacaqueV
  preserves all 64-bit patterns.

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
The float64 frames use version 002 (`GFMAC002`, `GFFAU002`, `GFFIDX02`).
Every segment stores its model id (1 byte), start/end timestamps (16 bytes),
minimum/maximum (16 bytes), and three byte vectors (timestamps, model values,
residuals), each with an 8-byte length. Thus segment overhead is 57 bytes plus
the three payloads. All bytes returned by `compress` are counted. IDs and error
fields are not serialized: the adapter always uses series id zero and error
zero, and these fields are not needed to decode values. This is a self-contained
codec frame, not an Arrow or Parquet file; no Arrow allocation overhead is
misreported as encoded storage.


The benchmark's indexed Fauna container instead has a 32-byte header and a
32-byte directory entry per native segment (first global position, batch base,
payload offset and payload length), followed by the unchanged 57-byte-plus-payload
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
segment/block boundary lengths, repeated/linear/periodic/random and decimal
inputs, bitwise MacaqueV random float64 patterns including nonfinite values,
float64 values that float32 cannot represent, negative-zero rejection and
wrong-count rejection. It uses explicit checks, so Release builds retain
validation. Round trips should also be checked on every benchmark dataset.

The FFI catches upstream Rust panics and returns an error; it does not silently
substitute another compressor. As with the research implementation, its frame
decoder is intended for trusted benchmark output, not hostile files.

`tests/recent/native_access.cpp` compares indexed queries with original full
Fauna decoding on the same fitting batches, including explicit PMC/Swing
residual-tail fixtures, segment/batch boundaries and serialized reopening.
The CLI test checks that external `-b` does not affect Fauna's native layout.
