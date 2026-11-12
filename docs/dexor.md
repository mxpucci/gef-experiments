# DeXOR integration

This is a C++ port of the **default Native DeXOR** codec, with one previous
decimal value, `rho=8`, no buffered-window extension, and no automatic switch
to exception-only mode. The paper's experiments also disable the latter
prior-knowledge mode. It is a sequential codec: any point-access benchmark
must scan its enclosing block from the beginning and report that fact.

Sources read:

* Lv et al., *DeXOR: Enabling XOR in Decimal Space for Streaming Lossless
  Compression of Floating-point Data*, VLDB 2026,
  [paper](https://www.vldb.org/pvldb/vol19/p849-li.pdf), especially §§4–5.
* [Official implementation](https://github.com/SuDIS-ZJU/DeXOR), pinned for this
  port to commit `06128b3df57cc1e8c53681231e235e22eb508c8d`:
  `src/main/java/algorithms/DeXOR/DeXORTools.java`,
  `encoder/DoubleDeXOREncoder.java`, and `decoder/DoubleDeXORDecoder.java`.

The official implementation is Java; no official C++, Rust, or Go DeXOR
implementation was found. The bundled TsFile C++ directory does not implement
DeXOR. Consequently this uses the user-authorized native port rather than a
Java runtime dependency. The upstream MIT notice is preserved in
`benchmark/DeXOR/LICENSE`.

## API and accounting

Include `benchmark/DeXOR/DeXOR.hpp`. Namespace `gef_compressors::dexor` exposes:

```cpp
std::vector<uint8_t> compress(const std::vector<double>& input);
std::vector<double> decompress(const std::vector<uint8_t>& bytes, size_t count);
std::vector<double> decompress_prefix(const std::vector<uint8_t>& bytes,
                                      size_t total_count, size_t prefix_count);
```

The serialized representation begins with an eight-byte big-endian count,
followed by the DeXOR bitstream, including all control bits, coordinate
metadata, exception data, and final zero byte-padding. Count **all returned
bytes** as compressed storage. Decoder `count` is checked against the stored
count; it is not uncounted metadata. Empty input occupies eight bytes.
There is no external dictionary or model. Reset codec state per benchmark
block, consistently with the other sequential competitors.

For point/range queries, `decompress_prefix` stops immediately after the
requested number of values, matching the existing streaming benchmarks. It
checks the total count in the header and rejects prefixes exceeding it. A
zero-length prefix reads only the header. Proper prefixes do not validate the
unread suffix; when the prefix equals the total count, the same complete
framing checks as `decompress` apply. This is sequential prefix scanning,
not native random access.

Compile this codec in a separate translation unit with C++17 or later and
`-fno-fast-math -ffp-contract=off`. GEF's exported `-ffast-math` must not affect
this TU. Disabling contraction ensures reconstruction arithmetic agrees
across compilers and machines. The header rejects `__FAST_MATH__` builds.


## Fidelity to upstream

The encoder and decoder follow `DoubleDeXOREncoder`/`DoubleDeXORDecoder` and
`DeXORTools` statement by statement, including upstream's **tolerance-based
acceptance** of decimal encodings (`comp(alpha + beta*pow, value, pow)`,
`DoubleDeXOREncoder.java:113`). Decoding is therefore not always bit-exact,
exactly as with the official Java. Java semantics are reproduced explicitly:
saturating `(long)` casts and `Math.round` (NaN → 0, ties toward +∞), the
`1e-23`/`1e-6` epsilons, unbounded tail search, and the high-precision tail
search (`getEnd_HP`, used when the previous tail exponent is below −12) with a
re-implementation of JDK 19+ `Double.toString` length/decimal-point position.
Fixed-width fields keep only their low bits, as upstream's `StreamWriter` does.

Known upstream behaviours that are therefore reproduced:

* Most inexact values differ by about one ULP (`alpha + sign*beta*10^q` rounding).
* The tail search treats values within `1e-6` of an integer multiple as
  integral, e.g. `19.0000005 → 19` and `39999.997 → 40000` (when the previous
  tail exponent is 4).
* The decoder re-applies the prefix's sign to the residual
  (`DoubleDeXORDecoder.java:109`), while only `|beta|` is written
  (`DoubleDeXOREncoder.java:120`). A value just below a carry of the shared
  prefix (previous `40000.002`, value `39999.997`) is reconstructed on the
  wrong side (`→ 40000.003`); upstream's acceptance test uses the signed beta
  and does not notice.
* The encoder derives prefixes from the original previous value while the
  decoder uses its decoded previous value, so an inexact value can perturb
  the following ones (e.g. `39999.987 → 40000.087` after the case above).
* NaN is treated as zero (`comp(NaN, 0) == 0`) and decodes as `+0.0`; `-0.0`
  decodes as `+0.0`.

On the full benchmark datasets the second to fourth behaviours cause errors above
1e-9 on LAT, LON and AP (see Validation). Following the benchmark policy
(patch only issues that prevent meaningful aggregate metrics), they are kept
and reported as validation warnings.

## Deviations from upstream

All deviations apply only where upstream would crash or emit an undecodable
stream:

* **Index guard.** Upstream indexes its `10^k` table without bounds checks and
  throws `ArrayIndexOutOfBoundsException` for some inputs (infinities, very
  small or large magnitudes, tail search running past `10^±23`), aborting the
  whole compression. Where Java would throw, the port encodes that value with
  DeXOR's own exception encoding (lossless) instead. Java output is unchanged
  for every input on which Java does not throw.
* **Tail-coordinate field.** Upstream writes `q+20` into 5 bits unchecked. For
  `q ∉ [-20, 11]` the field wraps and its own decoder loses synchronisation for
  the rest of the block. Such values use the exception encoding instead.
* **Decoder contraction counter.** Upstream's encoder resets `contract_step` on
  every exponent overflow, its decoder only while `EL < 10`; the port's decoder
  always resets it so both state machines agree (otherwise the decoder can lose
  synchronisation after ten overflows). This does not change the bitstream.
* **Framing and validation.** An 8-byte count header (counted as compressed
  size) and decoder checks for truncation, count mismatch, impossible
  coordinates/exponents, trailing bytes and nonzero padding; malformed input
  throws instead of reading out of bounds. Prefix decoding as above.

Earlier revisions of this port required bit-exact reconstruction, bounded the
tail search to `[-20, 11]`, dropped the `Double.toString` path, and rejected
over-wide suffixes. These were removed to follow the official implementation.

## Validation

`benchmark_results/20261109-dexor-upstream/` contains the comparison against
the unmodified official Java (OpenJDK 21) and its harnesses:

* 299 blocks of 1,000 values (298,746 values) from all 16 benchmark datasets:
  payload byte-identical to stock Java on 299/299 blocks, and decoded values
  identical to the stock Java decoder. 56,976 values (19%) decode inexactly,
  all within 2.3e-16 relative error.
* 3,009 synthetic vectors (decimal series of 0–16 decimals, high-precision
  values exercising `getEnd_HP`, large integers, random bit patterns, specials):
  byte-identical and decoder-identical on all 1,771 vectors where Java does not
  throw; Java throws on the other 1,238, which the port encodes instead.
* Full datasets (server, GCC 13.4, `full_upstream.csv`): 13 of 16 datasets decode
  within 2.3e-16 relative error. Values above 1e-9: LAT 135,738 (max 0.32),
  AP 5,648 (max 0.0101), LON 3,130 (max 0.01), out of 24.9M/137.7M/24.9M.
  Size: 6.6–30.7 bits/value, versus 11.6–36.2 for the previous exact port.

`tests/dexor_roundtrip.cpp` checks two bitstreams generated by the official Java
encoder (the second also pins the official decoder's output, including the
high-precision path and inexact values), tolerance-bounded round trips on
decimal data, the documented anomalies above, exact infinities, agreement of
prefix and full decoding, and rejection of malformed streams. It passes with
Apple Clang (ASan/UBSan) and GCC 13.4 `-O3 -march=native`.

```sh
c++ -std=c++17 -O2 -fno-fast-math -ffp-contract=off \
    tests/dexor_roundtrip.cpp -o dexor_roundtrip
./dexor_roundtrip
```
