# ALP integration

The codec is the pinned cwida/ALP submodule at
31ca0ed11c93c99d3f5b5c30e01a3e1c3832d3ce. The existing build-time fixes for
two AVX-512 intrinsic vector-type typos remain; no codec algorithm is rewritten.
The adapter uses the same upstream calls as the two scheme
branches in `lib/ALP/benchmarks/benchmark.cpp`: encoder sampling and scheme
selection, ALP or ALP-RD encoding, and FastLanes packing. The sampling call
differs from benchmark.cpp's own arguments (see below).
ALP uses the original fused FALP decoder and exception patcher. ALP-RD uses the
original unpackers and `rd_encoder::decode`. No substitute all-exceptions codec
is used. Dictionaries are fitted and stored once per native row group.

Native defaults are unchanged: 1024 values/vector and 100 vectors/row group.
Generic benchmark block size does not affect ALP. Sampling calls
`init(base, rg_start, n)` with the base column, absolute row-group offset and
total value count n. benchmark.cpp instead passes a shifted row-group pointer
together with an offset; we do not copy that call. The upstream sampler
handles a partial last vector (it skips one with fewer than 32 values unless
nothing else was sampled).

Upstream does not provide a complete indexed column file API. Integration is
still required: the authors' end-to-end example uses three streams and three
offsets per vector. We retain that design. The compact metadata is actually
materialized through upstream MemStorer; it counts:

- 8-byte total value count;
- one scheme byte per row group;
- for ALP-RD groups, two width bytes, one dictionary-count byte, and two bytes
  per dictionary entry;
- three 8-byte stream offsets per vector;
- 13 codec metadata bytes per ALP vector, or 2 per ALP-RD vector;
- exact packed payload bytes, exceptions and their positions.

ALP exceptions cost 8+2 bytes each; ALP-RD exceptions cost 2+2 bytes each.
The dictionary, offsets and metadata are not omitted to improve reported ratios.
C++ object padding, unused allocation capacity and fitting scratch space are not
serialized storage. This remains adapter-inclusive storage, not upstream's
payload-only ratio estimator or resident-memory usage.

Partial last vector: the standalone cwida/ALP kernels process only complete
1024-value vectors, and its benchmarks drop the remainder. The authors'
production integration (DuckDB `alp`/`alprd`) instead encodes the last vector
with its real count n: exp/fac search, exceptions and frame of reference over
n values, then bit-packing of n values rounded up to a 32-value group. The
adapter follows that approach with the upstream kernels:

- the n real values are padded to 1024 by repeating the last one. Padding adds
  no new minimum or maximum, and exceptions at positions < n are unchanged;
- the vector-level exp/fac search covers only the n values through upstream's
  `state::vector_size` field;
- pad exceptions (positions >= n) are dropped;
- only the FastLanes words that hold values [0, n) are stored, i.e.
  ceil(ceil(n/L)*bw/W) words in each of L lanes (16 lanes of 64-bit words, or
  64 lanes of 16-bit words for ALP-RD left parts). Before decoding, the stored
  words are copied into a zero-filled full block for the upstream unpackers.

This also covers columns shorter than one vector. The value count n comes from
the stored total count. No verbatim tail is kept.

On the full 16 datasets (Linux, GCC 13, -march=native, AVX-512 path), the
bitwise round trip, point reads at n-1 and range reads ending at n all pass.
Compared with the verbatim tail, sizes shrink by 11.6% on BP, 5.9% on BM and
0.3-0.5% on BT and BW, and by less than 0.1% elsewhere. US grows by 21 bytes:
its 18-value tail costs one 16-row FastLanes block plus 37 metadata bytes, more
than 144 raw bytes. See benchmark_results/20261109-alp-tail/sizes.csv.

The input converter and ALP adapter compile without fast-math or LTO, with FP
contraction disabled. Full output is checked bitwise outside timing; mismatches
remain warnings with validation_status=failed. Point/range validation also
flags failures. Timing counts executed queries, including partial-vector queries.

Validation: tests/alp_adapter.cpp exercises decimals, random binary64 patterns
and constants at tiny, vector, row-group and multiple-row-group boundaries.
It checks full bitwise validation and point/range results, including indices
in the partial vector. The size check calls the same upstream encoder with the
same padding, so it re-derives the adapter's own formula rather than giving an
independent bound. For the partial vector it also checks two things on the
real values only: that padding does not widen the frame of reference, and the
ALP-RD exception counts, using a scalar dictionary lookup.
The test is registered as `alp_adapter` in the main CMake build.

The previous ALP measurements are superseded pending rerun: they substituted
exceptions for ALP-RD, omitted the tail, and undercounted metadata. An
intermediate version stored the tail verbatim. No correction
factor can repair those measurements. Full-dataset validation and new timings
are required before publication.

Integer-overflow handling: upstream decoder::decode_value is also called by
find_top_k_combinations while evaluating candidate exponent/factor pairs. Some
candidate integer products overflow (e.g. -50 * 10^18). We compile ALP with
-fwrapv to define signed wrapping arithmetic without patching the codec. Local
ASan/UBSan checks of the adapter and inline upstream encoder pass with this
flag. A separate Linux comparison found identical sizes, exception counts and
scheme choices with and without -fwrapv on all 16 sample inputs. The linked
generated packing library in that local check is not instrumented.

The index is reconstructed from the counted metadata before decoding, so tests
cannot rely on uncounted fitting state. Tests include mixed ALP/RD row groups,
point/range reads across boundaries, and optional native-format sample files.
Linux GCC 13 validation passed on 5,507,968 values from 32 head/tail samples
covering all 16 datasets; BP, BM, BT and BW fit entirely within the sample limit.
These checks are correctness/size evidence, not published performance results.

An isolated Linux flag audit fed identical precomputed binary64 input to the
original encoder with strict FP flags versus -ffast-math. The latter increased
payload size on 12/16 dataset prefixes (up to 65.4% on DP). This is a separate
problem from input conversion: unsafe optimization inside ALP can change its
encoding decisions even when the input bits are correct. See the evidence in
benchmark_results/20261105-alp-fidelity. These are prefix payload comparisons,
not corrections that can be applied to the old full-dataset ratios.
