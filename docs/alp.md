# ALP integration

The codec is the pinned cwida/ALP submodule at
31ca0ed11c93c99d3f5b5c30e01a3e1c3832d3ce. The existing build-time fixes for
two AVX-512 intrinsic vector-type typos remain; no codec algorithm is rewritten.
The adapter follows the two scheme
branches in `lib/ALP/benchmarks/benchmark.cpp`: original encoder sampling and
scheme selection, original ALP or ALP-RD encoding, and original FastLanes packing.
ALP uses the original fused FALP decoder and exception patcher. ALP-RD uses the
original unpackers and `rd_encoder::decode`. No substitute all-exceptions codec
is used. Dictionaries are fitted and stored once per native row group.

Native defaults are unchanged: 1024 values/vector and 100 vectors/row group.
Generic benchmark block size does not affect ALP. Sampling uses the original
sampler API with the base column, absolute row-group offset and total complete
vector count (not a shifted pointer combined with an absolute offset).

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

The original kernels process complete vectors. The adapter stores the final
0..1023 values verbatim, counts every byte and serves reads from that retained
tail. This explicit boundary policy is our integration choice, not an upstream
partial-vector algorithm. It also handles columns shorter than one vector.
Compression timing includes retaining the tail and creating metadata.

The input converter and ALP adapter compile without fast-math or LTO, with FP
contraction disabled. Full output is checked bitwise outside timing; mismatches
remain warnings with validation_status=failed. Point/range validation also
flags failures. Timing counts executed queries, including tail queries.

Validation: tests/alp_adapter.cpp exercises decimals, random binary64 patterns
and constants at tiny, vector, row-group and multiple-row-group boundaries.
It checks full bitwise validation, point/range results and independently
recomputes encoded lengths through the original upstream encoder calls.
The test is registered as `alp_adapter` in the main CMake build.

The previous ALP measurements are superseded pending rerun: they substituted
exceptions for ALP-RD, omitted the tail, and undercounted metadata. No correction
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
