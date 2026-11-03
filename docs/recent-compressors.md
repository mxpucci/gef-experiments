# Recent compressors: integration and measurement

The implementation goal is partially complete. Fauna/Macaque, DeXOR, Pcodec,
Elf*, SElf* and FLEA have buildable adapters and benchmark entries. Kangaroo and
EAS require author source artifacts; they have no placeholder implementation.

| Entry | Source and provenance | Representation | Access through this adapter |
| --- | --- | --- | --- |
| `fauna` | [Official Rust import](fauna.md) | Exact float32 subset of input doubles | Full block, original models |
| `macaque` | [Official Rust import](fauna.md) | Exact float32 subset of input doubles | Full block, standalone MacaqueV |
| `dexor` | [C++ port of official Java](dexor.md) | Float64 | Prefix scan within block |
| `pcodec` | [Official Rust crate](pcodec.md) | Float64 | Full block |
| `pcodec_i64` | [Official Rust crate](pcodec.md) | Unshifted int64 | Full block |
| `elfstar` | [C++ port of official Java](elfstar.md) | Float64 | Prefix scan within block |
| `selfstar` | [C++ port of official Java](selfstar.md) | Float64 | Checkpoint restoration and prefix scan |
| `flea` | [Official C++ import](flea.md) | Unshifted, already scaled int64 | Full block/FFT |
| Kangaroo | [Source audit](kangaroo.md) | Pending | Pending |
| EAS | [Source audit](eas.md) | Pending | Pending |

Each requested compressor received a separate agent task; Elf* and SElf* shared
only their common C++ primitives. Each task read its supplied paper and checked
implementation provenance. Native implementations are imported with locked
versions or commit/checksum pins. Java is not needed to run any benchmark.

## Data and correctness

`pcodec_i64` and `flea` operate on the original int64 values before the harness
subtracts a minimum or converts fixed-decimal values to doubles. The other
entries receive the same float64 view as existing floating-point competitors.
The new harness validates decoded values bit for bit before emitting a CSV
result. This does not redefine the pre-existing int64-to-float64 conversion:
values not exactly representable by that view should be evaluated using an
integer entry when preservation of original int64 values matters.

Fauna/Macaque remain float32 implementations. Their double adapters explicitly
reject inexact narrowing. Full Fauna also rejects negative zero and nonfinite
values because its original model fitting does not preserve those bit patterns.
FLEA has documented count and magnitude bounds protecting unchecked upstream
integer arithmetic. Unsupported inputs produce an error and a nonzero process
status, rather than another codec's result. All new CSV entries therefore have
explicit representation limits; an absent result is not a zero-size result.

Full Fauna imports the original fitting/model pipeline and supplies synthetic
regular index timestamps, since input files contain values only. Both its
encoded timestamp/model metadata and Macaque residual bytes count.

## Measurement conventions

Independent blocks follow the existing `-b` option (default 1,000). Compressed
size includes every returned codec byte plus an explicit block directory:
16 bytes for total count and block size, and `8*(number_of_blocks+1)` bytes for
byte offsets. Allocator capacity and temporary scratch buffers are excluded.
The block directory remains part of the reported size even for codecs whose
payload redundantly stores its own count. These are storage sizes, not peak
memory measurements. Existing competitors retain their original accounting;
those historically omit some directory bytes, so small-block comparisons should
consider that difference (about 0.064 bits/value at blocks of 1,000).

SElf* uses one encoder per dataset so its statistics survive block boundaries.
Every independently decoded block stores all required decoder state, and that
checkpoint counts toward its size. Its internal adaptation score excludes the
access wrapper, as in the official streaming encoder. See its format document.

Point/range access uses a block directory. DeXOR/Elf*/SElf* stop decoding at the
requested index/end, matching the existing streaming competitors' prefix-scan
approach. Pcodec, Fauna/Macaque and FLEA currently use their full-block APIs.
None of these wrappers is described as native constant-time individual access.
Reported query counts are the actual number executed. Range output copying and
all API allocations are timed. Full decompression is separately timed after an
untimed complete reconstruction check.

Compression includes the C++ block-vector copy, allocation, imported API calls,
FFI/conversion costs and codec framing. Decompression includes output allocation
and adapter expansion to the benchmark representation. These are reproducible
adapter-inclusive timings, not the papers' isolated core-kernel rates. Float
throughput uses the harness's 8-byte logical input/output view, including for
Fauna/Macaque's eligible float32 values; it is not their native float32 byte
throughput. Consult per-codec documents for further restrictions.

The default 1,000-value block can substantially penalize Pcodec's compression
ratio; its authors recommend larger chunks. Use an additional larger-block
experiment for full-scan compression (for example `-b 262144`), and report that
block size and access cost, rather than treating it as equivalent to the
1,000-value access experiment.

New numerical codecs compile in a separate translation unit with
`-fno-fast-math -ffp-contract=off`. Those options do not propagate to GEF. The
pinned ALP AVX-512 header has two vector-type typos; CMake corrects only those
types in a generated build-tree copy so native GCC compilation succeeds.
The ALP submodule and its algorithms remain at the original revision.

## Reproducing builds and checks

On the target Linux x86-64 account:

```sh
./scripts/build_remote.sh
./build/LosslessBenchmark -c dexor,pcodec,pcodec_i64,elfstar,selfstar,flea -b 1000 data.bin
```

The helper provisions user-owned GCC/G++ 13.4 and Rust 1.87 compatible with the
server's glibc 2.17, builds the main benchmark and runs CTest. CMake >=3.22,
Git, curl, tar and initial network access are needed. It disables optional
Squash/LeCo, enables BUFF and the new codecs, and leaves system packages alone.
The original GitHub Actions workflow remains usable; compilation does not
require transferring ARM binaries to Linux.

The adapter tests can also run without the older baseline dependencies:

```sh
cmake -S tests/recent -B build-recent -DCMAKE_BUILD_TYPE=Release
cmake --build build-recent -j4
ctest --test-dir build-recent --output-on-failure
```

The full build additionally runs `recent_cli_smoke`: all eight new CLI entries,
short final blocks, cross-block queries, CSV rows, the exact int64 route,
float32 eligibility errors, and unknown-codec failure. Codec tests cover source
bitstream fixtures where available, special numerical values, changing SElf*
windows, prefix access and invalid inputs. Rust and FLEA tests exercise the
actual pinned upstream implementations. Checks remain active in Release.
