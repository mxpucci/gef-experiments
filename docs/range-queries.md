# Range-query benchmark

This benchmark measures how fast each codec returns `range` consecutive
decoded values that start at a random position. It follows the NeaTS paper
(Fig. 4: 10,000 random ranges per size, sizes 10·2^0 … 10·2^16, averaged over
the 11 largest datasets). Results are reported as throughput in decimal
MB/s instead of queries/s.

## Running

```sh
# one binary call (any codecs, any datasets)
build/LosslessBenchmark -c rle_gef,neats,alp -r 10,20,40,...,655360 \
    --range-queries 10000 --range-seed 1234 --range-trials 3 \
    --range-trials-csv trials.csv -o results.csv datasets/CT.bin

# campaign: one isolated, resumable process per (dataset, codec), then aggregate + plot
scripts/run_range_queries.py --binary build/LosslessBenchmarkFull \
    --dataset-dir datasets --largest 11 \
    --codecs neats,dac,rle_gef,b_star_gef_optimal,alp,lz4 \
    --trials 3 --taskset 4 --out results/range-queries

# re-aggregate / re-plot (directories from the driver and/or trials CSVs)
scripts/plot_range_queries.py results/range-queries --codecs rle_gef,neats,alp
```

Range-query options of `LosslessBenchmark*`:

| option | default | meaning |
|---|---|---|
| `-r <list>` | `10,100,1000,10000,100000` | range sizes; sizes `>= n` are skipped |
| `--range-queries N` | 10000 | queries per size, the same for every codec |
| `--range-seed S` | 1234 | `mt19937` seed of the shared starts |
| `--range-trials T` | 1 | timed repetitions of all N queries; `range_*_mbs` reports the median |
| `--range-batch-bytes B` | 262144 | output arena per timed batch (see below) |
| `--no-range-verify` | off | skip the untimed check of every query |
| `--range-trials-csv F` | none | one row per (codec, dataset, size, trial) |

The driver's defaults are the paper's setup: 17 sizes `10·2^k` (k = 0…16),
10,000 queries and seed 1234. It also defaults to 3 trials. `--largest 11` picks the 11 largest
`.bin` files: IT, US, ECG, WD, AP, UK, GE, LAT, LON, DP and CT. These are
the paper's 11 largest datasets. Outputs:

- `raw/<dataset>__<codec>.{csv,trials.csv,log}` is the per-case output. A case
  counts as complete only when it has one CSV row and every (size, trial)
  record. Complete cases are skipped on a rerun.
- `range-trials.csv` lists every trial.
- `range-per-dataset.csv` has the median over trials for each (codec, dataset, size).
- `range-summary.csv` has the arithmetic mean over datasets for each (codec, size),
  with the number of datasets.
- `range-throughput.{png,svg}` is a log-log plot of the mean MB/s against range size. A codec
  is plotted only at sizes covered by every dataset.
- `metadata.json` records the configuration, the binary's SHA-256 and the git commit.
- `failures.txt` lists cases that exited non-zero, are incomplete or failed validation.

## Methodology

Every codec adapter calls the same runner, `run_range_queries` in
`benchmark/range_queries.hpp`.

- **Same queries for everyone.** `main` draws the starts once per dataset
  and size: `mt19937(seed)`, uniform over `[0, n - range - 1]`, as in the
  paper's `simd_scan_speed()`. Every codec answers exactly these N starts in
  the same order. Before this change, Gorilla, Chimp, Chimp128, Elf, TSXor,
  Squash, ALP, LeCo, BUFF, Camel and the recent compressors used only the
  first 1,000 starts.
- **What is timed.** Only the queries are timed. Compression, the choice of
  starts and buffer allocation happen outside the timer, and so do output
  preparation and verification. Each query must write all `range` values
  into caller-owned memory. Nothing is cached between queries: no fully
  decoded column and no decoded blocks.
- **Output buffers and preparation.** Each query writes into its own
  `std::vector<T>` slot of length `range`. The slots come from an arena of
  `max(1, B / (range·8))` slots, capped at N, that is allocated once per size
  and reused. For each batch of slots, the runner zero-fills the slots
  (untimed), times the batch's queries, then moves to the next batch.
  Zeroing is required by NeaTS: `simd_scan` adds model predictions into its
  destination and skips the residual unpacking of 0-bit segments, so a
  reused, non-zeroed buffer returns wrong values. Every codec gets the same
  preparation, so no codec pays for or skips work the others don't.
  Batching keeps timer overhead to one clock pair per batch, which matters
  for 10-value queries. The 256 KiB default arena keeps the slots
  cache-resident.
- **Codec-internal work is timed.** This covers decoding whole native blocks
  or vectors, skipping to the start inside a block, and the codec's own
  temporary allocations. Harness scratch buffers, such as Squash's block
  decode buffer, are allocated once, outside timing.
- **Trials.** All N queries are run T times per size. The main CSV reports
  the median trial, and every trial goes to the trials CSV.
- **Correctness.** After the trials, the runner reruns every query untimed
  with the same preparation. Each value must match the codec's own full
  decompression under the shared round-trip rule in `benchmark_common.hpp`:
  bit-identical, or within an absolute difference of 1e-9. Full decompression is checked against the input
  separately by the same rule. For BUFF this means a range is compared with
  the full decode, not with the input. Any mismatch is reported on stderr and in
  `mismatched_queries`, and sets `validation_status=failed`; metrics are
  still emitted. The plot script drops mismatching rows.
- **Units.** The number of bytes counts only the requested output:
  `queries × range × 8`, since every codec returns 8-byte `int64`, `uint64`
  or `double` values. It excludes compressed bytes and the extra values of
  decoded blocks. The trials CSV gives `throughput_MB_per_s` (10^6 bytes/s),
  `throughput_MiB_per_s` and `queries_per_s`, the paper's unit. The legacy
  `range_<size>_mbs` columns of the main CSV are MiB/s, like every other
  `*_mbs` column there.
- **Aggregation.** For each (codec, dataset, size) the scripts take the
  median over trials. For each (codec, size) they take the arithmetic mean
  over datasets; the paper does not say which mean it uses.
- **Environment.** Pin one case at a time (`--taskset`) and build with
  OpenMP disabled (`-DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=TRUE`). With OpenMP,
  GEF's `get_elements` may decode large ranges in parallel. The driver sets
  `OMP_NUM_THREADS=1`.

### What a query does per codec

| codec(s) | query |
|---|---|
| `*_gef` | `get_elements(start, range, out)` on the uniform-partitioned structure |
| `neats` | `simd_scan(start, start+range, out)`, needs zeroed output |
| `dac` | `std::copy` over the SDSL `dac_vector` iterator |
| `alp` | decode every intersected 1024-value vector (FALP and exceptions, or ALP-RD), then copy |
| `leco` | `decodeArray8` of every intersected native partition, then copy |
| `buff` | `buff::get_range` on the single native column |
| `gorilla`, `chimp`, `chimp128`, `elf`, `tsxor` | decode intersected `-b` blocks, streaming up to the start |
| `lz4`, `zstd`, `brotli`, `xz`, `snappy` | Squash-decode intersected `-b` blocks into a preallocated buffer, then copy |
| `camel` | native C++ Camel decoder on intersected `-b` blocks, streaming up to the end of the range |
| `selfstar`, `elfstar`, `dexor`, `fauna`, `macaque`, `pcodec`, `pcodec_i64`, `flea` | decode intersected `-b` blocks (prefix decode when available), then copy |

## Differences from the NeaTS paper's harness

The reference is `simd_scan_speed()` in `main_simd.cpp` of and-gue/NeaTS at
commit 2d804ff.

- **Output allocation.** Their loop allocates a zero-initialized, 64-byte-aligned
  `std::vector` per query inside the timed region, so allocation and
  zeroing are counted for every codec. We exclude both, as described above.
  For fast decoders this is a large effect: on CT at 655,360 values, an
  earlier measurement of ALP dropped from about 11.9 GB/s to 7.5 GB/s with a
  fresh buffer per query.
- **Repetitions and checks.** They run a single pass and their correctness
  check is commented out. We run T trials and check every query.
- **Units.** Their code computes decimal MB/s, but Fig. 4 plots queries/s.
  We plot MB/s and keep `queries_per_s` in the CSVs.
- **ALP layout.** The paper describes wrapping ALP in independent
  1,000-value blocks, and its public repository has no ALP range code. We
  use ALP's native 1,024-value vectors and row groups (see `docs/alp.md`),
  so our ALP curve is not a reproduction of theirs.
- **Starts and query count** are the same: seed 1234, 10,000 queries,
  uniform over `[0, n - range - 1]`.

## Changes to the default benchmark

These changes apply to every codec. They also change the default
`range_*_mbs` numbers.

1. All codecs use the same number of queries (`--range-queries`, default
   10,000; previously 1,000 for many codecs), so slow codecs now take longer.
2. Every query writes into a zero-filled slot that was prepared outside
   timing. Previously each codec reused one hot buffer. NeaTS ranges were
   wrong before this change, because its reused buffer was not zeroed.
3. Every range query is verified against full decompression, untimed, and a
   mismatch fails validation. Previously the checks were partial: LeCo
   checked every query, ALP and BUFF their first 3, and the other codecs
   none.
4. Squash range queries no longer allocate their block buffer inside the
   timed loop.

Starts, seed and the CSV layout are unchanged. `--range-trials 1`, the
default, reproduces the single-pass measurement.
