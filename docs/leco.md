# LeCo

Paper: Liu, Zeng, Zhang, "LeCo: Lightweight Compression via Learning Serial
Correlations", SIGMOD 2024 ([arXiv 2306.15374](https://arxiv.org/abs/2306.15374)).
Source: <https://github.com/yhliu918/Learn-to-Compress>, submodule
`lib/Learn-to-Compress` pinned at `28bf8bf`.

## Variant

We run **LeCo-fix**: linear regressor, fixed-length partitions. When the paper
says just "LeCo", it means LeCo-fix. Figure 2 labels "LeCo" and "LeCo-var"
separately, and §4.5 says "LeCo (i.e., LeCo-fix)". The paper also shows
LeCo-var in §4.3. It compresses better only on some datasets: 28.2% better on
four locally-easy, globally-hard datasets and under 10% better on the rest. Its
random access is slower. We do not run LeCo-var.
The codec is the authors' unmodified `Codecset::Leco_int<uint64_t>` from
`headers/piecewise_fix_integer_template.h`. This is the same class and 64-bit
instantiation that their `experiments/leco_int_fix_template.cpp` runs on the
64-bit datasets (`fb`, `wiki`, `poisson`, `ml_timestamp`). Inputs are the
benchmark's shifted, non-negative int64 values.

## Partition size

The paper (§4.1) says: "For all the fixed-length partitioning methods, the
partition size is obtained through a quick sampling-based parameter search
described in Section 3.2.1." The released microbenchmark
(`benchmark/fix_int_benchmark.py`) does not run that search. Instead it passes
a hand-set partition *count* for each of the paper's datasets (1k, 100k or 1M,
which gives about 200 to 2000 values per partition). That table cannot cover
our datasets.

The only code the authors released for this search is `pick_block_size()` in
`headers/search_hyper.h`. Their `experiments/search_hyper_regress.cpp` calls it
with candidate sizes {200, 400, 800, 1600, 3200, 6400, 10000}, one sample of 1%
of the data, and keeps the candidate with the smallest encoded sample. We call
the same function with the same arguments, using LeCo's own codec, once per
dataset. Its time counts as compression time. `tools/import_leco_tuner.py`
regenerates `benchmark/LeCo/NativeTuning.hpp` from the pinned file. It asserts
each of these textual changes:

- 64-bit: the input pointer is `uint64_t`, and the ratio's denominator is 8
  bytes per value instead of 4. Without the second change, 64-bit samples can
  score above the initial minimum of 1.0, and then no candidate is selected.
- A small adapter replaces `codecfactory.h`, which only offers uint32 codecs,
  with the original uint64 `Leco_int`.
- Upstream bug fix: the sample's last, partial partition was read from
  `data + ...`, not from the sampled offset `data + pick_block*sample_size + ...`.

Two changes are in our caller, not in the authors' code:

- `srand(1)` before each search. Upstream never seeds `rand()` and runs one
  dataset per process. Reseeding gives the same choice regardless of the order
  in which datasets run.
- Fewer than 100 values give an empty 1% sample, which makes upstream assert.
  In that case we take the first candidate.

Below 20,000 values, the sample is shorter than every candidate, so the helper
itself returns 200.

## Storage and queries

Counted size includes:

- the encoded partitions
- one 8-byte offset per partition, plus one extra
- 16 bytes for the value count and partition size
- LeCo's `mul_add_diff_set` correction records, which are always empty in this
  codec version

Point queries call the native `randomdecodeArray8` on one element. Range
queries decode only the native partitions that overlap the range, using
`decodeArray8`. LeCo has no other range API, and we add no extra blocking.
After timing, every timed point query and every timed range query is checked
exactly against the input. The LeCo translation unit does not use fast-math.

The paper describes a fuller search (exponential search, then refinement, until
the result changes by less than 0.01%). The authors released no code for it, so
we do not re-implement it.
