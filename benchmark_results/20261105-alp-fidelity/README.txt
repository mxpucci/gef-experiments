ALP adapter fidelity validation, 2026-11-05

Canonical code: /Users/michelangelopucci/Downloads/gef-experiments.
No published benchmark results were replaced. Tests do not provide performance
measurements; corrected full-dataset runs are still required.

local-validation.csv and sanitized-validation.csv: 28 synthetic cases plus
16 saved real-data prefixes (156,782 values; BP is complete). CSV columns:
fixture, value_count, compressed_bits, validation. No header row.
linux-validation.csv: 28 synthetic cases plus 32 head/tail samples (up to
205,825 values each) covering all 16 datasets, 5,507,968 values in real samples.
BP, BM, BT and BW are each smaller than the limit, so head/tail are duplicates
of their full input. The totals count both copies.

All checks include bitwise full output, point/range checks, reopening counted
metadata, and independently computed byte counts through original encoder calls.

linux-flags-strict.csv and linux-flags-fast.csv compare the original encoder
with IDENTICAL precomputed binary64 samples, up to 205,825 values/dataset.
Columns: fixture, complete_vector_value_count, payload_bits, exceptions,
ALP_RD_vector_count. Headerless. Payload excludes wrapper metadata and tail.
flag-audit.cpp is the diagnostic source. Both variants use -O3 -march=native
-fwrapv with GCC 13.4. Strict adds -fno-fast-math -ffp-contract=off; the other
adds -ffast-math. This experiment isolates codec arithmetic from input division.

Local build: CMake alp_adapter_test and LosslessBenchmark succeeded.
CMake registers the regression test under alp_adapter. For local sanitizers,
clang++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-fast-math
-ffp-contract=off -fwrapv -Ibuild/alp-compat -Ilib/ALP/include
tests/alp_adapter.cpp build/libneats_alp.a -o /tmp/alp-test
UBSAN_OPTIONS=halt_on_error=1 /tmp/alp-test <sample files>
That local check instruments adapter/inline upstream encoder, not the linked
packing library. Without -fwrapv, UBSan identifies overflow in upstream candidate
evaluation. The flag defines wrapping semantics; upstream source is unchanged.

Linux validation was isolated under gef-experiments/build-alp-validation-20261105
on CPU 18. The other session's CPU 8 benchmark was not modified or stopped.

linux-flags-nowrap.csv is byte-identical to linux-flags-strict.csv: adding
-fwrapv did not change payload sizes, exception counts or scheme choices on
these 16 Linux samples. GCC release and Clang sanitizer local CSVs also agree.
