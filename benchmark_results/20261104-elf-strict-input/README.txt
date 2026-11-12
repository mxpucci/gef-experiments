User-authorized rerun: only Elf* and SElf*, all 16 full datasets, 2026-11-04.
The previous remaining-compressor run was stopped after 366 completed cases.
IT Brotli was interrupted; IT NeaTS was not started. Completed results retained.
The old automatic plot-completion watcher was stopped with that run.

Change: move scaled-integer-to-double conversion out of the fast-math benchmark
translation unit into benchmark/benchmark_input.cpp. Compile that file with
-fno-fast-math -ffp-contract=off -fno-lto and disable its CMake IPO property.
The codec implementations, block size (1000), checkpoints, directory accounting,
CPU pinning (2), query counts and range sizes are unchanged.
Original implementations: local C++ ports of authors' Java; NOT original Java.

Validation:
- New bit-pattern regression fails against old code (3/10 vs 3*0.1).
- Corrected function tested from a caller using fast-math and LTO on GCC 13.4.
- All 156,782 values in the 16 real-data prefixes match strict reference bits.
- Local conversion and Elf*/SElf* tests pass; server tests retained in tests.txt.
- Full-run per-case round-trip checks and metrics retained in cases/ and results.txt.

Run is isolated at ~/gef-benchmark-runs/20261104-elf-strict-input on the server.
No original paper plots/tables or preceding benchmark results are overwritten.
compare.py produces comparison.csv, comparison.txt and comparison-summary.json
from results.txt and ../20261103-native-access/results-all.txt.
Other floating-point-codec results using the old conversion remain provisional.

COMPLETED: all 32 full-dataset cases passed bitwise round-trip validation, with
zero warnings or failures. Mean compressed/original size over all 16 datasets:
Elf* 41.627327% -> 22.914623%; SElf* 42.355180% -> 23.796435%.
See audit.txt, comparison.txt and comparison.csv.
The user subsequently authorized all remaining compressors; that run is recorded
in ../20261104-all-strict-input and combines only the two corrected runs.
